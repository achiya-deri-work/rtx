"""GLU adapter for the shared persistent bandit/cost-model/local-search engine."""
from dataclasses import asdict,replace
import statistics
import time
from pathlib import Path

import torch

from .core import DiscreteKernelAdapter, KernelContext
from .legacy import DeviceFingerprint
from .outcomes import TrialOutcome, raise_if_fatal_device_context_error
from .recipes import HybridTuningPolicy, make_hybrid_autotuner
from .store import ResidualTuningStore
from ..glu import DEFAULT_GLU_CONFIG, _validate
from ..configs.glu import (normalize_glu_config, glu_config_from_dict,
                          MXFP8GLUPrequantConfig, prequant_glu_config_from_dict)
from ..kernels.mxfp8 import (
    FWD_SEARCH_SPACE, MXFP8Problem, normalize_fwd_config,
    fwd_config_id, fwd_config_to_dict, fwd_config_from_dict,
)

GLU_KERNEL_REVISION = 6
# Inherit all projection coordinates, including compound TMA/role/reuse moves.
# GLU has its own output schedule rather than pretending a plain GEMM store
# writes the same tensor. Unsupported transport combinations remain rejected.
GLU_SEARCH_SPACE = {key: values for key,values in FWD_SEARCH_SPACE.items()
                    if key not in ("epilogue", "epilogue_stages", "store_vec")}
GLU_SEARCH_SPACE["glu_epilogue"] = ("exchange", "registers")
GLU_SEARCH_SPACE["glu_store"] = ("direct", "tma")
GLU_SEARCH_SPACE["glu_store_stages"] = (1,2,3,4)
GLU_SEARCH_SPACE["glu_quant_vec"] = (1,2,4,8,16,32)
GLU_SEARCH_SPACE["glu_scratch"] = ("dedicated", "reuse_operands")
GLU_SEARCH_SPACE["glu_scratch_swizzle"] = ("none","xor_16b")
GLU_SEARCH_SPACE["glu_silu_math"] = ("division","shared_reciprocal","approx_reciprocal")
GLU_SEARCH_SPACE["glu_quant_epilogue"] = ("exchange", "fragment_registers")
GLU_SEARCH_SPACE["glu_tile"] = tuple(
    (m,n,am) for m in (32,64,128,256) for n in (128,256)
    for am in ((2,4,8) if m==128 else (8,) if m==256 else (2,)))


def _reference(x, weight, activation):
    # Independent PyTorch oracle: no RTX quantizer or GEMM is called.
    def dq(t):
        k = t.shape[-1]
        blocks = torch.nn.functional.pad(t.float(), (0, (-k) % 32)).reshape(t.shape[0], -1, 32)
        amax = blocks.abs().amax(-1, keepdim=True)
        bits = amax.view(torch.int32)
        exponent = ((bits >> 23) & 255) - 127 - 8
        exponent += (bits & 0x7fffff) > 0x600000
        scale = torch.exp2(exponent.clamp(-126, 127).float())
        return ((blocks / scale).clamp(-448,448).to(torch.float8_e4m3fn).float() * scale).flatten(1)
    gate, up = (dq(x) @ dq(weight).T).chunk(2, -1)
    return getattr(torch.nn.functional, activation)(gate) * up


def make_glu_adapter(x, weight, *, activation="silu", output_format="bf16",
                      warmup=5, samples=7, calls_per_sample=20, backend="cta"):
    """Bind actual operands and a separate numerical-policy tuning context.

    Timings cover the fused device kernel, excluding Python allocation and
    compilation. Each successful candidate passes an independent reference.
    """
    _validate(x, weight, activation, output_format)
    if backend not in ("cta","prequant"):
        raise ValueError("backend must be cta or prequant")
    from ..runtime import validate_runtime_environment
    validate_runtime_environment()
    if torch.cuda.get_device_capability(x.device)[0] != 12:
        raise RuntimeError("MXFP8 GLU requires SM12x")
    if min(warmup, samples, calls_per_sample) < 1:
        raise ValueError("benchmark counts must be positive")
    from ..kernels.mxfp8_glu import compile_mxfp8_glu, glu_rejection
    x = x.detach().view(-1, x.shape[-1])
    weight = weight.detach()
    problem = MXFP8Problem(x.shape[0], weight.shape[0], x.shape[1])
    h = problem.n // 2
    width = h if output_format == "bf16" else (h+31)//32*32
    expected = _reference(x, weight, activation)
    out = torch.empty((problem.m,width), device=x.device, dtype=torch.bfloat16 if output_format=="bf16" else torch.float8_e4m3fn)
    scales = torch.empty((problem.m,0 if output_format == "bf16" else (h+31)//32), device=x.device, dtype=torch.uint8)

    def evaluate(config):
        start = time.monotonic()
        try:
            if backend == "prequant":
                from ..kernels.mxfp8_glu_prequant import build_prequant_glu_runner
                launch = build_prequant_glu_runner(x,weight,out,scales,config,activation,output_format)
            else:
                launcher = compile_mxfp8_glu(problem, config, activation, output_format)
                def launch():
                    auxiliary = out if output_format == "bf16" else scales
                    launcher(x,weight,out,auxiliary,auxiliary,auxiliary,auxiliary)
        except Exception as exc:
            raise_if_fatal_device_context_error(exc)
            return TrialOutcome("compile_error", compile_ms=(time.monotonic()-start)*1000, error=str(exc))
        compile_ms = (time.monotonic()-start)*1000
        launch()
        actual = out.float()
        if output_format == "mxfp8":
            actual *= scales.view(torch.float8_e8m0fnu).float().repeat_interleave(32,-1)
        actual = actual[:, :h]
        error = (actual-expected).norm()/expected.norm().clamp_min(1e-20)
        limit = 0.004 if output_format == "bf16" else 0.04
        if not actual.isfinite().all() or error.item() > limit:
            return TrialOutcome("correctness_error", compile_ms=compile_ms, error=f"relative L2 {error.item():.6g} > {limit}")
        for _ in range(warmup):
            launch()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(calls_per_sample):
                launch()
        timings = []
        for _ in range(samples):
            start_event, end_event = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start_event.record()
            graph.replay()
            end_event.record()
            end_event.synchronize()
            timings.append(start_event.elapsed_time(end_event)/calls_per_sample)
        return TrialOutcome("ok", median_ms=statistics.median(timings), timings_ms=timings,
                            compile_ms=compile_ms, metadata={"relative_l2":error.item(), "calls_per_sample":calls_per_sample})

    def reject(config):
        reason = config.rejection(problem,output_format) if backend=="prequant" else glu_rejection(problem, config, output_format)
        return None if reason is None else ("implementation_rejected",reason)
    axes = GLU_SEARCH_SPACE
    def update(c,k,v):
        return normalize_glu_config(c, **{k:v})
    if backend == "prequant":
        from ..prequant_autotune import PREQUANT_SEARCH_SPACE
        axes = {k:v for k,v in PREQUANT_SEARCH_SPACE.items()
                if k not in ("epilogue", "global_l2_fetch")}
        axes["glu_epilogue"] = ({"gemm":{"glu_epilogue":"exchange"}}, {"gemm":{"glu_epilogue":"registers"}})
        axes["glu_quant_vec"] = tuple({"gemm":{"glu_quant_vec":v}} for v in (1,2,4,8))
        axes["glu_reduction"] = tuple({"gemm":{"reduction":v}} for v in ("shuffle","redux"))
        axes["glu_store"] = tuple({"gemm":{"glu_store":store,"glu_store_stages":stages}}
                                   for store,stages in (("direct",1),*(('tma',s) for s in (1,2,3,4))))
        def update(c,k,v):
            values = asdict(c)
            for key,value in v.items():
                if isinstance(value,dict):
                    value = dict(value)
                    if key=="gemm" and value.get("epilogue")=="tma":
                        value.pop("epilogue")
                        value.pop("store_vec",None)
                        value["glu_store"] = "tma"
                        value["glu_store_stages"] = value.pop("epilogue_stages",1)
                    old = values.get(key)
                    if old is None and key=="weight_quant":
                        old = asdict(c.resolved_weight_quant())
                    values[key] = {**(old or {}),**value}
                else:
                    values[key] = value
            return prequant_glu_config_from_dict(values)
    return DiscreteKernelAdapter(
        context=KernelContext("mxfp8_glu_prequant_fwd" if backend=="prequant" else "mxfp8_glu_fwd", GLU_KERNEL_REVISION,
                              {**asdict(problem), "activation":activation, "output_format":output_format},
                              device=DeviceFingerprint.current(x.device).as_dict(),
                              tags={"warmup":warmup,"samples":samples,"calls_per_sample":calls_per_sample}),
        initial_config=MXFP8GLUPrequantConfig() if backend=="prequant" else DEFAULT_GLU_CONFIG, axes=axes,
        config_id_fn=fwd_config_id, serialize_fn=fwd_config_to_dict,
        deserialize_fn=prequant_glu_config_from_dict if backend=="prequant" else glu_config_from_dict,
        update_fn=update,
        evaluator=evaluate, rejection_fn=reject,
    )


def tune_mxfp8_glu(x, weight, *, activation="silu", output_format="bf16",
                    output_dir="autotune_results/glu", max_trials=32,
                    time_budget_s=120, seed=0, progress=print, backend="auto", seed_history=None,tuning_policy=None):
    """Run/resume a bounded bandit portfolio; return its persistent winner.

    Explicitly call before torch.compile, then pass ``result.config`` to GLU.
    All evaluations are saved by the standard TuningStore. This experimental
    family does not share cache entries with ordinary linear projections.
    """
    tuning_started=time.monotonic()
    from .effort import AutotunePolicy
    if tuning_policy is not None:
        if not isinstance(tuning_policy,AutotunePolicy):raise TypeError('tuning_policy must be AutotunePolicy')
        tuning_policy=tuning_policy.for_pass()
        max_trials,time_budget_s,seed=tuning_policy.max_trials,tuning_policy.max_seconds,tuning_policy.seed
    if backend=="auto":
        if max_trials < 2:
            raise ValueError("auto backend search needs at least two trials")
        started=time.monotonic()
        results=[]
        for index,family in enumerate(("cta","prequant")):
            remaining=max(0.001,time_budget_s-(time.monotonic()-started))
            results.append(tune_mxfp8_glu(
                x,weight,activation=activation,output_format=output_format,output_dir=output_dir,
                max_trials=max_trials//2+(max_trials%2 if index else 0),
                time_budget_s=remaining/(2-index),seed=seed,progress=progress,
                backend=family,seed_history=seed_history,
                tuning_policy=None if tuning_policy is None else replace(tuning_policy,
                    max_trials=max_trials//2+(max_trials%2 if index else 0),
                    max_seconds=remaining/(2-index),forward_trials=None)))
        return min(results,key=lambda result:result.median_ms)
    with torch.cuda.device(x.device):
        measurement = {} if tuning_policy is None else {
            "warmup": tuning_policy.warmup,
            "samples": tuning_policy.samples,
            "calls_per_sample": tuning_policy.calls_per_sample,
        }
        adapter = make_glu_adapter(x, weight, activation=activation, output_format=output_format,
                                   backend=backend, **measurement)
        if seed_history is not None:
            import json
            candidates=[]
            for path in Path(seed_history).rglob("observations.jsonl"):
                for line in path.read_text().splitlines():
                    try:
                        record=json.loads(line)
                        features=record["features"]
                        if (record["family"]==adapter.context.family and record["outcome"]["status"]=="ok"
                            and all(features.get(f"context.workload.{key}")==value for key,value in adapter.context.workload.items() if isinstance(value,int))
                            and features.get(f"context.workload.activation={activation}")==1
                            and features.get(f"context.workload.output_format={output_format}")==1):
                            candidate=adapter.deserialize(record["config"])
                            if adapter.rejection(candidate) is None:
                                candidates.append((record["outcome"]["median_ms"],candidate))
                    except (ValueError,KeyError,TypeError):
                        continue
            if candidates:
                adapter.initial_config=min(candidates,key=lambda item:item[0])[1]
                if progress: progress("SEED prior winner; timing discarded and candidate revalidated")
        policy = HybridTuningPolicy(orchestration="bandit", max_trials=max_trials,
                                    time_budget_s=time_budget_s, seed=seed,
                                    confirmation_repeats=2, confirm_initial=True,
                                    use_pretrained=False, use_pairwise=False)
        store = ResidualTuningStore(Path(output_dir) / adapter.context.identifier,
                                    transfer_root=output_dir)
        return make_hybrid_autotuner(adapter, store,tuning_policy or policy, progress=progress,started_at=tuning_started).tune()


def main():
    import argparse
    import json
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=512)
    # Do not use --h: CuTe's diagnostic parser treats it as --help.
    parser.add_argument("--hidden", type=int, default=1536)
    parser.add_argument("--k", type=int, default=768)
    parser.add_argument("--activation", choices=("silu","gelu","relu"), default="silu")
    parser.add_argument("--output-format", choices=("bf16","mxfp8"), default="bf16")
    parser.add_argument("--trials", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=120)
    parser.add_argument("--output-dir", default="autotune_results/glu")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--backend", choices=("auto","cta","prequant"), default="auto")
    parser.add_argument("--seed-history")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    with torch.cuda.device(args.device):
        x = torch.randn(args.m,args.k,device=args.device,dtype=torch.bfloat16)
        w = torch.randn(2*args.hidden,args.k,device=args.device,dtype=torch.bfloat16)*0.02
        result = tune_mxfp8_glu(x,w,activation=args.activation,output_format=args.output_format,
                               max_trials=args.trials,time_budget_s=args.seconds,
                               output_dir=args.output_dir,seed=args.seed,backend=args.backend,
                               seed_history=args.seed_history,
                               progress=lambda text:print(text,flush=True))
    print(json.dumps(asdict(result), indent=2, default=str))


if __name__ == "__main__":
    main()
