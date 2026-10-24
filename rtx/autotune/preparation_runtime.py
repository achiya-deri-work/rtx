"""Cold-only MXFP8 preparation selection; hot runners launch kernels only."""
from dataclasses import replace
from pathlib import Path
import math
import statistics

import torch

from .adapters import make_mxfp8_fwd_adapter, make_mxfp8_prequant_adapter
from .first_hit import FirstHitPolicy
from .legacy import DeviceFingerprint, default_cache_dir
from .outcomes import TrialOutcome, raise_if_fatal_device_context_error
from .preparation import PreparationCandidate, make_preparation_adapter, tune_preparation, expand_preparation_candidates
from .preparation_mxfp8 import numerical_policy, execution_signature, fused_forward_preparation_seeds
from .store import JsonlTuningStore
from .measurement import runtime_timing_metadata


_MEASUREMENT_CONTRACT = {
    "timing": "cuda_graph", "unroll": 8, "replays": 10, "samples": 9,
    "warmup": 3, "graph_warmup_ms": 200., "graph_warmup_min_replays": 10,
    "graph_warmup_max_replays": 4096, "graph_pilot_replays": 2,
    "oracle": "fp32_original_bf16", "nrms_gate": .06,
    "absolute_rms_floor": 1e-6, "revision": 4, "timing_quality_revision": 1,
}


def _preparation_device_profile(device):
    """Stable runtime capacities for retrieval; no clock/power snapshots or SKU guesses."""
    from .hardware import device_properties
    profile = DeviceFingerprint.current(device).as_dict()
    properties = device_properties(device)
    for key in ("l2_cache_size", "memory_bus_width_bits", "shared_memory_per_block_optin",
                "shared_memory_per_multiprocessor", "regs_per_multiprocessor",
                "max_threads_per_multiprocessor", "max_blocks_per_multiprocessor"):
        value = properties.get(key)
        if isinstance(value, int) and value > 0:
            profile[key] = value
    return profile


def _graph_warmup_replays(graph_ms):
    """Bound calibration cost without assuming equal replay cost across shapes."""
    if not math.isfinite(graph_ms) or graph_ms <= 0:
        raise ValueError("graph pilot duration must be positive and finite")
    return min(_MEASUREMENT_CONTRACT["graph_warmup_max_replays"],
               max(_MEASUREMENT_CONTRACT["graph_warmup_min_replays"],
                   math.ceil(_MEASUREMENT_CONTRACT["graph_warmup_ms"] / graph_ms)))


def _warm_measurement_graph(graph):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(_MEASUREMENT_CONTRACT["graph_pilot_replays"]):
        graph.replay()
    end.record()
    end.synchronize()
    graph_ms = start.elapsed_time(end) / _MEASUREMENT_CONTRACT["graph_pilot_replays"]
    replays = _graph_warmup_replays(graph_ms)
    for _ in range(replays):
        graph.replay()
    return {"pilot_graph_ms": graph_ms, "warmup_replays": replays}


def select_forward_runner(x, weight, request):
    # CUDA events/capture queries without an explicit device use the current
    # device, which need not be the input's device in a multi-GPU application.
    with torch.cuda.device(x.device):
        return _select_forward_runner(x, weight, request)


def _select_forward_runner(x, weight, request):
    """Select once per shape/stream via the owning frontend runner cache.

    No new dispatcher ABI is needed: fused and materialized runners implement
    the same (x, weight, out) launch contract. Must never tune during capture.
    """
    from ..fp8 import _build_prequant_runner, _intern_prequant_config
    from ..kernels.mxfp8 import MXFP8Problem
    from ..runtime import load_kernel_symbol

    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("warm preparation selection before CUDA graph capture")
    problem = MXFP8Problem(int(x.shape[0]), int(weight.shape[0]), int(x.shape[1]))
    compile_fused = load_kernel_symbol("mxfp8_fwd", "compile_mxfp8_fwd")

    def build(candidate):
        if candidate.implementation == "materialized":
            return _build_prequant_runner(x, weight, _intern_prequant_config(candidate.config))
        return compile_fused(problem, candidate.config)

    # Lazy allocation: a persistent winner hit must not construct reference
    # tensors, benchmark buffers, or compile discarded alternatives.
    reference = None
    out = None

    def evaluate(name, config):
        nonlocal reference, out
        try:
            with torch.no_grad():
                if reference is None:
                    reference = x.float() @ weight.float().T
                    out = torch.empty((problem.m, problem.n), device=x.device, dtype=x.dtype)
                runner = build(PreparationCandidate(name, config))
                for _ in range(3):
                    runner(x, weight, out)
                torch.cuda.synchronize(x.device)
                error = out.float() - reference
                rms = error.square().mean().sqrt().item()
                rrms = reference.square().mean().sqrt().item()
                if not bool(out.isfinite().all()) or rms > max(.06 * rrms, 1e-6):
                    return TrialOutcome("correctness_error", error="preparation FP32 reference gate failed")
                stream = torch.cuda.Stream(device=x.device)
                stream.wait_stream(torch.cuda.current_stream(x.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        runner(x, weight, out)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(8):
                        runner(x, weight, out)
                warmup = _warm_measurement_graph(graph)
                samples = []
                for _ in range(9):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) / 80)
                return TrialOutcome("ok", median_ms=statistics.median(samples), timings_ms=samples,
                    max_abs_error=error.abs().max().item(), metadata={"rms": rms, "reference_rms": rrms,
                        "graph_warmup": warmup, **runtime_timing_metadata(samples)})
        except Exception as error:
            raise_if_fatal_device_context_error(error)
            return TrialOutcome("runtime_error", error=str(error))

    device = _preparation_device_profile(x.device)
    adapters = {
        "materialized": make_mxfp8_prequant_adapter(problem, lambda c: evaluate("materialized", c),
                                                    initial=request.initial, device=device,
                                                    tags={"measurement": _MEASUREMENT_CONTRACT}),
        "fused": make_mxfp8_fwd_adapter(problem, lambda c: evaluate("fused", c), device=device,
                                        tags={"measurement": _MEASUREMENT_CONTRACT}),
    }
    policy = numerical_policy("materialized", request.initial)
    fused_seeds = fused_forward_preparation_seeds()
    for seed in fused_seeds:
        if (adapters["fused"].rejection(seed) is None
                and numerical_policy("fused", seed) == policy):
            adapters["fused"].initial_config = seed
            break
    adapter = make_preparation_adapter(adapters, numerical_policy=policy, policy_of=numerical_policy,
                                       equivalence_key=execution_signature)
    candidates = [PreparationCandidate(name, child.initial_config) for name, child in adapters.items()]
    candidates.append(PreparationCandidate("materialized", replace(request.initial, quant_launches="separate")))
    candidates.extend(PreparationCandidate("fused", config) for config in fused_seeds)
    candidates = expand_preparation_candidates(adapter, candidates, per_seed=8)
    budget = request.policy
    if budget is not None and any(getattr(budget, field, None) for field in ("prior_artifact", "predictor_artifact")):
        raise ValueError("experimental preparation selection needs a preparation-specific qualified prior")
    result = tune_preparation(adapter,
        JsonlTuningStore(Path(request.cache_dir or default_cache_dir()) / "preparation"),
        candidates, score=lambda _: 0, refine=getattr(budget, "cache_behavior", "reuse") == "refine",
        policy=FirstHitPolicy(max_trials=getattr(budget,"forward_trials",None) or getattr(budget,"max_trials",None) or 8,
                              seconds=getattr(budget,"max_seconds",getattr(budget,"time_budget_s",30.0)),
                              pool_size=getattr(budget,"pool_size",128),
                              confirmation_repeats=getattr(budget,"confirmation_repeats",1)))
    if result.config is None or not math.isfinite(result.median_ms):
        raise RuntimeError("no valid preparation configuration")
    return build(result.config)


def select_backward_config(dy, x, weight, request, phase="backward"):
    with torch.cuda.device(x.device):
        return _select_backward_config(dy, x, weight, request, phase)


def _select_backward_config(dy, x, weight, request, phase="backward"):
    """Cold-only measured selection for paired or partial original-BF16 gradients."""
    from ..fp8_bwd import _build_bwd_runner, _build_partial_bwd_runner
    from ..kernels.mxfp8 import MXFP8Problem
    from ..kernels.mxfp8_bwd import DEFAULT_FUSED_MXFP8_BWD_CONFIG
    from .adapters import make_mxfp8_bwd_adapter
    from .preparation_mxfp8 import backward_plan_config

    if phase not in ("backward", "dx", "dw"):
        raise ValueError("invalid backward preparation phase")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("warm backward preparation selection before graph capture")
    dy, x, weight = dy.contiguous(), x.contiguous(), weight.contiguous()
    problem = MXFP8Problem(int(x.shape[0]), int(weight.shape[0]), int(x.shape[1]))
    names = ("dx", "dw") if phase == "backward" else (phase,)
    refs, outputs = None, None

    def evaluate(config):
        nonlocal refs, outputs
        try:
            with torch.no_grad():
                if refs is None:
                    refs = {name: (dy.float() @ weight.float() if name == "dx"
                                   else dy.float().T @ x.float()) for name in names}
                    outputs = {name: torch.empty_like(x if name == "dx" else weight) for name in names}
                if phase == "backward":
                    runner = _build_bwd_runner(problem, config, x.device)
                    def run():
                        runner(dy, x, weight, outputs["dx"], outputs["dw"])
                else:
                    runner = _build_partial_bwd_runner(problem, config, x.device, phase)
                    def run():
                        runner(dy, weight if phase == "dx" else x, outputs[phase])
                for _ in range(3):
                    run()
                torch.cuda.synchronize(x.device)
                metrics = {}
                for name in names:
                    error = outputs[name].float() - refs[name]
                    rms, rrms = error.square().mean().sqrt().item(), refs[name].square().mean().sqrt().item()
                    metrics[name] = {"rms": rms, "reference_rms": rrms, "max_abs": error.abs().max().item()}
                    if not bool(outputs[name].isfinite().all()) or rms > max(.06 * rrms, 1e-6):
                        return TrialOutcome("correctness_error", error=f"{name} FP32 reference gate failed")
                stream = torch.cuda.Stream(device=x.device)
                stream.wait_stream(torch.cuda.current_stream(x.device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        run()
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    for _ in range(8):
                        run()
                warmup = _warm_measurement_graph(graph)
                samples = []
                for _ in range(9):
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(10):
                        graph.replay()
                    end.record()
                    end.synchronize()
                    samples.append(start.elapsed_time(end) / 80)
                return TrialOutcome("ok", median_ms=statistics.median(samples), timings_ms=samples,
                    max_abs_error=max(v["max_abs"] for v in metrics.values()), metadata={"metrics": metrics,
                        "graph_warmup": warmup, **runtime_timing_metadata(samples)})
        except Exception as error:
            raise_if_fatal_device_context_error(error)
            return TrialOutcome("runtime_error", error=str(error))

    device = _preparation_device_profile(x.device)
    plans = (("staged", "fused_dx", "fused_dw", "fused") if phase == "backward"
             else ("staged", "fused_" + phase))
    adapters = {}
    for name in plans:
        initial = backward_plan_config(name)
        # The fully fused scalar implementation owns logical tails. Preserve
        # it when native-scale or TMA seeds cannot cover a ragged shape.
        if name != "staged" and initial.implementation_rejection(problem) is not None:
            initial = DEFAULT_FUSED_MXFP8_BWD_CONFIG
        adapters[name] = make_mxfp8_bwd_adapter(problem, evaluate, initial=initial,
            device=device, tags={"phase": phase, "measurement": _MEASUREMENT_CONTRACT})
    policy = numerical_policy("staged", adapters["staged"].initial_config)
    adapter = make_preparation_adapter(adapters, numerical_policy=policy, policy_of=numerical_policy,
                                       equivalence_key=lambda c: execution_signature(c, phase))
    seeds = [PreparationCandidate(name, child.initial_config) for name, child in adapters.items()]
    accept = None
    if phase != "backward":
        inactive = "dw" if phase == "dx" else "dx"
        accept = lambda c: getattr(c.config, inactive) == getattr(adapters[c.implementation].initial_config, inactive)
    candidates = expand_preparation_candidates(adapter, seeds, per_seed=8, accept=accept)
    from .preparation_mxfp8 import execution_plan
    budget = request.policy
    if budget is not None and any(getattr(budget, field, None) for field in ("prior_artifact", "predictor_artifact")):
        raise ValueError("experimental preparation selection needs a preparation-specific qualified prior")
    result = tune_preparation(adapter,
        JsonlTuningStore(Path(request.cache_dir or default_cache_dir()) / "preparation"),
        candidates, score=lambda _: 0, refine=getattr(budget,"cache_behavior","reuse") == "refine",
        plan_of=lambda c: execution_plan(c, phase),
        policy=FirstHitPolicy(max_trials=getattr(budget,"backward_trials",None) or getattr(budget,"max_trials",None) or 8,
            seconds=getattr(budget,"max_seconds",getattr(budget,"time_budget_s",30.0)),
            pool_size=getattr(budget,"pool_size",128), confirmation_repeats=getattr(budget,"confirmation_repeats",1)))
    if result.config is None or not math.isfinite(result.median_ms):
        raise RuntimeError("no valid backward preparation configuration")
    return result.config.config
