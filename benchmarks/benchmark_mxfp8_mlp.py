"""Matched aggressive-compile BF16 versus strict recomputed MXFP8 MLP."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import statistics

import torch
from torch.utils.checkpoint import checkpoint
from benchmarks.benchmark_torchao_fp8_rowwise import COMPILE_OPTIONS
from rtx.mlp import DEFAULT_MLP_CONFIG,MLP_CONFIG_REVISION,MXFP8MLPConfig,_parse,mxfp8_mlp
from rtx.autotune.mlp import reference


def measure(fn,inputs,dy,training):
    def run():
        if training:
            y=fn(*inputs)
            return y,torch.autograd.grad(y,inputs,dy)
        with torch.no_grad():
            return fn(*inputs)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(5):run()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        for _ in range(3):result=run()
    graph.replay();torch.cuda.synchronize()
    times=[]
    for _ in range(15):
        a,b=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        a.record();graph.replay();b.record();b.synchronize()
        times.append(a.elapsed_time(b)*1000/3)
    return {"median_us":statistics.median(times),"min_us":min(times),"samples_us":times}


def saved_state(fn, inputs):
    """Observe compiled autograd retention, without retaining extra tensors."""
    original = {t.untyped_storage().data_ptr() for t in inputs}
    saved = []
    extra = {}

    def pack(t):
        pointer = t.untyped_storage().data_ptr()
        saved.append({"shape":list(t.shape),"dtype":str(t.dtype),
                      "input_storage":pointer in original,"numel":t.numel()})
        if pointer not in original and t.numel():
            extra[pointer] = t.untyped_storage().nbytes()
        return t

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t:t):
        y = fn(*inputs)
    del y
    return {"tensors":saved,"non_input_saved_storage_bytes":sum(extra.values())}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape",default="128,128,256,128",help="M,K,H,O")
    p.add_argument("--activation",choices=("silu","gelu","relu"),default="silu")
    policy=p.add_mutually_exclusive_group()
    policy.add_argument("--history")
    policy.add_argument("--config-json",help="Explicit revisioned config or device/shape-scoped benchmark preset")
    p.add_argument("--history-phase",choices=("forward","train"),default="train")
    p.add_argument("--profile-forward",action="store_true",help="Warm then expose one candidate forward through cudaProfilerStart/Stop")
    p.add_argument("--profile-train",action="store_true",help="Expose one warmed candidate forward/backward through cudaProfilerStart/Stop")
    p.add_argument("--output",default="autotune_reports/mlp/baseline.json")
    args=p.parse_args();m,k,h,o=map(int,args.shape.split(","));torch.manual_seed(0)
    config=DEFAULT_MLP_CONFIG
    if args.config_json:
        payload=json.loads(Path(args.config_json).read_text())
        if "config" in payload:
            if payload.get("shape_m_k_h_o")!=[m,k,h,o] or payload.get("activation")!=args.activation:
                raise ValueError("benchmark preset does not match shape/activation")
            if payload.get("device")!=torch.cuda.get_device_name():
                raise ValueError("benchmark preset does not match this GPU; autotune locally instead")
            payload=payload["config"]
        config=MXFP8MLPConfig.from_dict(payload)
    if args.history:
        rows=[json.loads(line) for path in Path(args.history).rglob("observations.jsonl") for line in path.read_text().splitlines()]
        rows=[r for r in rows if r["outcome"]["status"]=="ok" and
              all(r["features"].get(f"context.workload.{key}")==value for key,value in {"m":m,"k":k,"h":h,"o":o}.items())
              and r["features"].get(f"context.workload.activation={args.activation}")==1
              and r["features"].get(f"context.workload.phase={args.history_phase}")==1]
        if not rows:raise ValueError("no matching verified MLP observations")
        winner=min(rows,key=lambda r:r["outcome"]["median_ms"])
        config=_parse(json.dumps({"revision":MLP_CONFIG_REVISION,**winner["config"]},sort_keys=True))
    inputs=tuple((torch.randn(*shape,device="cuda",dtype=torch.bfloat16)*scale).requires_grad_()
                 for shape,scale in (((m,k),.2),((2*h,k),.02),((o,h),.02)))
    dy=torch.randn(m,o,device="cuda",dtype=torch.bfloat16)*.1
    def bf16(x,gu,down):
        g,u=torch.nn.functional.linear(x,gu).chunk(2,-1)
        return torch.nn.functional.linear(getattr(torch.nn.functional,args.activation)(g)*u,down)
    baseline=torch.compile(bf16,fullgraph=True,dynamic=False,options=COMPILE_OPTIONS)
    def recomputed_bf16(x,gu,down):
        return checkpoint(bf16,x,gu,down,use_reentrant=True,preserve_rng_state=False)
    checkpointed=torch.compile(recomputed_bf16,fullgraph=True,dynamic=False,options=COMPILE_OPTIONS)
    candidate=torch.compile(lambda x,gu,w:mxfp8_mlp(x,gu,w,activation=args.activation,config=config),
                            fullgraph=True,dynamic=False,options=COMPILE_OPTIONS)
    with torch.no_grad():
        expected=reference(*inputs,dy,args.activation)
    y=candidate(*inputs)
    actual=(y,*torch.autograd.grad(y,inputs,dy))
    errors=[((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-20)).item()
            for a,b in zip(actual,expected)]
    if any(not a.isfinite().all() for a in actual) or max(errors)>.025:
        raise RuntimeError(f"candidate failed current-revision numerical gate: {errors}")
    del expected,actual,y
    if args.profile_train:
        for _ in range(5):
            value=candidate(*inputs)
            gradients=torch.autograd.grad(value,inputs,dy)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        with torch.cuda.nvtx.range("mlp/forward"):
            value=candidate(*inputs)
        with torch.cuda.nvtx.range("mlp/backward"):
            gradients=torch.autograd.grad(value,inputs,dy)
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return
    if args.profile_forward:
        with torch.no_grad():
            for _ in range(5):candidate(*inputs)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            with torch.cuda.nvtx.range("mlp/forward"):
                result=candidate(*inputs)
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
        return
    report={"shape_m_k_h_o":[m,k,h,o],"device":torch.cuda.get_device_name(),"torch":str(torch.__version__),
            "cuda":torch.version.cuda,"capability":torch.cuda.get_device_capability(),
            "mlp_config_revision":MLP_CONFIG_REVISION,
            "activation":args.activation,"config":asdict(config),"compile_options":COMPILE_OPTIONS,
            "timing":"CUDA graph replay; compilation and capture excluded", "forward_kernel_contract":2,
            "reference_relative_l2_y_dx_dgu_ddown":errors,
            "baseline_policy":"BF16 standard autograd, without activation recomputation",
            "checkpoint_policy":"BF16 reentrant whole-MLP checkpoint, compiled fullgraph"}
    baseline_y=baseline(*inputs)
    baseline_grads=torch.autograd.grad(baseline_y,inputs,dy)
    checkpoint_y=checkpointed(*inputs)
    checkpoint_grads=torch.autograd.grad(checkpoint_y,inputs,dy)
    checkpoint_errors=[]
    for a,b in zip((checkpoint_y,*checkpoint_grads),(baseline_y,*baseline_grads)):
        error=((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-20)).item()
        checkpoint_errors.append(error)
        if not a.isfinite().all() or error>.015:
            raise RuntimeError(f"checkpointed BF16 disagrees with baseline: {checkpoint_errors}")
    report["checkpoint_vs_bf16_relative_l2_y_dx_dgu_ddown"]=checkpoint_errors
    del baseline_y,baseline_grads,checkpoint_y,checkpoint_grads
    report["saved_state"]={name:saved_state(fn,inputs) for name,fn in
                           (("baseline",baseline),("checkpointed",checkpointed),("candidate",candidate))}
    for name in ("checkpointed","candidate"):
        if report["saved_state"][name]["non_input_saved_storage_bytes"]:
            raise RuntimeError(f"{name} retained hidden state; cannot claim equal retention")
    for phase in ("forward","train"):
        report[phase]={"baseline":measure(baseline,inputs,dy,phase=="train"),
                       "checkpointed":measure(checkpointed,inputs,dy,phase=="train"),
                       "candidate":measure(candidate,inputs,dy,phase=="train")}
        report[phase]["speedup"]=report[phase]["baseline"]["median_us"]/report[phase]["candidate"]["median_us"]
        report[phase]["speedup_vs_checkpointed"]=report[phase]["checkpointed"]["median_us"]/report[phase]["candidate"]["median_us"]
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))


if __name__=="__main__":main()
