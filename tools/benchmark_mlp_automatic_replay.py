"""Fresh-process first-hit bridge timing: fixed config, no search or manual graph."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import rtx
from rtx.autotune.mlp import reference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode",choices=("plain","replay"),required=True)
    parser.add_argument("--shape",default="512,768,1536,768")
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    from rtx._mlp_autotune import _register_replay_lowerings
    _register_replay_lowerings()
    m,k,h,o=map(int,args.shape.split(","))
    torch.manual_seed(953)
    layer=rtx.MXFP8MLP(k,h,o,device="cuda",autotune="cache",
        autotune_cache_dir=args.output.with_suffix(".cache"))
    x=(torch.randn(m,k,device="cuda",dtype=torch.bfloat16)*.1).requires_grad_()
    dy=torch.randn(m,o,device="cuda",dtype=torch.bfloat16)*.1
    inputs=(x,layer.gate_up_weight,layer.down_weight)
    with torch.no_grad():expected=reference(*inputs,dy,"silu")
    fn=torch.compile(layer,fullgraph=True,dynamic=False,
        options={"triton.cudagraphs":args.mode=="replay"})
    def run():
        torch.compiler.cudagraph_mark_step_begin()
        y=fn(x)
        return (y,*torch.autograd.grad(y,inputs,dy))
    torch.cuda.synchronize()
    t=time.perf_counter()
    actual=run()
    torch.cuda.synchronize()
    cold=time.perf_counter()-t
    errors=[]
    for a,b in zip(actual,expected,strict=True):
        assert a.isfinite().all()
        error=float((a.detach().float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
        assert error<.025,error
        errors.append(error)
    del actual,a
    for _ in range(8):run()
    torch.cuda.synchronize()
    from torch._dynamo.utils import counters
    before=dict(counters["stats"])
    torch.cuda.reset_peak_memory_stats()
    samples=[];walls=[]
    for _ in range(30):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        t=time.perf_counter()
        start.record()
        for _ in range(5):
            actual=run()
            del actual
        end.record();end.synchronize()
        walls.append((time.perf_counter()-t)*1000/5)
        samples.append(start.elapsed_time(end)/5)
    assert dict(counters["stats"])==before,"warm execution recompiled"
    report=dict(mode=args.mode,shape=[m,k,h,o],torch=str(torch.__version__),cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),seed=953,command=sys.argv,config=layer.config.to_dict(),
        cold_first_fwd_bwd_seconds=cold,samples_ms=samples,wall_samples_ms=walls,
        median_ms=statistics.median(samples),min_ms=min(samples),stdev_ms=statistics.stdev(samples),
        wall_median_ms=statistics.median(walls),relative_l2=errors,compiler_stats=before,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),live_bytes=torch.cuda.memory_allocated(),
        scope="fixed default config; CUDA event end-to-end host-fed training, not isolated kernel time; no search")
    with args.output.open("x") as f:json.dump(report,f,indent=2)
    print(json.dumps({k:report[k] for k in ("mode","shape","cold_first_fwd_bwd_seconds","median_ms","wall_median_ms")}),flush=True)


if __name__=="__main__":main()
