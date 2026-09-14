"""Fresh-process copy/direct bridge comparison; fixed config, not tuning quality.

Run once per --bridge value with distinct --output files. Compilation, warmup,
and capture are excluded from CUDA-event replay samples. The copy branch
reconstructs the previous private bridge without modifying production sources.
"""
import argparse
import json
from pathlib import Path
import statistics
import sys

import torch
import rtx


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--bridge",choices=("copy","direct"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--shape",default="64,128,256,64",help="M,K,H,O")
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    m,k,h,o=map(int,args.shape.split(","))
    torch.manual_seed(935)
    from rtx import _mlp_autotune as impl
    if args.bridge=="copy":
        from rtx.mlp import _backward,_backward_keys,_parse
        old={}
        def factory(key,activation,mask,shape,signature):
            identity=key,activation,mask,shape,signature
            if identity not in old:
                config=_parse(key);keys=_backward_keys(key,*shape)
                def execute(x,gu,down,dy,ox,og,od):
                    ctx=impl._BackwardContext((x,gu,down),activation,key,keys,
                        config.backward_recompute=="joint",config.backward_overlap,mask)
                    for out,grad in zip((ox,og,od),_backward(ctx,dy)[:3]):
                        if out is not None:out.copy_(grad)
                execute.__code__=execute.__code__.replace()
                compiled=torch.compile(execute,fullgraph=True,dynamic=False)
                def run(x,gu,down,dy):
                    outputs=tuple(torch.empty_like(t) if need else None for t,need in zip((x,gu,down),mask))
                    compiled(x,gu,down,dy,*outputs)
                    return outputs
                old[identity]=run
            return old[identity]
        impl._backward_runner=factory
    layer=rtx.MXFP8MLP(k,h,o,device="cuda",autotune="cache",autotune_cache_dir=args.output.with_suffix(".cache"))
    x=(torch.randn(m,k,device="cuda",dtype=torch.bfloat16)*.1).requires_grad_()
    dy=torch.randn(m,o,device="cuda",dtype=torch.bfloat16)*.1
    inputs=x,layer.gate_up_weight,layer.down_weight
    fn=torch.compile(layer,fullgraph=True,dynamic=False)
    def run():
        y=fn(x)
        return y,*torch.autograd.grad(y,inputs,dy)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):run()
    stream.synchronize()
    with torch.cuda.stream(stream):
        actual=run()
    stream.synchronize()
    from rtx.autotune.mlp import reference
    with torch.no_grad():expected=reference(*inputs,dy,"silu")
    errors=[]
    for a,b in zip(actual,expected):
        assert a.isfinite().all()
        delta=a.detach().float()-b.float()
        error=float(delta.norm()/b.float().norm().clamp_min(1e-12))
        assert error<.025
        errors.append(dict(relative_l2=error,max_abs=float(delta.abs().max())))
    del actual
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):held=run()
    samples=[]
    with torch.cuda.stream(stream):
        for _ in range(5):graph.replay()
        for _ in range(30):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(10):graph.replay()
            end.record();end.synchronize()
            samples.append(start.elapsed_time(end)/10)
    result=dict(bridge=args.bridge,shape=[m,k,h,o],seed=935,torch=str(torch.__version__),cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),config=layer.config.to_dict(),samples_ms=samples,
        median_ms=statistics.median(samples),min_ms=min(samples),stdev_ms=statistics.stdev(samples),
        numerical_errors=errors,scope="fixed-default-config forward+backward graph replay; no startup or autotuning",
        command=sys.argv)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as f:json.dump(result,f,indent=2)
    print(json.dumps({k:result[k] for k in ("bridge","shape","median_ms","min_ms","stdev_ms")}),flush=True)


if __name__=="__main__":main()
