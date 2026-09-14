"""Isolated nested-replay ownership and external-capture qualification."""
import argparse
import json
import traceback
from pathlib import Path

import torch
import rtx
from rtx.autotune.mlp import reference


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--case",choices=("outstanding","external"),required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    from rtx._mlp_autotune import _register_replay_lowerings,selected_mlp
    _register_replay_lowerings()
    torch.manual_seed(951)
    layer=rtx.MXFP8MLP(128,256,64,device="cuda",autotune="cache",
        autotune_cache_dir=args.output.with_suffix(".cache"))
    x=(torch.randn(64,128,device="cuda",dtype=torch.bfloat16)*.1).requires_grad_()
    dy=torch.randn(64,64,device="cuda",dtype=torch.bfloat16)*.1
    inputs=(x,layer.gate_up_weight,layer.down_weight)
    fn=torch.compile(lambda *v:selected_mlp(*v,"silu",layer._first_hit_spec,True)[0],
        fullgraph=True,dynamic=False,mode="reduce-overhead")
    report=dict(case=args.case,torch=str(torch.__version__),gpu=torch.cuda.get_device_name(),
                passed=False,errors=[])
    def check(actual,expected):
        assert len(actual)==len(expected)==4
        for a,b in zip(actual,expected):
            assert a.isfinite().all()
            error=float((a.detach().float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
            report["errors"].append(error)
            assert error<.025,error
    def run(values):
        y=fn(*values)
        return (y,*torch.autograd.grad(y,values,dy))
    try:
        if args.case=="outstanding":
            for _ in range(4):
                torch.compiler.cudagraph_mark_step_begin()
                run(inputs)
            torch.compiler.cudagraph_mark_step_begin()
            other=(x.detach().clone().add_(.01).requires_grad_(),*inputs[1:])
            with torch.no_grad():
                expected1=reference(*inputs,dy,"silu")
                expected2=reference(*other,dy,"silu")
            y1=fn(*inputs)
            snapshot=y1.detach().clone()
            y2=fn(*other)
            torch.testing.assert_close(y1,snapshot,rtol=0,atol=0)
            g2=torch.autograd.grad(y2,other,dy)
            g1=torch.autograd.grad(y1,inputs,dy)
            check((y1,*g1),expected1)
            check((y2,*g2),expected2)
            del y1,y2,g1,g2,snapshot
            for _ in range(10):
                torch.compiler.cudagraph_mark_step_begin()
                run(inputs)
            live=[]
            for _ in range(4):
                for _ in range(5):
                    torch.compiler.cudagraph_mark_step_begin()
                    run(inputs)
                torch.cuda.synchronize()
                live.append(torch.cuda.memory_allocated())
            report["live_bytes"]=live
            assert max(live)==min(live),live
        else:
            stream=torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                values=tuple(t.detach().requires_grad_() for t in inputs)
                for _ in range(4):
                    torch.compiler.cudagraph_mark_step_begin()
                    run(values)
            stream.synchronize()
            torch.compiler.cudagraph_mark_step_begin()
            graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph,stream=stream):
                actual=run(values)
            with torch.cuda.stream(stream):
                for _ in range(3):graph.replay()
            torch.cuda.current_stream().wait_stream(stream)
            with torch.no_grad():expected=reference(*inputs,dy,"silu")
            check(actual,expected)
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream),torch.no_grad():
                values[0].add_(.001)
                values[2].add_(.0001)
                graph.replay()
            torch.cuda.current_stream().wait_stream(stream)
            with torch.no_grad():expected=reference(*inputs,dy,"silu")
            check(actual,expected)
        torch.cuda.synchronize()
        report["passed"]=True
    except Exception:
        report["error"]=traceback.format_exc()
        raise
    finally:
        with args.output.open("x") as f:json.dump(report,f,indent=2)
        print(json.dumps(report),flush=True)


if __name__=="__main__":main()
