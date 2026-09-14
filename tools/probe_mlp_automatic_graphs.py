"""Probe automatic Inductor CUDA graphs, not manual capture or timing quality."""
import argparse
import json
import traceback
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import torch
import rtx


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--selection",choices=("explicit","first_hit"),required=True)
    p.add_argument("--output",type=Path,required=True)
    p.add_argument("--allocation-history",action="store_true")
    p.add_argument("--nested-replay",action="store_true",help="Compatibility flag; first-hit replay now follows compiler options automatically")
    p.add_argument("--forward-only-replay",action="store_true",help="Diagnostic isolation, not a runtime policy")
    p.add_argument("--shape",default="128,128,128,128",help="M,K,H,O")
    p.add_argument("--activation",choices=("silu","gelu"),default="silu")
    p.add_argument("--phase",choices=("train","forward"),default="train")
    p.add_argument("--mask",choices=("all","x","gu","down"),default="all")
    p.add_argument("--tune",action="store_true",help="Exercise actual eight-trial first-hit tuning")
    args=p.parse_args()
    if args.forward_only_replay and not args.nested_replay:
        p.error("--forward-only-replay requires --nested-replay")
    if args.allocation_history:
        from torch._inductor import config
        config.triton.cudagraph_trees_history_recording=True
        from torch._inductor import cudagraph_trees
        original_check=cudagraph_trees.check_memory_pool
        def diagnose(*values,**kwargs):
            try:return original_check(*values,**kwargs)
            except RuntimeError as exc:
                import gc,re
                pointers={int(v) for v in re.findall(r"Data Pointer: (\d+)",str(exc))}
                for obj in gc.get_objects():
                    if type(obj) not in (torch.Tensor,torch.nn.Parameter):continue
                    try:
                        if not obj.is_cuda or obj.data_ptr() not in pointers:continue
                    except RuntimeError:continue
                    refs=[]
                    for owner in gc.get_referrers(obj):
                        if isinstance(owner,dict):
                            refs.append([str(k) for k,v in owner.items() if v is obj])
                        else:refs.append(type(owner).__name__)
                    print("UNTRACKED_TENSOR",tuple(obj.shape),str(obj.dtype),refs,flush=True)
                raise
        cudagraph_trees.check_memory_pool=diagnose
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.manual_seed(947)
    from rtx import mlp
    if args.nested_replay:
        from rtx import _mlp_autotune
        _mlp_autotune._register_replay_lowerings()
    if args.forward_only_replay:
        from rtx import _mlp_autotune
        original_backward=_mlp_autotune._selected_backward
        def without_replay(*values,**kwargs):
            kwargs["replay"]=False
            return original_backward(*values,**kwargs)
        _mlp_autotune._selected_backward=without_replay
    from torch._dynamo.utils import counters
    from rtx.autotune.mlp import reference
    m,k,h,o=map(int,args.shape.split(","))
    mask=tuple(args.phase=="train" and args.mask in ("all",name) for name in ("x","gu","down"))
    policy=rtx.AutotunePolicy(max_trials=8,max_seconds=90,search_schedule="progressive",
        cache_behavior="refine",finalist_rounds=2,confirmation_repeats=0,warmup=1,samples=3,calls_per_sample=2)
    layer=rtx.MXFP8MLP(k,h,o,device="cuda",activation=args.activation,
        autotune="off" if args.selection=="explicit" else ("balanced" if args.tune else "cache"),
        tuning_policy=policy,autotune_cache_dir=args.output.with_suffix(".cache"))
    layer.gate_up_weight.requires_grad_(mask[1]);layer.down_weight.requires_grad_(mask[2])
    x=(torch.randn(m,k,device="cuda",dtype=torch.bfloat16)*.1).requires_grad_(mask[0])
    dy=torch.randn(m,o,device="cuda",dtype=torch.bfloat16)*.1
    inputs=x,layer.gate_up_weight,layer.down_weight
    requested=tuple(t for t in inputs if t.requires_grad)
    def oracle():
        with torch.no_grad():
            y,*grads=reference(*inputs,dy,args.activation)
        return (y,*(g for g,need in zip(grads,mask) if need))
    expected=oracle()
    compiled=torch.compile(layer,mode="reduce-overhead",fullgraph=True,dynamic=False)
    calls=[];backward_calls=[];steps=[]
    original=mlp._run_forward
    def forward(*values,**kwargs):
        calls.append(len(calls))
        return original(*values,**kwargs)
    report=dict(selection=args.selection,torch=str(torch.__version__),cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),shape=[m,k,h,o],seed=947,activation=args.activation,
        phase=args.phase,mask=args.mask,tune=args.tune,
        steps=steps,data_update_steps=[3,6],nested_replay=args.selection=="first_hit",
        legacy_nested_replay_flag=args.nested_replay,
        forward_only_replay=args.forward_only_replay,passed=False,
        scope="automatic reduce-overhead; no manual graph")
    try:
        # A MagicMock side_effect retains all call arguments, including graph
        # pool output buffers, and invalidates the ownership experiment.
        with ExitStack() as stack:
            stack.enter_context(patch("rtx.mlp._run_forward",new=forward))
            stack.enter_context(torch.set_grad_enabled(args.phase=="train"))
            from rtx import fp8_bwd
            for name in ("_launch_bwd_out","_launch_dx_out","_launch_dw_out"):
                original_launch=getattr(fp8_bwd,name)
                def counted(*values,_launch=original_launch,**kwargs):
                    backward_calls.append(1)
                    return _launch(*values,**kwargs)
                stack.enter_context(patch.object(fp8_bwd,name,new=counted))
            for step in range(8):
                torch.compiler.cudagraph_mark_step_begin()
                if step in (3,6):
                    with torch.no_grad():
                        x.add_(.002)
                        layer.down_weight.add_(.0001)
                        expected=oracle()
                before=len(calls);before_backward=len(backward_calls)
                y=compiled(x)
                grads=torch.autograd.grad(y,requested,dy) if requested else ()
                errors=[]
                for a,b in zip((y,*grads),expected):
                    assert a.isfinite().all()
                    error=float((a.detach().float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
                    assert error<.025,error
                    errors.append(error)
                del a,y,grads
                torch.cuda.synchronize()
                steps.append(dict(step=step,forward_launcher_calls=len(calls)-before,
                    backward_launcher_calls=len(backward_calls)-before_backward,relative_l2=errors))
                print(json.dumps(steps[-1]),flush=True)
        report["passed"]=True
    except Exception:
        report["error"]=traceback.format_exc()
        raise
    finally:
        report["inductor_counters"]=dict(counters["inductor"])
        with args.output.open("x") as f:json.dump(report,f,indent=2,default=str)


if __name__=="__main__":main()
