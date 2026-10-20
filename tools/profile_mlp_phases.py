"""Narrow warmed selected-direct MLP timeline; run under Nsight Systems.

Profiling annotations do not change precision, selected config or stream policy.
Ordinary graph timing lives in the replacement benchmark, not this trace.
"""
import argparse
from functools import wraps
import hashlib
import json
from pathlib import Path

import torch
import rtx.mlp as mlp
import rtx.fp8_bwd as bwd
from rtx._mlp_autotune import selected_mlp
from rtx.autotune.mlp import reference


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--field',default='challenger')
    parser.add_argument('--shape',default='11712,256,768,256')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--host-profile',type=Path,
                        help='Optional warmed main-thread cProfile artifact; diagnostic, not timing')
    args=parser.parse_args()
    if args.output.exists(): raise FileExistsError(args.output)
    if args.host_profile is not None and args.host_profile.exists():
        raise FileExistsError(args.host_profile)
    data=json.loads(args.config.read_text())
    config=mlp.MXFP8MLPConfig.from_dict(data[args.field])
    spec=json.dumps(dict(mode='fixed',initial=config.to_dict()),sort_keys=True)
    torch.manual_seed(317)
    m,k,h,o=map(int,args.shape.split(','))
    values=tuple((torch.randn(*shape,device='cuda',dtype=torch.bfloat16)*scale).requires_grad_()
                 for shape,scale in (((m,k),.2),((2*h,k),.02),((o,h),.02)))
    dy=torch.randn(m,o,device='cuda',dtype=torch.bfloat16)*.1
    fn=torch.compile(lambda *v:selected_mlp(*v,'silu',spec,True)[0],
                     fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    def pair():
        y=fn(*values)
        return y,*torch.autograd.grad(y,values,dy)
    actual=pair()
    with torch.no_grad(): expected=reference(*values,dy,'silu')
    errors=[]
    for a,b in zip(actual,expected):
        error=float((a.detach().float()-b.detach().float()).norm()/b.detach().float().norm().clamp_min(1e-12))
        assert bool(a.isfinite().all()) and error<.025,error
        errors.append(error)
    del actual,expected
    for _ in range(5): pair()
    torch.cuda.synchronize()
    def annotate(module,name):
        original=getattr(module,name)
        @wraps(original)
        def wrapped(*a,**kw):
            suffix=''
            if module is bwd:
                suffix=f':x={tuple(a[1].shape)}:w={tuple(a[2].shape)}'
            with torch.cuda.nvtx.range(name+suffix):
                return original(*a,**kw)
        setattr(module,name,wrapped)
    for name in ('_run_forward','_run_derivative'):
        annotate(mlp,name)
    for name in ('_launch_bwd_out','_launch_dx_out','_launch_dw_out'):
        annotate(bwd,name)
    # Wrapping launchers can invalidate host guards / lazy metadata paths.
    # Warm the instrumented callable, not only the unwrapped implementation.
    for _ in range(5): pair()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    profiler=None
    if args.host_profile is not None:
        import cProfile
        profiler=cProfile.Profile()
        profiler.enable()
    for index in range(3):
        with torch.cuda.nvtx.range(f'pair_{index}'):
            result=pair()
    torch.cuda.synchronize()
    if profiler is not None:
        profiler.disable()
        args.host_profile.parent.mkdir(parents=True,exist_ok=True)
        profiler.dump_stats(str(args.host_profile))
    torch.cuda.cudart().cudaProfilerStop()
    report=dict(shape=[m,k,h,o],config=config.to_dict(),errors=errors,
                torch=str(torch.__version__),cuda=torch.version.cuda,
                config_sha256=hashlib.sha256(args.config.read_bytes()).hexdigest(),
                scope='three warmed selected-direct pairs; profiler times diagnostic',
                host_profile=None if args.host_profile is None else str(args.host_profile),
                host_profile_scope='calling thread only; autograd worker Python is not included')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream: json.dump(report,stream,indent=2)


if __name__=='__main__': main()
