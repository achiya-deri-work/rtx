"""Controlled warm-launch Python A/B; kernels and config stay identical."""
import argparse
import inspect
import json
from pathlib import Path

import torch
import rtx._mlp_autotune as selection
import rtx.mlp as mlp
from rtx.mlp import MXFP8MLPConfig
from rtx.autotune.mlp import make_mlp_adapter
from rtx.autotune.deployment_qualification import qualify_pair
from rtx.autotune.replacement_benchmark import source_manifest


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--field',default='challenger')
    p.add_argument('--shape',default='20480,256,768,256')
    p.add_argument('--phase',choices=('train','forward'),default='train')
    p.add_argument('--component',choices=('metadata','nested_dispatch'),default='metadata')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.component=='nested_dispatch' and args.phase!='train':
        p.error('nested dispatch comparison requires training')
    if args.output.exists(): raise FileExistsError(args.output)
    sources=source_manifest()
    cfg=MXFP8MLPConfig.from_dict(json.loads(args.config.read_text())[args.field])
    m,k,h,o=map(int,args.shape.split(','))
    torch.manual_seed(1729)
    values=tuple(torch.randn(*s,device='cuda',dtype=torch.bfloat16)*scale
                 for s,scale in (((m,k),.2),((2*h,k),.02),((o,h),.02)))
    adapter=make_mlp_adapter(*values,initial=cfg,phase=args.phase,
        measurement_mode='selected_direct',warmup=3,samples=7,calls_per_sample=3)
    cached=selection._spec_mode_and_fixed_key
    original_backward=mlp._backward
    baseline_source=None
    if args.component=='nested_dispatch':
        # Reconstruct only the two former dispatch sites, without modifying
        # production files or the registered operators' implementation tables.
        baseline_source=inspect.getsource(original_backward)
        replacements={
            '_independent_bwd_launch as _mxfp8_bwd_compiler_visible':
                'independent_bwd as _mxfp8_bwd_compiler_visible',
            'derivative = _derivative_launch if independent_outputs else _derivative_op':
                'derivative = _derivative_op',
        }
        for old,new in replacements.items():
            if baseline_source.count(old)!=1:
                raise RuntimeError(f'baseline reconstruction no longer matches: {old}')
            baseline_source=baseline_source.replace(old,new)
        namespace=dict(vars(mlp))
        exec(compile(baseline_source,'<nested_dispatch_baseline>','exec'),namespace)
        nested_backward=namespace['_backward']
        # Separate evaluators ensure both paths receive independent initial
        # recipe checks rather than reusing one evaluator's validated result.
        adapters={name:make_mlp_adapter(*values,initial=cfg,phase=args.phase,
            measurement_mode='selected_direct',warmup=3,samples=7,calls_per_sample=3)
            for name in ('nested','launch_helpers')}
    records=[]
    def evaluate(name):
        # The uncached fixed path repeats loads/dumps just as the prior source
        # did. No GPU scratch, precision, schedule or token ownership changes.
        if args.component=='metadata':
            selection._spec_mode_and_fixed_key=cached if name=='cached' else cached.__wrapped__
            result=adapter.evaluate(cfg)
        else:
            mlp._backward=nested_backward if name=='nested' else original_backward
            # Cached closures capture _backward. Rebuild host closures outside
            # the evaluator's warmup/timing; retain identical GPU scratch.
            selection.BACKWARDS.clear()
            result=adapters[name].evaluate(cfg)
        records.append(dict(candidate=name,outcome=result.as_dict()))
        print(json.dumps(records[-1]),flush=True)
        return result
    try:
        names=('uncached','cached') if args.component=='metadata' else ('nested','launch_helpers')
        winner,decision=qualify_pair(*names,evaluate)
    finally:
        selection._spec_mode_and_fixed_key=cached
        mlp._backward=original_backward
        if args.component=='nested_dispatch': selection.BACKWARDS.clear()
    if source_manifest()!=sources: raise RuntimeError('RTX sources changed')
    report=dict(shape=[m,k,h,o],phase=args.phase,mode='selected_direct',
        config=cfg.to_dict(),seed=1729,torch=str(torch.__version__),
        cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        sources=sources,records=records,winner=winner,decision=decision,
        component=args.component,baseline_source=baseline_source,
        scope='fixed-config Python launch A/B only; not BF16 or application comparison')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream: json.dump(report,stream,indent=2)


if __name__=='__main__': main()
