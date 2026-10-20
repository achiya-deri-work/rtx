"""Paired deployment-mode qualification; one context per fresh process."""
import argparse
from dataclasses import replace
import json
import hashlib
from pathlib import Path
import time

import torch

from rtx.autotune.deployment_qualification import qualify_pair
from rtx.autotune.replacement_benchmark import source_manifest
from rtx.autotune.mlp import make_mlp_adapter, materialized_dw_config, materialized_dx_config
from rtx.mlp import DEFAULT_MLP_CONFIG, MXFP8MLPConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shape',default='128,128,256,128',help='M,K,H,O')
    p.add_argument('--phase',choices=('forward','train'),default='train')
    p.add_argument('--warmup',type=int,default=3)
    p.add_argument('--samples',type=int,default=7)
    p.add_argument('--calls-per-sample',type=int,default=3)
    p.add_argument('--mode',choices=('outer_capture','compiled_direct','selected_direct','private_replay'),default='selected_direct')
    p.add_argument('--initial',type=Path)
    p.add_argument('--initial-field',default='config')
    p.add_argument('--challenger',type=Path)
    p.add_argument('--challenger-field',default='config')
    p.add_argument('--gate-backend',choices=('prequant',),help='Use prequant gate with onchip down; preserves backward configs unless explicitly overridden')
    p.add_argument('--recompute-backend',choices=('prequant','onchip'))
    p.add_argument('--dw-splits',type=int,choices=(2,4,8,16,32))
    p.add_argument('--set-axis',action='append',default=[],metavar='NAME=JSON',
                   help='Apply an existing autotuner axis to the challenger')
    p.add_argument('--baseline-set-axis',action='append',default=[],metavar='NAME=JSON',
                   help='Normalize the incumbent before cloning the challenger; both resolved configs are recorded')
    p.add_argument('--materialize-backward',action='store_true',
                   help='Normalize both backward GEMMs to row-major prequantized operands; useful for ragged winner transfer')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.warmup<1 or args.samples<3 or args.calls_per_sample<1:
        p.error('require warmup >= 1, samples >= 3, calls-per-sample >= 1')
    if args.output.exists():
        raise FileExistsError(args.output)
    kernel_sources=source_manifest()
    torch.manual_seed(1729)
    m,k,h,o=map(int,args.shape.split(','))
    values=tuple(torch.randn(*shape,device='cuda',dtype=torch.bfloat16)*scale
                 for shape,scale in (((m,k),.2),((2*h,k),.02),((o,h),.02)))
    def config(path,field='config'):
        data=json.loads(path.read_text())
        if field != 'config' and field not in data:
            raise ValueError(f'missing config field {field}')
        return MXFP8MLPConfig.from_dict(data.get(field,data))
    reference_cache={}
    options=dict(phase=args.phase,measurement_mode=args.mode,warmup=args.warmup,samples=args.samples,
                 calls_per_sample=args.calls_per_sample,_reference_cache=reference_cache)
    baseline=make_mlp_adapter(*values,initial=config(args.initial,args.initial_field) if args.initial else DEFAULT_MLP_CONFIG,**options)
    incumbent=baseline.initial_config
    for setting in args.baseline_set_axis:
        axis,value=setting.split('=',1)
        incumbent=baseline.update_fn(incumbent,axis,json.loads(value))
    if args.baseline_set_axis:
        baseline=make_mlp_adapter(*values,initial=incumbent,**options)
        incumbent=baseline.initial_config
    candidate=incumbent
    if args.challenger:
        candidate=config(args.challenger,args.challenger_field)
    elif args.gate_backend or args.recompute_backend or args.dw_splits or args.set_axis:
        if args.gate_backend:
            candidate=replace(incumbent,gate_backend=args.gate_backend,down_backend='onchip')
    elif args.phase=='train':
        candidate=replace(incumbent,gate_backward=materialized_dw_config(incumbent.gate_backward),
            down_backward=materialized_dx_config(materialized_dw_config(incumbent.down_backward)))
    if args.recompute_backend:
        candidate=replace(candidate,recompute_backend=args.recompute_backend)
    if args.materialize_backward:
        candidate=replace(candidate,**{group:materialized_dx_config(
            materialized_dw_config(getattr(candidate,group)))
            for group in ('gate_backward','down_backward')})
    if args.dw_splits:
        for group in ('gate_backward','down_backward'):
            candidate=baseline.update_fn(candidate,f'{group}.dw_split_count',args.dw_splits)
    for setting in args.set_axis:
        axis,value=setting.split('=',1)
        candidate=baseline.update_fn(candidate,axis,json.loads(value))
    # Forward-only defaults compare identical configs: a no-promotion control.
    challenger=make_mlp_adapter(*values,initial=candidate,**options)
    records=[]
    def evaluate(name):
        adapter,c=(baseline,incumbent) if name=='incumbent' else (challenger,candidate)
        rejection=adapter.rejection(c)
        if rejection:
            raise RuntimeError(rejection)
        started=time.monotonic()
        result=adapter.evaluate(c)
        torch.cuda.synchronize()
        row=dict(candidate=name,elapsed_s=time.monotonic()-started,outcome=result.as_dict(),
                 allocated=torch.cuda.memory_allocated(),reserved=torch.cuda.memory_reserved())
        records.append(row)
        print(json.dumps(row),flush=True)
        if result.status!='ok':
            raise RuntimeError(result.error)
        return result
    torch.cuda.reset_peak_memory_stats()
    failure=None
    try:
        winner,decision=qualify_pair('incumbent','challenger',evaluate)
    except Exception as exc:
        # Preserve rejected/failed experiments, including the passing incumbent
        # samples already collected. A partial run never becomes a promotion.
        winner=None
        decision=dict(promoted=False,incomplete=True)
        failure=dict(type=type(exc).__name__,message=str(exc))
    report=dict(shape=[m,k,h,o],phase=args.phase,mode=args.mode,torch=torch.__version__,
                cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),seed=1729,
                incumbent=incumbent.to_dict(),challenger=candidate.to_dict(),
                winner=winner,decision=decision,records=records,failure=failure,
                peak_allocated=torch.cuda.max_memory_allocated())
    report['environment_qualified']=False
    report['installs_runtime_winner']=False
    report['arguments']={key:str(value) if isinstance(value,Path) else value
                         for key,value in vars(args).items()}
    sources=[Path('rtx/mlp.py'),Path('rtx/autotune/mlp.py'),Path(__file__),
             Path('rtx/kernels/mxfp8_glu_prequant.py')]
    sources.extend(path for path in (args.initial,args.challenger) if path is not None)
    report['source_sha256']={str(path):hashlib.sha256(path.read_bytes()).hexdigest()
                             for path in sources}
    if source_manifest()!=kernel_sources:
        raise RuntimeError('RTX sources changed during qualification')
    report['rtx_source_sha256']=kernel_sources
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2)
    if failure:
        raise SystemExit(f"qualification failed; evidence retained at {args.output}: {failure}")


if __name__=='__main__':
    main()
