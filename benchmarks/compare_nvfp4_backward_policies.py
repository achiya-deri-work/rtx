"""Matched NVFP4 forward, original-BF16-source MXFP8 versus MS-EDEN backward.

Fullgraph compiled forward/autograd; graph timings include FP4 RNG generation.
This is not a packed-source or delayed-history comparison.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import torch
import rtx
from benchmarks.profile_preparation_phases import measure
from benchmarks.study_nvfp4_precision import metrics
from rtx.autotune.measurement import runtime_timing_metadata
from rtx.autotune.posthoc import preparation_resource_seeds, PosthocPreparationSchedule
from rtx.experimental_ms_eden import ms_eden_linear
from rtx.nvfp4_inference_autotune import dynamic_config_from_dict


def load_schedule(path,shape,phase):
    path=Path(path)
    if path.is_dir():
        selections=list(path.glob('selection-*.json'))
        if len(selections)!=1:
            raise ValueError('provide an explicit selection file when zero/multiple selections exist')
        path=selections[0]
    value=json.loads(path.read_text())
    context=value['context']
    if tuple(context['workload'][key] for key in ('m','n','k'))!=tuple(shape):
        raise ValueError('backward schedule shape mismatch')
    tags=context['tags']
    if tags['phase']!=phase or tags['source']!='original_bf16' or tags['packed_reader']:
        raise ValueError('backward schedule phase/source mismatch')
    if (context['kernel_revision']!=7 or tags['layout']!='mma128'
            or context.get('family')!='ms_eden_preparation_research'
            or tags.get('policy')!='shared_sign_hadamard128_e8m4_rtz_eden_scale_sr'):
        raise ValueError('backward schedule revision/layout mismatch')
    schedule=PosthocPreparationSchedule(**value['config'])
    if schedule.gemm.scale_layout!='mma128' or schedule.reader_mode!='direct':
        raise ValueError('backward schedule payload contradicts policy metadata')
    return schedule


@torch.no_grad()
def gradient_phases(dy,operand,phase,schedule):
    from benchmarks.study_ms_eden_backward import prepare
    from rtx.fp4 import compile_nvfp4_gemm
    from rtx.kernels.nvfp4_posthoc import compile_posthoc_scale_product
    a=prepare(dy,phase=='dw','mma128',schedule)
    b=prepare(operand,True,'mma128',schedule)
    signs=torch.where(torch.rand(128,device=dy.device)<.5,-1.,1.)
    m,n=(dy.shape[0] if phase=='dx' else dy.shape[1]),operand.shape[1]
    k=dy.shape[1] if phase=='dx' else dy.shape[0]
    out=torch.empty(m,n,device=dy.device,dtype=dy.dtype)
    alpha=torch.empty(1,device=dy.device)
    product=compile_posthoc_scale_product()
    gemm=compile_nvfp4_gemm(rtx.NVFP4Problem(m,n,k),schedule.gemm)
    def preparation():
        a['run'](signs); b['run'](signs)
        product(a['factor'],b['factor'],alpha)
    def multiply():
        gemm(a['packed'],b['packed'],a['scales'],b['scales'],out,alpha)
    preparation(); multiply()
    oracle=a['decode']()@b['decode']().T
    error=metrics(out,oracle)
    if not error['finite'] or error['normalized_rms']>.005:
        raise RuntimeError(f'phase packed oracle failed: {error}')
    def rng():
        return (torch.where(torch.rand(128,device=dy.device)<.5,-1.,1.),
                torch.rand(dy.numel()//16,device=dy.device),
                torch.rand(operand.numel()//16,device=dy.device))
    compiled_rng=torch.compile(rng,fullgraph=True,dynamic=False)
    timings={name:measure(fn,time_based_warmup=True) for name,fn in
             (('rng',compiled_rng),('preparation',preparation),('gemm',multiply))}
    return dict(timings=timings,packed_metrics=error,
        quality={name:runtime_timing_metadata(value['samples_ms']) for name,value in timings.items()},
        contract='Isolated phases use fixed RNG inputs; RNG measured separately; not additive.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--forward-config',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--dx-seed',type=int,choices=range(4),default=0)
    p.add_argument('--dw-seed',type=int,choices=range(4),default=0)
    p.add_argument('--dx-selection',type=Path)
    p.add_argument('--dw-selection',type=Path)
    p.add_argument('--phase-breakdown',action='store_true')
    args=p.parse_args()
    document=json.loads(args.forward_config.read_text())
    m,n,k=document['shape']
    config=dynamic_config_from_dict(document['config'])
    if config.x_scale_region_rows <= 0 or config.weight_scale_region_rows <= 0:
        p.error('this matched experiment requires an explicit JIT-region forward config')
    torch.backends.cuda.matmul.allow_tf32=False
    torch.manual_seed(17)
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16).requires_grad_()
    w=(torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02).requires_grad_()
    dy=torch.randn(m,n,device='cuda',dtype=torch.bfloat16)*.25
    with torch.no_grad():
        refs=(x.float()@w.float().T,dy.float()@w.float(),dy.float().T@x.float())
    sm=torch.cuda.get_device_properties(0).multi_processor_count
    dx=preparation_resource_seeds((m,n,k),'dx',sm)[args.dx_seed]
    dw=preparation_resource_seeds((m,n,k),'dw',sm)[args.dw_seed]
    if args.dx_selection: dx=load_schedule(args.dx_selection,(m,n,k),'dx')
    if args.dw_selection: dw=load_schedule(args.dw_selection,(m,n,k),'dw')
    dx_json,dw_json=json.dumps(asdict(dx)),json.dumps(asdict(dw))
    policy=rtx.NVFP4ScaleConfig(tensor_scale_mode=config.tensor_scale_mode)
    results={}
    expected_forward=None
    for mode in ('mxfp8','ms_eden'):
        torch.compiler.reset()
        def forward(a,b):
            if mode=='mxfp8':
                return rtx.nvfp4_linear(a,b,scaling='jit_row_region',dynamic_config=config,
                    scale_config=policy,autotune='off',backward_config=rtx.DEFAULT_MXFP8_BWD_CONFIG)
            return ms_eden_linear(a,b,forward_scaling='jit_row_region',forward_config=config,
                forward_outer_mode=config.tensor_scale_mode,
                dx_schedule_json=dx_json,dw_schedule_json=dw_json)
        compiled=torch.compile(forward,fullgraph=True,dynamic=False)
        def step():
            y=compiled(x,w)
            return y,torch.autograd.grad(y,(x,w),dy)
        y,grads=step()
        if expected_forward is None:
            expected_forward=y.detach().clone()
        else:
            torch.testing.assert_close(y,expected_forward,rtol=0,atol=0)
        errors={name:metrics(value.detach(),ref) for name,value,ref in zip(('forward','dx','dw'),(y,*grads),refs)}
        for name,value in errors.items():
            gate=.25 if name=='forward' or mode=='ms_eden' else .06
            if not value['finite'] or value['normalized_rms']>gate:
                raise RuntimeError(f'{mode} {name} reference gate failed: {value}')
        del y,grads
        timing=measure(step,time_based_warmup=True)
        results[mode]=dict(metrics=errors,full_forward_backward=timing,
                          timing_quality=runtime_timing_metadata(timing['samples_ms']))
    result=dict(shape=[m,n,k],forward_config=asdict(config),source_policy='original_bf16',
        dx_schedule=asdict(dx),dw_schedule=asdict(dw),seed_indices=[args.dx_seed,args.dw_seed],
        results=results,seed=17,torch=torch.__version__,cuda=torch.version.cuda,
        device=str(torch.cuda.get_device_properties(0)),
        selection_sources={phase:str(path) if path else None for phase,path in
                           (('dx',args.dx_selection),('dw',args.dw_selection))},
        contract='Identical NVFP4 JIT-region forward; FP4 backward RNG included; explicit schedule provenance recorded.')
    if args.phase_breakdown:
        result['ms_eden_phases']={phase:gradient_phases(dy,operand,phase,schedule)
            for phase,operand,schedule in (('dx',w,dx),('dw',x,dw))}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result,handle,indent=2)
    print(json.dumps({key:{'us':value['full_forward_backward']['median_ms']*1000,
        'nrms':{name:err['normalized_rms'] for name,err in value['metrics'].items()}}
        for key,value in results.items()}),flush=True)


if __name__=='__main__':
    main()
