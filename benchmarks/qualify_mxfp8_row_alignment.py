"""Same-tensor MXFP8 forward alignment A/B; no external model dependencies."""
import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics

import torch


def measure(function, *, samples=7, replays=10):
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):function()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            output=function()
    stream.synchronize()
    begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
    begin.record()
    for _ in range(5):graph.replay()
    end.record();end.synchronize()
    pilot=begin.elapsed_time(end)/5
    for _ in range(min(4096,max(10,math.ceil(100/max(pilot,1e-6))))):graph.replay()
    timings=[]
    for _ in range(samples):
        begin.record()
        for _ in range(replays):graph.replay()
        end.record();end.synchronize()
        timings.append(begin.elapsed_time(end)/replays)
    result=tuple(t.clone() for t in output) if isinstance(output,tuple) else output.clone()
    graph.reset()
    return result,dict(median_ms=statistics.median(timings),min_ms=min(timings),samples_ms=timings)


def metrics(actual,reference):
    delta=actual.float()-reference
    rms=float(delta.square().mean().sqrt())
    ref_rms=float(reference.square().mean().sqrt())
    result=dict(finite=bool(actual.isfinite().all()),rms=rms,reference_rms=ref_rms,
                relative_l2=rms/max(ref_rms,1e-12),max_abs=float(delta.abs().max()))
    assert result['finite'] and result['relative_l2']<.08,result
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape',action='append',required=True,help='M,N,K; repeat for a ladder')
    parser.add_argument('--variants',default='auto,historical_fused,row_major,native64')
    parser.add_argument('--historical-cache',type=Path)
    parser.add_argument('--backward-selections',type=Path,help='explicit measured schedules, never imported as cache evidence')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--compiled',action='store_true',help='default-mode fullgraph compilation')
    args=parser.parse_args()
    import rtx
    import rtx.fp8 as f
    from rtx.prequant_autotune import update_prequant_config,PREQUANT_SEARCH_SPACE
    from rtx.kernels.mxfp8 import fwd_config_from_dict,MXFP8Problem
    torch.manual_seed(31103);torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    root=Path(__file__).resolve().parents[1]
    report=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=str(torch.cuda.get_device_properties(0)),
        seed=31103,compiled=args.compiled,mode='default',oracle='FP32 original BF16 operands',
        source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                       for p in (root/'rtx').rglob('*.py')},rows=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for shape in args.shape:
        m,n,k=map(int,shape.split(','));problem=MXFP8Problem(m,n,k)
        x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.1
        w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.1
        with torch.no_grad():reference=x.float()@w.float().T
        for variant in args.variants.split(','):
            if args.compiled:torch._dynamo.reset()
            keywords=dict(autotune=False)
            config=None
            if variant=='fused':
                config=f.DEFAULT_MXFP8_FWD_CONFIG
                keywords.update(backend='fused',forward_config=config)
            elif variant=='historical_fused':
                if args.historical_cache is None:continue
                paths=list(args.historical_cache.glob(
                    f'runtime_winners/mxfp8_fused_fwd/*/m{m}_n{n}_k{k}_hot_default_*.json'))
                if len(paths)!=1:
                    print('SKIP historical',shape,paths,flush=True);continue
                stored=json.loads(paths[0].read_text())
                config=fwd_config_from_dict(stored['config'])
                keywords.update(backend='fused',forward_config=config)
            elif variant=='row_major':
                initial=f.DEFAULT_MXFP8_PREQUANT_CONFIG
                config=replace(initial,quant=replace(initial.quant,scale_layout='row_major'),
                    gemm=replace(initial.gemm,scale_layout='row_major',scale_role='consumers',
                                 epilogue='direct',store_vec=1))
                keywords.update(backend='materialized',dynamic_config=config)
            elif variant=='native64':
                config=update_prequant_config(f.DEFAULT_MXFP8_PREQUANT_CONFIG,
                                             PREQUANT_SEARCH_SPACE['layout_transport'][1])
                keywords.update(backend='materialized',dynamic_config=config)
            elif variant!='auto':raise ValueError(variant)
            reason=(config.implementation_rejection(problem) if variant in ('fused','historical_fused')
                    else config.rejection(problem) if config is not None else None)
            if reason:
                print('SKIP',variant,shape,reason,flush=True);continue
            def forward():return rtx.mxfp8_linear(x,w,**keywords)
            function=torch.compile(forward,fullgraph=True,dynamic=False) if args.compiled else forward
            with torch.no_grad():
                actual,timing=measure(function)
                row=dict(shape=[m,n,k],variant=variant,config=None if config is None else asdict(config),
                         **timing,metrics=metrics(actual,reference))
            report['rows'].append(row)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print('RESULT',shape,variant,json.dumps(timing),row['metrics'],flush=True)
            del actual,function,forward
            rtx.clear_runtime_caches(preserve_compiled=True)
            gc.collect()
        if args.backward_selections:
            from rtx import fp8_bwd as b
            from rtx.bwd_autotune import bwd_config_from_dict
            saved=json.loads(args.backward_selections.read_text())
            selected=next((row for row in saved if row['key']==[[m,k],[n,k]]),None)
            if selected is not None:
                original=bwd_config_from_dict(selected['config'])
                dx_seed=f._DEFAULT_MXFP8_NATIVE64_CONFIG
                leg=replace(original.dx,backend='decomposed',quant_launches='separate',
                    quant_a=replace(original.dx.quant_a,scale_layout='mma64',native_scale_store='scalar'),
                    quant_b=replace(original.dx.resolved_quant_b(),scale_layout='mma128',native_scale_store='scalar'),
                    gemm=dx_seed.gemm)
                candidate=replace(original,dx=leg,quant_schedule='per_matmul')
                variants=[('historical',original)]
                if candidate.implementation_rejection(problem) is None:variants.append(('native64_dx',candidate))
                dy=torch.randn(m,n,device='cuda',dtype=torch.bfloat16)*.1
                references=dict(dx=dy.float()@w.float(),dw=dy.float().T@x.float())
                for variant,backward in variants:
                    key=b._intern_bwd_config(backward)
                    calls=dict(dx=lambda:b._mxfp8_dx_compiler_visible(dy,x,w,key),
                               dw=lambda:b._mxfp8_dw_compiler_visible(dy,x,w,key),
                               backward=lambda:b.mxfp8_linear_backward(dy,x,w,config=backward,autotune=False))
                    for phase,function in calls.items():
                        if args.compiled:
                            torch._dynamo.reset()
                            function=torch.compile(function,fullgraph=True,dynamic=False)
                        actual,timing=measure(function)
                        errors=([metrics(a,r) for a,r in zip(actual,references.values())] if phase=='backward'
                                else [metrics(actual,references[phase])])
                        row=dict(shape=[m,n,k],variant=variant,phase=phase,config=asdict(backward),
                                 **timing,metrics=errors)
                        report['rows'].append(row)
                        args.output.write_text(json.dumps(report,indent=2)+'\n')
                        print('BACKWARD_RESULT',shape,variant,phase,json.dumps(timing),errors,flush=True)
                    rtx.clear_runtime_caches(preserve_compiled=True)
                del dy,references
        del x,w,reference
    print('PASS',flush=True)


if __name__=='__main__':main()
