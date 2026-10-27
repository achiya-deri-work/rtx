"""Matched backward diagnosis; timings exclude compile and input construction.

Public graphs are authoritative. Isolated phase timings change cache residency
and must not be added to explain exact public latency. No kernel policy changes.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

SHAPES = ((512,512,512), (4096,512,512), (4096,1536,512),
          (4096,2048,512), (4096,512,1024), (32768,512,512),
          (4096,2048,2048), (8192,1536,1536))


def resolve_schedules(args):
    """Pin diagnostic execution to a measured candidate, not a fresh default."""
    from benchmarks.compare_ms_eden_amortization import schedule_pair
    design=getattr(args,'design',None)
    if design is not None:
        return {phase:json.dumps(asdict(getattr(design.schedule,phase)),sort_keys=True)
                for phase in ('dx','dw')},design.schedule.backward_pipeline
    manifest=getattr(args,'schedule_manifest',None)
    if manifest is None:
        return schedule_pair({'producer_signs':'register'}),args.backward_pipeline
    if args.mode not in ('public','fp4'):
        raise ValueError('schedule manifest requires public or fp4 diagnostic mode')
    from rtx.autotune.ms_eden_pipeline import MSEdenPipelineSchedule
    record=json.loads(Path(manifest).read_text())
    candidate=record.get('candidate',{})
    if 'schedules' in record and 'candidate_pipeline' in record:
        # Profile the exact measured comparator configuration, including its
        # per-run row width/store overrides rather than an older seed manifest.
        candidate = {phase: json.loads(record['schedules']['candidate'][phase])
                     for phase in ('dx','dw')}
        candidate['backward_pipeline'] = record['candidate_pipeline']
    if set(candidate)!={'dx','dw','backward_pipeline'}:
        raise ValueError('manifest must contain explicit candidate dx/dw/pipeline')
    config=MSEdenPipelineSchedule(**candidate)
    return {phase:json.dumps(asdict(getattr(config,phase)),sort_keys=True)
            for phase in ('dx','dw')},config.backward_pipeline


def load_design_selection(args):
    """Reject out-of-band substitutions before entering profiler capture."""
    from rtx.autotune.ms_eden_design import MSEdenDesign, make_ms_eden_design_adapter
    if (args.mode!='public' or args.schedule_manifest is not None
            or args.preparation_variant!='retained' or args.scale_finalizer!='retained'
            or args.backward_pipeline!='serial'):
        raise ValueError('design selection requires public mode without other schedule overrides')
    record=json.loads(args.design_selection.read_text())
    value=record.get('candidate')
    required={'schedule','transpose_loader','finalizer'}
    if not isinstance(value,dict) or not required<=set(value) or set(value)-required-{'tma_tile_rows','row_transform'}:
        raise ValueError('selection must contain the complete design candidate')
    design=MSEdenDesign(**value)
    shape=tuple(map(int,args.shape.split(',')))
    rejection=make_ms_eden_design_adapter(shape,lambda _:None,allow_row_major=True).rejection(design)
    if rejection:
        raise ValueError(rejection[1])
    return design


def worker(args):
    import torch
    from benchmarks.compare_ms_eden_amortization import schedule_pair, options
    from benchmarks.profile_preparation_phases import measure
    from rtx.autotune.measurement import runtime_timing_metadata
    shape = tuple(map(int, args.shape.split(',')))
    schedules,backward_pipeline = resolve_schedules(args)
    if args.mode == 'paired':
        import math
        import random
        import statistics
        from benchmarks.compare_mixed_nvfp4_backward import run_worker, paired_statistics
        records,bundles={},{}
        shared=None
        for policy in ('stable_mxfp8','fp4_fp4'):
            records[policy],bundles[policy]=run_worker(options(shape,schedules,policy=policy),
                capture_only=True,shared_inputs=shared)
            shared=bundles[policy]['inputs']
        graphs={key:value['pins'][1] for key,value in bundles.items()}
        packed_pins={}
        if args.graph_batch>1:
            for key,bundle in bundles.items():
                x,w,dy=bundle['inputs']
                # run_worker deliberately mutates/restores inputs for guards.
                # Build a fresh autograd context at the current version counter.
                stream=torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    forward_output=bundle['pins'][4](x,w)
                    torch.autograd.grad(forward_output,(x,w),dy,retain_graph=True)
                stream.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):
                    for _ in range(args.graph_batch):
                        grads=torch.autograd.grad(forward_output,(x,w),dy,retain_graph=True)
                graph.replay(); torch.cuda.synchronize()
                previous=tuple(g.clone() for g in grads)
                graph.replay(); torch.cuda.synchronize()
                for a,b in zip(previous,grads):
                    if not b.isfinite().all(): raise RuntimeError('nonfinite packed-graph gradient')
                    if (not torch.equal(a,b)) != (key=='fp4_fp4'):
                        raise RuntimeError('packed-graph RNG contract failed')
                references=(dy.float()@w.detach().float(),dy.float().T@x.detach().float())
                gate=.25 if key=='fp4_fp4' else .06
                if any(float((g.float()-r).norm()/r.norm())>gate for g,r in zip(grads,references)):
                    raise RuntimeError('packed-graph numerical contract failed')
                packed_pins[key]=(grads,stream,forward_output)
                graphs[key]=graph
        deadline=time.monotonic()+5.
        while time.monotonic()<deadline:
            for graph in graphs.values(): graph.replay()
            torch.cuda.synchronize()
        def batch(graph,count):
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(count): graph.replay()
            end.record(); end.synchronize()
            return start.elapsed_time(end)/count/args.graph_batch
        replays=max(10,math.ceil(4/(args.graph_batch*min(batch(g,10) for g in graphs.values()))))
        rng=random.Random(941)
        rounds=[]
        samples=[]
        for index in range(10):
            order=list(graphs); rng.shuffle(order)
            values=[]
            for sequence in (order,list(reversed(order))):
                value={key:batch(graphs[key],replays) for key in sequence}
                values.append(value)
                samples.append(dict(round=index,order=sequence,ms=value))
            rounds.append(dict(reference_ms=statistics.mean(v['stable_mxfp8'] for v in values),
                candidate_ms=statistics.mean(v['fp4_fp4'] for v in values)))
        return dict(shape=shape,records=records,samples=samples,replays=replays,graph_batch=args.graph_batch,
            median_ms={key:statistics.median(s['ms'][key] for s in samples) for key in graphs},
            paired=paired_statistics(rounds),scope='Randomized AB/BA complete public backward graphs; caller RNG included')
    if args.mode == 'fp4':
        from benchmarks.study_ms_eden_backward import run_study
        from rtx.autotune.posthoc import PosthocPreparationSchedule
        return run_study(shape,args.phase,'mma128',
            schedule=PosthocPreparationSchedule(**json.loads(schedules[args.phase])),
            preparation_phases=True,precondition_seconds=1.,profile=args.profile)
    if args.mode == 'public':
        from benchmarks.compare_mixed_nvfp4_backward import run_worker
        opts=options(shape,schedules,policy=args.policy)
        opts.backward_pipeline=backward_pipeline
        result,bundle=run_worker(opts,capture_only=True)
        graph=bundle['pins'][1]
        deadline=time.monotonic()+1.
        while time.monotonic()<deadline:
            for _ in range(16): graph.replay()
            torch.cuda.synchronize()
        from rtx.autotune.measurement import measure_cuda_batches
        from benchmarks.compare_mixed_nvfp4_backward import timing_summary
        attempts=[]
        for attempt in range(3):
            samples,batching=measure_cuda_batches(graph.replay,samples=15,
                calls_per_sample=40,device=bundle['inputs'][0].device)
            attempts.append(timing_summary(samples,batching))
            if attempts[-1]['sampling']['collection']['stationary']: break
            deadline=time.monotonic()+2.
            while time.monotonic()<deadline:
                for _ in range(16): graph.replay()
                torch.cuda.synchronize()
        result['backward_timing']=attempts[-1]
        result['timing_attempts']=attempts
        if args.profile:
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            with torch.cuda.nvtx.range('backward/'+args.policy):
                for _ in range(10): graph.replay()
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            result['profile_scope']='10 warmed public backward graph replays, includes caller RNG'
        result['timing_quality']=runtime_timing_metadata(result['backward_timing']['samples_ms'])
        return result
    from rtx.fp8_bwd import _build_bwd_runner, _default_bwd_config, _MatmulRunner
    from rtx.kernels.mxfp8 import MXFP8Problem
    m,n,k=shape
    torch.manual_seed(941)
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.25
    w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02
    dy=torch.randn(m,n,device='cuda',dtype=torch.bfloat16)*.25
    dx,dw=torch.empty_like(x),torch.empty_like(w)
    config=_default_bwd_config(MXFP8Problem(*shape))
    runner=_build_bwd_runner(MXFP8Problem(*shape),config,x.device)
    phases={}
    if runner.quad_quant is not None:
        phases['paired_preparation']=lambda:runner.quantize_quad(dy,x,w)
    for name,leg,operand,out in (('dx',runner.dx,w,dx),('dw',runner.dw,x,dw)):
        if isinstance(leg,_MatmulRunner):
            if runner.quad_quant is None:
                phases[name+'_preparation']=lambda leg=leg,operand=operand:leg.quantize(dy,operand)
            phases[name+'_gemm_including_split_reduction']=lambda leg=leg,out=out:leg.matmul(out)
        else:
            phases[name+'_fused']=lambda leg=leg,operand=operand,out=out:leg(dy,operand,out)
    phases['complete_paired']=lambda:runner(dy,x,w,dx,dw)
    phases['complete_paired']()
    errors={}
    for name,out,ref in (('dx',dx,dy.float()@w.float()),('dw',dw,dy.float().T@x.float())):
        nrms=float((out.float()-ref).norm()/ref.norm().clamp_min(1e-12))
        if not out.isfinite().all() or nrms>.06: raise RuntimeError((name,nrms))
        errors[name]=nrms
    timings={name:measure(fn,time_based_warmup=True) for name,fn in phases.items()}
    if args.profile:
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        phases['complete_paired']()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
    return dict(shape=shape,config=asdict(config),errors=errors,timings=timings,
        quality={n:runtime_timing_metadata(t['samples_ms']) for n,t in timings.items()},
        caveat='Actual default paired runner; isolated phases are nonadditive; GEMM includes split reduction when configured')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--mode',choices=('suite','public','paired','fp4','fp8'),default='suite')
    p.add_argument('--shape',default='512,512,512')
    p.add_argument('--phase',choices=('dx','dw'),default='dx')
    p.add_argument('--policy',choices=('fp4_fp4','stable_mxfp8'),default='fp4_fp4')
    p.add_argument('--profile',action='store_true')
    p.add_argument('--combo-kernels',action='store_true',
                   help='public only: scoped compiler RNG combo fusion, no runtime default changes')
    p.add_argument('--preparation-variant', choices=('retained', 'compact', 'compact_fma', 'tma_xor5_1', 'tma_columns_1'),
                   default='retained', help='public only: research producer, never a runtime default')
    p.add_argument('--backward-pipeline',choices=('serial','paired'),default='serial')
    p.add_argument('--graph-batch',type=int,choices=(1,8),default=1)
    p.add_argument('--scale-finalizer',choices=('retained','dual_both'),default='retained',
                   help='public only: match the research fused scale-finalizer topology')
    p.add_argument('--schedule-manifest',type=Path,
                   help='public/fp4 only: pin exact candidate schedules and pipeline from comparison manifest')
    p.add_argument('--design-selection',type=Path,
                   help='public only: exact structural design selection, including loader/finalizer')
    args=p.parse_args()
    args.design=load_design_selection(args) if args.design_selection is not None else None
    if args.combo_kernels and args.mode!='public':
        p.error('--combo-kernels requires --mode public')
    if args.scale_finalizer!='retained' and (args.mode!='public' or args.schedule_manifest is None):
        p.error('dual finalizer requires public mode and exact schedule manifest')
    if args.preparation_variant != 'retained' and (args.mode != 'public' or args.schedule_manifest is None):
        p.error('research preparation requires public mode and an exact schedule manifest')
    output=Path(args.output)
    if args.mode!='suite':
        from contextlib import ExitStack
        with ExitStack() as contexts:
            dispatch_records=[]
            if args.design is not None and args.policy=='fp4_fp4':
                from benchmarks.ms_eden_design_dispatch import dispatch
                dispatch_records=contexts.enter_context(dispatch(args.design))
            if args.preparation_variant != 'retained':
                from benchmarks.experiment_ms_eden_bank_layout import patched
                contexts.enter_context(patched(args.preparation_variant))
            if args.scale_finalizer=='dual_both':
                from benchmarks.experiment_ms_eden_dual_finalizer import patched as dual_patch
                contexts.enter_context(dual_patch('both'))
            if args.combo_kernels:
                import torch._inductor.config as compiler_config
                contexts.enter_context(compiler_config.patch(combo_kernels=True,combo_kernels_pointwise_only=True))
            result=worker(args)
            if args.design is not None:
                if args.policy=='fp4_fp4' and args.design.row_transform=='fma':
                    rows=[r for r in dispatch_records if r.get('transpose') is False]
                    if not rows or any(r.get('selected')!='row_fma' for r in rows):
                        raise RuntimeError('profile did not establish requested row FMA implementation')
                if args.policy=='fp4_fp4' and args.design.transpose_loader!='native':
                    transposes=[r for r in dispatch_records if r.get('transpose') is True]
                    if not transposes or any(r.get('selected')!=args.design.transpose_loader
                            or r.get('tile_rows',32)!=args.design.tma_tile_rows for r in transposes):
                        raise RuntimeError('profile did not establish requested TMA implementation')
                result['design']=asdict(args.design)
                result['design_dispatch']=dispatch_records
                root=Path(__file__).resolve().parents[1]
                result['design_source_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                    for name in ('benchmarks/ms_eden_design_dispatch.py',
                        'benchmarks/experiment_ms_eden_bank_layout.py','benchmarks/experiment_ms_eden_dual_finalizer.py',
                        'rtx/kernels/ms_eden_tma_preparation.py','rtx/kernels/ms_eden_dual_finalizer.py',
                        'rtx/kernels/ms_eden_bank_layout.py','rtx/kernels/nvfp4_posthoc.py')}
        if args.combo_kernels:
            result['compiler_options']={'combo_kernels':True,'combo_kernels_pointwise_only':True}
        result['preparation_variant'] = args.preparation_variant
        result['scale_finalizer'] = args.scale_finalizer
        if args.scale_finalizer!='retained':
            root=Path(__file__).resolve().parents[1]
            result['finalizer_source_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                for name in ('rtx/kernels/ms_eden_dual_finalizer.py',
                             'benchmarks/experiment_ms_eden_dual_finalizer.py')}
        if args.preparation_variant != 'retained':
            root = Path(__file__).resolve().parents[1]
            result['preparation_source_sha256'] = {
                name: hashlib.sha256((root/name).read_bytes()).hexdigest()
                for name in ('rtx/kernels/ms_eden_bank_layout.py', 'rtx/kernels/ms_eden_tma_preparation.py', 'benchmarks/experiment_ms_eden_bank_layout.py')}
        result['command']=sys.argv
        result['driver_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if args.schedule_manifest is not None:
            result['schedule_manifest_sha256']=hashlib.sha256(args.schedule_manifest.read_bytes()).hexdigest()
        output.parent.mkdir(parents=True,exist_ok=True)
        with output.open('x') as handle: json.dump(result,handle,indent=2)
        print('DONE',output,flush=True)
        return
    output.mkdir(parents=True,exist_ok=True)
    for shape in SHAPES:
        tag='_'.join(map(str,shape))
        jobs=[('public',policy,None) for policy in ('stable_mxfp8','fp4_fp4')]
        jobs += [('fp8',None,None),('fp4',None,'dx'),('fp4',None,'dw')]
        for mode,policy,phase in jobs:
            name=f'{tag}-{mode}-{policy or phase or "paired"}'
            target=output/(name+'.json')
            if target.exists(): continue
            cmd=[sys.executable,'-m','benchmarks.profile_backward_subparts','--mode',mode,
                 '--shape',','.join(map(str,shape)),'--output',str(target)]
            if policy: cmd+=['--policy',policy]
            if phase: cmd+=['--phase',phase]
            print('START',name,flush=True)
            with (output/(name+f'-{time.time_ns()}.log')).open('x') as log:
                run=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
            print('FINISH',name,'status',run.returncode,flush=True)
            if run.returncode: raise RuntimeError(f'Failed {name}; inspect log')


if __name__=='__main__': main()
