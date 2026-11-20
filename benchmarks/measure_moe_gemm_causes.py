"""Matched packed-GEMM diagnosis; no production policy or kernel changes.

Vary one geometry axis, measure both pure GEMMs and native quantization,
and expose a warmed single-kernel profiler boundary. BF16 and MXFP8 pure
GEMMs consume exactly the same decoded quantized values.
"""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import time


CASES=dict(small_tokens=(32,256,384,128),expert_thin=(32,3712,384,128),
    long_k=(32,3712,384,512),wide_n=(32,3712,1536,128),
    wide_both=(32,3712,1536,512),shared_down=(1,29504,384,256))
CASES['expert_dx']=(32,3712,384,256)


def templates(adapter):
    configs=[adapter.initial_config]
    for geometry,depth,transport in (
        ((128,128,128,4,2),2,'tma'),((128,128,128,4,2),3,'tma'),
        ((64,128,128,2,2),1,'tma'),((64,128,128,2,2),2,'tma'),
        ((64,64,128,2,2),2,'tma'),((128,64,128,4,2),1,'tma'),
        ((64,128,128,2,2),1,'cooperative')):
        c=adapter.update_fn(adapter.initial_config,'prepared.down.geometry',geometry)
        c=adapter.update_fn(c,'prepared.down.stages',depth)
        c=adapter.update_fn(c,'prepared.down.mma_schedule','preload' if depth==1 else 'interleaved')
        c=adapter.update_fn(c,'prepared.down.transport',transport)
        if adapter.rejection(c) is None and adapter.config_id(c) not in {adapter.config_id(v) for v in configs}:
            configs.append(c)
    return configs


def epilogue_templates(adapter):
    result=[];seen=set()
    for geometry in (None,(64,128,128,2,2)):
        base=adapter.initial_config
        if geometry:
            base=adapter.update_fn(base,'prepared.down.geometry',geometry)
            # Do not bypass the quarantined 64-row single-buffer policy.
            base=adapter.update_fn(base,'prepared.down.stages',2)
        for epilogue in (('direct',1,1),('tma',1,1),('tma',4,1),('hybrid',1,1),('hybrid',4,1)):
            c=adapter.update_fn(base,'prepared.down.epilogue_transport',epilogue)
            key=adapter.config_id(c)
            if key not in seen and adapter.rejection(c) is None:
                seen.add(key);result.append(c)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case',choices=(*CASES,'all'),default='all')
    p.add_argument('--dimensions',nargs=4,type=int,metavar=('BATCH','M','N','K'),
                   help='Override one case with exact packed GEMM dimensions')
    p.add_argument('--gemm-policy',type=Path,help='Exact MXFP8GemmConfig JSON, e.g. the selected common-MLP down schedule')
    p.add_argument('--source-result',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--capture',choices=('none','counters','timeline','torch'),default='none')
    p.add_argument('--precision',choices=('bf16','mxfp8','bf16_out'),default='mxfp8')
    p.add_argument('--record',type=Path,help='Freeze this prior measurement winner during profiler injection')
    p.add_argument('--layout-panel',action='store_true',help='Add BF16 logical-stride/output controls; layout copies stay outside timing')
    p.add_argument('--epilogue-panel',action='store_true',help='Measure existing direct/TMA store templates at paired geometries')
    p.add_argument('--source-stage',choices=('down','gate_dx'),default='down')
    args=p.parse_args()
    if args.dimensions and (args.case=='all' or min(args.dimensions)<1):
        p.error('--dimensions requires one case and four positive extents')
    if args.dimensions:CASES[args.case]=tuple(args.dimensions)
    if args.capture!='none' and (args.case=='all' or args.record is None):
        p.error('Profiling requires one case and a frozen --record')
    if args.output.exists():raise FileExistsError(args.output)
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='0'
    import torch
    from benchmarks.benchmark_moe_execution import _python_source_digest
    from benchmarks.profile_moe import hardware_attributes,statistics_ms
    from benchmarks.study_moe_priors import atomic_json
    from rtx import _moe_prepared
    from rtx.autotune.core import Proposal
    from rtx.autotune.effort import AutotunePolicy,make_effort_autotuner
    from rtx.autotune.measurement import measure_cuda_batches
    from rtx.autotune.moe import make_moe_adapter
    from rtx.autotune.outcomes import TrialOutcome,raise_if_fatal_device_context_error
    from rtx.autotune.preparation_runtime import _warm_measurement_graph
    from rtx.autotune.store import ResidualTuningStore
    from rtx.moe import MXFP8MoEConfig
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.kernels.mxfp8_moe_prepared import compile_expert_packed_gemm,compile_expert_quant
    saved=json.loads(args.source_result.read_text())
    initial=MXFP8MoEConfig.from_dict(saved['expert_config'])
    if args.gemm_policy:
        from rtx.configs.mxfp8 import MXFP8GemmConfig
        doc=json.loads(args.gemm_policy.read_text())
        initial=replace(initial,prepared=replace(initial.prepared,down=MXFP8GemmConfig(**doc)))
    if args.source_stage!='down':
        initial=replace(initial,prepared=replace(initial.prepared,down=getattr(initial.prepared,args.source_stage)))
    assert initial.prepared.quant.scale_layout=='row_major'
    stream=torch.cuda.Stream();torch.cuda.set_stream(stream)
    report=dict(status='running',argv=__import__('sys').argv,
        runtime_digest=_python_source_digest(Path.cwd()),hardware=hardware_attributes(),
        torch=str(torch.__version__),cuda=torch.version.cuda,cases={},
        graph_repetitions=32 if args.capture=='none' else 1,
        scope='Synthetic values at real GEMM geometries; pure BF16 uses decoded MXFP8 values; graph timing excludes preparation unless labelled')
    atomic_json(args.output,report)
    def capture(call):
        for _ in range(3):call()
        stream.synchronize();g=torch.cuda.CUDAGraph()
        with torch.cuda.graph(g,stream=stream):
            for _ in range(32):call()
        _warm_measurement_graph(g)
        return g
    def errors(actual,expected):
        delta=actual.float()-expected.float()
        nrms=float(delta.norm()/expected.float().norm().clamp_min(1e-12));maximum=float(delta.abs().max())
        assert bool(actual.isfinite().all()) and (nrms<.003 or maximum<2e-5),(nrms,maximum)
        return dict(nrms=nrms,max_abs=maximum)
    for name in (CASES if args.case=='all' else [args.case]):
        e,m,n,k=CASES[name];torch.manual_seed(819)
        a=torch.randn(e,m,k,device='cuda',dtype=torch.bfloat16)*.3
        b=torch.randn(e,n,k,device='cuda',dtype=torch.bfloat16)*.03
        qa,qb=_moe_prepared.prepare(((a,'row',False),(b,'row',False)),initial.prepared.quant)
        decode=lambda pair:(pair[0].float().view(e,-1,k//32,32)*pair[1].float().unsqueeze(-1)).flatten(2).bfloat16()
        da,db=decode(qa),decode(qb)
        expected=torch.bmm(da.float(),db.float().transpose(-1,-2)).bfloat16()
        out=torch.empty(e,m,n,device='cuda',dtype=torch.bfloat16)
        bo=torch.empty_like(out);raw_out=torch.empty_like(out)
        def bfmm(a,b,out):return torch.bmm(a,b.transpose(-1,-2),out=out)
        bf=torch.compile(bfmm,fullgraph=True,dynamic=False,
            options={'triton.cudagraphs':False,'max_autotune':True,'autotune_in_subproc':True})
        bf(da,db,bo);errors(bo,expected)
        bf(a,b,raw_out)
        def functional_mm(a,b):return torch.bmm(a,b.transpose(-1,-2))
        functional=torch.compile(functional_mm,fullgraph=True,dynamic=False,
            options={'triton.cudagraphs':False,'max_autotune':True,'autotune_in_subproc':True})
        hold={}
        def free_output():hold['pure']=functional(da,db)
        def free_raw():hold['raw']=functional(a,b)
        free_output();errors(hold['pure'],expected)
        shape=(m*e//min(4,e),n,e,m,min(4,e),k)
        adapter=make_moe_adapter(shape,None,initial=initial,phase='forward',
            device={'multiprocessor_count':torch.cuda.get_device_properties(0).multi_processor_count})
        # Pure-GEMM labels must never masquerade as whole-expert training data.
        adapter=replace(adapter,context=replace(adapter.context,family='mxfp8_expert_linear_micro',
            workload=dict(batch=e,m=m,n=n,k=k,phase='forward'),
            tags=dict(measurement='pure_gemm_outer_graph',seed=819)),extra_features_fn=None)
        if args.gemm_policy:
            # A common-MLP schedule uses the common GEMM contract, not the
            # additional grouped-MoE register-role policy. Do not retune its
            # registers merely to fit an unrelated diagnostic adapter.
            def generic_rejection(c):
                why=c.prepared.down.rejection(MXFP8Problem(m,n,k))
                return None if why is None else ('implementation_rejected',why)
            adapter=replace(adapter,context=replace(adapter.context,family='mxfp8_packed_gemm_micro'),
                            rejection_fn=generic_rejection)
        runners={}
        def runner(c):
            key=adapter.config_id(c)
            if key not in runners:
                tensors=(qa[0].flatten(0,1),qb[0].flatten(0,1),
                         qa[1].flatten(0,1),qb[1].flatten(0,1),out.flatten(0,1))
                if args.gemm_policy and e==1:
                    from rtx.kernels.mxfp8_gemm import compile_mxfp8_gemm
                    fn=compile_mxfp8_gemm(MXFP8Problem(m,n,k),c.prepared.down)
                    runners[key]=lambda:fn(*tensors)
                else:
                    fn=compile_expert_packed_gemm(e,MXFP8Problem(m,n,k),c.prepared.down,
                        transport=c.prepared.transports[1])
                    runners[key]=lambda:fn(*tensors,tensors[-1])
            return runners[key]
        pool=epilogue_templates(adapter) if args.epilogue_panel else templates(adapter)
        def evaluate(c):
            start=time.monotonic();graph=None
            try:
                call=runner(c);call();check=errors(out,expected)
                graph=capture(call)
                values,meta=measure_cuda_batches(graph.replay,samples=7,calls_per_sample=1,
                    device=a.device,min_batch_ms=3.)
                graph.replay();stream.synchronize();errors(out,expected)
                values=[v/32 for v in values]
                return TrialOutcome('ok',median_ms=statistics.median(values),timings_ms=values,
                    compile_ms=(time.monotonic()-start)*1000,metadata=dict(errors=check,measurement=meta))
            except Exception as ex:
                raise_if_fatal_device_context_error(ex)
                return TrialOutcome('compile_error',error=f'{type(ex).__name__}: {ex}')
            finally:
                if graph is not None:graph.reset()
        if args.record:
            prior=json.loads(args.record.read_text())['cases'][name]
            winner=MXFP8MoEConfig.from_dict(prior['winner'])
        else:
            indexed={adapter.config_id(c):c for c in pool}
            focused=replace(adapter,axes={'template':tuple(indexed)},evaluator=evaluate,
                update_fn=lambda c,key,value:indexed[value])
            policy=AutotunePolicy(max_trials=len(pool),max_seconds=120,confirmation_repeats=0,
                                 exploration_trials=0,pool_size=16)
            store=ResidualTuningStore(args.output.parent/'micro-residuals'/name)
            tuner=make_effort_autotuner(focused,store,policy,candidates=pool,
                progress=lambda s:print(name,s,flush=True))
            tuner.coverage_proposals=[Proposal(c,'structural') for c in pool[1:]]
            winner=tuner.tune().config
        fp=runner(winner);fp();errors(out,expected)
        if args.capture!='none':
            call=fp if args.precision=='mxfp8' else (free_output if args.precision=='bf16' else lambda:bf(da,db,bo))
            end=time.monotonic()+.3
            while time.monotonic()<end:call()
            stream.synchronize()
            if args.capture=='torch':
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA]) as prof:
                    call();stream.synchronize()
                prof.export_chrome_trace(str(args.output.with_suffix('.trace.json')))
            else:
                torch.cuda.cudart().cudaProfilerStart()
                with torch.cuda.nvtx.range(name+'/'+args.precision):
                    for _ in range(3 if args.capture=='timeline' else 1):call()
                stream.synchronize();torch.cuda.cudart().cudaProfilerStop()
            check=errors(out if args.precision=='mxfp8' else (hold['pure'] if args.precision=='bf16' else bo),expected)
            report['cases'][name]=dict(shape=[e,m,n,k],winner=winner.to_dict(),capture=args.precision,errors=check)
            report['status']='complete';atomic_json(args.output,report);return
        specs=((m,k,'row',False),(n,k,'row',False))
        quant=compile_expert_quant(e,specs,initial.prepared.quant)
        sources=(a.flatten(0,1),b.flatten(0,1))
        destinations=(qa[0].flatten(0,1),qb[0].flatten(0,1))
        scales=(qa[1].flatten(0,1),qb[1].flatten(0,1))
        def prepare():quant(sources[0],sources,destinations,scales)
        def prepared():prepare();fp()
        operations=dict(bf16=free_output,bf16_raw=free_raw,
                        mxfp8=fp,quantization=prepare,quant_plus_gemm=prepared)
        layouts=dict(bf16_a=list(da.stride()),bf16_b=list(db.stride()),bf16_out=list(bo.stride()))
        if args.layout_panel:
            ac=da.transpose(-1,-2).contiguous().transpose(-1,-2)
            orow=torch.empty_like(bo)
            def free_output_col():hold['col']=functional(ac,db)
            extra=dict(bf16_mandatory_out=lambda:bf(da,db,bo),
                       bf16_acol_orow=lambda:bf(ac,db,orow),bf16_acol_free_output=free_output_col)
            for label,call in extra.items():call()
            for value in (orow,*hold.values()):errors(value,expected)
            layouts.update(bf16_acol=list(ac.stride()),
                noncontiguous_out_control='Unsupported by Dynamo; not substituted with eager',
                bf16_free_output=list(hold['pure'].stride()),bf16_acol_free_output=list(hold['col'].stride()))
            operations.update(extra)
        graphs={label:capture(call) for label,call in operations.items()}
        timings={label:[] for label in graphs}
        order=list(graphs)
        for visit in range(4):
            for label in (order if visit%2==0 else reversed(order)):
                g=graphs[label];_warm_measurement_graph(g)
                values,meta=measure_cuda_batches(g.replay,samples=7,calls_per_sample=1,
                    device=a.device,min_batch_ms=3.)
                values=[v/32 for v in values]
                timings[label].append(dict(visit=visit,**statistics_ms(values),**meta))
        medians={label:statistics.median(row['median_ms'] for row in rows) for label,rows in timings.items()}
        report['cases'][name]=dict(shape=[e,m,n,k],useful_flops=2*e*m*n*k,
            winner=winner.to_dict(),production=adapter.initial_config.to_dict(),
            template_count=len(pool),median_ms=medians,rounds=timings,
            pure_speedup=medians['bf16']/medians['mxfp8'],
            dynamic_speedup=medians['bf16_raw']/medians['quant_plus_gemm'],
            errors=errors(out,expected),layouts=layouts,
            pure_reference='FP32 BMM of exactly decoded E4M3/E8M0 values; BF16 final cast')
        atomic_json(args.output,report);print('RESULT',name,medians,flush=True)
        for graph in graphs.values():graph.reset()
        del graphs;gc.collect();torch.cuda.synchronize()
    report['status']='complete';atomic_json(args.output,report)


if __name__=='__main__':main()
