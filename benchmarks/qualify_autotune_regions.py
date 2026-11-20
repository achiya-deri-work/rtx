"""Fresh routing-only causal panel or actual progressive-budget search.

Use a disposable process under a GPU flock. Independent FP64 transport/hard/
metadata/adjoint evaluator is shared with production, not a private timer.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time


def whole_pair(baseline,candidate,output,dependency=None):
    if dependency and dependency.exists():
        if any(r.get('exit_code')==76 for r in json.loads(dependency.read_text())):
            raise RuntimeError('Prior device fault; no follow-on GPU work')
    saved={name:json.loads(path.read_text()) for name,path in (('ordinary',baseline),('regions',candidate))}
    for name,value in saved.items():
        if value.get('status')!='complete' or value['memory']['live_growth_bytes']:
            raise ValueError(('Unqualified arm',name))
        if not value['validation'].get('independent_quantized_fp64_experts'):raise ValueError('Missing independent oracle')
    for key in ('input','parameters','upstream'):
        if saved['ordinary']['identities'][key]!=saved['regions']['identities'][key]:raise ValueError(('Unmatched',key))
    import statistics
    import torch
    from benchmarks.profile_moe_routing_cost import construct,tensor_results
    from benchmarks.benchmark_moe_composition import capture_step,compare
    from benchmarks.benchmark_moe_execution import _tensor_digest
    from benchmarks.profile_moe_active_budget import write
    from rtx.autotune.measurement import measure_cuda_batches
    from rtx.autotune.preparation_runtime import _warm_measurement_graph
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    graphs={};owners={};checks={};live={};rounds={name:[] for name in saved}
    with torch.cuda.stream(stream):
        for name,value in saved.items():
            calls,_,data=construct(value,include_components=False)
            for name_in_result,name_in_data in (('expert_ids','ids'),('inverse','inverse')):
                if _tensor_digest(data[name_in_data])!=value['identities']['routing'][name_in_result]:
                    raise ValueError(('Actual selected route differs from saved arm',name,name_in_result))
            step=tensor_results(calls['whole_step']);owners[name]=(calls,data)
            reference=tuple(v.detach().clone() for v in step());stream.synchronize()
            graph,actual=capture_step(step,stream);graph.replay();stream.synchronize()
            checks[name]=compare(actual,reference);graphs[name]=(graph,actual)
            samples=[]
            for _ in range(5):
                result=step();del result
                stream.synchronize();samples.append(torch.cuda.memory_allocated())
            if len(set(samples))!=1:raise ValueError(('Live growth',name,samples))
            live[name]=samples
        for visit in range(4):
            for name in (list(saved) if visit%2==0 else list(reversed(saved))):
                _warm_measurement_graph(graphs[name][0])
                times,metadata=measure_cuda_batches(graphs[name][0].replay,samples=7,calls_per_sample=1,
                    min_batch_ms=10.,device='cuda')
                rounds[name].append(dict(visit=visit,median_ms=statistics.median(times),samples_ms=times,measurement=metadata))
                print('PAIRED',visit,name,statistics.median(times),flush=True)
        medians={name:statistics.median(v['median_ms'] for v in values) for name,values in rounds.items()}
        write(output,dict(complete=True,rounds=rounds,median_ms=medians,
            speedup=medians['ordinary']/medians['regions'],checks=checks,live_allocated_bytes=live,
            helper_revision=2, source_digest=saved['ordinary']['source_digest'],
            hard_ids_identical=saved['ordinary']['identities']['routing']==saved['regions']['identities']['routing'],
            caveat='Per-arm numerical oracles passed independently; differing hard IDs are not fixed-route equivalence',
            baseline=str(baseline),candidate=str(candidate)))
        for graph,_ in graphs.values():graph.reset()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('panel','search','whole-pair'),default='panel')
    p.add_argument('--tokens',type=int);p.add_argument('--experts',type=int)
    p.add_argument('--top-k',type=int,default=4);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--prior',type=Path);p.add_argument('--trials',type=int,nargs='+',default=[8,16,32])
    p.add_argument('--opening',choices=('ordinary','regions'),default='regions')
    p.add_argument('--prior-search',choices=('hybrid','retrieval','ranker'),default='hybrid')
    p.add_argument('--seed',type=int,default=29421)
    p.add_argument('--baseline',type=Path);p.add_argument('--candidate',type=Path);p.add_argument('--dependency',type=Path)
    args=p.parse_args()
    if args.mode=='whole-pair':return whole_pair(args.baseline,args.candidate,args.output,args.dependency)
    if not args.tokens or not args.experts:p.error('--tokens and --experts required')
    args.output.mkdir(parents=True,exist_ok=True)
    import torch
    from rtx.routing import RoutingKernelConfig
    from rtx.autotune.routing import RoutingEvaluator,make_routing_adapter,make_flat_effort_autotuner
    from rtx.autotune.effort import AutotunePolicy
    from rtx.autotune.hardware import static_device_profile
    from rtx.autotune.store import ResidualTuningStore
    from rtx.autotune.outcomes import raise_if_fatal_device_context_error
    from benchmarks.profile_moe_active_budget import identity,write
    initial=RoutingKernelConfig(qb_backend='native_column',metadata_backend='cute',quota_threshold=0)
    policy=AutotunePolicy.preset('balanced',max_trials=8,max_seconds=180,search_schedule='progressive',
        candidate_mix='mixed',samples=7,calls_per_sample=1,confirmation_repeats=0,
        prior_artifact=str(args.prior) if args.prior else None,allow_research_prior=bool(args.prior))
    policy=replace(policy,prior_search=args.prior_search,region_coverage=args.opening=='regions')
    source=identity(Path.cwd());started=time.monotonic()
    torch.manual_seed(args.seed)
    scores=torch.nn.functional.logsigmoid(torch.randn(args.tokens,args.experts,device='cuda')*1.5)
    spec=dict(initial=initial.to_dict(),workload=dict(top_k=args.top_k,sinkhorn_iterations=5,
        dual_iterations=5,repair_rounds=64))
    evaluator=RoutingEvaluator(scores,spec,'train',policy)
    adapter=make_routing_adapter(args.tokens,args.experts,args.top_k,evaluator.evaluate,initial=initial,
        device=static_device_profile(scores.device),tags=dict(study='region_development',source=source['runtime'],
            seed=args.seed,opening=args.opening))
    record=dict(source=source,workload=adapter.context.as_dict(),started_setup_s=time.monotonic()-started,
                mode=args.mode,opening=args.opening,prior=str(args.prior),results=[])
    write(args.output/'manifest.json',record)
    try:
        if args.mode=='panel':
            # Independently generated small physical panel: two ownership
            # levels per direction, two QB families; never a winner table.
            from rtx.routing import SinkhornKernelConfig
            forward=(initial.forward,replace(initial.forward,threads=256,row_groups=8,column_groups=1))
            backward=(initial.backward,replace(initial.backward,threads=256,row_groups=4,column_groups=1))
            cells=[(f,b,q) for f in range(2) for b in range(2) for q in ('native_column','torch')]
            for f,b,q in cells:
                c=adapter.deserialize(replace(initial,forward=forward[f],backward=backward[b],qb_backend=q).to_dict())
                if adapter.rejection(c):continue
                now=time.monotonic();outcome=evaluator.evaluate(c)
                record['results'].append(dict(forward=f,backward=b,qb=q,config=adapter.serialize(c),
                    outcome=outcome.as_dict(),elapsed_s=time.monotonic()-now))
                write(args.output/'result.json',record)
                print('CELL',f,b,q,outcome.status,outcome.median_ms,flush=True)
        else:
            store=ResidualTuningStore(args.output/'residuals')
            for budget in args.trials:
                active=replace(policy,max_trials=budget);evaluator.policy=active
                now=time.monotonic()
                tuner=make_flat_effort_autotuner(adapter,store,active,region_coverage=args.opening=='regions',
                    progress=lambda value:print(value,flush=True))
                result=tuner.tune()
                record['results'].append(dict(budget=budget,elapsed_s=time.monotonic()-now,
                    config=adapter.serialize(result.config),median_ms=result.median_ms,
                    evaluated_this_launch=result.evaluated_trials,checkpoint=tuner.checkpoint_key))
                write(args.output/'result.json',record)
                print('BUDGET',budget,result.median_ms,flush=True)
    except Exception as error:
        raise_if_fatal_device_context_error(error);raise
    finally:evaluator.close()
    record['complete']=True;record['invocation_s']=time.monotonic()-started
    write(args.output/'result.json',record)


if __name__=='__main__':main()
