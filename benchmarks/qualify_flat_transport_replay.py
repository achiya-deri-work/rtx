"""Paired saved-plan timing after explicit source and FP64 requalification.

No new tuning. Expert/shared plans and fixture tensors must match across arms.
Original observations are never rewritten or relabeled with a new source hash.
"""
import argparse
import json
import statistics
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study',type=Path,required=True)
    parser.add_argument('--gate',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    import torch
    from benchmarks.benchmark_moe_execution import _python_source_digest
    from benchmarks.profile_moe_routing_cost import construct,tensor_results
    from benchmarks.benchmark_moe_composition import capture_step,compare
    from benchmarks.profile_moe_active_budget import write
    from rtx.autotune.measurement import measure_cuda_batches
    from rtx.autotune.preparation_runtime import _warm_measurement_graph

    source=_python_source_digest(Path.cwd())
    gate=json.loads(args.gate.read_text())
    if not gate['passed'] or gate['source_digest']!=source:
        raise ValueError('Strict full-router requalification required on this exact source')
    arms={f'{arm}{budget}':args.study/arm/f'budget-{budget}.json'
          for arm in ('baseline','candidate') for budget in (8,32)}
    qualified={Path(r['fixture']).resolve() for r in gate['results'] if r['passed']}
    saved={name:json.loads(path.read_text()) for name,path in arms.items()}
    first=next(iter(saved.values()))
    for name,value in saved.items():
        if arms[name].resolve() not in qualified or value.get('status')!='complete':
            raise ValueError(('Missing qualification',name))
        if value['memory']['live_growth_bytes'] or not value['validation']['independent_quantized_fp64_experts']:
            raise ValueError(('Original qualification failed',name))
        for key in ('input','parameters','upstream'):
            if value['identities'][key]!=first['identities'][key]:raise ValueError(('Unmatched fixture',key))
        for key in ('expert_config','shared_config','router_affinity_layout'):
            if value[key]!=first[key]:raise ValueError(('Unmatched fixed component',key))
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    owners={};graphs={};checks={};live={};rounds={name:[] for name in arms}
    with torch.cuda.stream(stream):
        for name,original in saved.items():
            # The original immutable record is retained below. construct's
            # source guard checks this separately requalified execution view.
            execution={**original,'source_digest':source}
            calls,_,data=construct(execution,include_components=False)
            owners[name]=(calls,data)
            step=tensor_results(calls['whole_step'])
            reference=tuple(v.detach().clone() for v in step());stream.synchronize()
            graph,actual=capture_step(step,stream);graph.replay();stream.synchronize()
            checks[name]=compare(actual,reference);graphs[name]=(graph,actual)
            del reference
            samples=[]
            for _ in range(5):
                result=step();del result
                stream.synchronize();samples.append(torch.cuda.memory_allocated())
            # A released compiler temporary can lower the first sample; it
            # is not a leak. Require a stable final plateau and no net growth.
            if len(set(samples[-3:]))!=1 or samples[-1]>samples[0]:
                raise ValueError(('Live memory growth',name,samples))
            live[name]=samples
        order=list(arms)
        for visit in range(4):
            for name in order[visit:]+order[:visit]:
                _warm_measurement_graph(graphs[name][0])
                times,meta=measure_cuda_batches(graphs[name][0].replay,samples=9,calls_per_sample=1,
                                               min_batch_ms=10.,device='cuda')
                rounds[name].append(dict(visit=visit,median_ms=statistics.median(times),
                                         samples_ms=times,measurement=meta))
                print('PAIRED',visit,name,statistics.median(times),flush=True)
        medians={name:statistics.median(r['median_ms'] for r in rs) for name,rs in rounds.items()}
        write(args.output,dict(complete=True,source_digest=source,
            original_sources={n:v['source_digest'] for n,v in saved.items()},
            source_requalification=str(args.gate.resolve()),fixtures={n:str(p.resolve()) for n,p in arms.items()},
            median_ms=medians,rounds=rounds,checks=checks,live_allocated_bytes=live,
            improvements=dict(new8_vs_old8=1-medians['candidate8']/medians['baseline8'],
                new32_vs_old32=1-medians['candidate32']/medians['baseline32'],
                new32_vs_new8=1-medians['candidate32']/medians['candidate8']),
            caveat='Per-arm strict oracles; hard assignment hashes may differ across configurations'))
        for graph,_ in graphs.values():graph.reset()


if __name__=='__main__':main()
