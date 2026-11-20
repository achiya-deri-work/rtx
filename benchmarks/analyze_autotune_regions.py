"""Audit actual-budget whole-layer ablations; never sum component timings."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from benchmarks.profile_moe_active_budget import write


def journal_prefixes(root):
    """Search evidence only; a search winner need not pass deployment checks."""
    from rtx.autotune.outcomes import TrialOutcome
    result=[]
    for shard in sorted(root.glob('prospective-gpu[01]')):
        manifest=json.loads((shard/'manifest.json').read_text())
        for context in manifest['contexts']:
            for arm in sorted((shard/context['id']).glob('*')):
                if not arm.is_dir():continue
                for path in sorted(arm.rglob('observations.jsonl')):
                    records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                    if not records:continue
                    outcomes=[TrialOutcome(**r['outcome']) for r in records]
                    eligible=[(i,r,o) for i,(r,o) in enumerate(zip(records,outcomes),1) if o.timing_eligible]
                    best_all=min((o.median_ms for _,_,o in eligible),default=None)
                    for budget in (8,16,32):
                        if len(records)<budget:continue
                        candidates=[(i,r,o) for i,r,o in eligible if i<=budget]
                        winner=min(candidates,key=lambda item:item[2].median_ms) if candidates else None
                        first=next((i for i,_,o in eligible if best_all is not None and o.median_ms<=best_all*1.02),None)
                        row=dict(context=context['id'],arm=arm.name,family=records[0]['family'],
                            journal_context_id=records[0]['context_id'],budget=budget,
                            total_observed=len(records),timing_eligible=len(candidates),
                            unique_configurations=len({r['config_id'] for r in records[:budget]}),
                            best_ms=None if winner is None else winner[2].median_ms,
                            winner_strategy=None if winner is None else winner[1]['strategy'],
                            own_observed_regret=None if winner is None else winner[2].median_ms/best_all-1,
                            first_within_2pct_of_own_observed=first,
                            elapsed_s=sum(r['elapsed_s'] for r in records[:budget]),
                            ineligible_elapsed_s=sum(r['elapsed_s'] for r,o in zip(records[:budget],outcomes[:budget])
                                                     if not o.timing_eligible),
                            journal=str(path.resolve()),scope='search only; no deployment promotion implied')
                        result.append(row)
    return result


def export_table(rows,path):
    if not rows:return
    columns=list(dict.fromkeys(k for row in rows for k in row))
    with path.with_suffix('.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=columns);writer.writeheader();writer.writerows(rows)
    import pyarrow as pa
    import pyarrow.parquet as pq
    pq.write_table(pa.Table.from_pylist(rows),path.with_suffix('.parquet'),compression='zstd')


def analyze(root,output):
    rows=[];failures=[];identities={};context_hashes={}
    for shard in sorted(root.glob('prospective-gpu[01]')):
        manifest=json.loads((shard/'manifest.json').read_text())
        identities[shard.name]=manifest['identity']
        if (shard/'progress.json').exists():
            failures.extend(dict(shard=shard.name,**r) for r in json.loads((shard/'progress.json').read_text()) if r['exit_code'])
        for context in manifest['contexts']:
            for path in sorted((shard/context['id']).glob('*/budget-*.json')):
                value=json.loads(path.read_text())
                if value.get('status')!='complete':continue
                if value['source_digest']!=manifest['identity']['runtime']:raise ValueError(('Runtime drift',path))
                if value['driver_sha256']!=manifest['identity']['drivers']['benchmark_moe_varlen.py']:
                    raise ValueError(('Driver drift',path))
                if value['memory']['live_growth_bytes']:raise ValueError(('Live memory growth',path))
                if not value['validation'].get('independent_quantized_fp64_experts'):raise ValueError('Missing oracle')
                signature={k:value['identities'][k] for k in ('input','upstream','parameters')}
                if context['id'] in context_hashes and context_hashes[context['id']]!=signature:
                    raise ValueError(('Unmatched input/parameters',context['id']))
                context_hashes[context['id']]=signature
                budget=value['tuning_budgets']['router']
                row=dict(context=context['id'],layout=context['layout'],arm=path.parent.name,budget=budget,
                    source=value['source_digest'],record_path=str(path.resolve()),
                    hard_ids=json.dumps(value['identities']['routing'],sort_keys=True),
                    live_growth_bytes=value['memory']['live_growth_bytes'])
                row.update({name+'_ms':statistics.median(r['median_ms'] for r in visits)
                            for name,visits in value['timings'].items()})
                for name,counts in value['tuning_counts'].items():
                    row[name+'_evaluated_this_launch']=counts['evaluated_trials']
                    row[name+'_search_s']=counts['elapsed_s']
                    row[name+'_invocation_s']=counts.get('invocation_s',counts['elapsed_s'])
                row['autotune_invocation_s']=sum(row[name+'_invocation_s'] for name in ('expert','shared','router'))
                rows.append(row)
    for row in rows:
        control=next((r for r in rows if r['context']==row['context'] and r['budget']==row['budget'] and r['arm']=='ordinary'),None)
        if control:
            row['whole_speedup_vs_same_budget']=control['mxfp8_step_ms']/row['mxfp8_step_ms']
            row['router_speedup_vs_same_budget']=control['router_step_ms']/row['router_step_ms']
            row['hard_ids_match_control']=row['hard_ids']==control['hard_ids']
        row['whole_speedup_vs_compact_bf16']=row['bf16_compact_checkpoint_step_ms']/row['mxfp8_step_ms']
    output.mkdir(parents=True,exist_ok=True)
    paired={}
    for path in sorted(root.glob('paired-gpu?.json')):
        paired[path.stem]=json.loads(path.read_text())
    prefixes=journal_prefixes(root)
    write(output/'summary.json',dict(rows=rows,failures=failures,paired=paired,identities=identities,
        journal_prefixes=prefixes,
        caveats=['Both boards are RTX5090; no cross-SKU claim',
                 'Each budget is per component, not a global sum of eight calls',
                 'Fresh winner caches; shared compilation caches and rotated treatment order',
                 'Cold invocation is caller autotuning time, not total outer compile/reference setup',
                 'Hard-ID changes are recorded; independent per-arm oracles do not imply bitwise policy identity',
                 'Phase scopes overlap and cannot be summed to reconstruct whole latency']))
    export_table(rows,output/'whole_layer')
    export_table(prefixes,output/'journal_prefixes')
    print(json.dumps(dict(complete_rows=len(rows),failed_jobs=len(failures),paired=list(paired))),flush=True)


def analyze_flat_transport_cycle(root,output):
    """Keep original search evidence distinct from final-source replay."""
    import statistics
    from rtx.autotune.outcomes import TrialOutcome
    from benchmarks.profile_moe_active_budget import write
    rows=[];prefixes=[];paired={};gates={};original_gates={}
    for gpu in (0,1):
        shard=root/f'gpu{gpu}'
        manifest=json.loads((shard/'manifest.json').read_text())
        replay=json.loads((root/f'paired-qualified-gpu{gpu}.json').read_text())
        gate=json.loads((root/f'gate-qualified-gpu{gpu}.json').read_text())
        if not replay['complete'] or not gate['passed'] or replay['source_digest']!=gate['source_digest']:
            raise ValueError(('Final qualification incomplete',gpu))
        paired[str(gpu)]=replay;gates[str(gpu)]=gate
        original_gates[str(gpu)]=json.loads((root/f'gate-gpu{gpu}.json').read_text())
        identity=None;fixed=None
        for arm in ('baseline','candidate'):
            for budget in (8,32):
                path=shard/arm/f'budget-{budget}.json';value=json.loads(path.read_text())
                if value['source_digest']!=manifest['identities'][arm]['runtime']:
                    raise ValueError(('Original source drift',path))
                signature={k:value['identities'][k] for k in ('input','parameters','upstream')}
                plans={k:value[k] for k in ('expert_config','shared_config','router_affinity_layout')}
                if identity is None:identity=signature;fixed=plans
                if identity!=signature or fixed!=plans:raise ValueError(('Unmatched components',path))
                row=dict(gpu=gpu,arm=arm,budget=budget,tokens=sum(value['lengths']),**value['case'],
                    original_search_source=value['source_digest'],replay_source=replay['source_digest'],
                    original_whole_ms=statistics.median(v['median_ms'] for v in value['timings']['mxfp8_step']),
                    original_router_ms=statistics.median(v['median_ms'] for v in value['timings']['router_step']),
                    paired_whole_ms=replay['median_ms'][f'{arm}{budget}'],
                    live_growth_bytes=value['memory']['live_growth_bytes'],
                    router_invocation_s=value['tuning_counts']['router']['invocation_s'],
                    router_evaluated_this_launch=value['tuning_counts']['router']['evaluated_trials'])
                rows.append(row)
            journals=list((shard/arm/'tuning').rglob('observations.jsonl'))
            for path in journals:
                records=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                if not records or records[0]['family']!='balanced_routing':continue
                if len(records)!=32 or len({r['config_id'] for r in records})!=32:
                    raise ValueError(('Expected 32 distinct actual trials',path))
                for budget in (8,16,32):
                    admitted=records[:budget]
                    eligible=[r for r in admitted if TrialOutcome(**r['outcome']).timing_eligible]
                    winner=min(eligible,key=lambda r:r['outcome']['median_ms']) if eligible else None
                    prefixes.append(dict(gpu=gpu,arm=arm,budget=budget,distinct=len(admitted),
                        timing_eligible=len(eligible),best_ms=winner['outcome']['median_ms'] if winner else None,
                        winner_strategy=winner['strategy'] if winner else None,
                        elapsed_s=sum(r['elapsed_s'] for r in admitted),
                        later_transport_trials=sum(r['strategy']=='flat_transport' for r in admitted),
                        scope='Original source search, not final-source retuning',journal=str(path)))
    output.mkdir(parents=True,exist_ok=True)
    write(output/'summary.json',dict(rows=rows,prefixes=prefixes,paired=paired,final_gates=gates,
        original_gates=original_gates,caveats=[
            'Two unseen flat shapes on two RTX5090 boards, not cross-SKU qualification',
            'Original search sources retained; final timings requalify fixed selected plans without retuning',
            'Expert/shared configurations and fixture tensors fixed across arms',
            'Per-arm hard IDs may differ; independent numerical checks are not bitwise cross-plan equivalence',
            'Old near-constant scalar-gradient failure retained, fixed by centered selected-softmax adjoint',
            'Whole-MoE gains are relative to the old tuner, not automatically relative to BF16']))
    export_table(rows,output/'whole_layer');export_table(prefixes,output/'journal_prefixes')
    print(json.dumps(dict(whole_records=len(rows),prefix_records=len(prefixes),strict_records=sum(len(g['results']) for g in gates.values()))))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--flat-transport-cycle',action='store_true')
    args=p.parse_args()
    if args.flat_transport_cycle:analyze_flat_transport_cycle(args.root,args.output)
    else:analyze(args.root,args.output)


if __name__=='__main__':main()
