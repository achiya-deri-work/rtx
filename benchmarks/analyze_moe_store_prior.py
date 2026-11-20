"""Summarize actual bounded searches; never infer trials from a seed pool."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import statistics


def search_summary(folder,pattern):
    from rtx.autotune.outcomes import TrialOutcome
    paths=list(folder.glob(pattern))
    if len(paths)!=1:raise ValueError(f'Expected one residual context, got {paths}')
    rows=[json.loads(line) for line in paths[0].read_text().splitlines() if line.strip()]
    identifiers=[r['config_id'] for r in rows]
    if len(identifiers)!=len(set(identifiers)):raise AssertionError('Repeated config evaluations')
    result=dict(observations=len(rows),outcomes=dict(Counter(r['outcome']['status'] for r in rows)),
                timing_eligible_observations=sum(TrialOutcome.from_dict(r['outcome']).timing_eligible for r in rows),
                journal=str(paths[0]),prefixes={})
    for budget in (8,16,32):
        if len(rows)<budget:continue
        prefix=rows[:budget]
        valid=[r for r in prefix if TrialOutcome.from_dict(r['outcome']).timing_eligible]
        if not valid:raise ValueError(f'No eligible timing in first {budget} observations')
        best=min(valid,key=lambda r:r['outcome']['median_ms'])
        store_indices=[]
        for i,r in enumerate(prefix):
            c=r['config']
            if (c.get('family')=='grouped_prepared' and c['prepared']['down']['epilogue'] in ('tma','hybrid')
                    and c['prepared']['forward_engine']=='packed') or (
                    c.get('down_backend')=='prequant' and c['down_gemm']['epilogue'] in ('tma','hybrid')):
                store_indices.append(i+1)
        result['prefixes'][str(budget)]=dict(best_ms=best['outcome']['median_ms'],
            timing_eligible_observations=len(valid),
            best_config_id=best['config_id'],config_ids=[r['config_id'] for r in prefix],
            first_store_trial=next(iter(store_indices),None),
            first_store_outcome=None if not store_indices else prefix[store_indices[0]-1]['outcome']['status'],
            strategies=dict(Counter(r['strategy'] for r in prefix)),
            initial_ms=prefix[0]['outcome'].get('median_ms'))
    if '16' in result['prefixes']:
        a,b=(result['prefixes'][str(n)]['best_ms'] for n in (8,16))
        result['eight_trial_regret_vs_sixteen']=a/b-1
    if '32' in result['prefixes']:
        for budget in (8,16):
            result[f'{budget}_trial_regret_vs_32']=(result['prefixes'][str(budget)]['best_ms']/
                                                  result['prefixes']['32']['best_ms']-1)
    return result


def summarize(root):
    result={}
    for folder in sorted(root.iterdir()):
        if not folder.is_dir() or not (folder/'result8.json').exists():continue
        reports={}
        for budget in (8,16,32):
            path=folder/f'result{budget}.json'
            if not path.exists():continue
            d=json.loads(path.read_text())
            if d['status']!='complete':raise ValueError(f'Incomplete {path}')
            if d['memory']['live_growth_bytes']!=0:raise AssertionError(path)
            if not d['validation']['all_finite']:raise AssertionError(path)
            timings={k:statistics.median(r['median_ms'] for r in v) for k,v in d['timings'].items()}
            reports[str(budget)]=dict(timings_ms=timings,tuning_counts=d['tuning_counts'],
                tokens=d['total_tokens'],lengths=d['lengths'],case=d['case'],
                physical_capacity=d['expert_plan']['physical_capacity'],
                native_vs_bf16_speedup=timings['bf16_checkpoint_step']/timings['mxfp8_step'],
                validation=d['validation'],memory=d['memory'],source_digest=d['source_digest'])
        result[folder.name]=dict(reports=reports,
            expert=search_summary(folder,'tuning/moe_effort/*/*/observations.jsonl'),
            shared=search_summary(folder,'shared-tuning/*/*/observations.jsonl'))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=summarize(args.root)
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    for key,row in result.items():
        print(key,*(f'{role}8/16={row[role].get("eight_trial_regret_vs_sixteen",float("nan")):.2%}'
                    for role in ('expert','shared')))


if __name__=='__main__':main()
