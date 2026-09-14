"""Separate proposal coverage from selection using complete measured pools.

Development diagnosis only: these inspected outcomes must not become held-out
evidence for a policy subsequently designed from this report.
"""
import argparse
import json
import math
from pathlib import Path


def ranking_diagnosis(reference, ranking, observations, probes=(1, 2, 4, 8, 16)):
    """Screening-only quality of frozen rank prefixes, not simulated search.

    Failed reference candidates consume positions but supply no latency label.
    Ranking quality and the actual scheduler's exposure are reported separately.
    """
    ids=[r['config_id'] for r in reference]
    if len(set(ids)) != len(ids) or len(ranking) != len(ids) or set(ranking) != set(ids):
        raise ValueError('ranking must permute the unique complete reference pool')
    times={}
    for row in reference:
        outcome=row['outcome'];value=outcome.get('median_ms')
        if outcome['status']=='ok':
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                raise ValueError('invalid successful reference timing')
            times[row['config_id']]=value
    if not times:raise ValueError('no successful reference candidates')
    visited={r['config_id']:i+1 for i,r in enumerate(observations)}
    if len(visited)!=len(observations) or not set(visited)<=set(ids):
        raise ValueError('duplicate or foreign observed candidate')
    best=min(times.values())
    first=next((i+1 for i,cid in enumerate(ranking) if times.get(cid,math.inf)<=best*1.05),None)
    rows=[]
    for count in probes:
        if type(count) is not int or count<1:raise ValueError('probe counts must be positive integers')
        prefix=ranking[:count]
        successful=[times[cid] for cid in prefix if cid in times]
        rows.append(dict(requested_probes=count,available_positions=len(prefix),
            successful_positions=len(successful),best_reference_excess=(min(successful)/best-1) if successful else None))
    return dict(first_reference_within_5pct_rank=first,
        actual_trial_for_that_candidate=None if first is None else visited.get(ranking[first-1]),
        prefix_quality=rows,
        leading_candidates=[dict(rank=i+1,config_id=cid,actual_trial=visited.get(cid),
            reference_excess=times[cid]/best-1 if cid in times else None)
            for i,cid in enumerate(ranking[:16])],
        limitation='Rank-only screening excludes initial/protected/exploration costs and is not a replay of adaptive search.')


def diagnose(reference, observations, selected, budget):
    if isinstance(budget, bool) or not isinstance(budget, int) or budget <= 0:
        raise ValueError('budget must be a positive integer')
    times={}
    for row in reference:
        cid=row['config_id']
        if cid in times:raise ValueError('duplicate reference configuration')
        outcome=row['outcome'];value=outcome.get('median_ms')
        if outcome['status']=='ok':
            if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
                raise ValueError('invalid successful reference timing')
            times[cid]=value
        else:times[cid]=None
    prefix=observations[:budget]
    if len(prefix)!=budget or len({r['config_id'] for r in prefix})!=budget:
        raise ValueError('incomplete or duplicate budget prefix')
    if any(r['config_id'] not in times for r in prefix):
        raise ValueError('reference missing a proposed candidate')
    eligible=[r for r in prefix if r['outcome']['status']=='ok' and times[r['config_id']] is not None]
    if selected not in {r['config_id'] for r in eligible}:
        raise ValueError('selected configuration lacks successful prefix/reference evidence')
    best=min(v for v in times.values() if v is not None)
    best_seen=min(times[r['config_id']] for r in eligible)
    first_near=next((i+1 for i,r in enumerate(prefix) if r in eligible and
                    times[r['config_id']]<=best*1.05),None)
    protected=[r for r in prefix if r['strategy']=='initial_neighborhood']
    return dict(budget=budget,selected=selected,reference_best_ms=best,
        best_eligible_proposed_ms=best_seen,selected_reference_ms=times[selected],
        proposal_coverage_excess=best_seen/best-1,
        selection_excess_given_proposals=times[selected]/best_seen-1,
        first_eligible_within_5pct_trial=first_near,
        unsuccessful_search_trials=sum(r['outcome']['status']!='ok' for r in prefix),
        protected_coordinate_count=len({r.get('coordinate') for r in protected}),
        protected=[dict(config_id=r['config_id'],coordinate=r.get('coordinate'),
            reference_excess=None if times[r['config_id']] is None else times[r['config_id']]/best-1)
            for r in protected])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root');parser.add_argument('--output',required=True)
    args=parser.parse_args();root=Path(args.root)
    from .first_hit_costs import read_jsonl
    from .matched_audit import audit
    from .matched_gpu import write_new
    checked=audit(root)
    if not checked['complete']:raise ValueError(checked['issues'])
    pool=json.loads((root/'pool.json').read_text())
    reference=read_jsonl(root/'reference/journal/observations.jsonl')
    if {r['config_id'] for r in reference}!={r['config_id'] for r in pool['configs']}:
        raise ValueError('incomplete full-pool reference')
    by_config={json.dumps(r['config'],sort_keys=True):r['config_id'] for r in pool['configs']}
    rows=[];ranking_reports={}
    rankings=json.loads((root/'rankings.json').read_text())['rankings']
    for treatment in ('ordinary','portfolio','rules','portfolio_ml'):
        observations=read_jsonl(root/'results'/treatment/'observations.jsonl')
        ranking_reports[treatment]=ranking_diagnosis(reference,rankings[treatment],observations)
        for budget in (8,16,32,64):
            result=json.loads((root/'results'/f'{treatment}-{budget}.json').read_text())
            selected=by_config[json.dumps(result['winner'],sort_keys=True)]
            rows.append(dict(treatment=treatment,**diagnose(reference,observations,selected,budget)))
    write_new(args.output,dict(pool_id=pool['pool_id'],rows=rows,rankings=ranking_reports,runtime_promotion=False,
        limitation='Complete reference screening times, not paired inference or global optimality. '
                   'Inspected development evidence; do not evaluate a derived policy on these same labels.'))
    for r in rows:print(r['treatment'],r['budget'],round(r['proposal_coverage_excess']*100,3),
                       round(r['selection_excess_given_proposals']*100,3),r['first_eligible_within_5pct_trial'])


if __name__=='__main__':main()
