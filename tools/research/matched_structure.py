"""Development-only structural associations in complete measured pools.

Candidate counts are not independent performance replicates. Associations do
not establish causal effects or authorize pruning or runtime promotion.
"""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics


def associations(configs, reference, ranking, fields, *, opening=8):
    ids = [c['config_id'] for c in configs]
    refs = [r['config_id'] for r in reference]
    if (not ids or len(ids)!=len(set(ids)) or len(refs)!=len(set(refs)) or
            set(ids)!=set(refs) or len(ranking)!=len(ids) or set(ranking)!=set(ids)):
        raise ValueError('require unique complete matched pool/reference/ranking')
    if type(opening) is not int or opening < 1 or not fields or len(fields)!=len(set(fields)):
        raise ValueError('invalid opening or structural fields')
    times = {}
    for r in reference:
        outcome = r['outcome']
        if outcome['status']=='ok':
            value=outcome.get('median_ms')
            if type(value) not in (int,float) or not math.isfinite(value) or value<=0:
                raise ValueError('invalid successful timing')
            times[r['config_id']]=value
    if not times: raise ValueError('no successful reference')
    best=min(times.values())
    groups=defaultdict(list)
    for row in configs:
        values=[]
        for field in fields:
            value=row['config']
            for key in field.split('.'):
                if not isinstance(value,dict) or key not in value:
                    raise ValueError('missing structural field: '+field)
                value=value[key]
            values.append(value)
        groups[json.dumps(values,sort_keys=True)].append(row['config_id'])
    ranks={cid:i+1 for i,cid in enumerate(ranking)}
    result=[]
    for key,members in sorted(groups.items()):
        successful=[times[cid] for cid in members if cid in times]
        near=[cid for cid in members if times.get(cid,math.inf)<=best*1.05]
        result.append(dict(values=json.loads(key),candidates=len(members),successful=len(successful),
            failed=len(members)-len(successful),near_5pct=len(near),
            top_opening_count=sum(ranks[cid]<=opening for cid in members),
            first_rank=min(ranks[cid] for cid in members),
            first_near_rank=min((ranks[cid] for cid in near),default=None),
            median_reference_excess=statistics.median(successful)/best-1 if successful else None,
            best_reference_excess=min(successful)/best-1 if successful else None))
    return dict(fields=fields,opening=opening,pool_size=len(ids),reference_best_ms=best,groups=result,
        runtime_promotion=False,causal_effect=False,
        limitation='Within inspected finite pools; correlated candidates are not independent replicates. '
                   'Opening counts describe ranking only, not the adaptive scheduler.')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots',nargs='+',type=Path)
    parser.add_argument('--fields',nargs='+',default=['gemm.scale_layout','gemm.scale_role','gemm.epilogue'])
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    reports=[]
    for root in args.roots:
        paths=[root/'pool.json',root/'rankings.json',root/'reference/journal/observations.jsonl']
        pool=json.loads(paths[0].read_text()); rankings=json.loads(paths[1].read_text())
        if rankings['pool_id']!=pool['pool_id']: raise ValueError('pool identity mismatch')
        reference=[json.loads(s) for s in paths[2].read_text().splitlines()]
        reports.append(dict(root=str(root),pool_id=pool['pool_id'],context=pool['context'],
            provenance={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
            treatments={name:associations(pool['configs'],reference,ranking,args.fields)
                        for name,ranking in rankings['rankings'].items()}))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as sink: json.dump(dict(schema=1,reports=reports),sink,indent=2,allow_nan=False)
    print('SAVED',args.output)


if __name__=='__main__':main()
