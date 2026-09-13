"""Audit and summarize prospective policy journals from directories or ZIPs."""
import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import zipfile
import pyarrow as pa
import pyarrow.parquet as pq
from .core import stable_id
from .first_hit_study import atomic_json


def documents(paths):
    for path in map(Path,paths):
        if path.suffix=='.zip':
            with zipfile.ZipFile(path) as z:
                names=set(z.namelist())
                for name in sorted(names):
                    if not name.endswith('/trial.json'):continue
                    journal=name[:-len('trial.json')]+'residuals/observations.jsonl'
                    yield json.loads(z.read(name)),[json.loads(l) for l in z.read(journal).splitlines()] if journal in names else []
        else:
            for file in sorted(path.rglob('trial.json')):
                journal=file.parent/'residuals/observations.jsonl'
                yield json.loads(file.read_text()),[json.loads(l) for l in journal.read_text().splitlines()] if journal.exists() else []


def summarize(paths,output):
    groups=defaultdict(dict);seen=set();duplicates=0
    for trial,observations in documents(paths):
        identity=stable_id(trial)
        if identity in seen:duplicates+=1;continue
        seen.add(identity)
        context=json.loads(json.dumps(trial['context']))
        context['tags'].pop('prior_policy',None)
        key=stable_id([context,trial['prior_sha256']])
        groups[key][trial['task']['policy']]=(trial,observations)
    records=[];mismatches=[]
    for group,arms in groups.items():
        pools={t['pool_id'] for t,_ in arms.values()}
        if len(pools)!=1:mismatches.append(group);continue
        valid=[]
        for _,rows in arms.values():
            for r in rows:
                o=r['outcome'];stable=o.get('metadata',{}).get('sampling',{}).get('collection',{}).get('stationary')
                if o['status']=='ok' and stable is not False:valid.append(o['median_ms'])
        oracle=min(valid) if valid else None
        for policy,(trial,rows) in arms.items():
            for budget in (4,8,16,32,64,128):
                prefix=rows[:budget];good=[]
                for r in prefix:
                    o=r['outcome'];stationary=o.get('metadata',{}).get('sampling',{}).get('collection',{}).get('stationary')
                    if o['status']=='ok' and stationary is not False:good.append(o['median_ms'])
                records.append(dict(group_id=group,family=trial['task']['family'],policy=policy,
                    prior_sha256=trial['prior_sha256'],pool_id=trial['pool_id'],budget=budget,
                    observed=len(prefix),budget_reached=len(rows)>=budget,
                    compared_policies=len(arms),complete_policy_group=set(arms)==set(trial.get('policies',['ordered','random','retrieval','hybrid'])),
                    best_ms=min(good) if good else None,regret=(min(good)/oracle-1) if good and oracle else None,
                    failures=sum(r['outcome']['status']!='ok' for r in prefix),
                    observed_evaluation_seconds=sum(r['elapsed_s'] for r in prefix)))
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    if records:
        with (root/'policies.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        pq.write_table(pa.Table.from_pylist(records),root/'policies.parquet')
    atomic_json(root/'audit.json',dict(groups=len(groups),duplicate_trials=duplicates,pool_mismatches=mismatches,
        limitations=['Regret uses the union of observed stable successes, not an exhaustive oracle',
                     'Compare only budget_reached=true rows in complete matched policy groups',
                     'Evaluation seconds may include GPU-lease waits; caches are shared']))
    print(f'SUMMARY groups={len(groups)} mismatches={len(mismatches)} output={root}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('paths',nargs='+');p.add_argument('--output',required=True)
    a=p.parse_args();summarize(a.paths,a.output)


if __name__=='__main__':main()
