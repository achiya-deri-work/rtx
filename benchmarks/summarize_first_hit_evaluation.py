"""Summarize offline replay regret without implying prospective speedups."""
import argparse
import ast
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np
from rtx.autotune.first_hit_study import atomic_json


def summarize(path,output):
    document=json.loads(Path(path).read_text());groups=defaultdict(list)
    for row in document['metrics']:
        if row['status']!='offline_replay_successful_candidates_only':continue
        family,revision,semantics,_=ast.literal_eval(row['family'])
        semantic=dict(semantics)
        def category(name):
            return next((k.split('=',1)[1] for k in semantic if k.startswith('context.workload.'+name+'=')),'all')
        device='shape_holdout'
        if row['split']=='device':
            hardware=dict(ast.literal_eval(row['heldout']))
            device=str(int(hardware['context.device.multiprocessor_count']))+' SMs'
        groups[(family,revision,category('precision'),category('phase'),device,row['budget'])].append(row)
    summary=[]
    for key,rows in sorted(groups.items()):
        result=dict(zip(('family','revision','precision','phase','heldout_device','budget'),key))
        result['contexts']=len(rows)
        for policy in ('ranker','retrieval','ordered_portfolio','random'):
            values=np.array([r[policy+'_regret'+('_mean' if policy=='random' else '')] for r in rows])
            result[policy+'_mean_regret_percent']=float(values.mean()*100)
            result[policy+'_median_regret_percent']=float(np.median(values)*100)
            result[policy+'_p90_regret_percent']=float(np.quantile(values,.9)*100)
        summary.append(result)
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    atomic_json(root/'summary.json',dict(source=str(path),split=document['split'],groups=summary,
        limitations=document['limitations'],deployment_enabled=False))
    if summary:
        with (root/'summary.csv').open('w',newline='') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(summary[0]));writer.writeheader();writer.writerows(summary)
    print(f'SUMMARY groups={len(summary)} output={root}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('evaluation');p.add_argument('--output',required=True)
    a=p.parse_args();summarize(a.evaluation,a.output)


if __name__=='__main__':main()
