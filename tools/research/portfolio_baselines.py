"""Measure portfolio value separately from learned ranking; observed pools only."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import numpy as np
from .evidence_dataset import load
from .portfolio_study import contract, portfolio_ids, splits
from tools.research_effort import identity
from tools.predictor_study import atomic


def run(dataset,study):
    study=Path(study)
    rows=load(dataset)
    for row in rows:
        row['shape']=identity([row['features'].get('context.workload.m'),row['features'].get('context.workload.k')])
    results=[]
    for protocol,tr,te in splits(rows):
        key=hashlib.sha256(protocol.encode()).hexdigest()[:12]
        portfolios=json.loads((study/f'portfolio-{key}.json').read_text())['portfolios']
        supported={contract(r) for r in tr}
        contexts=defaultdict(list)
        for row in te:
            if contract(row) in supported:contexts[row['context']].append(row)
        for context,values in contexts.items():
            values.sort(key=lambda r:r['config_id'])
            observed={r['config_id']:i for i,r in enumerate(values)}
            seeds=portfolio_ids(portfolios,values[0])
            available=[observed[cid] for cid in seeds if cid in observed]
            times=np.asarray([r['ms'] for r in values]);best=float(times.min())
            rng=np.random.default_rng(int(context[:8],16))
            for budget in (8,16,32,64):
                if len(values)<budget:continue
                prefix=available[:max(1,budget//4)]
                remaining=[i for i in range(len(values)) if i not in prefix]
                regrets=[]
                for _ in range(64):
                    ids=prefix+list(rng.choice(remaining,budget-len(prefix),replace=False))
                    regrets.append(float(times[ids].min()/best-1))
                results.append(dict(protocol=protocol,context=context,shape=values[0]['shape'],
                    budget=budget,mean_regret=statistics.mean(regrets),
                    portfolio_proposed=len(seeds),portfolio_measured=len(available)))
        print('BASELINE',protocol,flush=True)
    atomic(study/'portfolio_random_metrics.json',results)
    groups=defaultdict(list)
    for row in results:groups[(row['protocol'],row['budget'])].append(row['mean_regret'])
    atomic(study/'portfolio_random_summary.json',[dict(protocol=p,budget=b,contexts=len(v),mean_regret=statistics.mean(v))
                                                for (p,b),v in groups.items()])


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset',required=True);p.add_argument('--study',required=True)
    a=p.parse_args();run(a.dataset,a.study)
