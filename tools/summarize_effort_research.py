"""Paired, workload-clustered summaries for research_effort.py outputs."""
import argparse,csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def summarize(root):
    root=Path(root);rows=pq.read_table(root/'metrics.parquet').to_pylist();summaries=[]
    rng=np.random.default_rng(1729)
    for partition in ('all','family','device'):
        groups=defaultdict(list)
        for r in rows:groups[(r['protocol'],r['model'],r['budget'],r.get(partition,'all'))].append(r)
        for (protocol,model,budget,value),rs in groups.items():
            regret=np.array([r['regret'] for r in rs]);random=np.array([r['random_regret'] for r in rs])
            clusters=defaultdict(list)
            for r in rs:clusters[r['shape']].append(r['random_regret']-r['regret'])
            sums=np.array([sum(v) for v in clusters.values()]);counts=np.array([len(v) for v in clusters.values()])
            ids=rng.integers(0,len(sums),size=(2000,len(sums)))
            ci=np.quantile(sums[ids].sum(1)/counts[ids].sum(1),[.025,.975])
            summaries.append(dict(protocol=protocol,model=model,budget=budget,partition=partition,value=value,
                contexts=len(rs),workload_clusters=len(clusters),mean_regret=float(regret.mean()),
                median_regret=float(np.median(regret)),p90_regret=float(np.quantile(regret,.9)),
                within2=float(np.mean(regret<=.02)),random_mean_regret=float(random.mean()),
                paired_gain_ci_low=float(ci[0]),paired_gain_ci_high=float(ci[1])))
    pq.write_table(pa.Table.from_pylist(summaries),root/'summary.parquet')
    with (root/'summary.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    (root/'summary.json').write_text(json.dumps(summaries,indent=2))
    for r in summaries:
        if r['partition']=='all' and r['budget']==8:print(json.dumps(r))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root');summarize(p.parse_args().root)
