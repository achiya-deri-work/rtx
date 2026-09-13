"""Post-collection pool ceiling versus frozen-model selection diagnostics."""
import argparse
from collections import Counter,defaultdict
import json
from pathlib import Path
import statistics

from tools.predictor_study import atomic
from tools.research.structural_sampling import pool_diagnostics


def analyze(root,artifact):
    root=Path(root)
    records=[json.loads(p.read_text()) for p in sorted((root/'tasks').glob('*/result.json'))]
    numerics=defaultdict(list)
    for r in records:
        for name,v in r.get('numerics',{}).items():
            if v['finite']:numerics[name].append(v)
    panels=defaultdict(list);latencies={};by_id={}
    for r in records:
        t=r['task']
        if t['kind']!='panel':continue
        cid=t['config_id'];by_id[cid]=r
        for source in t['sources']:panels[source].append(cid)
        if r['status']=='ok':
            latencies[cid]=statistics.median([r['outcome']['median_ms'],*[v['median_ms'] for v in r['verification']]])
    result=dict(completed=len(records),statuses=dict(Counter(r['task']['kind']+':'+r['status'] for r in records)),
        numerics={k:dict(cases=len(v),max_normalized_rms=max(x['normalized_rms'] for x in v),
                        max_absolute_error=max(x['max_abs'] for x in v)) for k,v in numerics.items()},
        panels={},rankers={},automatically_reinstated=False)
    for name,ids in panels.items():
        times=[latencies[i] for i in ids if i in latencies]
        result['panels'][name]=dict(attempted=len(ids),successful=len(times),
            failures=dict(Counter(by_id[i]['status'] for i in ids if by_id[i]['status']!='ok')),
            best_ms=min(times,default=None),median_ms=statistics.median(times) if times else None)
    if artifact and by_id:
        from rtx._portable_predictor import load_predictor
        from rtx.autotune.adapters import make_mxfp8_prequant_adapter
        from rtx.kernels.mxfp8 import MXFP8Problem
        representative=next(r for r in by_id.values() if 'hardware' in r)
        adapter=make_mxfp8_prequant_adapter(MXFP8Problem(*representative['task']['shape']),lambda c:None,
                                          device=representative['hardware'])
        model=load_predictor(artifact,research=True)
        result['predictor_digest']=model.digest
        if model.bundle(adapter) is None:raise ValueError('model contract does not match panel')
        for name,ids in panels.items():
            configs=[adapter.deserialize(by_id[i]['task']['config']) for i in ids]
            for head in ('residual','mlp'):
                ordered=model.order(adapter,configs,head)
                selected=[adapter.config_id(c) for c in ordered[:8]]
                result['rankers'][name+':'+head]=dict(pool_diagnostics(latencies,ids,selected),
                    selected_failed_trials=sum(by_id[i]['status']!='ok' for i in selected),
                    trials=len(selected))
    atomic(root/'coverage_analysis.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('output')
    p.add_argument('--predictor',default='autotune_priors/tournament_predictors_v1.json.gz')
    a=p.parse_args();analyze(a.output,a.predictor)
