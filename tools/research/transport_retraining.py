"""Historical whole-(M,K) cross-validation of existing model choices.

Development screening only. No inspected prospective pool is loaded or fitted.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time


def shape_fold(row, folds=3):
    if type(folds) is not int or folds<2: raise ValueError('at least two folds required')
    values=[row['features']['context.workload.'+k] for k in ('m','k')]
    if any(type(v) not in (int,float) or v<=0 or int(v)!=v for v in values):
        raise ValueError('positive integral workload required')
    key=json.dumps([int(v) for v in values])
    return int(hashlib.sha256(key.encode()).hexdigest()[:16],16)%folds


def main():
    from threadpoolctl import threadpool_limits
    from .evidence_dataset import load
    from .models import FittedCandidate,REGISTRY
    from .portfolio_study import fit_portfolios,score,summary
    from tools.research_effort import identity
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    def save(name,value):
        with (args.output/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False)
    rows=[r for r in load(args.dataset) if r['split']=='train' and r['family']=='mxfp8_prequant_fwd']
    for r in rows:r['shape']=identity([r['features']['context.workload.m'],r['features']['context.workload.k']])
    save('plan.json',dict(dataset=str(args.dataset),rows=len(rows),folds=3,
        dataset_manifest_sha256=hashlib.sha256((args.dataset/'manifest.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        models=['extra_trees','hist_family'],views=['shape','interactions'],balance_contexts=[False,True],
        variant=0,threads=args.threads,runtime_promotion=False,
        scope='Historical train-only whole-MK cross-validation; observed stable pools, not adaptive search or prospective proof'))
    all_metrics=[]
    with threadpool_limits(limits=args.threads):
        for fold in range(3):
            train=[r for r in rows if shape_fold(r)!=fold]
            test=[r for r in rows if shape_fold(r)==fold]
            assert train and test and not ({r['shape'] for r in train}&{r['shape'] for r in test})
            portfolio=fit_portfolios(train)
            save(f'fold-{fold}.json',dict(train_rows=len(train),test_rows=len(test),
                train_shapes=sorted({r['shape'] for r in train}),test_shapes=sorted({r['shape'] for r in test})))
            for name in ('extra_trees','hist_family'):
                for view in ('shape','interactions'):
                    for balanced in (False,True):
                        model=FittedCandidate(REGISTRY[name],view,0,args.threads)
                        started=time.monotonic();model.fit(train,balance_contexts=balanced)
                        fit_s=time.monotonic()-started
                        started=time.monotonic();predictions=model.predict(test)
                        predict_s=time.monotonic()-started
                        key=f'{name}-{view}-balanced{int(balanced)}'
                        metrics=score(test,predictions,portfolio,'whole_mk_cv',key)
                        all_metrics.extend(metrics)
                        save(f'fold-{fold}-{key}.json',dict(fit_s=fit_s,predict_s=predict_s,
                            features=model.feature_count,metrics=metrics,
                            predictions=[dict(context=r['context'],config_id=r['config_id'],score=float(p))
                                         for r,p in zip(test,predictions)]))
                        print('FIT',fold,key,'fit_s',round(fit_s,3),'predict_s',round(predict_s,3),flush=True)
    save('summary.json',summary(all_metrics))
    print('DONE',args.output,flush=True)


if __name__=='__main__':main()
