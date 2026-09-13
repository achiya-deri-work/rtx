"""Reproducible chronological / device-disjoint cost-model development study.

CPU-only; no production model promotion, no counterfactual online-UCB claims.
Inputs are immutable observation journals in ZIP archives or directories.
"""
import argparse,json,math,hashlib,time,zipfile,os
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import HistGradientBoostingRegressor,ExtraTreesRegressor


def identity(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()[:24]

def rows(paths):
    for path in map(Path,paths):
        if path.suffix=='.zip':
            with zipfile.ZipFile(path) as z:
                for n in sorted(z.namelist()):
                    if n.endswith('/observations.jsonl'):
                        for line in z.read(n).splitlines():yield json.loads(line)
        else:
            for f in sorted(path.rglob('observations.jsonl')):
                for line in f.read_text().splitlines():yield json.loads(line)


def features(f,sku=True):
    result={}
    for k,v in f.items():
        keep=k.startswith(('config.','context.workload.'))
        if sku:keep|=k.startswith(('derived.','context.device.properties.','context.device.architecture.','context.device.sku.'))
        if not keep or any(w in k.lower() for w in ('compiled','spill','telemetry','measured','source','name=','sku_family=')):continue
        if not math.isfinite(v):continue
        result[k]=math.copysign(math.log1p(abs(v)),v) if '=' not in k else v
    return result


def load(paths):
    counts=Counter();groups=defaultdict(list);seen=set()
    for r in rows(paths):
        oid=r.get('observation_id')
        if oid and oid in seen:counts['duplicate']+=1;continue
        seen.add(oid);counts['raw']+=1
        o=r['outcome'];counts[o['status']]+=1
        if o['status']!='ok':continue
        if o.get('metadata',{}).get('sampling',{}).get('collection',{}).get('stationary') is False:
            counts['unstable']+=1;continue
        ms=o.get('median_ms')
        if not isinstance(ms,(int,float)) or not math.isfinite(ms) or ms<=0:continue
        f=r['features'];c=r['config'];family=r['family']
        # Apply narrowly established numerical quarantines, not learned
        # performance pruning. Keep counts; never use these as winners.
        if family=='nvfp4_weight_prequant_fwd' and c.get('quant_x',{}).get('scale_reciprocal','direct')!='direct':
            counts['quarantined']+=1;continue
        gemms=[c.get('gemm',{})]
        if family=='mxfp8_recompute_mlp':gemms+=[c.get('gate_prequant',{}).get('gemm',{}),c.get('down_gemm',{})]
        if any(g.get('tile_m')==64 and g.get('stages')==1 and g.get('scale_role','consumers')=='consumers' and g.get('scale_layout','row_major')=='row_major' for g in gemms):
            counts['quarantined']+=1;continue
        device=next((k.split('=',1)[1] for k in f if k.startswith('context.device.name=')),None)
        if device is None:device=next((k.split('=',1)[1] for k in f if k.startswith('context.device.sku.sku_family=')),'unknown')
        semantic={k:v for k,v in f.items() if k.startswith('context.workload.') or
                  k.startswith('context.regime=') or k.startswith('context.tags.') and any(k.startswith('context.tags.'+n+'=') for n in ('precision','phase','scaling','scale_policy','operand_state','activation'))}
        ctx=identity([family,r['kernel_revision'],device,semantic])
        key=(ctx,r['config_id'])
        groups[key].append((r,ms,device,semantic))
    result=[]
    for (ctx,cid),values in groups.items():
        r,_,device,semantic=values[0]
        result.append(dict(context=ctx,config_id=cid,family=r['family'],revision=r['kernel_revision'],device=device,
            shape=identity([r['family'],semantic]),features=r['features'],ms=float(np.median([x[1] for x in values])),replicates=len(values)))
    counts['aggregated']=len(result)
    return result,dict(counts)


def metrics(test,pred,protocol,model):
    groups=defaultdict(list)
    for i,r in enumerate(test):groups[r['context']].append(i)
    result=[];rng=np.random.default_rng(1729)
    for ctx,ids in groups.items():
        if len(ids)<8:continue
        oracle=min(test[i]['ms'] for i in ids)
        order=sorted(ids,key=lambda i:pred[i])
        for b in (8,16,32):
            if len(ids)<b:continue
            regret=min(test[i]['ms'] for i in order[:b])/oracle-1
            random_regrets=[min(test[i]['ms'] for i in rng.choice(ids,b,replace=False))/oracle-1 for _ in range(32)]
            result.append(dict(protocol=protocol,model=model,context=ctx,shape=test[ids[0]]['shape'],device=test[ids[0]]['device'],family=test[ids[0]]['family'],budget=b,pool=len(ids),regret=regret,random_regret=float(np.mean(random_regrets))))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--train',nargs='+',required=True);p.add_argument('--test',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=a.threads)
    out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
    train,train_audit=load(a.train);test,test_audit=load(a.test)
    audit=dict(train=train_audit,test=test_audit,train_paths=a.train,test_paths=a.test,
        protocols='chronological and train-device-disjoint; no tuning on test',
        limits=['Observed successful candidate pools only; NOT off-policy evaluation of adaptive UCB',
                'This prospective study was previously inspected: development evidence, not untouched qualification',
                'Measurements without explicit stationarity flags remain eligible',
                'MLP quarantine conservatively filters stored inactive schedules too',
                'Replicated timings use configuration medians; no cold compile latency claim'])
    (out/'audit.json').write_text(json.dumps(audit,indent=2));print('AUDIT',audit,flush=True)
    all_metrics=[];fits=[]
    devices=sorted({r['device'] for r in test})
    for heldout in [None,*devices]:
        tr=[r for r in train if heldout is None or r['device']!=heldout]
        supported={(r['family'],r['revision']) for r in tr}
        te=[r for r in test if (heldout is None or r['device']==heldout) and (r['family'],r['revision']) in supported]
        protocol='chronological' if heldout is None else 'heldout:'+heldout
        context_medians=defaultdict(list)
        for r in tr:context_medians[r['context']].append(math.log(r['ms']))
        centers={k:np.median(v) for k,v in context_medians.items()}
        y=np.array([math.log(r['ms'])-centers[r['context']] for r in tr])
        for name,sku in [('hist_shape_config',False),('hist_sku_derived',True),('extra_trees_sku',True),('family_hist_sku',True)]:
            start=time.monotonic();vec=DictVectorizer(sparse=False,dtype=np.float32)
            def fs(r):return {**features(r['features'],sku),f'family={r["family"]}':1.,f'revision={r["revision"]}':1.}
            X=vec.fit_transform([fs(r) for r in tr]);T=vec.transform([fs(r) for r in te])
            model=(ExtraTreesRegressor(n_estimators=48,max_depth=18,min_samples_leaf=4,max_features=.2,n_jobs=a.threads,random_state=1729)
                if name.startswith('extra') else HistGradientBoostingRegressor(max_iter=100,max_depth=6,min_samples_leaf=20,l2_regularization=1.,random_state=1729))
            if name.startswith('family'):
                pred=np.zeros(len(te))
                for family in sorted({r['family'] for r in tr}):
                    train_ids=[i for i,r in enumerate(tr) if r['family']==family]
                    test_ids=[i for i,r in enumerate(te) if r['family']==family]
                    if not test_ids:continue
                    # Same hyperparameters, distinct family heads; no model
                    # selection or fitting uses held-out timing values.
                    model.fit(X[train_ids],y[train_ids]);pred[test_ids]=model.predict(T[test_ids])
            else:
                model.fit(X,y);pred=model.predict(T)
            fit=dict(protocol=protocol,model=name,train=len(tr),test=len(te),features=X.shape[1],seconds=time.monotonic()-start)
            fits.append(fit);all_metrics.extend(metrics(te,pred,protocol,name));print('FIT',fit,flush=True)
            pq.write_table(pa.Table.from_pylist(all_metrics),out/'metrics.parquet')
            (out/'fits.json').write_text(json.dumps(fits,indent=2))
    import sklearn
    (out/'environment.json').write_text(json.dumps({'sklearn':sklearn.__version__,'numpy':np.__version__,'seed':1729,'threads':a.threads},indent=2))
    print('DONE',out,flush=True)


if __name__=='__main__':main()
