"""Versioned, JSON-only portable priors; learned pruning is always soft.

Models are frozen offline. A prior returns candidate order, never a runtime
winner, and never changes adapter legality or numerical validation.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import gzip
import json
import math
from pathlib import Path
import random

import numpy as np

from .core import stable_id
from .cost_model import GradientBoostedCostModel, GradientBoostedFeasibilityModel
from .first_hit import bounded_candidates, model_features
from .pretrained import ContextRankingModel, _observation_documents, _usable_for_latency, load_offline_observations
from .outcomes import TrialOutcome

SCHEMA_VERSION = 1
FEATURE_REVISION = 1
POLICY_NAMES = {'precision','activation','phase','silu_math','scaling','operand_state',
                'outer_scale_scope','pass','scale_policy'}


def semantic_contract(family, revision, features):
    policy={k:v for k,v in features.items() if k.startswith('context.regime=') or
            (k.startswith(('context.workload.','context.tags.'))
             and k.split('=',1)[0].rsplit('.',1)[-1] in POLICY_NAMES)}
    return {'family':family,'revision':revision,'policy':policy}


def prior_features(features):
    # Hardware ratios already supplied by each adapter remain available.
    # No recorded resource counters, software identities or output timings.
    return model_features(features)


def shape_distance(a,b):
    return sum(abs(math.log2(max(1,a.get('context.workload.'+k,1))/
                              max(1,b.get('context.workload.'+k,1)))) for k in ('m','n','k','h','o'))


def aggregate(rows):
    """Repeated confirmations do not receive extra training weight."""
    groups=defaultdict(list)
    for row in rows:
        if row.successful and not _usable_for_latency(row):continue
        groups[(row.context_id,row.config_id,row.outcome.status)].append(row)
    result=[]
    for group in groups.values():
        row=group[0]
        if row.successful:
            row=replace(row,outcome=replace(row.outcome,
                median_ms=float(np.median([r.score for r in group])),timings_ms=[]))
        result.append(replace(row,features=prior_features(row.features)))
    return result


def train_bundle(rows, contract):
    raw=rows;rows=aggregate(rows)
    ranker=ContextRankingModel(GradientBoostedCostModel(n_estimators=16,ensembles=2,max_depth=3))
    ranker.fit(rows)
    feasible=GradientBoostedFeasibilityModel(n_estimators=16,ensembles=2,
        negative_statuses=('compile_error','implementation_rejected'))
    negatives=sum(r.outcome.status in feasible.negative_statuses for r in rows)
    if negatives>=8:feasible.fit(rows)
    # This is preparation cost, NOT pure compiler cost: historical compile_ms
    # includes setup and may include GPU-lease waits. Never use it to hard prune.
    preparation=[]
    for r in rows:
        ms=r.outcome.compile_ms
        if ms is not None and math.isfinite(ms) and ms>0 and r.successful:
            preparation.append(replace(r,outcome=TrialOutcome('ok',median_ms=ms)))
    cost=GradientBoostedCostModel(n_estimators=12,ensembles=2,max_depth=3)
    cost.fit(preparation)
    winners={}
    for r in rows:
        if r.successful and (r.context_id not in winners or r.score<winners[r.context_id].score):
            winners[r.context_id]=r
    return dict(contract=contract,ranker=ranker.state_dict() if ranker.fitted else None,
        feasibility=feasible.state_dict() if feasible.fitted else None,
        preparation_cost=cost.state_dict() if cost.fitted else None,
        status_counts=dict(Counter(r.outcome.status for r in rows)),
        training_ids_hash=stable_id(sorted(r.observation_id for r in raw)),
        training_contexts=sorted({r.context_id for r in raw}),
        winners=[dict(context_id=r.context_id,config_id=r.config_id,config=r.serialized_config,
                      features=r.features) for r in winners.values()],
        limitations=['Preparation cost includes setup/cache/lease effects',
                     'Feasibility is uncalibrated and used only for soft ordering',
                     'No learned rule is a hard rejection'])


class FrozenPrior:
    def __init__(self,document,*,research=False):
        if document.get('schema_version')!=SCHEMA_VERSION or document.get('feature_revision')!=FEATURE_REVISION:
            raise ValueError('unsupported portable prior schema/features')
        if not research and not document.get('deployment_enabled',False):
            raise ValueError('prior is research-only until prospective qualification')
        self.document=document;self._models={}

    @classmethod
    def load(cls,path,**kwargs):
        path=Path(path)
        opener=gzip.open if path.suffix=='.gz' else open
        with opener(path,'rt') as stream:return cls(json.load(stream),**kwargs)

    def bundle(self,adapter):
        key=stable_id(semantic_contract(adapter.context.family,adapter.context.kernel_revision,
                                       adapter.context.features()))
        return self.document['bundles'].get(key)

    def retrieval(self,adapter,limit=32):
        bundle=self.bundle(adapter)
        if bundle is None:return []
        target=adapter.context.features();candidates=[]
        for row in sorted(bundle['winners'],key=lambda r:shape_distance(r['features'],target)):
            try:candidates.append(adapter.deserialize(row['config']))
            except (ValueError,TypeError,KeyError):continue
        return bounded_candidates(adapter,candidates,limit)

    def order(self,adapter,pool,*,policy='hybrid',seed=0,mix=(2,4,2),
              feasibility_penalty=.25,preparation_penalty=.05):
        """Return a full permutation. Truncation is an explicit caller budget."""
        if policy not in ('ordered','random','retrieval','ranker','hybrid'):
            raise ValueError('unknown prior policy')
        if len(mix)!=3 or any(not isinstance(n,int) or n<0 for n in mix) or sum(mix)<1:
            raise ValueError('mix must contain three nonnegative integer slot counts with positive sum')
        if any(not math.isfinite(x) or x<0 for x in (feasibility_penalty,preparation_penalty)):
            raise ValueError('soft penalties must be finite and nonnegative')
        pool=list(pool)
        if not pool:return []
        if policy=='ordered':return pool
        if policy=='random':
            random.Random(seed).shuffle(pool);return pool
        bundle=self.bundle(adapter)
        if bundle is None:return pool
        ids=[adapter.config_id(c) for c in pool];positions={cid:i for i,cid in enumerate(ids)}
        retrieved=[]
        for r in sorted(bundle['winners'],key=lambda r:shape_distance(r['features'],adapter.context.features())):
            i=positions.get(r['config_id'])
            if i is not None and i not in retrieved:retrieved.append(i)
        # Short retrieval budgets need neither model deserialization nor
        # candidate feature extraction/prediction.
        if policy=='retrieval':
            return [pool[i] for i in retrieved+[i for i in range(len(pool)) if i not in retrieved]]
        key=stable_id(bundle['contract'])
        if key not in self._models:
            self._models[key]=tuple(cls.from_state_dict(bundle[name]) if bundle.get(name) else None
                for name,cls in (('ranker',ContextRankingModel),('feasibility',GradientBoostedFeasibilityModel),
                                 ('preparation_cost',GradientBoostedCostModel)))
        ranker,feasibility,cost=self._models[key]
        features=[prior_features(adapter.features(c)) for c in pool]
        predicted,uncertainty=ranker.predict(features) if ranker else (np.ones(len(pool)),np.ones(len(pool)))
        probability=feasibility.predict(features)[0] if feasibility else np.ones(len(pool))
        preparation=cost.predict(features)[0] if cost else np.ones(len(pool))
        # Penalties are deliberately bounded. Sparse failure evidence cannot
        # erase a promising configuration or overrule static legality.
        score=np.log(np.maximum(predicted,1e-12))+feasibility_penalty*(1-np.clip(probability,0,1))
        score+=preparation_penalty*np.clip(np.log(np.maximum(preparation,1e-9)/max(float(np.median(preparation)),1e-9)),-2,2)
        ranked=sorted(range(len(pool)),key=lambda i:(float(score[i]),ids[i]))
        if policy=='ranker':order=ranked
        else:
            # 2 retrieval, 4 model, 2 diverse slots per block of eight.
            # Diversity uses feature-space distance; no prediction calibration
            # is assumed from the bagged model's uncertainty estimate.
            keys=sorted(set().union(*(f.keys() for f in features)))
            matrix=np.array([[math.log1p(abs(f.get(k,0))) for k in keys] for f in features])
            matrix=matrix[:,np.ptp(matrix,axis=0)>0]
            nearest=np.full(len(pool),np.inf)
            order=[];remaining=set(range(len(pool)))
            for slot in range(len(pool)):
                position=slot%sum(mix)
                source=retrieved if position<mix[0] else ranked
                available=[i for i in source if i in remaining]
                if position>=mix[0]+mix[1] and order:
                    chosen=max(sorted(remaining),key=lambda i:nearest[i])
                else:chosen=available[0] if available else next(i for i in ranked if i in remaining)
                remaining.remove(chosen);order.append(chosen)
                nearest=np.minimum(nearest,np.abs(matrix-matrix[chosen]).sum(axis=1))
        return [pool[i] for i in order]


def build(paths,output):
    rows,report=load_offline_observations(paths)
    contracts={}
    for _,r in _observation_documents(paths):
        contracts[r['context_id']]=semantic_contract(r['family'],r['kernel_revision'],r['features'])
    grouped=defaultdict(list)
    for row in rows:grouped[stable_id(contracts[row.context_id])].append(row)
    bundles={}
    for index,(key,group) in enumerate(sorted(grouped.items())):
        c=contracts[group[0].context_id]
        print(f'TRAIN {index+1}/{len(grouped)} {c["family"]} rows={len(group)}',flush=True)
        bundles[key]=train_bundle(group,c)
    document=dict(schema_version=SCHEMA_VERSION,feature_revision=FEATURE_REVISION,
        deployment_enabled=False,training_data_sha256=report['dataset_sha256'],
        source_paths=[str(p) for p in paths],ingestion={k:v for k,v in report.items()
            if k not in ('source_files','observation_identity_hashes')},bundles=bundles)
    path=Path(output);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with gzip.open(temporary,'wt') as stream:json.dump(document,stream,sort_keys=True)
    temporary.replace(path)
    print(f'PRIOR {path} bundles={len(bundles)} bytes={path.stat().st_size}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('paths',nargs='+')
    parser.add_argument('--output',required=True)
    args=parser.parse_args();build(args.paths,args.output)


if __name__=='__main__':main()
