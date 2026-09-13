"""Device-held-out shortlist replay, including recorded failed candidates."""
import argparse
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
import time
import numpy as np
from .core import stable_id
from .first_hit_study import atomic_json,grouped_key
from .pretrained import load_offline_observations,_observation_documents
from .prior import FrozenPrior,semantic_contract,train_bundle,aggregate,SCHEMA_VERSION,FEATURE_REVISION


def evaluate(paths,output):
    rows,ingestion=load_offline_observations(paths);contracts={}
    for _,r in _observation_documents(paths):
        contracts[r['context_id']]=semantic_contract(r['family'],r['kernel_revision'],r['features'])
    groups=defaultdict(list)
    for r in rows:groups[stable_id(contracts[r.context_id])].append(r)
    metrics=[];partitions=[];root=Path(output);root.mkdir(parents=True,exist_ok=True)
    for number,(key,data) in enumerate(sorted(groups.items())):
        contract=contracts[data[0].context_id]
        devices=sorted({grouped_key(r,'device') for r in data})
        print(f'EVALUATE {number+1}/{len(groups)} {contract["family"]} devices={len(devices)}',flush=True)
        for device in devices:
            train=[r for r in data if grouped_key(r,'device')!=device]
            if not train:continue
            test=aggregate([r for r in data if grouped_key(r,'device')==device])
            bundle=train_bundle(train,contract)
            prior=FrozenPrior(dict(schema_version=SCHEMA_VERSION,feature_revision=FEATURE_REVISION,
                deployment_enabled=False,bundles={key:bundle}),research=True)
            contexts=defaultdict(list)
            for r in test:contexts[r.context_id].append(r)
            partitions.append(dict(contract=contract,heldout=str(device),
                train_contexts=sorted({r.context_id for r in train}),test_contexts=sorted(contexts)))
            for context,observations in contexts.items():
                # Static rejections are removed by the runtime adapter too.
                # Conflicting outcomes for one config are not a deterministic
                # counterfactual; omit them and retain the raw evidence.
                by_config=defaultdict(list)
                for r in observations:by_config[r.config_id].append(r)
                pool=[v[0] for v in by_config.values() if len(v)==1 and v[0].outcome.status!='implementation_rejected']
                pool.sort(key=lambda r:(r.sequence,r.config_id))
                valid=[r for r in pool if r.successful]
                if not valid or len(pool)<8:continue
                oracle=min(r.score for r in valid)
                base={**pool[0].features,**contract['policy']}
                adapter=SimpleNamespace(context=SimpleNamespace(family=contract['family'],
                    kernel_revision=contract['revision'],features=lambda:base),
                    config_id=lambda r:r.config_id,features=lambda r:r.features)
                for policy in ('ordered','retrieval','ranker','hybrid','random'):
                    for seed in range(20 if policy=='random' else 1):
                        start=time.perf_counter();order=prior.order(adapter,pool,policy=policy,seed=seed)
                        ordering_ms=(time.perf_counter()-start)*1000
                        for budget in (4,8,16):
                            chosen=order[:budget];success=[r for r in chosen if r.successful]
                            first=next((i+1 for i,r in enumerate(chosen) if r.successful),None)
                            metrics.append(dict(family=contract['family'],contract_id=key,context=context,
                                heldout_device=str(device),policy=policy,seed=seed,budget=budget,
                                candidates=len(pool),retained=len(chosen),first_valid_trial=first,
                                failures=sum(not r.successful for r in chosen),
                                regret=min(r.score for r in success)/oracle-1 if success else None,
                                retains_within_two_percent=any(r.score<=oracle*1.02 for r in success),
                                ordering_ms=ordering_ms))
        # Durable partial output allows inspection during the CPU-only study.
        atomic_json(root/'evaluation.json',dict(schema_version=1,deployment_enabled=False,
            completed_groups=number+1,total_groups=len(groups),partitions=partitions,metrics=metrics,
            training_data_sha256=ingestion['dataset_sha256'],
            limitations=['Observed pools only; no cold compiler time claim',
                         'Unstable successful timings and conflicting outcomes omitted',
                         'Research policy selection, not prospective qualification']))
    print(f'DONE {root}',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('paths',nargs='+');p.add_argument('--output',required=True)
    a=p.parse_args();evaluate(a.paths,a.output)


if __name__=='__main__':main()
