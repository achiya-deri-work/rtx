"""Current-study budget curves, exact conditional effects and held-out ranking.

Uses the shared autotune observation loader/features. No automatic deployment;
model rankings are tested on logged candidate pools, not prospective searches.
"""
from __future__ import annotations

import argparse
from collections import Counter,defaultdict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import statistics

from benchmarks.profile_moe_active_budget import write
from benchmarks.analyze_moe_priors import curve
from rtx.autotune.pretrained import load_offline_observations
from rtx.autotune.prior import prior_features


def flatten(value,prefix=''):
    result={}
    if isinstance(value,dict):
        for key,item in value.items():result.update(flatten(item,prefix+'.'+key if prefix else key))
    elif isinstance(value,(list,tuple)):
        for index,item in enumerate(value):result.update(flatten(item,f'{prefix}[{index}]'))
    else:result[prefix]=value
    return result


def semantic_geometry(row):
    """One shape across boards/replicates must stay in the same CV fold."""
    workload={k:v for k,v in row.features.items() if k.startswith('context.workload.')
              or k=='offline.moe.sequences'}
    if not workload:raise ValueError('Missing component shape features')
    value=(row.family,row.kernel_revision,workload)
    return hashlib.sha256(json.dumps(value,sort_keys=True).encode()).hexdigest()[:24]


def canonical_features(row):
    features=prior_features(row.features)
    config=row.serialized_config;p=config.get('prepared',{})
    if (config.get('family')=='grouped_prepared' and p.get('input_preparation')!='gather_bf16'
            and p.get('forward_engine')!='onchip'):
        features={k:v for k,v in features.items() if not k.startswith('config.gather_mode=')}
        features['config.gather_mode=flat']=1.
    return features


def exact_effects(rows):
    """Compare only configurations identical in every other serialized field."""
    by_context=defaultdict(list)
    for row in rows:
        if row.outcome.timing_eligible:by_context[row.context_id].append(row)
    effects=[]
    for context,items in by_context.items():
        values=[(row,flatten(row.serialized_config)) for row in items]
        for index,(a,af) in enumerate(values):
            for b,bf in values[index+1:]:
                keys=set(af)|set(bf)
                changed=[k for k in keys if af.get(k)!=bf.get(k)]
                if len(changed)!=1:continue
                key=changed[0];av,bv=af.get(key),bf.get(key)
                left,right=a,b
                if str(av)>str(bv):left,right=b,a;av,bv=bv,av
                effects.append(dict(context=context,family=left.family,axis=key,a=av,b=bv,
                    a_config=left.config_id,b_config=right.config_id,a_ms=left.score,b_ms=right.score,
                    b_over_a=right.score/left.score))
    groups=defaultdict(list)
    for row in effects:groups[(row['family'],row['axis'],str(row['a']),str(row['b']))].append(row)
    summary=[]
    for (family,axis,a,b),items in groups.items():
        by_owner=defaultdict(list)
        for row in items:by_owner[row['context']].append(row['b_over_a'])
        ratios=[statistics.median(v) for v in by_owner.values()]
        summary.append(dict(family=family,axis=axis,a=a,b=b,matched_pairs=len(items),
            contexts=len(ratios),median_b_over_a=statistics.median(ratios),
            b_wins=sum(v<.99 for v in ratios),a_wins=sum(v>1.01 for v in ratios)))
    return effects,sorted(summary,key=lambda r:(-r['contexts'],r['family'],r['axis']))


def adopted_residual_root(root,document):
    """Keep raw provenance immutable while allowing the whole bundle to move."""
    declared=Path(document['residual_root'])
    relocated=root.parent/declared.parent.name/declared.name
    target=relocated if relocated.exists() else declared
    source=target/'result.json'
    if not source.exists() or hashlib.sha256(source.read_bytes()).hexdigest()!=document['sha256']:
        raise ValueError(('Missing or changed adopted pilot',target))
    return target


def load_studies(roots):
    from benchmarks.study_moe_heuristics import result_validation
    observations=[];owners={};audits=[]
    for root in roots:
        manifest=json.loads((root/'manifest.json').read_text())
        for context in manifest['contexts']:
            folder=root/context['id'];path=folder/'result.json'
            if not path.exists() or json.loads(path.read_text()).get('status')!='complete':continue
            result_validation(json.loads(path.read_text()),context,manifest['identity'],manifest['settings'])
            if (folder/'adoption.json').exists():
                folder=adopted_residual_root(root,json.loads((folder/'adoption.json').read_text()))
            rows,audit=load_offline_observations([folder]);audits.append(audit)
            if audit.get('malformed',0):raise ValueError(('Malformed residuals',folder,audit))
            unique={}
            for row in sorted(rows,key=lambda r:(r.finished_at,r.sequence,r.observation_id)):
                unique.setdefault((row.context_id,row.config_id),row)
            for row in unique.values():
                original=row.context_id
                # Invocation-local prefixes must not be inflated by another
                # workload that happens to share this component geometry.
                new=hashlib.sha256((str(root.resolve())+context['id']+original).encode()).hexdigest()[:24]
                owners[new]=dict(workload=context['id'],partition=context['partition'],
                                 original_context=original,component_geometry=semantic_geometry(row),root=str(root.resolve()))
                observations.append(replace(row,context_id=new,features=canonical_features(row)))
    return observations,owners,audits


def budget_score(items,order,budget=8):
    best=min((r.score for r in items if r.outcome.timing_eligible),default=None)
    selected=min((items[i].score for i in order[:budget] if items[i].outcome.timing_eligible),default=None)
    return None if best is None or selected is None else selected/best-1


def prospective_results(roots):
    """Actual finite-budget searches, not a reranking of logged pools."""
    from benchmarks.study_moe_heuristics import result_validation
    rows=[];missing=[]
    for root in roots:
        if not (root/'manifest.json').exists():
            missing.append(str(root));continue
        manifest=json.loads((root/'manifest.json').read_text())
        for context in manifest['contexts']:
            arms={}
            for mode in ('control','prior'):
                path=root/context['id']/mode/'result.json'
                if not path.exists():continue
                value=json.loads(path.read_text())
                if value.get('status')!='complete':continue
                result_validation(value,context,manifest['identity'],manifest['settings'])
                arms[mode]=value
            if len(arms)!=2:
                missing.append(str(root/context['id']));continue
            control,prior=arms['control'],arms['prior']
            for name in ('input','upstream','parameters'):
                if control['identities'][name]!=prior['identities'][name]:
                    raise ValueError(('Unmatched prospective comparison',context['id'],name))
            phases=('mxfp8_step','router_step','expert_step','shared_step')
            timings={mode:{key:statistics.median(v['median_ms'] for v in value['timings'][key])
                           for key in phases} for mode,value in arms.items()}
            rows.append(dict(workload=context['id'],layout=context['layout'],root=str(root),
                source_digest=control['source_digest'],prior_sha256=manifest['prior_sha256'],
                hard_ids_identical=control['identities']['routing']==prior['identities']['routing'],
                timings_ms=timings,prior_over_control={key:timings['prior'][key]/timings['control'][key] for key in phases},
                evaluated_trials={mode:{key:value['evaluated_trials'] for key,value in arm['tuning_counts'].items()}
                                  for mode,arm in arms.items()}))
    return dict(complete=not missing,rows=rows,missing=missing,deployment_enabled=False,
        caveats=['Fresh actual searches; two HOME boards of the same SKU',
                 'Arms run in separate processes, not interleaved timing visits',
                 'Hard routing equality is recorded, never inferred from soft numerical agreement'])


def ranking_study(rows,owners):
    import numpy as np
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.ensemble import ExtraTreesRegressor
    from sklearn.tree import DecisionTreeRegressor,export_text
    from sklearn.model_selection import GroupKFold
    reports={};rules={}
    for family in sorted({r.family for r in rows}):
        eligible_contexts=Counter(r.context_id for r in rows if r.family==family)
        items=[r for r in rows if r.family==family and eligible_contexts[r.context_id]>=8]
        train=[r for r in items if owners[r.context_id]['partition']=='discovery']
        if len({owners[r.context_id]['component_geometry'] for r in train})<4:continue
        grouped=defaultdict(list)
        for row in items:grouped[row.context_id].append(row)
        scale={key:statistics.median(r.score for r in value if r.outcome.timing_eligible)
               for key,value in grouped.items() if any(r.outcome.timing_eligible for r in value)}
        train=[r for r in train if r.context_id in scale]
        features=DictVectorizer(sparse=False)
        matrix=features.fit_transform([r.features for r in train])
        target=np.array([math.log(r.score/scale[r.context_id]) if r.outcome.timing_eligible else
                         math.log(100) for r in train])
        groups=np.array([owners[r.context_id]['component_geometry'] for r in train])
        configurations={'shallow_tree':lambda:DecisionTreeRegressor(max_depth=5,min_samples_leaf=8,random_state=0),
            'extra_trees':lambda:ExtraTreesRegressor(n_estimators=96,max_depth=14,min_samples_leaf=3,
                                                    max_features=.8,n_jobs=6,random_state=0)}
        models={};results={}
        for name,factory in configurations.items():
            oof=np.zeros(len(train))
            for fit,test in GroupKFold(n_splits=min(4,len(set(groups)))).split(matrix,target,groups):
                model=factory().fit(matrix[fit],target[fit]);oof[test]=model.predict(matrix[test])
            predictions={(r.context_id,r.config_id):float(v) for r,v in zip(train,oof)}
            model=factory().fit(matrix,target);models[name]=model
            unseen=[r for r in items if owners[r.context_id]['partition']=='holdout']
            if unseen:
                scores=model.predict(features.transform([r.features for r in unseen]))
                predictions.update({(r.context_id,r.config_id):float(v) for r,v in zip(unseen,scores)})
            evaluations=[]
            train_original=set(groups)
            for context,value in grouped.items():
                ordered=sorted(value,key=lambda r:(r.finished_at,r.sequence,r.observation_id))
                if any((r.context_id,r.config_id) not in predictions for r in ordered):continue
                # Protect the actual incumbent and spend only seven slots on
                # model proposals. Invalid rows consume slots, as in reality.
                ranked=[0]+sorted(range(1,len(ordered)),key=lambda i:predictions[(context,ordered[i].config_id)])
                rng=np.random.default_rng(0)
                random_scores=[budget_score(ordered,[0,*rng.permutation(range(1,len(ordered)))]) for _ in range(200)]
                evaluations.append(dict(context=context,**owners[context],candidates=len(ordered),
                    known_component_geometry=owners[context]['component_geometry'] in train_original,
                    model_regret8=budget_score(ordered,ranked),
                    actual_prefix_regret8=budget_score(ordered,list(range(len(ordered)))),
                    random_regret8_median=float(np.median([v for v in random_scores if v is not None]))
                        if any(v is not None for v in random_scores) else None,
                    chosen_config_ids=[ordered[i].config_id for i in ranked[:8]]))
            results[name]=evaluations
        tree=models['shallow_tree']
        rules[family]=export_text(tree,feature_names=list(features.get_feature_names_out()),decimals=4)
        reports[family]=dict(discovery_rows=len(train),discovery_geometries=len(set(groups)),
            models=results,important_features=[dict(feature=str(features.get_feature_names_out()[i]),
                importance=float(models['extra_trees'].feature_importances_[i])) for i in
                np.argsort(models['extra_trees'].feature_importances_)[-20:][::-1]])
    return reports,rules


def freeze_discovery_prior(rows,owners,roots,path):
    """Use the library's portable JSON trainer, never pickle sklearn objects."""
    import gzip
    from rtx.autotune.core import stable_id
    from rtx.autotune.prior import train_bundle,semantic_contract,SCHEMA_VERSION,FEATURE_REVISION
    expected={(str(root.resolve()),r['id']) for root in roots for r in
              json.loads((root/'manifest.json').read_text())['contexts'] if r['partition']=='discovery'}
    seen={(v['root'],v['workload']) for v in owners.values() if v['partition']=='discovery'}
    if seen!=expected:raise ValueError('Finish all discovery contexts before freezing a prior')
    training=[r for r in rows if owners[r.context_id]['partition']=='discovery']
    grouped=defaultdict(list);contracts={}
    for row in training:
        contract=semantic_contract(row.family,row.kernel_revision,row.features)
        key=stable_id(contract);contracts[key]=contract;grouped[key].append(row)
    bundles={}
    for key,items in grouped.items():
        print('TRAIN portable prior',contracts[key]['family'],len(items),flush=True)
        bundles[key]=train_bundle(items,contracts[key])
    digest=stable_id(sorted((r.observation_id,r.config_id,r.outcome.status,r.outcome.median_ms) for r in training))
    document=dict(schema_version=SCHEMA_VERSION,feature_revision=FEATURE_REVISION,
        deployment_enabled=False,training_data_sha256=digest,bundles=bundles,
        training_workloads=sorted(seen),source_paths=[str(p.resolve()) for p in roots],
        feature_canonicalization='prepared non-onchip FP8 gather_mode aliases -> flat; revision 1',
        qualification='Discovery-only; research artifact requires prospective held-out qualification',
        scope='Two HOME RTX 5090 boards; no cross-SKU deployment claim')
    path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+'.tmp')
    with gzip.open(temporary,'wt') as out:json.dump(document,out,sort_keys=True,allow_nan=False)
    temporary.replace(path)
    return dict(path=str(path.resolve()),training_observations=len(training),bundles=len(bundles),deployment_enabled=False)


def analyze(roots,output,*,models=True,freeze_prior=None):
    rows,owners,audits=load_studies(roots);output.mkdir(parents=True,exist_ok=True)
    groups=defaultdict(list)
    for row in rows:groups[row.context_id].append(row)
    curves=[dict(context=key,family=items[0].family,**owners[key],**curve(items)) for key,items in groups.items()]
    pairs,effects=exact_effects(rows)
    ranking,rules=ranking_study(rows,owners) if models else ({},{})
    prior=freeze_discovery_prior(rows,owners,roots,freeze_prior) if freeze_prior else None
    report=dict(schema=1,deployment_enabled=False,observations=len(rows),contexts=len(groups),
        portable_prior=prior,
        loading=audits,budget_curves=curves,matched_effects=effects,ranking=ranking,
        caveats=['No hard pruning or runtime policy installed',
                 'Exact one-field matches are controlled configuration evidence, not isolated kernel attribution',
                 'Model rankings use logged pools; prospective search still required',
                 'OOF folds group by semantic component geometry across boards/replicates, not random observation rows',
                 'Holdout known_component_geometry is true when another architecture shared that component',
                 'Failed/unstable candidates consume finite ranking slots; only timing_eligible outcomes score',
                 'Both devices are HOME RTX 5090; no cross-SKU generalization demonstrated'])
    write(output/'summary.json',report);write(output/'matched_pairs.json',pairs)
    write(output/'tree_rules.json',rules)
    import pyarrow as pa
    import pyarrow.parquet as pq
    export=[dict(context=r.context_id,**owners[r.context_id],family=r.family,config_id=r.config_id,
        eligible=r.outcome.timing_eligible,status=r.outcome.status,median_ms=r.outcome.median_ms,
        strategy=r.strategy,elapsed_s=r.elapsed_s,features=json.dumps(r.features,sort_keys=True),
        config=json.dumps(r.serialized_config,sort_keys=True)) for r in rows]
    if export:pq.write_table(pa.Table.from_pylist(export),output/'observations.parquet',compression='zstd')
    # Whole-layer measurements stay separate from component search objectives.
    whole=[]
    for root in roots:
        manifest=json.loads((root/'manifest.json').read_text())
        for context in manifest['contexts']:
            path=root/context['id']/'result.json'
            if not path.exists():continue
            value=json.loads(path.read_text())
            if value.get('status')!='complete':continue
            item=dict(workload=context['id'],partition=context['partition'],layout=context['layout'],
                      samples=context['samples'],total_tokens=context['total_tokens'],**context['case'],
                      source_digest=value['source_digest'],result_path=str(path.resolve()),
                      live_growth_bytes=value['memory']['live_growth_bytes'])
            item.update({name+'_ms':statistics.median(v['median_ms'] for v in visits)
                         for name,visits in value['timings'].items()})
            item['compact_checkpoint_speedup']=item['bf16_compact_checkpoint_step_ms']/item['mxfp8_step_ms']
            whole.append(item)
    if whole:
        import csv
        pq.write_table(pa.Table.from_pylist(whole),output/'whole_layer.parquet',compression='zstd')
        with (output/'whole_layer.csv').open('w',newline='') as handle:
            writer=csv.DictWriter(handle,fieldnames=list(dict.fromkeys(k for row in whole for k in row)))
            writer.writeheader();writer.writerows(whole)
    print(json.dumps(dict(observations=len(rows),contexts=len(groups),matched_pairs=len(pairs),
                         model_families=list(ranking))),flush=True)
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('roots',nargs='+',type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--no-models',action='store_true')
    p.add_argument('--freeze-prior',type=Path,help='Freeze shared-library research prior using completed discovery only')
    p.add_argument('--prospective',nargs='*',type=Path,default=[])
    args=p.parse_args();analyze(args.roots,args.output,models=not args.no_models,freeze_prior=args.freeze_prior)
    if args.prospective:write(args.output/'prospective.json',prospective_results(args.prospective))


if __name__=='__main__':main()
