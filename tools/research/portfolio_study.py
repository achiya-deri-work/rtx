"""Train-only compact portfolios and grouped, observed-pool model evaluation."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import time

import numpy as np
import joblib

from tools.predictor_study import atomic
from .evidence_dataset import load
from .models import FittedCandidate, REGISTRY
from tools.research_effort import identity


def contract(row):
    semantic = {k:v for k,v in row['features'].items() if '=' in k and
                k.startswith(('context.workload.', 'context.tags.precision=', 'context.tags.phase=',
                              'context.tags.scaling=', 'context.tags.scale_policy=', 'context.tags.operand_state='))}
    return json.dumps([row['family'], row['revision'], semantic], sort_keys=True)


def fit_portfolios(rows, size=16, tolerance=.05):
    """Greedy coverage of contexts within 5% of their TRAIN best observed.

    Unmeasured configurations cover no context; that is lack of evidence,
    not a slow-runtime label. Config payloads remain revision-bound.
    """
    contexts = defaultdict(list)
    for row in rows:
        contexts[row['context']].append(row)
    banks = defaultdict(lambda: defaultdict(set))
    configs = {}
    for context, values in contexts.items():
        best = min(r['ms'] for r in values)
        for row in values:
            if row['ms'] <= best * (1+tolerance):
                for bucket in ('*', row['shape_class']):
                    banks[(contract(row), bucket)][row['config_id']].add(context)
                configs[row['config_id']] = row['config']
    portfolios = {}
    for (key, bucket), candidates in banks.items():
        covered, selected = set(), []
        all_contexts = set().union(*candidates.values())
        for _ in range(size):
            remaining = [cid for cid in candidates if cid not in selected]
            if not remaining:
                break
            winner = min(remaining, key=lambda cid:(-len(candidates[cid]-covered), cid))
            if not candidates[winner]-covered:
                break
            selected.append(winner)
            covered.update(candidates[winner])
        portfolios[json.dumps([key, bucket])] = dict(config_ids=selected,
            configs=[configs[cid] for cid in selected], training_contexts=len(all_contexts),
            covered_training_contexts=len(covered))
    return portfolios


def portfolio_ids(portfolios, row):
    entries = [portfolios.get(json.dumps([contract(row), bucket]), {}).get('config_ids', [])
               for bucket in (row['shape_class'], '*')]
    return list(dict.fromkeys(cid for entry in entries for cid in entry))


def splits(rows):
    train = [r for r in rows if r['split']=='train']
    later = [r for r in rows if r['split']!='train']
    yield 'chronological_development', train, later
    held = {r['shape'] for r in rows if int(r['shape'][:8],16)%5==0}
    yield 'unseen_shapes', [r for r in rows if r['shape'] not in held], [r for r in rows if r['shape'] in held]
    yield 'unseen_shape_class:contraction', [r for r in rows if r['shape_class']!='contraction'], [r for r in rows if r['shape_class']=='contraction']
    for device in sorted({r['device'] for r in rows}):
        yield 'heldout_sku:'+device, [r for r in rows if r['device']!=device], [r for r in rows if r['device']==device]


def score(test, predictions, portfolios, protocol, model):
    contexts = defaultdict(list)
    for i,row in enumerate(test):
        contexts[row['context']].append(i)
    results = []
    for context, ids in contexts.items():
        ids.sort(key=lambda i:test[i]['config_id'])
        base = test[ids[0]]
        measured = {test[i]['config_id']:i for i in ids}
        seeds = portfolio_ids(portfolios, base)
        available = [measured[cid] for cid in seeds if cid in measured]
        order = sorted(ids, key=lambda i:(predictions[i], test[i]['config_id']))
        best = min(test[i]['ms'] for i in ids)
        good = {i for i in ids if test[i]['ms'] <= best*1.05}
        rng = np.random.default_rng(int(context[:8],16))
        for budget in (8,16,32,64):
            if len(ids) < budget:
                continue
            random_regrets = [min(test[i]['ms'] for i in rng.choice(ids,budget,replace=False))/best-1
                              for _ in range(64)]
            for policy in ('ranker', 'portfolio_then_ranker'):
                prefix = available[:max(1,budget//4)] if policy=='portfolio_then_ranker' else []
                selected = [*prefix, *(i for i in order if i not in prefix)][:budget]
                regret = min(test[i]['ms'] for i in selected)/best-1
                results.append(dict(protocol=protocol, model=model, policy=policy, context=context,
                    shape=base['shape'], family=base['family'], revision=base['revision'],
                    device=base['device'], budget=budget, pool=len(ids), regret=regret,
                    good_hit=bool(good.intersection(selected)),
                    random_regret=statistics.mean(random_regrets),
                    probability_beating_random=sum(regret < r for r in random_regrets)/len(random_regrets),
                    portfolio_measured=len(available), portfolio_proposed=len(seeds)))
    return results


def summary(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row['protocol'],row['model'],row['policy'],row['budget'])].append(row)
    result = []
    rng = np.random.default_rng(1729)
    for (protocol,model,policy,budget), values in grouped.items():
        shapes = defaultdict(list)
        for v in values:
            shapes[v['shape']].append(v['regret']-v['random_regret'])
        clusters = [statistics.mean(v) for v in shapes.values()]
        ci = np.quantile([np.mean(rng.choice(clusters,len(clusters),replace=True)) for _ in range(500)], [.025,.975])
        result.append(dict(protocol=protocol,model=model,policy=policy,budget=budget,
            contexts=len(values),shape_clusters=len(clusters),mean_regret=statistics.mean(v['regret'] for v in values),
            worst_regret=max(v['regret'] for v in values),good_hit_rate=statistics.mean(v['good_hit'] for v in values),
            mean_random_regret=statistics.mean(v['random_regret'] for v in values),
            shape_weighted_regret_delta=statistics.mean(clusters),shape_bootstrap_delta_ci=ci.tolist()))
    return result


def run(args):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=args.threads)
    output = Path(args.output)
    output.mkdir(parents=True,exist_ok=False)
    rows = load(args.dataset)
    # Hold the entire (M,K) group out, including different output widths,
    # families, cache regimes and numerical policies on all devices.
    for row in rows:
        row['shape'] = identity([row['features'].get('context.workload.m'),
                                 row['features'].get('context.workload.k')])
    dataset_hash = hashlib.sha256((Path(args.dataset)/'manifest.json').read_bytes()).hexdigest()
    protocols = list(splits(rows))
    atomic(output/'plan.json',dict(dataset=str(args.dataset),dataset_sha256=dataset_hash,
        models=args.models,views=args.views,threads=args.threads,protocols=[p for p,_,_ in protocols],
        split_group='whole_M_K_across_families_devices_and_output_widths',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime_promotion=False,prospective=False,
        limitations=['Observed stable candidate pools only; not adaptive UCB or compile-failure simulation.',
                     'Missing portfolio timings are reported, never imputed.',
                     'These previously inspected datasets support development, not release qualification.']))
    # Stage 3 completes for every split before any Stage 4 model is fitted.
    prepared = []
    for protocol,tr,te in protocols:
        supported = {contract(r) for r in tr}
        test = [r for r in te if contract(r) in supported]
        key = hashlib.sha256(protocol.encode()).hexdigest()[:12]
        portfolio = fit_portfolios(tr)
        atomic(output/f'portfolio-{key}.json',dict(protocol=protocol,portfolios=portfolio,
            train_rows=len(tr),test_rows=len(test),unsupported_test_rows=len(te)-len(test)))
        prepared.append((protocol,key,tr,test,portfolio))
        print('PORTFOLIO',protocol,len(tr),len(test),flush=True)
    all_metrics=[]
    for protocol,key,tr,te,portfolio in prepared:
        if not tr or not te:
            continue
        for name in args.models:
            for view in args.views:
                started=time.monotonic()
                model=FittedCandidate(REGISTRY[name],view,0,args.threads)
                model.fit(tr,balance_contexts=True)
                fit_s=time.monotonic()-started
                started=time.monotonic()
                predictions=model.predict(te)
                predict_s=time.monotonic()-started
                metrics=score(te,predictions,portfolio,protocol,name+':'+view)
                atomic(output/f'metrics-{key}-{name}-{view}.json',dict(protocol=protocol,model=name,view=view,
                    fit_s=fit_s,predict_s=predict_s,features=model.feature_count,rows=metrics))
                joblib.dump(model,output/f'model-{key}-{name}-{view}.joblib',compress=3)
                all_metrics.extend(metrics)
                atomic(output/'summary.json',summary(all_metrics))
                print('MODEL',protocol,name,view,f'fit={fit_s:.1f}s predict={predict_s:.1f}s',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--models',nargs='+',default=['hist_family','extra_trees'],choices=tuple(REGISTRY))
    parser.add_argument('--views',nargs='+',default=['shape','interactions'],choices=['shape','hardware','interactions'])
    run(parser.parse_args())
