"""Freeze all ranking treatments before any prospective candidate timing."""
import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np

from tools.research_effort import identity as stable_id
from rtx._portable_predictor import load_predictor
from .conditional_heuristics import predict_rules


def freeze_rankings(manifest, predictor, rules, *, excluded_mk, seed=1729, retrieval_prefix=2):
    if type(retrieval_prefix) is not int or not 0 <= retrieval_prefix <= 32:
        raise ValueError('retrieval prefix must be an integer in [0, 32]')
    started = time.monotonic()
    context = manifest['context']
    configs = sorted(manifest['configs'], key=lambda row: row['config_id'])
    ids = [row['config_id'] for row in configs]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError('empty or duplicate matched candidates')
    m, k = (context['workload'][key] for key in ('m', 'k'))
    if (m, k) in {tuple(pair) for pair in excluded_mk}:
        raise ValueError('prospective geometry was already observed or inspected')
    # The artifact's own compatibility matcher handles categorical tags;
    # ambiguous or missing contracts are errors, never fallback treatments.
    adapter = SimpleNamespace(context=SimpleNamespace(
        family=context['family'], kernel_revision=context['kernel_revision'],
        features=lambda: configs[0]['features']))
    bundle = predictor.bundle(adapter)
    if bundle is None:
        raise ValueError('ML/portfolio contract unsupported or ambiguous')
    rows = [dict(family=context['family'], revision=context['kernel_revision'],
                 context=stable_id(context), config_id=c['config_id'],
                 features={**bundle['contract']['policy'], **c['features']}) for c in configs]
    scores = predict_rules(rules, rows, heldout=True)
    ml_scores = predictor.predict(rows, 'trees')
    if not np.isfinite(scores).all() or not np.isfinite(ml_scores).all():
        raise ValueError('nonfinite matched ranking predictions')
    available = set(ids)
    target = rows[0]['features']
    def distance(row):
        return sum(abs(math.log2(max(1, row['features'].get('context.workload.'+key, 1)) /
                                 max(1, target.get('context.workload.'+key, 1))))
                   for key in ('m', 'n', 'k', 'h', 'o'))
    if 'runtime_search_space_revision' in context.get('tags',{}):
        # Preparation uses the actual adapter's deserializer/canonical IDs.
        # Historical IDs can differ only because ignored L2 fields changed;
        # intersecting their raw IDs with this projected pool loses retrieval.
        retrieved = manifest.get('runtime_retrieval_ids')
        if (context['tags']['runtime_search_space_revision'] != 1 or
                not isinstance(retrieved,list) or
                any(not isinstance(cid,str) or cid not in available for cid in retrieved) or
                len(retrieved)!=len(set(retrieved))):
            raise ValueError('missing or invalid projected retrieval identity')
    else:
        retrieved = list(dict.fromkeys(row['config_id'] for row in
            sorted(bundle['winners'], key=lambda row: (distance(row), row['config_id']))
            if row['config_id'] in available))
    prefix = retrieved[:retrieval_prefix]
    ordinary = list(ids)
    random.Random(seed).shuffle(ordinary)
    def prepend(order):
        return prefix + [cid for cid in order if cid not in prefix]
    rankings = dict(ordinary=ordinary, portfolio=prepend(ordinary),
        rules=[ids[i] for i in sorted(range(len(ids)), key=lambda i: (scores[i], ids[i]))],
        portfolio_ml=prepend([ids[i] for i in sorted(range(len(ids)), key=lambda i: (ml_scores[i], ids[i]))]))
    def score_audit(values):
        unique, counts = np.unique(values, return_counts=True)
        return dict(distinct_scores=len(unique), largest_tie=int(counts.max()),
                    minimum=float(unique[0]), maximum=float(unique[-1]),
                    tie_break='config_id', candidates=len(values))
    return dict(schema=1, pool_id=manifest['pool_id'], seed=seed, rankings=rankings,
                portfolio_available=len(retrieved), portfolio_prefix=prefix,
                retrieval_prefix=retrieval_prefix,
                ranking_setup_s=time.monotonic()-started,
                excluded_mk_digest=stable_id(sorted({tuple(pair) for pair in excluded_mk})),
                runtime_promotion=False, training_admission=False,
                score_audit=dict(rules=score_audit(scores), portfolio_ml=score_audit(ml_scores)),
                policy=f'up to {retrieval_prefix} retrieved candidates then full seeded/random or ML order; rules standalone')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('pool', 'predictor', 'rules', 'exclusions', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--seed', type=int, default=1729)
    parser.add_argument('--retrieval-prefix', type=int, choices=range(33), default=2)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise ValueError('frozen ranking exists; choose a new output')
    started = time.monotonic()
    manifest = json.loads(Path(args.pool).read_text())
    with gzip.open(args.rules, 'rt') as stream:
        rules = json.load(stream)
    predictor = load_predictor(args.predictor, research=True)
    document = freeze_rankings(manifest, predictor, rules,
        excluded_mk=json.loads(Path(args.exclusions).read_text()), seed=args.seed,
        retrieval_prefix=args.retrieval_prefix)
    document['provenance'] = {name: hashlib.sha256(Path(getattr(args, name)).read_bytes()).hexdigest()
                              for name in ('pool', 'predictor', 'rules', 'exclusions')}
    document['source_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    document['total_setup_s'] = time.monotonic()-started
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(document, stream, allow_nan=False, indent=2)
    print('FROZEN', output, document['total_setup_s'], flush=True)


if __name__ == '__main__':
    main()
