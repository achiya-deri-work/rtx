"""Refit on older data only; export and independently verify data-only predictors."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import warnings

import numpy as np
from threadpoolctl import threadpool_limits

from rtx._portable_predictor import PortablePredictor, SCHEMA, FEATURE_REVISION, contract
from tools.research.data import load_dataset
from tools.research.models import FittedCandidate, REGISTRY
from tools.research_effort import rows


def export_model(fitted):
    result = {}
    for family, model in fitted.models.items():
        if fitted.spec.name == 'small_mlp':
            scale, mlp = model.steps[0][1], model.steps[1][1]
            if mlp.activation != 'relu' or mlp.out_activation_ != 'identity':
                raise ValueError('unsupported MLP activation')
            result[family] = dict(mean=scale.mean_.tolist(), scale=scale.scale_.tolist(),
                layers=[dict(weight=w.tolist(), bias=b.tolist()) for w, b in zip(mlp.coefs_, mlp.intercepts_)])
        else:
            trees = []
            # The private sklearn reader is isolated to this exporter and
            # checked numerically. Runtime never imports sklearn internals.
            for predictors in model._predictors:
                if len(predictors) != 1:
                    raise ValueError('unsupported multi-output tree')
                nodes = predictors[0].nodes
                if nodes['is_categorical'].any():
                    raise ValueError('native categorical splits are unsupported')
                trees.append([[float(n['value']), int(n['feature_idx']), float(n['num_threshold']),
                               int(n['left']), int(n['right']), int(n['is_leaf']), int(n['missing_go_to_left'])]
                              for n in nodes])
            result[family] = dict(baseline=float(model._baseline_prediction.ravel()[0]), trees=trees)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=8)
    args = parser.parse_args()
    threadpool_limits(limits=args.threads)
    source = json.loads(Path(args.sources).read_text())
    train, ta, counts, _ = load_dataset(source['train_paths'])
    test, va, test_counts, _ = load_dataset(source['test_paths'])
    if {r['observation_id'] for r in ta} & {r['observation_id'] for r in va}:
        raise ValueError('training/evaluation observation overlap')
    winners = {}
    for r in train:
        if r['context'] not in winners or r['ms'] < winners[r['context']]['ms']:
            winners[r['context']] = r
    needed = {(r['family'], r['revision'], r['config_id']) for r in winners.values()}
    configs = {}
    for r in rows(source['train_paths']):
        key = (r['family'], r['kernel_revision'], r['config_id'])
        if key in needed:
            configs[key] = r['config']
    bundles = {}
    for r in winners.values():
        c = contract(r['family'], r['revision'], r['features'])
        key = json.dumps(c, sort_keys=True)
        bundles.setdefault(key, dict(contract=c, winners=[]))['winners'].append(dict(
            config=configs[(r['family'], r['revision'], r['config_id'])], config_id=r['config_id'],
            features={k: v for k, v in r['features'].items() if k.startswith('context.workload.')}))
    document = dict(schema=SCHEMA, feature_revision=FEATURE_REVISION, deployment_enabled=False,
                    bundles=list(bundles.values()), models={}, feature_names=[])
    report = dict(train_counts=counts, evaluation_counts=test_counts, heads={}, source_manifest=source)
    for head, name in [('residual', 'roofline_residual'), ('mlp', 'small_mlp')]:
        print('FIT', head, len(train), flush=True)
        with warnings.catch_warnings(record=True) as caught:
            fitted = FittedCandidate(REGISTRY[name], 'interactions', 1, args.threads).fit(train)
        names = fitted.vectorizer.get_feature_names_out().tolist()
        if document['feature_names'] and document['feature_names'] != names:
            raise ValueError('head feature maps differ')
        document['feature_names'] = names
        document['models'][head] = export_model(fitted)
        portable = PortablePredictor(document, research=True)
        eligible = [r for r in test if r['family'] in fitted.models]
        expected = fitted.predict(eligible)
        actual = portable.predict(eligible, head)
        np.testing.assert_allclose(actual, expected, atol=1e-5, rtol=1e-5)
        # Group ordering checks supplement numerical equality. Near ties may
        # move under floating arithmetic; report them rather than concealing.
        contexts = {}
        for i, r in enumerate(eligible):
            contexts.setdefault(r['context'], []).append(i)
        same, total = 0, 0
        for ids in contexts.values():
            if len(ids) >= 8:
                total += 1
                same += set(np.array(ids)[np.argsort(expected[ids], kind='stable')[:8]]) == set(np.array(ids)[np.argsort(actual[ids], kind='stable')[:8]])
        report['heads'][head] = dict(rows=len(eligible), max_abs_error=float(np.max(np.abs(actual-expected))),
            top8_sets_equal=int(same), top8_contexts=total, warnings=sorted({str(w.message) for w in caught}))
        print('VERIFIED', head, report['heads'][head], flush=True)
    import sklearn
    document['provenance'] = dict(sklearn_version=sklearn.__version__, numpy_version=np.__version__,
        train_sources=source['train_paths'], training_rows=len(train), variant=1,
        feature_source_sha256=hashlib.sha256(Path('tools/research/data.py').read_bytes()).hexdigest())
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    with gzip.open(temp, 'wt') as stream:
        json.dump(document, stream, allow_nan=False)
    temp.replace(path)
    report['artifact_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix('.verification.json').write_text(json.dumps(report, indent=2))
    print('DONE', path, flush=True)


if __name__ == '__main__':
    main()
