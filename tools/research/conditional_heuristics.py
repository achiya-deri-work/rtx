"""Interpretable, train-only ranking rules; never hard-prune legal candidates.

Same observed pools and portfolio prefixes as portfolio_study. Historical
records cannot reconstruct unmeasured initial neighborhoods, so this evaluates
ranking only, not the new runtime guard or an unbiased prospective policy.
"""
import argparse
import hashlib
import json
from pathlib import Path

from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeRegressor, export_text
import numpy as np

from .coverage_audit import context_weights
from .data import feature_view
from .evidence_dataset import load
from .models import labels
from .portfolio_study import contract, splits, score, summary
from tools.predictor_study import atomic
from tools.research_effort import identity


def fit_rules(rows, depth=4):
    vectorizer = DictVectorizer(sparse=False, dtype=np.float32)
    features = vectorizer.fit_transform([feature_view(r, 'interactions') for r in rows])
    targets = labels(rows)
    weights = np.asarray(context_weights(rows))
    heads = {}
    # Keep revisions and numerical policies separate. Shared physical features
    # may generalize across SKUs; an unsupported semantic contract may not.
    for key in sorted({contract(r) for r in rows}):
        ids = np.array([i for i, r in enumerate(rows) if contract(r) == key])
        model = DecisionTreeRegressor(max_depth=depth, min_samples_leaf=32,
                                      random_state=1729)
        model.fit(features[ids], targets[ids], sample_weight=weights[ids])
        heads[key] = model
    return vectorizer, heads


def export_rules(rows):
    """Fit only explicitly admitted training rows; emit data, not a pickle."""
    if not rows or any(row.get('split') != 'train' for row in rows):
        raise ValueError('rule export requires nonempty, exclusively training evidence')
    vectorizer, heads = fit_rules(rows)
    trees = {}
    for key, model in heads.items():
        tree = model.tree_
        trees[key] = [dict(feature=int(tree.feature[i]), threshold=float(tree.threshold[i]),
                           left=int(tree.children_left[i]), right=int(tree.children_right[i]),
                           value=float(tree.value[i].ravel()[0]))
                      for i in range(tree.node_count)]
    document = dict(schema=1, view='interactions', deployment_enabled=False,
                    feature_names=vectorizer.get_feature_names_out().tolist(), trees=trees,
                    training_contexts=sorted({r['context'] for r in rows}),
                    training_mk=sorted({(r['features'].get('context.workload.m'),
                                         r['features'].get('context.workload.k')) for r in rows}))
    matrix = vectorizer.transform([feature_view(r, 'interactions') for r in rows])
    expected = np.empty(len(rows))
    contracts = [contract(r) for r in rows]
    for key, model in heads.items():
        indices = [i for i, value in enumerate(contracts) if value == key]
        expected[indices] = model.predict(matrix[indices])
    actual = predict_rules(document, rows, heldout=False)
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-12)
    return document


def predict_rules(document, rows, *, heldout=True):
    """Exact tree traversal with sklearn's float32 input conversion contract."""
    if document.get('schema') != 1 or document.get('view') != 'interactions':
        raise ValueError('unsupported rule artifact')
    contexts = set(document['training_contexts'])
    geometries = {tuple(value) for value in document['training_mk']}
    predictions = []
    for row in rows:
        mk = (row['features'].get('context.workload.m'), row['features'].get('context.workload.k'))
        if heldout and (row['context'] in contexts or mk in geometries):
            raise ValueError('evaluation overlaps training context or whole-(M,K) geometry')
        key = contract(row)
        if key not in document['trees']:
            raise ValueError('rule artifact does not support this semantic/revision contract')
        features = feature_view(row, document['view'])
        values = np.asarray([features.get(name, 0.) for name in document['feature_names']], dtype=np.float32)
        nodes = document['trees'][key]
        index = 0
        for _ in range(len(nodes)):
            node = nodes[index]
            if node['left'] == -1:
                predictions.append(node['value'])
                break
            index = node['left'] if values[node['feature']] <= node['threshold'] else node['right']
        else:
            raise ValueError('invalid cyclic rule tree')
    return np.asarray(predictions)


def run(dataset, study, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    rows = load(dataset)
    for row in rows:
        row['shape'] = identity([row['features'].get('context.workload.m'),
                                 row['features'].get('context.workload.k')])
    atomic(output/'plan.json', dict(depth=4, min_samples_leaf=32,
        dataset_sha256=hashlib.sha256((Path(dataset)/'manifest.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        runtime_promotion=False, hard_pruning=False,
        limitation='Adaptive observational data; ranking-only development study, not causal rules.'))
    all_metrics = []
    for protocol, train, test in splits(rows):
        supported = {contract(r) for r in train}
        test = [r for r in test if contract(r) in supported]
        vectorizer, heads = fit_rules(train)
        features = vectorizer.transform([feature_view(r, 'interactions') for r in test])
        predictions = np.zeros(len(test))
        contracts = [contract(r) for r in test]
        for key, model in heads.items():
            ids = np.array([i for i, c in enumerate(contracts) if c == key], dtype=int)
            if len(ids):
                predictions[ids] = model.predict(features[ids])
        key = hashlib.sha256(protocol.encode()).hexdigest()[:12]
        portfolios = json.loads((Path(study)/f'portfolio-{key}.json').read_text())['portfolios']
        rules = {c: export_text(m, feature_names=list(vectorizer.get_feature_names_out()),
                                 decimals=5) for c, m in heads.items()}
        atomic(output/f'rules-{key}.json', dict(protocol=protocol, rules=rules,
            feature_units='Continuous features are signed log1p; leaves predict centered log latency.'))
        metrics = score(test, predictions, portfolios, protocol, 'conditional_tree_depth4')
        atomic(output/f'metrics-{key}.json', metrics)
        all_metrics.extend(metrics)
        print('RULES', protocol, len(train), len(test), flush=True)
    atomic(output/'summary.json', summary(all_metrics))


def export_dataset(dataset, output):
    """Create a new immutable train-only artifact and its provenance report."""
    import gzip
    import time
    started = time.monotonic()
    output = Path(output)
    if output.exists():
        raise ValueError('artifact exists; use a new version')
    rows = load(dataset)
    train = [r for r in rows if r['split'] == 'train']
    document = export_rules(train)
    document['provenance'] = dict(
        dataset_manifest_sha256=hashlib.sha256((Path(dataset)/'manifest.json').read_bytes()).hexdigest(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        training_rows=len(train), training_split='train', runtime_promotion=False,
        export_parity_rows=len(train), maximum_parity_tolerance=1e-12,
        elapsed_s=time.monotonic()-started)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('xb') as raw, gzip.open(raw, 'wt') as stream:
        json.dump(document, stream, allow_nan=False)
    print('EXPORTED', output, json.dumps(document['provenance']), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('dataset', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--study')
    parser.add_argument('--export-only', action='store_true')
    args = parser.parse_args()
    if args.export_only:
        export_dataset(args.dataset, args.output)
    elif args.study:
        run(args.dataset, args.study, args.output)
    else:
        parser.error('--study is required unless --export-only is selected')
