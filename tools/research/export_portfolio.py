"""Export an audited historical portfolio/model as an opt-in NumPy-only prior."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from rtx._portable_predictor import PortablePredictor, contract, SCHEMA, FEATURE_REVISION
from tools.export_tournament_predictors import export_model
from tools.predictor_study import atomic
from .evidence_dataset import load
from .portfolio_study import fit_portfolios, portfolio_ids


def tree_models(fitted):
    if fitted.spec.name == 'hist_family':
        return export_model(fitted)
    if fitted.spec.name != 'extra_trees':
        raise ValueError('only validated histogram/extra-tree exports supported')
    models = {}
    for family, forest in fitted.models.items():
        trees = []
        for estimator in forest.estimators_:
            tree = estimator.tree_
            trees.append([[float(tree.value[i].ravel()[0])/len(forest.estimators_),
                int(tree.feature[i]),float(tree.threshold[i]),int(tree.children_left[i]),
                int(tree.children_right[i]),int(tree.children_left[i] == -1),0]
                for i in range(tree.node_count)])
        models[family] = dict(baseline=0.,trees=trees)
    return models


def supported_splits(rows, families):
    train = [r for r in rows if r['split']=='train' and r['family'] in families]
    test = [r for r in rows if r['split']!='train' and r['family'] in families]
    if not train or not test or any(not any(r['family']==f for r in test) for f in families):
        raise ValueError('each exported family requires train and verification evidence')
    if any(not any(r['family']==f for r in train) for f in families):
        raise ValueError('each exported family requires training evidence')
    return train,test


def export(dataset, model_path, output):
    rows = load(dataset)
    fitted = joblib.load(model_path)  # Only our locally produced artifact.
    # A family-specific fit must not advertise unrelated retrieval bundles.
    train,test = supported_splits(rows, fitted.models)
    portfolio = fit_portfolios(train)
    bundles = {}
    seen = set()
    for row in train:
        if row['config_id'] not in portfolio_ids(portfolio,row):
            continue
        c = contract(row['family'],row['revision'],row['features'])
        key = json.dumps(c,sort_keys=True)
        if (key,row['config_id']) in seen:
            continue
        seen.add((key,row['config_id']))
        bundles.setdefault(key,dict(contract=c,winners=[]))['winners'].append(dict(
            config=row['config'],config_id=row['config_id'],
            features={k:v for k,v in row['features'].items() if k.startswith('context.workload.')}))
    document = dict(schema=SCHEMA,feature_revision=FEATURE_REVISION,deployment_enabled=False,
        bundles=list(bundles.values()),feature_names=fitted.vectorizer.get_feature_names_out().tolist(),
        models={'trees':tree_models(fitted)},provenance=dict(
            dataset_manifest_sha256=hashlib.sha256((Path(dataset)/'manifest.json').read_bytes()).hexdigest(),
            model_sha256=hashlib.sha256(Path(model_path).read_bytes()).hexdigest(),
            historical_only=True,runtime_promotion=False,training_split='train'))
    portable = PortablePredictor(document,research=True)
    eligible = [r for r in test if r['family'] in fitted.models]
    # Deterministic bounded verification includes every supported family.
    check=[]
    for family in sorted(fitted.models):
        check.extend([r for r in eligible if r['family']==family][:256])
    expected=fitted.predict(check)
    actual=portable.predict(check,'trees')
    np.testing.assert_allclose(actual,expected,atol=1e-5,rtol=1e-5)
    output=Path(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        raise ValueError('artifact exists; use a new version')
    with gzip.open(output,'wt') as stream:
        json.dump(document,stream,allow_nan=False)
    atomic(output.with_suffix('.verification.json'),dict(rows=len(check),
        maximum_abs_error=float(np.max(np.abs(actual-expected))),
        artifact_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),deployment_enabled=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--model',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    export(args.dataset,args.model,args.output)
