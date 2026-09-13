"""Run a resumable CPU cost-model tournament on immutable observation bundles.

python -m tools.research.tournament --sources <effort_ml_v2/audit.json> --output ...
No production artifact is trained or installed by this command.
"""
import argparse
from collections import defaultdict
import hashlib
import importlib.metadata
import json
from pathlib import Path
import time
import warnings

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, mean_absolute_error
from threadpoolctl import threadpool_limits

from tools.research_effort import metrics
from .data import VIEWS, feature_view, inner_split, load_dataset, split_rows
from .models import FittedCandidate, REGISTRY, Retrieval


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    temp.replace(path)


def atomic_parquet(path, rows):
    temp = path.with_suffix(path.suffix + '.tmp')
    pq.write_table(pa.Table.from_pylist(rows), temp)
    temp.replace(path)


def selection_score(rows, pred):
    values = metrics(rows, pred, 'inner', 'candidate')
    eight = [v['regret'] for v in values if v['budget'] == 8]
    if not eight:
        raise ValueError('inner validation has no contexts with eight observed candidates')
    return float(np.mean(eight))


def evaluate_model(train, test, spec, view, threads):
    fitting, validation = inner_split(train)
    supported = {(r['family'], r['revision']) for r in fitting}
    validation = [r for r in validation if (r['family'], r['revision']) in supported]
    if not fitting or not validation:
        raise ValueError('empty grouped inner split')
    scores, warning_messages = [], []
    started = time.monotonic()
    for variant in spec.variants:
        with warnings.catch_warnings(record=True) as caught:
            fitted = FittedCandidate(spec, view, variant, threads).fit(fitting)
        pred = fitted.predict(validation)
        scores.append(selection_score(validation, pred))
        warning_messages.extend(str(w.message) for w in caught)
        del fitted
    chosen = int(np.argmin(scores))
    with warnings.catch_warnings(record=True) as caught:
        model = FittedCandidate(spec, view, spec.variants[chosen], threads).fit(train)
    warning_messages.extend(str(w.message) for w in caught)
    fit_seconds = time.monotonic() - started
    predict_started = time.monotonic()
    predictions = model.predict(test)
    predict_seconds = time.monotonic() - predict_started
    # Include Python feature extraction/vectorization, not just tree traversal.
    groups = defaultdict(list)
    for r in test:
        groups[r['context']].append(r)
    timings = []
    for rs in list(groups.values())[:10]:
        pool = rs[:128]
        start = time.monotonic()
        model.predict(pool)
        timings.append((time.monotonic()-start)*1e3)
    details = dict(inner_regret8=scores, chosen_variant=spec.variants[chosen],
                   fit_seconds=fit_seconds, predict_seconds=predict_seconds,
                   pool_prediction_ms_median=float(np.median(timings)),
                   pool_prediction_ms_max=max(timings), feature_count=model.feature_count,
                   train_rows=len(train), test_rows=len(test),
                   inner_train_shapes=len({r['shape'] for r in fitting}),
                   inner_validation_shapes=len({r['shape'] for r in validation}),
                   warnings=sorted(set(warning_messages)))
    return predictions, details


def auxiliary(train, test, protocol, threads):
    """Failure and evaluation-cost heads; never encode failure as high latency."""
    tr, te = split_rows(train, test, protocol)
    vec = DictVectorizer(sparse=False, dtype=np.float32)
    X = vec.fit_transform([feature_view(r, 'interactions') for r in tr])
    T = vec.transform([feature_view(r, 'interactions') for r in te])
    y, target = np.array([r['failed'] for r in tr]), np.array([r['failed'] for r in te])
    result = dict(protocol=protocol, train_attempts=len(tr), test_attempts=len(te),
                  train_failures=int(y.sum()), test_failures=int(target.sum()))
    if len(np.unique(y)) > 1 and len(te):
        clf = HistGradientBoostingClassifier(max_iter=60, max_depth=4,
            l2_regularization=10, early_stopping=False, random_state=1729).fit(X, y)
        pred = clf.predict_proba(T)[:, 1]
        baseline = np.full(len(te), float(y.mean()))
        result['failure'] = dict(brier=float(brier_score_loss(target, pred)),
            baseline_brier=float(brier_score_loss(target, baseline)),
            log_loss=float(log_loss(target, pred, labels=[0, 1])),
            average_precision=float(average_precision_score(target, pred)) if target.sum() else None,
            test_prevalence=float(target.mean()))
    else:
        result['failure'] = {'skipped': 'insufficient classes or evaluation rows'}
    ids = [i for i, r in enumerate(tr) if r['evaluation_s'] is not None]
    tids = [i for i, r in enumerate(te) if r['evaluation_s'] is not None]
    if ids and tids:
        yt = np.log([tr[i]['evaluation_s'] for i in ids])
        truth = np.log([te[i]['evaluation_s'] for i in tids])
        reg = HistGradientBoostingRegressor(max_iter=60, max_depth=4,
            l2_regularization=10, early_stopping=False, random_state=1729).fit(X[ids], yt)
        pred = reg.predict(T[tids])
        result['evaluation_cost'] = dict(train_rows=len(ids), test_rows=len(tids),
            log_mae=float(mean_absolute_error(truth, pred)),
            baseline_log_mae=float(mean_absolute_error(truth, np.full(len(truth), np.median(yt)))),
            median_multiplicative_error=float(np.median(np.exp(np.abs(truth-pred)))))
    else:
        result['evaluation_cost'] = {'skipped': 'missing evaluator elapsed_s'}
    return result


def collect_summary(root):
    records = []
    for path in sorted((root/'trials').glob('*.parquet')):
        records.extend(pq.read_table(path).to_pylist())
    if records:
        atomic_parquet(root/'metrics.parquet', records)
        from tools.summarize_effort_research import summarize
        summarize(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', required=True, help='audit.json with train_paths and test_paths')
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--models', nargs='+', choices=list(REGISTRY), default=list(REGISTRY))
    parser.add_argument('--views', nargs='+', choices=VIEWS, default=list(VIEWS))
    parser.add_argument('--protocols', nargs='+', default=['chronological'])
    parser.add_argument('--auxiliary', action='store_true')
    parser.add_argument('--summarize', action='store_true')
    args = parser.parse_args()
    threadpool_limits(limits=args.threads)
    root = Path(args.output)
    (root/'trials').mkdir(parents=True, exist_ok=True)
    if args.summarize:
        collect_summary(root)
        return
    sources = json.loads(Path(args.sources).read_text())
    digests = {}
    for source in sources['train_paths'] + sources['test_paths']:
        path = Path(source)
        digest = hashlib.sha256()
        files = [path] if path.is_file() else sorted(path.rglob('observations.jsonl'))
        for item in files:
            digest.update(str(item.relative_to(path) if path.is_dir() else item.name).encode())
            with item.open('rb') as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b''):
                    digest.update(chunk)
        digests[source] = digest.hexdigest()
    code = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in Path(__file__).parent.glob('*.py')}
    contract = {'source_manifest': sources, 'source_sha256': digests, 'code_sha256': code,
                'seed': 1729, 'threads': args.threads,
                'versions': {p: importlib.metadata.version(p) for p in ('numpy', 'scikit-learn', 'pyarrow', 'xgboost-cpu', 'catboost')}}
    manifest = root/'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != contract:
        raise ValueError('research contract changed; use a new output directory')
    atomic_json(manifest, contract)
    print('LOAD older training bundles', flush=True)
    train, train_attempts, train_counts, train_hardware = load_dataset(sources['train_paths'])
    print('LOAD newer evaluation bundles', flush=True)
    test, test_attempts, test_counts, test_hardware = load_dataset(sources['test_paths'])
    atomic_json(root/'audit.json', dict(train=train_counts, test=test_counts,
        train_hardware=train_hardware, test_hardware=test_hardware,
        limits=['Previously inspected evaluation data: development, not untouched qualification',
                'Latency budgets rank observed successful pools, not failure-inclusive UCB replay',
                'Pre-evaluation telemetry may reflect prior workload; not a causal power intervention',
                'Quarantine conservatively checks stored inactive MLP configs too']))
    protocols = (['chronological', 'unseen_shapes'] + ['heldout:'+d for d in sorted({r['device'] for r in test})]
                 if args.protocols == ['all'] else args.protocols)
    print('AUDIT', train_counts, test_counts, flush=True)
    for protocol in protocols:
        tr, te = split_rows(train, test, protocol)
        # Each result is its own atomically published shard. Interruption never
        # invalidates completed fits; resume skips them without fitting again.
        choices = [('retrieval', 'hardware')] + [(name, view) for view in args.views for name in args.models]
        for name, view in choices:
            token = hashlib.sha256(f'{protocol}:{name}:{view}'.encode()).hexdigest()[:20]
            destination = root/'trials'/token
            if destination.with_suffix('.json').exists() and destination.with_suffix('.parquet').exists():
                print('RESUME', protocol, name, view, flush=True)
                continue
            print('START', protocol, name, view, flush=True)
            started = time.monotonic()
            try:
                if name == 'retrieval':
                    model = Retrieval().fit(tr)
                    t = time.monotonic()
                    pred = model.predict(te)
                    detail = dict(fit_seconds=t-started, predict_seconds=time.monotonic()-t)
                else:
                    pred, detail = evaluate_model(tr, te, REGISTRY[name], view, args.threads)
                if not np.isfinite(pred).all():
                    raise ValueError('nonfinite model prediction')
                rows = metrics(te, pred, protocol, name+'/'+view)
                atomic_parquet(destination.with_suffix('.parquet'), rows)
                atomic_json(destination.with_suffix('.json'), dict(protocol=protocol, model=name, view=view, **detail))
                eight = [r['regret'] for r in rows if r['budget'] == 8]
                print('DONE', protocol, name, view, 'regret8=', float(np.mean(eight)),
                      'seconds=', round(time.monotonic()-started, 2), flush=True)
            except Exception as exc:
                atomic_json(destination.with_suffix('.error.json'), {'error': repr(exc)})
                print('ERROR', protocol, name, view, repr(exc), flush=True)
        if args.auxiliary:
            target = root/('auxiliary-'+hashlib.sha256(protocol.encode()).hexdigest()[:12]+'.json')
            if not target.exists():
                print('AUXILIARY', protocol, flush=True)
                atomic_json(target, auxiliary(train_attempts, test_attempts, protocol, args.threads))
    collect_summary(root)
    print('DONE tournament', root, flush=True)


if __name__ == '__main__':
    main()
