"""Context-balancing ablation on older data; no runtime model promotion.

Each fit writes an independent atomic JSON residual. Prospective GPU studies
are excluded: this measures a training-weighting change, not deployed UCB.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import statistics
import time

from tools.predictor_study import atomic
from tools.research.data import load_dataset, split_rows
from tools.research.models import FittedCandidate, REGISTRY
from tools.research_effort import metrics


def run(args):
    from threadpoolctl import threadpool_limits
    threadpool_limits(limits=args.threads)
    output = Path(args.output)
    if (output/'plan.json').exists():
        raise ValueError('choose a fresh output; do not mix model/source revisions')
    source = json.loads(Path(args.sources).read_text())
    train, _, train_audit, _ = load_dataset(source['train_paths'])
    test, _, test_audit, _ = load_dataset(source['test_paths'])
    hashes = {str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in (
        Path(__file__), Path('tools/research/models.py'), Path('tools/research/data.py'))}
    atomic(output/'plan.json', dict(schema=1, sources=source,
        train_audit=train_audit, test_audit=test_audit, source_hashes=hashes,
        models=args.models, protocols=args.protocols, threads=args.threads,
        quarantine='unchanged legacy', runtime_promotion=False,
        caveat='Repeated historical holdout analysis; not new unbiased prospective evidence.'))
    summary = []
    for protocol in args.protocols:
        tr, te = split_rows(train, test, protocol)
        if not tr or not te:
            raise ValueError(f'empty split: {protocol}')
        for name in args.models:
            for balanced in (False, True):
                start = time.monotonic()
                model = FittedCandidate(REGISTRY[name], 'interactions', 0, args.threads)
                model.fit(tr, balance_contexts=balanced)
                fit_s = time.monotonic()-start
                predictions = model.predict(te)
                rows = metrics(te, predictions, protocol, name)
                key = hashlib.sha256(f'{protocol}/{name}/{balanced}'.encode()).hexdigest()[:16]
                atomic(output/(key+'.json'), dict(protocol=protocol, model=name,
                    balanced=balanced, fit_s=fit_s, train_rows=len(tr), test_rows=len(te), metrics=rows))
                by_budget = defaultdict(list)
                for row in rows:
                    by_budget[row['budget']].append(row['regret'])
                for budget, regrets in sorted(by_budget.items()):
                    summary.append(dict(protocol=protocol, model=name, balanced=balanced,
                        budget=budget, contexts=len(regrets), mean_regret=statistics.mean(regrets),
                        median_regret=statistics.median(regrets), worst_regret=max(regrets), fit_s=fit_s))
                atomic(output/'summary.json', dict(rows=summary, runtime_promotion=False))
                print(f'DONE {protocol} {name} balanced={balanced} fit={fit_s:.1f}s', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', default='autotune_reports/effort_ml_v2/audit.json')
    parser.add_argument('--output', required=True)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--models', nargs='+', choices=tuple(REGISTRY), default=['hist_family', 'small_mlp'])
    parser.add_argument('--protocols', nargs='+', default=['chronological', 'unseen_shapes'])
    run(parser.parse_args())
