"""Merge completed tournament shards, preserving paired context statistics."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from tools.summarize_effort_research import summarize


def merge(roots, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    metrics, fits, errors, auxiliary = {}, [], [], []
    for root in map(Path, roots):
        for path in sorted((root/'trials').glob('*.parquet')):
            for row in pq.read_table(path).to_pylist():
                key = (row['protocol'], row['model'], row['context'], row['budget'])
                if key in metrics and metrics[key] != row:
                    raise ValueError(f'conflicting metric shard: {key}')
                metrics[key] = row
        for path in sorted((root/'trials').glob('*.json')):
            result = {**json.loads(path.read_text()), 'source': str(path)}
            (errors if path.name.endswith('.error.json') else fits).append(result)
        for path in sorted(root.glob('auxiliary-*.json')):
            auxiliary.append({**json.loads(path.read_text()), 'source': str(path)})
    if not metrics:
        raise ValueError('no completed model metrics')
    pq.write_table(pa.Table.from_pylist(list(metrics.values())), output/'metrics.parquet')
    summarize(output)
    summary = json.loads((output/'summary.json').read_text())
    scorecard = sorted((r for r in summary if r['partition'] == 'all'),
                       key=lambda r: (r['protocol'], r['budget'], r['mean_regret']))
    with (output/'leaderboard.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scorecard[0]))
        writer.writeheader()
        writer.writerows(scorecard)
    (output/'fits.json').write_text(json.dumps(fits, indent=2))
    (output/'errors.json').write_text(json.dumps(errors, indent=2))
    (output/'auxiliary.json').write_text(json.dumps(auxiliary, indent=2))
    # Choose among learned candidates using ONLY their older-data inner
    # validation score. Outer regret is attached afterwards, never consulted
    # by this selection. Retrieval has no inner tournament score here.
    selected = {}
    for fit in fits:
        if 'inner_regret8' not in fit:
            continue
        protocol = fit['protocol']
        score = min(fit['inner_regret8'])
        current = selected.get(protocol)
        if current is None or score < current['inner_regret8']:
            selected[protocol] = dict(model=fit['model']+'/'+fit['view'],
                                      inner_regret8=score, variant=fit['chosen_variant'])
    for protocol, choice in selected.items():
        choice['outer_budget8'] = next((r for r in scorecard if r['protocol'] == protocol
                                       and r['model'] == choice['model'] and r['budget'] == 8), None)
    (output/'inner_selected.json').write_text(json.dumps(selected, indent=2))
    comparisons = []
    grouped = defaultdict(list)
    for r in metrics.values():
        if r['model'] != 'retrieval/hardware':
            grouped[(r['protocol'], r['model'], r['budget'])].append(r)
    for (protocol, model, budget), rows in grouped.items():
        clusters = defaultdict(list)
        wins = []
        for r in rows:
            baseline = metrics.get((protocol, 'retrieval/hardware', r['context'], budget))
            if baseline is None:
                continue
            gain = baseline['regret'] - r['regret']
            clusters[r['shape']].append(gain)
            wins.append(gain > 0)
        if not clusters:
            continue
        sums = np.array([sum(v) for v in clusters.values()])
        counts = np.array([len(v) for v in clusters.values()])
        ids = np.random.default_rng(1729).integers(0, len(sums), (2000, len(sums)))
        low, high = np.quantile(sums[ids].sum(1)/counts[ids].sum(1), [.025, .975])
        comparisons.append(dict(protocol=protocol, model=model, budget=budget,
            contexts=sum(counts).item(), shape_clusters=len(clusters),
            mean_gain_over_retrieval=float(sums.sum()/counts.sum()),
            gain_ci_low=float(low), gain_ci_high=float(high),
            strictly_beats_retrieval_fraction=float(np.mean(wins))))
    (output/'versus_retrieval.json').write_text(json.dumps(comparisons, indent=2))
    # Fixed cohort for effort curves: increasing budgets must not silently
    # drop difficult contexts and manufacture apparent improvement.
    common = {(r['protocol'], r['model'], r['context']) for r in metrics.values() if r['budget'] == 32}
    curves = defaultdict(list)
    for r in metrics.values():
        if (r['protocol'], r['model'], r['context']) in common:
            curves[(r['protocol'], r['model'], r['budget'])].append(r['regret'])
    curve_rows = [dict(protocol=p, model=m, budget=b, contexts=len(v),
                       mean_regret=float(np.mean(v)), p90_regret=float(np.quantile(v, .9)))
                  for (p, m, b), v in sorted(curves.items())]
    with (output/'fixed_cohort_budget_curves.csv').open('w') as stream:
        if curve_rows:
            writer = csv.DictWriter(stream, fieldnames=list(curve_rows[0]))
            writer.writeheader()
            writer.writerows(curve_rows)
    lines = ['# Model tournament scorecard', '',
             'Completed shards only; a live run may still be incomplete.', '',
             'Regret is against the best observed accepted candidate, not the full search space.', '',
             '| Protocol | Model/features | Budget | Mean regret | P90 | Within 2% |',
             '|---|---|---:|---:|---:|---:|']
    for r in scorecard:
        lines.append(f"| {r['protocol']} | {r['model']} | {r['budget']} | "
                     f"{100*r['mean_regret']:.2f}% | {100*r['p90_regret']:.2f}% | {100*r['within2']:.1f}% |")
    lines += ['', f'Fit records: {len(fits)}. Error records: {len(errors)}.', '',
              'See summary.csv for paired workload-cluster confidence intervals and family/device breakdowns.',
              'CPU timing is diagnostic under potentially concurrent load; no runtime model was promoted.']
    (output/'README.md').write_text('\n'.join(lines)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('roots', nargs='+')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    merge(args.roots, args.output)
