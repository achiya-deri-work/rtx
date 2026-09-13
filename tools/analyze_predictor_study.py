"""Paired prospective results, retaining missing/failure and transfer caveats."""
import argparse
from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import random
import statistics

from tools.predictor_study import atomic, summarize


def comparisons(rows):
    paired = defaultdict(dict)
    for row in rows:
        if row['matched_complete'] and (row['contract_compatible'] or row['treatment'] in ('existing', 'mixed')):
            paired[(row['family'], row['shape'], row['replicate'], row['budget'])][row['treatment']] = row
    results = []
    for budget in sorted({k[-1] for k in paired}):
        for control in ('existing','retrieval','residual_mlp','mixed'):
            for treatment in ('retrieval','residual','residual_mlp','mixed','mixed_residual_mlp'):
                if control == treatment:
                    continue
                records=[]
                for key, group in paired.items():
                    if key[-1] != budget or control not in group or treatment not in group:
                        continue
                    a,b = group[control],group[treatment]
                    records.append(dict(shape=key[1], family=key[0],
                        log_speedup=math.log(a['verified_ms']/b['verified_ms']),
                        tuning_delta_s=b['tuning_wall_s']-a['tuning_wall_s']))
                if not records:
                    continue
                # Cluster resampling preserves dependent families/replicates
                # on each geometry. Few geometries cannot support strong CIs.
                clusters=defaultdict(list)
                for r in records:
                    clusters[r['shape']].append(r['log_speedup'])
                rng=random.Random(20260915)
                keys=sorted(clusters)
                boot=[]
                if len(keys)>=2:
                    for _ in range(2000):
                        sample=[x for k in rng.choices(keys,k=len(keys)) for x in clusters[k]]
                        boot.append(math.exp(statistics.mean(sample)))
                    boot.sort()
                results.append(dict(budget=budget,control=control,treatment=treatment,
                    pairs=len(records),geometries=len(keys),
                    geometric_mean_speedup=math.exp(statistics.mean(r['log_speedup'] for r in records)),
                    worst_speedup=min(math.exp(r['log_speedup']) for r in records),
                    fraction_beating_control_by_2pct=sum(math.exp(r['log_speedup'])>1.02 for r in records)/len(records),
                    mean_tuning_delta_s=statistics.mean(r['tuning_delta_s'] for r in records),
                    exploratory_cluster_ci95=None if not boot else [boot[49],boot[1949]]))
    return results


def analyze(output):
    output=Path(output)
    summarize(output)
    summary=json.loads((output/'summary.json').read_text())
    rows=summary['rows']
    paired=comparisons(rows)
    report=dict(planned=summary['planned'],completed=summary['completed'],
        statuses=dict(Counter(r['status'] for r in rows)),
        incompatible_cells=sum(r['contract_compatible'] is False for r in rows),
        partial_budget_cells=sum(r['attempts'] is not None and r['attempts']<r['budget'] for r in rows),
        comparisons=paired, deployment_enabled=False,
        limitations=['Observed regret is against matched study winners, not an exhaustive oracle.',
            'Compiler caches are shared; process startup is fresh, compiler-cache coldness is not guaranteed.',
            'Few held-out geometries and one SKU do not qualify default deployment.',
            'Retrieval control includes reserved random exploration and local search.',
            'Incompatible model contracts are fallback controls, excluded from learned comparisons.'])
    atomic(output/'analysis.json',report)
    lines=['# Prospective predictor comparison','',
        f'Completed {summary["completed"]}/{summary["planned"]} cells. Research only; no default promotion.','',
        '| Trials | Control | Candidate | Pairs | Geomean speedup | Worst speedup | Added tuning seconds |',
        '|---:|---|---|---:|---:|---:|---:|']
    for r in paired:
        lines.append(f'| {r["budget"]} | {r["control"]} | {r["treatment"]} | {r["pairs"]} | '
            f'{r["geometric_mean_speedup"]:.3f}x | {r["worst_speedup"]:.3f}x | {r["mean_tuning_delta_s"]:+.2f} |')
    lines += ['', 'Speedup >1 is faster. Latencies use three post-search verification measurements.', '',
              *['- '+s for s in report['limitations']], '',
              'Per-family/shape failures, compiler time, proposal time, confirmation time and numerical outcomes remain in summary.csv, summary.parquet and task journals.', '']
    (output/'analysis.md').write_text('\n'.join(lines))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output')
    analyze(parser.parse_args().output)
