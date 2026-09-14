"""Summarize paired policy timing separately from first-hit search cost."""
import argparse
import json
from pathlib import Path
import statistics
from tools.predictor_study import atomic


def summarize(root):
    root = Path(root)
    tasks = {}
    for path in (root/'tasks').glob('*/result.json'):
        row = json.loads(path.read_text())
        task = row['task']
        tasks[(*[task[k] for k in ('family','category','budget','replicate')],
               task['treatment'])] = row
    rows=[]
    for path in (root/'confirmation').glob('*/result.json'):
        result=json.loads(path.read_text())
        confirmation=result['confirmation']
        if confirmation['invalid'] or confirmation['status']!='complete':
            raise ValueError(f'incomplete or failed confirmation: {path}')
        for policy in confirmation['normalized_scores']:
            original=tasks[(*result['context'],policy)]
            timings=[r['outcomes'][policy]['outcome']['median_ms'] for r in confirmation['rounds']]
            ratios=[r['outcomes'][policy]['outcome']['median_ms'] /
                    r['outcomes']['existing']['outcome']['median_ms'] for r in confirmation['rounds']]
            rows.append(dict(context=result['context'],shape=original['task']['shape'],
                policy=policy,median_ms=statistics.median(timings),min_ms=min(timings),
                paired_ratio_median=statistics.median(ratios),paired_ratio_min=min(ratios),
                paired_ratio_max=max(ratios),tuning_wall_s=original['tuning_wall_s'],
                worker_wall_s=original['total_wall_s'],trials=original['evaluated_trials'],
                independent_initial_numerics=original.get('independent_initial_numerics')))
    atomic(root/'summary.json',dict(rows=rows,runtime_promotion=False,
        comparison='Selected configurations versus ordinary autotuning, NOT BF16.',
        caveats=['Two shapes and one replicate are a pilot, not release qualification.',
                 'CPU historical research ran concurrently; first-hit wall times are descriptive, not isolated.',
                 'Prepared-graph paired timing excludes search, compilation and setup.']))
    return rows


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    for row in summarize(p.parse_args().output):
        print(row['shape'],row['policy'],f"{row['median_ms']*1000:.3f} us",
              f"ratio={row['paired_ratio_median']:.5f}",f"tune={row['tuning_wall_s']:.2f}s")
