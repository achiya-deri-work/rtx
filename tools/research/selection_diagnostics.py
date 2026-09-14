"""Separate recorded candidate coverage from selection and timing evidence."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

from tools.regression_replay import documents
from tools.predictor_study import atomic
from .evidence_dataset import timing_quality


def audit_source(source):
    groups = defaultdict(dict)
    observations = defaultdict(list)
    pools = defaultdict(dict)
    for name, text in documents(source):
        task_id = name.split('/tasks/')[1].split('/')[0]
        if name.endswith('/result.json'):
            result = json.loads(text)
            task = result['task']
            key = (task['family'], json.dumps(task['shape'], sort_keys=True), task['regime'],
                   task['budget'], task['replicate'])
            groups[key][task['treatment']] = result
        elif name.endswith('/observations.jsonl'):
            observations[task_id].extend(json.loads(line) for line in text.splitlines())
        else:
            for line in text.splitlines():
                session = json.loads(line)
                pools[task_id].update((session.get('metadata', {}).get('candidate_pool') or {}).get('origins', {}))
    rows = []
    for key, group in groups.items():
        baseline = group.get('existing')
        if baseline is None or baseline['status'] != 'ok':
            continue
        records = observations[baseline['task']['id']]
        winner = next((r['config_id'] for r in records if r['config'] == baseline['winner_config']), None)
        for treatment, result in group.items():
            if treatment == 'existing' or result['status'] != 'ok':
                continue
            task_id = result['task']['id']
            trials = observations[task_id]
            origins = pools[task_id]
            measured = [r for r in trials if r['config_id'] == winner]
            stable = all(timing_quality(o) == 'stable' for r in (baseline, result) for o in r.get('verification', []))
            stable &= bool(baseline.get('verification') and result.get('verification'))
            stable &= max(baseline.get('verification_cv', 1), result.get('verification_cv', 1)) <= .05
            complete = baseline.get('attempts') == key[3] and result.get('attempts') == key[3]
            rows.append(dict(source=str(source), device=result['hardware']['name'], family=key[0],
                shape=result['task']['shape'], category=result['task']['category'], budget=key[3],
                replicate=key[4], treatment=treatment, complete_budget=complete,
                contract_compatible=result.get('contract_compatible'), verification_stable=stable,
                baseline_over_candidate=baseline['verified_ms']/result['verified_ms'],
                baseline_config_id=winner, pool_recorded=bool(origins),
                baseline_in_pool=winner in origins if origins and winner is not None else None,
                baseline_measured=bool(measured), baseline_measurement_statuses=[r['outcome']['status'] for r in measured],
                ranking_vs_scheduler='not_identifiable_without_full_proposal_ranks',
                time_to_first_valid_s=result.get('time_to_first_valid_s'),
                tuning_wall_s=result.get('tuning_wall_s'),
                compile_failures=sum(r['outcome']['status']=='compile_error' for r in trials),
                failed_evaluation_s=sum(r.get('elapsed_s', 0) for r in trials if r['outcome']['status']!='ok'),
                attempted_trials=len(trials)))
    return rows


def run(sources, output):
    output = Path(output)
    rows = []
    for i, source in enumerate(sources):
        part = audit_source(source)
        atomic(output/f'source-{i}.json', part)
        rows.extend(part)
        print('DIAGNOSE', source, len(part), flush=True)
    eligible = [r for r in rows if r['complete_budget'] and r['verification_stable'] and r['contract_compatible']]
    regressions = [r for r in eligible if r['baseline_over_candidate'] < .98]
    counts = Counter('missing_pool' if r['baseline_in_pool'] is False and not r['baseline_measured']
                     else 'generated_not_measured' if r['baseline_in_pool'] and not r['baseline_measured']
                     else 'measured_but_not_selected' if r['baseline_measured'] else 'pool_unknown'
                     for r in regressions)
    atomic(output/'summary.json', dict(matched=len(rows), eligible=len(eligible),
        regressions_over_2pct=len(regressions), mechanisms=dict(counts),
        limits=['Timing stability uses within-cell evidence; repeated-context instability needs dataset audit.',
                'Absent proposal ranks prevent distinguishing ranking from scheduler starvation.',
                'No latency is imputed for an unmeasured pool candidate.']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', nargs='+', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.sources, args.output)
