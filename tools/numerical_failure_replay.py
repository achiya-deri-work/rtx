"""Replay quarantined study observations without relaxing their original gates."""
import argparse
import json
from pathlib import Path

from tools.predictor_study import atomic, build_adapter
from rtx.autotune.outcomes import raise_if_fatal_device_context_error


def replay(root, index, device, repeats):
    case = json.loads((root / 'plan.json').read_text())['cases'][index]
    result = dict(case=case, outcomes=[], status='incomplete')
    try:
        for _ in range(repeats):
            # Successful adapters cache a prepared graph; evaluate() on that
            # graph repeats timing, not correctness. Rebuild for each check.
            adapter, _ = build_adapter(case['task'],
                dict(warmup=1, samples=3, calls_per_sample=2), device)
            config = adapter.deserialize(case['observation']['config'])
            result['rejection'] = adapter.rejection(config)
            if result['rejection'] is not None:
                result['status'] = 'statically_rejected'
                return
            outcome = adapter.evaluate(config)
            result['outcomes'].append(outcome.as_dict())
            atomic(root / f'case-{index}.json', result)
            print(outcome.status, outcome.error, flush=True)
            if outcome.error:
                raise_if_fatal_device_context_error(outcome.error)
        result['status'] = 'complete'
    finally:
        atomic(root / f'case-{index}.json', result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', type=int, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if args.repeats < 1 or args.case < 0:
        parser.error('repeats must be positive and case nonnegative')
    replay(args.output, args.case, args.device, args.repeats)
