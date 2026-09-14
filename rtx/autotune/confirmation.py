"""Paired finalist confirmation, separate from candidate-search trial budgets.

This primitive does not promote runtime winners. Callers own finalist selection,
deadline reservation and persistence. Every evaluation is emitted immediately;
only complete rounds participate in the paired decision.
"""
import math
import random
import statistics
import time

from .outcomes import TrialOutcome, raise_if_fatal_device_context_error


def confirm_finalists(evaluate, finalists, *, rounds=4, seed=0,
                      deadline=None, record=None):
    """Compare ``{config_id: config}`` finalists in balanced forward/reverse order.

    ``evaluate`` must perform correctness validation and return a TrialOutcome.
    A caller reusing prepared graphs must validate them before entry. Deadline
    is an absolute monotonic timestamp; an admitted round finishes atomically
    unless a fatal device fault occurs. Failed/nonfinite candidates cannot win.
    Confirmation evaluations are extra measurements, not new unique configs.
    """
    if type(rounds) is not int or rounds < 2 or rounds % 2:
        raise ValueError('rounds must be a positive even integer of at least two')
    if len(finalists) < 2:
        raise ValueError('at least two distinct finalists are required')
    if deadline is not None and not math.isfinite(deadline):
        raise ValueError('deadline must be finite')
    rng = random.Random(seed)
    keys = list(finalists)
    rows = []
    invalid = set()
    timing_unreliable = set()
    for pair in range(rounds // 2):
        if deadline is not None and time.monotonic() >= deadline:
            break
        order = keys.copy()
        rng.shuffle(order)
        # Admit balanced pairs together, so a deadline cannot bias order.
        for iteration, current in enumerate((order, order[::-1])):
            row = dict(round=2 * pair + iteration, order=current, outcomes={})
            for key in current:
                if key in invalid:
                    continue
                started = time.monotonic()
                try:
                    outcome = evaluate(finalists[key])
                except Exception as exc:
                    outcome = TrialOutcome('runtime_error', error=f'{type(exc).__name__}: {exc}')
                if (not outcome.successful or not math.isfinite(outcome.median_ms)
                        or outcome.median_ms <= 0):
                    invalid.add(key)
                # Timing instability is not a numerical failure and must not
                # blacklist a legal configuration from subsequent searches.
                sampling = (outcome.metadata or {}).get('sampling', {})
                if sampling.get('collection', {}).get('stationary') is False:
                    timing_unreliable.add(key)
                item = dict(round=row['round'], config_id=key,
                            elapsed_s=time.monotonic() - started, outcome=outcome.as_dict())
                row['outcomes'][key] = item
                if record is not None:
                    record(item)
                if outcome.error:
                    raise_if_fatal_device_context_error(outcome.error)
            rows.append(row)
    valid = [key for key in keys if key not in invalid and key not in timing_unreliable]
    scores = {}
    for key in valid:
        ratios = []
        for row in rows:
            if not all(k in row['outcomes'] for k in valid):
                continue
            times = [row['outcomes'][k]['outcome']['median_ms'] for k in valid]
            ratios.append(row['outcomes'][key]['outcome']['median_ms'] / statistics.median(times))
        if ratios:
            scores[key] = statistics.median(ratios)
    return dict(status='complete' if len(rows) == rounds else 'deadline',
                winner=min(scores, key=scores.get) if scores else None,
                normalized_scores=scores, invalid=sorted(invalid), rounds=rows,
                timing_unreliable=sorted(timing_unreliable),
                evaluations=sum(len(row['outcomes']) for row in rows))
