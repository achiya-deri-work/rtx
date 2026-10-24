"""Durable paired winner decisions, separate from raw search observations."""
import json
import math
from pathlib import Path
import statistics

from .confirmation import confirm_finalists
from .core import stable_id

PROMOTION_REVISION = 3


def decision_events(store, context_id, kind='winner_decision'):
    """Read this residual only; malformed tails cannot hide earlier decisions."""
    if store.path is None:
        records = getattr(store, 'events', ())
    else:
        path = Path(store.path)/'events.jsonl'
        records = []
        if path.exists():
            for line in path.read_text().splitlines():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    for record in records:
        payload = record.get('payload', record)
        if record.get('kind') == kind and payload.get('context_id') == context_id:
            yield payload


def promote(adapter, store, history, session_id, *, rounds, margin, deadline=None, seed=0):
    """Return a confirmed selection or retain the incumbent on inconclusive timing.

    Numerical failures disqualify a finalist regardless of its speed. The
    margin is a practical effect threshold, not a statistical confidence claim.
    Raw search observations and scheduler history are never modified.
    """
    if type(rounds) is not int or rounds < 2 or rounds % 2:
        raise ValueError('promotion rounds must be even and at least two')
    if not math.isfinite(margin) or not 0 <= margin < 1:
        raise ValueError('promotion margin must be finite and in [0, 1)')
    evidence = stable_id([(o.config_id, o.outcome.as_dict()) for o in history.current])
    contract = stable_id(dict(schema=PROMOTION_REVISION, rounds=rounds, margin=margin, seed=seed))
    failed = {event['config_id'] for event in decision_events(store, adapter.context.identifier,
              'winner_confirmation') if event.get('outcome', {}).get('status') != 'ok'}
    # Feasibility is not latency evidence. In particular, a noisy initial
    # schedule must not become the incumbent after search found a measured
    # alternative. Keep the same eligibility contract as SearchHistory.best.
    eligible = {o.config_id:o for o in history.successful
                if o.outcome.timing_eligible and o.config_id not in failed
                and adapter.rejection(o.config) is None}
    if not eligible:
        raise RuntimeError('no legal successful candidate for promotion')
    previous = list(decision_events(store, adapter.context.identifier))
    for decision in reversed(previous):
        cid = decision.get('config_id')
        if (decision.get('contract') == contract and decision.get('evidence') == evidence
                and (decision.get('confirmation') is None or
                     decision['confirmation'].get('status') == 'complete')
                and cid in eligible):
            return {**decision, 'reused': True}
    incumbent = next((eligible[d['config_id']] for d in reversed(previous)
                      if d.get('promotion_revision') == PROMOTION_REVISION
                      and d.get('config_id') in eligible), None)
    if incumbent is None:
        incumbent = eligible.get(adapter.config_id(adapter.initial_config)) or next(iter(eligible.values()))
    ranked = sorted(eligible.values(), key=lambda o:(o.score,o.config_id))
    finalists = {incumbent.config_id:incumbent.config}
    for item in ranked:
        finalists[item.config_id] = item.config
        if len(finalists) == 3:
            break
    confirmation = None
    chosen = incumbent.config_id
    median = incumbent.score
    reason = 'only_candidate'
    if len(finalists) > 1:
        confirmation = confirm_finalists(adapter.evaluate, finalists, rounds=rounds, seed=seed,
            deadline=deadline, record=lambda row:store.record_event(session_id,
                'winner_confirmation', dict(context_id=adapter.context.identifier,
                    evidence=evidence, contract=contract, **row)))
        valid = [cid for cid in finalists if cid not in confirmation['invalid']]
        if not valid:
            store.record_event(session_id, 'winner_promotion_failed', dict(
                context_id=adapter.context.identifier, evidence=evidence, confirmation=confirmation))
            raise RuntimeError('all winner finalists failed confirmation')
        rounds_done = confirmation['rounds']
        if chosen not in valid:
            chosen = confirmation['winner']
            if chosen is None:
                raise RuntimeError('incumbent failed and no confirmed replacement exists')
            reason = 'incumbent_invalid'
        else:
            reason = 'incumbent_retained'
            # Require the complete requested comparison, not a lucky partial
            # deadline-limited round. An invalid incumbent is handled above.
            unreliable = set(confirmation['timing_unreliable'])
            if chosen in unreliable:
                reason = 'timing_inconclusive'
            if confirmation['status'] == 'complete' and chosen not in unreliable:
                candidates = []
                for cid in valid:
                    if cid in unreliable:
                        continue
                    ratios = [r['outcomes'][cid]['outcome']['median_ms'] /
                              r['outcomes'][incumbent.config_id]['outcome']['median_ms']
                              for r in rounds_done]
                    # A mean/median benefit that reverses with order is not
                    # sufficient evidence to replace a working incumbent.
                    if max(ratios) < 1 and statistics.median(ratios) < 1-margin:
                        candidates.append((statistics.median(ratios), cid))
                if candidates:
                    _,chosen = min(candidates)
                    reason = 'paired_improvement'
        timings = [r['outcomes'][chosen]['outcome']['median_ms'] for r in rounds_done
                   if chosen in r['outcomes'] and chosen not in confirmation['timing_unreliable']]
        if timings:
            median = statistics.median(timings)
    decision = dict(context_id=adapter.context.identifier, contract=contract, evidence=evidence,
        promotion_revision=PROMOTION_REVISION,
        config_id=chosen, median_ms=median, reason=reason, reused=False,
        confirmation=confirmation, completed_trials=len(history.current))
    store.record_event(session_id, 'winner_decision', decision)
    return decision
