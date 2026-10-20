"""Conservative deployment-mode challenger qualification, independent of search."""
import math
import statistics


def qualify_pair(incumbent, challenger, evaluate, *, rounds=6, margin=.01):
    """Alternate A/B order. A lower-tail paired speedup must clear the margin.

    The evaluator owns correctness, warmup, and the deployment execution mode.
    Both the paired lower quartile and aggregate-median gain must clear the
    margin. This consistency gate is a decision rule, not a confidence interval.
    Failures or inconclusive improvements retain a verified incumbent.
    """
    if type(rounds) is not int or rounds < 4 or rounds % 2:
        raise ValueError('qualification rounds must be even and >=4')
    if not math.isfinite(margin) or not 0 <= margin < 1:
        raise ValueError('qualification margin must be in [0,1)')
    samples={'incumbent':[], 'challenger':[]}
    failures=[]
    for i in range(rounds):
        order=('incumbent','challenger') if i%2==0 else ('challenger','incumbent')
        for name in order:
            outcome=evaluate(incumbent if name=='incumbent' else challenger)
            value=outcome.median_ms
            if outcome.status!='ok' or value is None or not math.isfinite(value) or value<=0:
                failures.append(dict(round=i,candidate=name,status=outcome.status,error=outcome.error))
            else:
                samples[name].append(value)
    if len(samples['incumbent']) != rounds:
        raise RuntimeError('deployment incumbent did not pass qualification; no safe promotion')
    speedups=[]
    promote=False
    aggregate_speedup=None
    if len(samples['challenger'])==rounds:
        speedups=[a/b-1 for a,b in zip(samples['incumbent'],samples['challenger'])]
        aggregate_speedup=statistics.median(samples['incumbent'])/statistics.median(samples['challenger'])-1
        promote=sorted(speedups)[rounds//4] > margin and aggregate_speedup > margin
    return (challenger if promote else incumbent), dict(
        promoted=promote, rounds=rounds,margin=margin,samples_ms=samples,
        paired_speedups=speedups,failures=failures,
        decision_rule='paired_lower_quartile_and_aggregate_median_speedup',
        decision_revision=2,aggregate_speedup=aggregate_speedup,
        median_ms=statistics.median(samples['challenger' if promote else 'incumbent']))
