"""Full-pool screening statistics; measured finite-pool regret, not a global optimum."""
import math
import statistics
import numpy as np


def summarize_reference(records, pool_ids, *, draws=2000, seed=1729):
    ids=list(pool_ids)
    by_id={r['config_id']:r for r in records}
    if len(by_id)!=len(records) or len(set(ids))!=len(ids) or set(by_id)!=set(ids):
        raise ValueError('reference must contain exactly one observation per frozen candidate')
    if type(draws) is not int or draws < 1:
        raise ValueError('positive integer draws required')
    ids.sort()
    times=[]
    for cid in ids:
        outcome=by_id[cid]['outcome']
        value=outcome.get('median_ms')
        if outcome['status']=='ok':
            if value is None or not math.isfinite(value) or value<=0:
                raise ValueError('successful reference has invalid timing')
            times.append(value)
        else:
            # Infinity here denotes an unsuccessful draw, not an imputed
            # runtime label. It is never serialized or used to fit a model.
            times.append(math.inf)
    valid=[t for t in times if math.isfinite(t)]
    if not valid:
        raise ValueError('reference contains no successful candidates')
    best=min(valid)
    rng=np.random.default_rng(seed)
    summaries=[]
    for budget in (8,16,32,64):
        if budget>len(times):
            continue
        sampled=[min(times[i] for i in rng.choice(len(times),budget,replace=False)) for _ in range(draws)]
        regrets=[t/best-1 for t in sampled if math.isfinite(t)]
        summaries.append(dict(budget=budget,draws=draws,
            no_valid_probability=1-len(regrets)/draws,
            mean_regret_given_valid=statistics.mean(regrets) if regrets else None,
            median_regret_given_valid=statistics.median(regrets) if regrets else None,
            within_5pct_probability=sum(t<=best*1.05 for t in sampled)/draws))
    return dict(candidates=len(ids),valid=len(valid),best_screened_ms=best,
        within_2pct=sum(t<=best*1.02 for t in valid),within_5pct=sum(t<=best*1.05 for t in valid),
        uniform_without_replacement=summaries,
        limitation='screening timings, not paired confirmation; one finite pool, not global optimum')
