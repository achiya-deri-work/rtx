"""Bounded candidate mixtures shared by first-hit search and offline studies."""
from collections import Counter
import random

from rtx._sampling import StructuralSamplingPolicy, raw_portfolio, select_portfolio
from .core import Proposal


def mixed_pool(adapter, *, limit=128, seed=1729, retrieved=(), seeds=(),
               structural_proposals=512, skip_discarded_generation=False, protected=()):
    """Reserve slots for retrieval, structural coverage and local candidates.

    The adapter's config ID is authoritative for active-schedule deduplication.
    Unfilled quotas are backfilled, never with illegal or duplicate candidates.
    Structural distance is categorical, not a performance prediction.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError('limit must be positive')
    if type(structural_proposals) is not int or structural_proposals < 1:
        raise ValueError('structural_proposals must be positive')
    # Explicit caller anchors are not ordinary local-fill candidates. Keep
    # them before retrieval/structural quotas; a full pool previously silently
    # discarded every seed after the initial configuration.
    seeds = list(seeds)
    pool = []; seen = set(); origins = {}; counts = Counter()

    def take(name, configs, quota):
        accepted = 0
        for config in configs:
            if len(pool) >= limit or accepted >= quota:
                break
            if adapter.rejection(config) is not None:
                continue
            key = adapter.config_id(config)
            if key in seen:
                continue
            seen.add(key); pool.append(config); origins[key] = name
            counts[name] += 1; accepted += 1

    # Adapter opt-in only: default callers keep the historical seed order.
    # Protect a few complete regions before a long mutation/seed list can
    # fill the pool. These are still validated proposals, never forced wins.
    protected=list(protected)
    if protected:
        take('seed',[adapter.initial_config],1)
        take('region',protected,min(5,max(0,limit-1)))
        take('retrieval',retrieved,min(2,max(0,limit-len(pool))))
        take('seed',seeds,limit)
    else:take('seed',[adapter.initial_config,*seeds],limit)
    if skip_discarded_generation and len(pool)==limit:
        # Explicit anchors have always had first priority. Generating and
        # ranking thousands of assignments that cannot enter a full pool
        # wastes first-hit budget without changing a single candidate.
        return pool,dict(schema=1,origins=origins,counts=dict(counts),
            structural_statuses={},construction_failures={},generation_skipped='explicit_pool_full')
    raw = raw_portfolio(adapter, StructuralSamplingPolicy(
        min(limit, structural_proposals), structural_proposals, seed))
    structural, audit = select_portfolio(adapter, raw, limit)
    local = adapter.sample(random.Random(seed), limit, [adapter.initial_config])
    sources = [('retrieval', list(retrieved), limit // 4),
               ('structural', structural, max(1, limit // 4)),
               ('local', list(local), limit)]
    for name, configs, quota in sources:
        take(name, configs, quota)
    for name, configs, _ in sources:
        take(name, configs, limit)
    return pool, dict(schema=1, origins=origins, counts=dict(counts),
                     **({'regions':dict(Counter(adapter.region_fn(c) for c in pool))}
                        if getattr(adapter,'region_fn',None) else {}),
                     structural_statuses=dict(Counter(x['status'] for x in audit)),
                     construction_failures=raw['construction_failures'])


class StructuralSearch:
    """A finite broad-exploration arm; UCB still owns allocation."""
    name = 'structural'

    def __init__(self, pool):
        self.pool = tuple(pool)

    def propose(self, adapter, history, rng, limit):
        return [Proposal(c, self.name) for c in self.pool
                if adapter.config_id(c) not in history.seen_ids][:limit]

    def observe(self, observation):
        pass
