"""Bounded candidate mixtures shared by first-hit search and offline studies."""
from collections import Counter
import random

from rtx._sampling import StructuralSamplingPolicy, raw_portfolio, select_portfolio
from .core import Proposal


def mixed_pool(adapter, *, limit=128, seed=1729, retrieved=(), seeds=(),
               structural_proposals=512):
    """Reserve slots for retrieval, structural coverage and local candidates.

    The adapter's config ID is authoritative for active-schedule deduplication.
    Unfilled quotas are backfilled, never with illegal or duplicate candidates.
    Structural distance is categorical, not a performance prediction.
    """
    if type(limit) is not int or limit < 1:
        raise ValueError('limit must be positive')
    if type(structural_proposals) is not int or structural_proposals < 1:
        raise ValueError('structural_proposals must be positive')
    raw = raw_portfolio(adapter, StructuralSamplingPolicy(
        min(limit, structural_proposals), structural_proposals, seed))
    structural, audit = select_portfolio(adapter, raw, limit)
    local = adapter.sample(random.Random(seed), limit, [adapter.initial_config])
    sources = [('seed', [adapter.initial_config, *seeds], 1),
               ('retrieval', list(retrieved), limit // 4),
               ('structural', structural, max(1, limit // 4)),
               ('local', list(local), limit)]
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

    for name, configs, quota in sources:
        take(name, configs, quota)
    for name, configs, _ in sources:
        take(name, configs, limit)
    return pool, dict(schema=1, origins=origins, counts=dict(counts),
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
