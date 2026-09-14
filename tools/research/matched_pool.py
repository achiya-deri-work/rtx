"""Research-only finite candidate universes for matched policy comparisons.

This is an experimental evaluation constraint, never a production pruning
rule. It contains no latency-based filtering or labels for missing candidates.
"""
from dataclasses import replace
import random

from rtx.autotune.core import Proposal, stable_id


class MatchedPoolAdapter:
    def __init__(self, adapter, candidates):
        self.base = adapter
        self._configs = {}
        for config in candidates:
            rejection = adapter.rejection(config)
            if rejection is not None:
                raise ValueError(f'frozen pool contains illegal candidate: {rejection}')
            self._configs.setdefault(adapter.config_id(config), config)
        if adapter.config_id(adapter.initial_config) not in self._configs:
            raise ValueError('frozen pool must contain the initial configuration')
        self.pool = tuple(self._configs[key] for key in sorted(self._configs))
        self.pool_id = stable_id(dict(context=adapter.context.as_dict(),
                                      initial_config_id=adapter.config_id(adapter.initial_config),
                                      config_ids=sorted(self._configs)))
        self.context = replace(adapter.context, tags={**adapter.context.tags,
            'research_matched_pool': self.pool_id})
        self.initial_config = adapter.initial_config

    def __getattr__(self, name):
        return getattr(self.base, name)

    def rejection(self, config):
        if self.base.config_id(config) not in self._configs:
            return ('implementation_rejected', 'outside frozen research candidate pool')
        return self.base.rejection(config)

    def evaluate(self, config):
        # Direct evaluation, including finalist verification, must not bypass
        # the same membership gate enforced for search proposals.
        reason = self.rejection(config)
        if reason is not None:
            raise ValueError(str(reason))
        return self.base.evaluate(config)

    def neighbors(self, config):
        for coordinate, value, candidate in self.base.neighbors(config):
            if self.rejection(candidate) is None:
                yield coordinate, value, candidate

    def sample(self, rng, count, seeds=()):
        if type(count) is not int or count < 0:
            raise ValueError('candidate count must be a nonnegative integer')
        # Ignore seeds for uniform exploration; local search has its own arm.
        return rng.sample(self.pool, min(count, len(self.pool)))

    def protected_prefix(self, count=2, *, order='axis_major'):
        if type(count) is not int or count < 0:
            raise ValueError('prefix count must be a nonnegative integer')
        if order not in ('axis_major', 'structural_interleaved'):
            raise ValueError('unknown protected prefix order')
        if order == 'structural_interleaved':
            # Reuse the library's deterministic component/axis coverage. No
            # timing labels, predictor scores or incumbent enter this plan.
            from rtx.autotune.coverage import coverage_plan
            plan, _ = coverage_plan(self)
            return tuple(replace(p, strategy='initial_neighborhood') for p in plan[:count])
        result = []
        seen = {self.config_id(self.initial_config)}
        for coordinate, value, candidate in self.neighbors(self.initial_config):
            if len(result) >= count:
                break
            cid = self.config_id(candidate)
            if cid in seen:
                continue
            seen.add(cid)
            result.append(Proposal(candidate, 'initial_neighborhood',
                parent_config_id=self.config_id(self.initial_config),
                coordinate=coordinate, coordinate_value=value))
        return tuple(result)

    def manifest(self, *, prefix_order='axis_major'):
        return dict(schema=1, pool_id=self.pool_id,
            context=self.context.as_dict(),
            configs=[dict(config_id=self.config_id(c), config=self.serialize(c),
                          features=dict(self.features(c))) for c in self.pool],
            prefix_order=prefix_order,
            protected_prefix=[self.config_id(p.config) for p in self.protected_prefix(order=prefix_order)],
            production_pruning=False, missing_latency_imputation=False)


class FrozenRanking:
    """Identical proposal mechanics for each pre-measurement ranking treatment."""

    name = 'frozen_ranking'

    def __init__(self, config_ids):
        self.config_ids = tuple(config_ids)

    def propose(self, adapter, history, rng, limit):
        result = []
        for rank, cid in enumerate(self.config_ids):
            if cid in history.seen_ids:
                continue
            result.append(Proposal(adapter._configs[cid], self.name,
                                   metadata={'frozen_rank': rank}))
            if len(result) >= limit:
                break
        return result

    def observe(self, observation):
        pass


def make_matched_tuner(adapter, store, *, treatment, ranking=None,
                       max_trials=8, max_seconds=60., seed=1729,
                       finalist_rounds=2, prefix_order='axis_major', ranked_opening=0):
    """Compose a controlled ranking ablation using the production search engine.

    Ranking must be a complete permutation frozen before evaluation. Missing
    prior coverage must be explicitly filled by the caller, never dropped or
    assigned a fictitious latency. Ordinary uses a seeded random permutation.
    This isolates ranking, not the full deployed policy's extra startup work.
    """
    from rtx.autotune.effort import ProgressiveBanditScheduler
    from rtx.autotune.orchestrator import AutotuneOrchestrator
    from rtx.autotune.core import TuningBudget
    from rtx.autotune.strategies import CoordinateLocalSearch, RandomSearch

    if not isinstance(adapter, MatchedPoolAdapter):
        raise TypeError('matched study requires a frozen pool adapter')
    if treatment not in ('ordinary', 'portfolio', 'rules', 'portfolio_ml'):
        raise ValueError('unknown matched treatment')
    if type(max_trials) is not int or max_trials not in (8, 16, 32, 64):
        raise ValueError('matched budgets must be 8/16/32/64')
    if type(finalist_rounds) is not int or finalist_rounds < 2 or finalist_rounds % 2:
        raise ValueError('matched finalists require a positive even paired round count')
    if type(ranked_opening) is not int or not 0 <= ranked_opening <= 4:
        raise ValueError('ranked opening must be an integer in [0, 4] to reserve exploration by trial 8')
    if treatment == 'ordinary':
        if ranking is not None:
            raise ValueError('ordinary ranking is generated independently from the seed')
        ranking = sorted(adapter._configs)
        random.Random(seed).shuffle(ranking)
    elif ranking is None:
        raise ValueError('learned/retrieval treatment requires an explicit frozen ranking')
    ranking = tuple(ranking)
    if len(ranking) != len(adapter.pool) or set(ranking) != set(adapter._configs):
        raise ValueError('ranking must be a complete permutation of the common pool')
    coverage = adapter.protected_prefix(order=prefix_order)
    if ranked_opening:
        seen={adapter.config_id(adapter.initial_config), *(adapter.config_id(p.config) for p in coverage)}
        opening=[]
        for rank,cid in enumerate(ranking):
            if cid in seen:continue
            opening.append(Proposal(adapter._configs[cid], 'frozen_ranking',
                metadata={'frozen_rank':rank, 'protected_ranked_opening':True}))
            seen.add(cid)
            if len(opening)==ranked_opening:break
        coverage=(*coverage,*opening)
    identity = dict(schema=1, pool_id=adapter.pool_id, treatment=treatment,
                    ranking=ranking, seed=seed,
                    coverage=[adapter.config_id(p.config) for p in coverage],
                    schedule_revision=1, anchor_tolerance=.01)
    if prefix_order != 'axis_major':
        identity['prefix_order'] = prefix_order
    if ranked_opening:
        identity['ranked_opening'] = ranked_opening
    tuner = AutotuneOrchestrator(adapter, store,
        [CoordinateLocalSearch(anchor_tolerance=.01), RandomSearch(), FrozenRanking(ranking)],
        ProgressiveBanditScheduler(exploration=1., warmup_trials=0),
        TuningBudget(max_trials, max_seconds), seed=seed,
        max_trials_includes_resumed=True, transfer_history=False,
        coverage_proposals=coverage)
    tuner.checkpoint_key = stable_id(identity)
    tuner.effort_contract = stable_id(identity)
    tuner.finalist_rounds = finalist_rounds
    tuner.finalist_reserve_s = max_seconds * .2
    tuner.pool_audit = identity
    return tuner


def make_reference_tuner(adapter, store, *, max_seconds=1800., seed=1729):
    """Enumerate the entire frozen universe independently of learned scores."""
    from rtx.autotune.core import TuningBudget
    from rtx.autotune.orchestrator import AutotuneOrchestrator, SequentialScheduler
    from rtx.autotune.strategies import RandomSearch
    if not isinstance(adapter, MatchedPoolAdapter):
        raise TypeError('reference enumeration requires a frozen pool')
    ordered = list(adapter.pool)
    random.Random(seed).shuffle(ordered)
    tuner = AutotuneOrchestrator(adapter, store, [RandomSearch()],
        SequentialScheduler((('random', None),)),
        TuningBudget(len(ordered), max_seconds), seed=seed,
        max_trials_includes_resumed=True, transfer_history=False,
        coverage_proposals=[Proposal(c,'reference_enumeration') for c in ordered])
    tuner.checkpoint_key = stable_id(dict(pool=adapter.pool_id, seed=seed,
                                         purpose='full_pool_reference_v1'))
    return tuner
