"""Explicit, portable composition of frozen-prior candidate orderings.

This is a research policy builder, not a runtime winner cache or a benchmark.
The caller supplies family routing, validates candidates, and owns time/trial
budgets. Stages count proposed unique schedules, including failed evaluations;
they must not be interpreted as a promised number of successful benchmarks.
"""
from dataclasses import dataclass, field
from typing import Mapping

_POLICIES=frozenset(("ordered","random","retrieval","ranker","hybrid"))


@dataclass(frozen=True)
class PriorStage:
    policy: str
    trials: int | None = None

    def __post_init__(self):
        if self.policy not in _POLICIES:
            raise ValueError(f"unknown prior policy: {self.policy}")
        if self.trials is not None and (type(self.trials) is not int or self.trials<1):
            raise ValueError("stage trials must be a positive integer or None")


@dataclass(frozen=True)
class StagedPriorPolicy:
    stages: tuple[PriorStage,...]

    def __post_init__(self):
        object.__setattr__(self,"stages",tuple(self.stages))
        if not self.stages or self.stages[-1].trials is not None:
            raise ValueError("the last stage must retain the remaining pool (trials=None)")
        if any(s.trials is None for s in self.stages[:-1]):
            raise ValueError("only the final stage may be unbounded")

    def order(self,prior,adapter,pool,*,seed=0,max_candidates=None,**selector):
        """Order a prefix lazily; do not invoke later stages for a small budget.

        Without max_candidates this returns a full permutation, never a hard
        pruning decision. Duplicate IDs are rejected to keep budgets honest.
        Unknown contracts retain caller order and never load a learned model.
        """
        if max_candidates is not None and (type(max_candidates) is not int or max_candidates<0):
            raise ValueError("max_candidates must be a nonnegative integer or None")
        pool=list(pool);ids=[adapter.config_id(c) for c in pool]
        if len(ids)!=len(set(ids)):
            raise ValueError("candidate pool contains duplicate configuration IDs")
        limit=len(pool) if max_candidates is None else min(max_candidates,len(pool))
        if not limit or prior.bundle(adapter) is None:
            return pool[:limit]
        result=[];seen=set()
        for stage in self.stages:
            if len(result)>=limit:break
            order=prior.order(adapter,pool,policy=stage.policy,seed=seed,**selector)
            count=0
            for config in order:
                cid=adapter.config_id(config)
                if cid in seen:continue
                result.append(config);seen.add(cid);count+=1
                if len(result)>=limit or stage.trials is not None and count>=stage.trials:break
        return result


@dataclass(frozen=True)
class FamilyPriorPolicy:
    """User-supplied routing; the generic library contains no GPU/family rules."""
    default: StagedPriorPolicy
    routes: Mapping[str,StagedPriorPolicy] = field(default_factory=dict)

    def order(self,prior,adapter,pool,**kwargs):
        policy=self.routes.get(adapter.context.family,self.default)
        return policy.order(prior,adapter,pool,**kwargs)
