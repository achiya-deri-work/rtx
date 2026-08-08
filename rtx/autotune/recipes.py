"""Opinionated compositions built solely from the public autotuning contracts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .bandit import AdaptiveBanditScheduler
from .core import ConfigT, KernelAdapter, TuningBudget
from .cost_model import GradientBoostedCostModel, GradientBoostedFeasibilityModel
from .orchestrator import (
    AutotuneOrchestrator,
    ConfirmationPolicy,
    SequentialScheduler,
)
from .store import TuningStore
from .strategies import CostModelGuidedSearch, CostModelLocalSearch, RandomSearch


@dataclass(frozen=True, slots=True)
class HybridTuningPolicy:
    orchestration: Literal["sequential", "bandit"] = "sequential"
    max_trials: int = 512
    time_budget_s: float = 1800.0
    cost_model_trials: int = 320
    model_warmup: int = 32
    model_pool_size: int = 4096
    model_refit_interval: int = 16
    model_exploration: float = 0.15
    model_estimators: int = 24
    model_ensembles: int = 3
    model_max_depth: int = 3
    model_min_leaf: int = 4
    model_max_features: int = 64
    model_max_thresholds: int = 10
    feasibility_estimators: int = 16
    feasibility_ensembles: int = 3
    feasibility_max_depth: int = 3
    feasibility_min_leaf: int = 3
    feasibility_exploration: float = 0.5
    minimum_optimistic_feasibility: float = 0.05
    local_beam_width: int = 3
    local_model_refit_interval: int = 32
    bandit_exploration: float = 1.25
    confirmation_repeats: int = 0
    confirmation_ratio: float = 0.0
    confirm_initial: bool = False
    seed: int = 0
    resume: bool = True
    max_trials_includes_resumed: bool = True
    transfer_history: bool = True


def make_hybrid_autotuner(
    adapter: KernelAdapter[ConfigT],
    store: TuningStore[ConfigT],
    policy: HybridTuningPolicy = HybridTuningPolicy(),
    *,
    progress=None,
) -> AutotuneOrchestrator[ConfigT]:
    """Build GBT-guided global search followed by/bandited with local search."""

    random_search = RandomSearch[ConfigT]()
    feasibility = GradientBoostedFeasibilityModel(
        n_estimators=policy.feasibility_estimators,
        ensembles=policy.feasibility_ensembles,
        max_depth=policy.feasibility_max_depth,
        min_leaf=policy.feasibility_min_leaf,
        max_features=policy.model_max_features,
        max_thresholds=policy.model_max_thresholds,
        seed=policy.seed ^ 0x5EED,
    )
    learned = CostModelGuidedSearch[ConfigT](
        model=GradientBoostedCostModel(
            n_estimators=policy.model_estimators,
            ensembles=policy.model_ensembles,
            max_depth=policy.model_max_depth,
            min_leaf=policy.model_min_leaf,
            max_features=policy.model_max_features,
            max_thresholds=policy.model_max_thresholds,
            seed=policy.seed,
        ),
        warmup=random_search,
        min_observations=policy.model_warmup,
        pool_size=policy.model_pool_size,
        refit_interval=policy.model_refit_interval,
        exploration=policy.model_exploration,
        feasibility_model=feasibility,
        feasibility_exploration=policy.feasibility_exploration,
        minimum_optimistic_feasibility=policy.minimum_optimistic_feasibility,
    )
    local = CostModelLocalSearch[ConfigT](
        model=learned.model,
        feasibility_model=feasibility,
        beam_width=policy.local_beam_width,
        exploration=policy.model_exploration * 0.25,
        refit_interval=policy.local_model_refit_interval,
        feasibility_exploration=policy.feasibility_exploration,
        minimum_optimistic_feasibility=policy.minimum_optimistic_feasibility,
    )
    if policy.orchestration == "sequential":
        strategies = [learned, local]
        scheduler = SequentialScheduler(
            ((learned.name, policy.cost_model_trials), (local.name, None))
        )
    else:
        strategies = [random_search, learned, local]
        scheduler = AdaptiveBanditScheduler(
            exploration=policy.bandit_exploration,
            warmup_trials=policy.model_warmup,
            warmup_arm=random_search.name,
        )
    return AutotuneOrchestrator(
        adapter,
        store,
        strategies,
        scheduler,
        TuningBudget(policy.max_trials, policy.time_budget_s),
        seed=policy.seed,
        resume=policy.resume,
        max_trials_includes_resumed=policy.max_trials_includes_resumed,
        transfer_history=policy.transfer_history,
        confirmation=ConfirmationPolicy(
            repeats=policy.confirmation_repeats,
            contender_ratio=policy.confirmation_ratio,
            confirm_initial=policy.confirm_initial,
        ),
        progress=progress,
    )


__all__ = ["HybridTuningPolicy", "make_hybrid_autotuner"]
