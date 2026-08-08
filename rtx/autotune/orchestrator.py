"""Budgeted orchestration for interchangeable autotuning strategies."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import random
import statistics as python_statistics
import time
from typing import Callable, Generic, Mapping, Protocol, Sequence

from .core import (
    ComposableTuningResult,
    ConfigT,
    KernelAdapter,
    Observation,
    Proposal,
    SearchHistory,
    TuningBudget,
    evaluate_proposal,
    utc_now,
)
from .legacy import TrialOutcome
from .store import TuningStore
from .strategies import SearchStrategy


@dataclass(slots=True)
class ArmStatistics:
    pulls: int = 0
    successes: int = 0
    reward_sum: float = 0.0
    elapsed_s: float = 0.0
    unavailable_until: int = 0
    effective_pulls: float = 0.0
    effective_reward_sum: float = 0.0
    prior_pulls: float = 0.0
    prior_reward_sum: float = 0.0
    last_reward: float = 0.0

    @property
    def mean_reward(self) -> float:
        return 0.0 if self.pulls == 0 else self.reward_sum / self.pulls

    @property
    def bandit_pulls(self) -> float:
        return self.effective_pulls + self.prior_pulls

    @property
    def bandit_mean_reward(self) -> float:
        pulls = self.bandit_pulls
        if pulls <= 0:
            return 0.0
        return (self.effective_reward_sum + self.prior_reward_sum) / pulls

    def as_dict(self) -> dict[str, object]:
        return {
            "pulls": self.pulls,
            "successes": self.successes,
            "reward_sum": self.reward_sum,
            "elapsed_s": self.elapsed_s,
            "effective_pulls": self.effective_pulls,
            "effective_reward_sum": self.effective_reward_sum,
            "prior_pulls": self.prior_pulls,
            "prior_reward_sum": self.prior_reward_sum,
            "last_reward": self.last_reward,
            "unavailable_until": self.unavailable_until,
        }


@dataclass(frozen=True, slots=True)
class ConfirmationPolicy:
    """Repeat promising measurements before they can become incumbents."""

    repeats: int = 0
    contender_ratio: float = 0.0
    confirm_initial: bool = False

    def __post_init__(self) -> None:
        if self.repeats < 0:
            raise ValueError("confirmation repeats cannot be negative")
        if self.contender_ratio < 0:
            raise ValueError("confirmation contender ratio cannot be negative")


class StrategyScheduler(Protocol):
    name: str

    def select(
        self,
        names: Sequence[str],
        statistics: Mapping[str, ArmStatistics],
        trial_index: int,
    ) -> str | None: ...


@dataclass(slots=True)
class UCB1Scheduler:
    """Multi-armed bandit over complete search strategies."""

    exploration: float = 1.25
    latency_aware: bool = True
    name: str = "ucb1_bandit"

    def select(
        self,
        names: Sequence[str],
        statistics: Mapping[str, ArmStatistics],
        trial_index: int,
    ) -> str | None:
        available = [
            name for name in names if statistics[name].unavailable_until <= trial_index
        ]
        if not available:
            return None
        for name in available:
            if statistics[name].pulls == 0:
                return name
        total = max(1, sum(statistics[name].pulls for name in available))

        def score(name: str) -> float:
            arm = statistics[name]
            exploitation = arm.mean_reward
            if self.latency_aware and arm.elapsed_s > 0:
                exploitation /= max(1e-6, arm.elapsed_s / arm.pulls)
            return exploitation + self.exploration * math.sqrt(
                math.log(total + 1) / arm.pulls
            )

        return max(available, key=score)


def _feature_number(observation: Observation, name: str) -> float | None:
    value = observation.features.get(name)
    if value is None or not math.isfinite(float(value)):
        return None
    return float(value)


def _observation_similarity(
    observation: Observation,
    active_features: Mapping[str, float],
) -> float:
    """Conservative workload/device similarity for contextual virtual pulls."""

    distance = 0.0
    dimensions = 0
    for key in ("context.workload.m", "context.workload.n", "context.workload.k"):
        left = _feature_number(observation, key)
        right = active_features.get(key)
        if left is None or right is None or left <= 0 or right <= 0:
            continue
        distance += abs(math.log2(left / float(right)))
        dimensions += 1
    shape_similarity = math.exp(-distance / max(1, dimensions))

    regime_similarity = 1.0
    regime_keys = {
        key for key, value in observation.features.items()
        if key.startswith("context.regime=") and value > 0
    }
    active_regime_keys = {
        key for key, value in active_features.items()
        if key.startswith("context.regime=") and value > 0
    }
    if regime_keys and active_regime_keys and regime_keys != active_regime_keys:
        regime_similarity = 0.45

    sm_similarity = 1.0
    sm_key = "context.device.multiprocessor_count"
    left_sm = _feature_number(observation, sm_key)
    right_sm = active_features.get(sm_key)
    if left_sm is not None and right_sm is not None and left_sm > 0 and right_sm > 0:
        sm_similarity = math.exp(-abs(math.log2(left_sm / float(right_sm))))
    return max(0.02, shape_similarity * regime_similarity * sm_similarity)


@dataclass(slots=True)
class AdaptiveBanditScheduler:
    """Discounted, cost-aware contextual UCB over search strategies.

    Current-context observations are replayed on resume. Observations from
    similar contexts contribute only capped virtual pulls, so transfer can
    break ties but cannot overwhelm measurements from the active context.
    """

    exploration: float = 0.35
    discount: float = 0.985
    warmup_trials: int = 32
    warmup_arm: str = "random"
    transfer_prior_strength: float = 3.0
    cost_scale_s: float = 2.0
    name: str = "adaptive_contextual_bandit"

    def __post_init__(self) -> None:
        if self.exploration < 0:
            raise ValueError("bandit exploration cannot be negative")
        if not 0 < self.discount <= 1:
            raise ValueError("bandit discount must be in (0, 1]")
        if self.warmup_trials < 0 or self.transfer_prior_strength < 0:
            raise ValueError("bandit warmup/prior strength cannot be negative")
        if self.cost_scale_s <= 0:
            raise ValueError("bandit cost scale must be positive")

    def reward(self, before: float, observation: Observation) -> float:
        cost = 0.04 * math.tanh(max(0.0, observation.elapsed_s) / self.cost_scale_s)
        if not observation.successful:
            return -0.20 - cost
        improvement = 0.0
        if math.isfinite(before) and observation.score > 0:
            improvement = math.tanh(max(0.0, math.log(before / observation.score)))
        uncertainty = observation.metadata.get("predicted_std_ms", 0.0)
        predicted = observation.metadata.get("predicted_ms", 0.0)
        try:
            information = float(uncertainty) / max(float(predicted), 1e-9)
        except (TypeError, ValueError):
            information = 0.0
        return 0.015 + improvement + 0.015 * math.tanh(max(0.0, information)) - cost

    def _decay(self, statistics: Mapping[str, ArmStatistics]) -> None:
        for arm in statistics.values():
            arm.effective_pulls *= self.discount
            arm.effective_reward_sum *= self.discount

    def update(
        self,
        statistics: Mapping[str, ArmStatistics],
        arm_name: str,
        reward: float,
        observation: Observation,
    ) -> None:
        self._decay(statistics)
        arm = statistics[arm_name]
        arm.pulls += 1
        arm.successes += int(observation.successful)
        arm.elapsed_s += observation.elapsed_s
        arm.reward_sum += reward
        arm.effective_pulls += 1.0
        arm.effective_reward_sum += reward
        arm.last_reward = reward

    def initialize(
        self,
        statistics: Mapping[str, ArmStatistics],
        history: SearchHistory,
        active_features: Mapping[str, float],
    ) -> None:
        """Rebuild exact active state and contextual priors from JSONL history."""

        incumbents: dict[str, float] = {}
        transfer_rewards: dict[str, list[tuple[float, float]]] = {
            name: [] for name in statistics
        }
        ordered = sorted(
            history.observations,
            key=lambda item: (item.finished_at, item.session_id, item.sequence),
        )
        for observation in ordered:
            before = incumbents.get(observation.context_id, math.inf)
            if observation.successful:
                incumbents[observation.context_id] = min(before, observation.score)
            if observation.strategy not in statistics:
                continue
            reward = self.reward(before, observation)
            if observation.context_id == history.active_context_id:
                self.update(statistics, observation.strategy, reward, observation)
            else:
                similarity = _observation_similarity(observation, active_features)
                transfer_rewards[observation.strategy].append((similarity, reward))

        for name, values in transfer_rewards.items():
            if not values or self.transfer_prior_strength <= 0:
                continue
            weight = sum(item[0] for item in values)
            if weight <= 0:
                continue
            mean = sum(similarity * reward for similarity, reward in values) / weight
            pulls = self.transfer_prior_strength * weight / (weight + 8.0)
            statistics[name].prior_pulls = pulls
            statistics[name].prior_reward_sum = pulls * mean

    def scores(
        self,
        names: Sequence[str],
        statistics: Mapping[str, ArmStatistics],
        trial_index: int,
    ) -> dict[str, float | None]:
        available = [
            name for name in names if statistics[name].unavailable_until <= trial_index
        ]
        if not available:
            return {name: None for name in names}
        total = max(1.0, sum(statistics[name].bandit_pulls for name in available))
        result: dict[str, float | None] = {name: None for name in names}
        for name in available:
            arm = statistics[name]
            pulls = arm.bandit_pulls
            if pulls <= 0:
                result[name] = math.inf
            else:
                result[name] = arm.bandit_mean_reward + self.exploration * math.sqrt(
                    math.log(total + 1.0) / pulls
                )
        return result

    def select(
        self,
        names: Sequence[str],
        statistics: Mapping[str, ArmStatistics],
        trial_index: int,
    ) -> str | None:
        available = [
            name for name in names if statistics[name].unavailable_until <= trial_index
        ]
        if not available:
            return None
        if trial_index < self.warmup_trials and self.warmup_arm in available:
            return self.warmup_arm
        # Every real arm gets a local measurement before contextual priors are trusted.
        for name in available:
            if statistics[name].pulls == 0:
                return name
        scores = self.scores(names, statistics, trial_index)
        return max(available, key=lambda name: float(scores[name]))

    def snapshot(
        self,
        names: Sequence[str],
        statistics: Mapping[str, ArmStatistics],
        trial_index: int,
    ) -> dict[str, object]:
        raw_scores = self.scores(names, statistics, trial_index)
        return {
            "scheduler": self.name,
            "trial": trial_index,
            "discount": self.discount,
            "exploration": self.exploration,
            "scores": {
                name: (
                    None
                    if value is None or not math.isfinite(value)
                    else value
                )
                for name, value in raw_scores.items()
            },
            "arms": {name: statistics[name].as_dict() for name in names},
        }


@dataclass(slots=True)
class SequentialScheduler:
    """Deterministic strategy stages such as cost-model search then local search."""

    stages: Sequence[tuple[str, int | None]]
    name: str = "sequential"

    def select(
        self,
        names: Sequence[str],
        statistics: Mapping[str, ArmStatistics],
        trial_index: int,
    ) -> str | None:
        consumed = 0
        for name, budget in self.stages:
            if name not in names:
                raise ValueError(f"sequential scheduler references unknown strategy {name!r}")
            if budget is None or trial_index < consumed + budget:
                return (
                    name
                    if statistics[name].unavailable_until <= trial_index
                    else None
                )
            consumed += budget
        return self.stages[-1][0] if self.stages else None


def _restore_observation(
    adapter: KernelAdapter[ConfigT],
    record: Mapping[str, object],
) -> Observation[ConfigT] | None:
    try:
        serialized = dict(record["config"])  # type: ignore[arg-type]
        config = adapter.deserialize(serialized)
        outcome = TrialOutcome.from_dict(record["outcome"])  # type: ignore[arg-type]
        return Observation(
            observation_id=str(record["observation_id"]),
            session_id=str(record["session_id"]),
            sequence=int(record["sequence"]),
            context_id=str(record["context_id"]),
            family=str(record["family"]),
            kernel_revision=int(record["kernel_revision"]),
            config_id=str(record["config_id"]),
            config=config,
            serialized_config=serialized,
            features={str(key): float(value) for key, value in dict(record["features"]).items()},  # type: ignore[arg-type]
            strategy=str(record["strategy"]),
            outcome=outcome,
            started_at=str(record["started_at"]),
            finished_at=str(record["finished_at"]),
            elapsed_s=float(record["elapsed_s"]),
            parent_config_id=(
                None if record.get("parent_config_id") is None else str(record["parent_config_id"])
            ),
            coordinate=None if record.get("coordinate") is None else str(record["coordinate"]),
            coordinate_value=record.get("coordinate_value"),
            metadata=dict(record.get("metadata", {})),  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError):
        return None


class AutotuneOrchestrator(Generic[ConfigT]):
    """Evaluate, persist, and route budget among arbitrary search strategies."""

    def __init__(
        self,
        adapter: KernelAdapter[ConfigT],
        store: TuningStore[ConfigT],
        strategies: Sequence[SearchStrategy[ConfigT]],
        scheduler: StrategyScheduler,
        budget: TuningBudget = TuningBudget(),
        *,
        seed: int = 0,
        resume: bool = True,
        max_trials_includes_resumed: bool = False,
        transfer_history: bool = True,
        confirmation: ConfirmationPolicy = ConfirmationPolicy(),
        progress: Callable[[str], None] | None = None,
    ) -> None:
        if not strategies:
            raise ValueError("at least one search strategy is required")
        names = [strategy.name for strategy in strategies]
        if len(names) != len(set(names)):
            raise ValueError("strategy names must be unique")
        self.adapter = adapter
        self.store = store
        self.strategies = {strategy.name: strategy for strategy in strategies}
        self.scheduler = scheduler
        self.budget = budget
        self.seed = seed
        self.resume = resume
        self.max_trials_includes_resumed = max_trials_includes_resumed
        self.transfer_history = transfer_history
        self.confirmation = confirmation
        self.progress = progress

    def _log(self, started: float, message: str) -> None:
        if self.progress is not None:
            self.progress(f"[{time.monotonic() - started:8.2f}s] {message}")

    @staticmethod
    def _reward(before: float, observation: Observation[ConfigT]) -> float:
        if not observation.successful:
            return -0.01
        if not math.isfinite(before):
            return 0.05
        improvement = max(0.0, math.log(before / observation.score))
        return 0.002 + improvement

    def tune(self) -> ComposableTuningResult[ConfigT]:
        started = time.monotonic()
        rng = random.Random(self.seed)
        restored: list[Observation[ConfigT]] = []
        if self.resume:
            source_context = None if self.transfer_history else self.adapter.context
            for record in self.store.records(source_context):
                if (
                    record.get("family") != self.adapter.context.family
                    or int(record.get("kernel_revision", -1))
                    != self.adapter.context.kernel_revision
                ):
                    continue
                observation = _restore_observation(self.adapter, record)
                if observation is not None:
                    restored.append(observation)
        history = SearchHistory(restored, self.adapter.context.identifier)
        resumed_active = (
            len(history.current) if self.max_trials_includes_resumed else 0
        )
        session_id = self.store.start_session(
            self.adapter.context,
            {
                "scheduler": self.scheduler.name,
                "strategies": list(self.strategies),
                "budget": {
                    "max_trials": self.budget.max_trials,
                    "time_budget_s": self.budget.time_budget_s,
                },
                "seed": self.seed,
                "resumed_observations": len(restored),
                "resumed_active_budget": resumed_active,
                "max_trials_includes_resumed": self.max_trials_includes_resumed,
            },
        )
        statistics = {name: ArmStatistics() for name in self.strategies}
        initialize = getattr(self.scheduler, "initialize", None)
        if initialize is not None:
            initialize(
                statistics,
                history,
                self.adapter.context.features(),
            )
        evaluated = 0

        def confirm(
            observation: Observation[ConfigT],
            proposal: Proposal[ConfigT],
            incumbent_score: float,
        ) -> None:
            policy = self.confirmation
            if policy.repeats == 0 or not observation.successful:
                return
            is_initial = proposal.strategy == "initial"
            contender = math.isfinite(incumbent_score) and (
                observation.score
                <= incumbent_score * (1.0 + policy.contender_ratio)
            )
            if not ((is_initial and policy.confirm_initial) or contender):
                return
            screen = observation.outcome
            outcomes = [screen]
            confirmation_started = time.monotonic()
            for _ in range(policy.repeats):
                try:
                    outcome = self.adapter.evaluate(proposal.config)
                except Exception as exc:
                    outcome = TrialOutcome(
                        "runtime_error",
                        error=f"{type(exc).__name__}: {exc}"[:4000],
                    )
                outcomes.append(outcome)
            successful = [item for item in outcomes if item.successful]
            timings = [
                timing
                for item in successful
                for timing in item.timings_ms
            ]
            if timings:
                median_ms = float(python_statistics.median(timings))
            elif successful:
                median_ms = float(
                    python_statistics.median(
                        float(item.median_ms) for item in successful
                    )
                )
            else:
                median_ms = None
            if median_ms is not None:
                compile_values = [
                    float(item.compile_ms)
                    for item in successful
                    if item.compile_ms is not None
                ]
                error_values = [
                    float(item.max_abs_error)
                    for item in successful
                    if item.max_abs_error is not None
                ]
                observation.outcome = TrialOutcome(
                    "ok",
                    median_ms=median_ms,
                    timings_ms=timings,
                    compile_ms=(
                        None if not compile_values else sum(compile_values)
                    ),
                    max_abs_error=(
                        None if not error_values else max(error_values)
                    ),
                )
            confirmation_elapsed = time.monotonic() - confirmation_started
            observation.elapsed_s += confirmation_elapsed
            observation.finished_at = utc_now()
            observation.metadata["confirmation"] = {
                "repeats": policy.repeats,
                "screen_outcome": screen.as_dict(),
                "outcomes": [item.as_dict() for item in outcomes[1:]],
                "successful_runs": len(successful),
                "elapsed_s": confirmation_elapsed,
            }

        def run(proposal: Proposal[ConfigT]) -> Observation[ConfigT]:
            nonlocal evaluated
            before = math.inf if history.best is None else history.best.score
            observation = evaluate_proposal(
                self.adapter,
                proposal,
                session_id=session_id,
                sequence=evaluated,
            )
            confirm(observation, proposal, before)
            evaluated += 1
            history.observations.append(observation)
            self.store.record_observation(observation)
            arm = statistics.get(proposal.strategy)
            if arm is not None:
                reward_fn = getattr(self.scheduler, "reward", self._reward)
                reward = float(reward_fn(before, observation))
                update = getattr(self.scheduler, "update", None)
                if update is not None:
                    update(statistics, proposal.strategy, reward, observation)
                else:
                    arm.pulls += 1
                    arm.successes += int(observation.successful)
                    arm.elapsed_s += observation.elapsed_s
                    arm.reward_sum += reward
            self._log(
                started,
                f"SAVE {proposal.strategy} {observation.config_id} "
                f"{observation.outcome.status}"
                + ("" if observation.outcome.median_ms is None else f" {observation.score * 1000:.3f}us")
                + (
                    ""
                    if observation.outcome.error is None
                    else " "
                    + " ".join(observation.outcome.error.splitlines())[:240]
                ),
            )
            return observation

        initial_id = self.adapter.config_id(self.adapter.initial_config)
        if (
            initial_id not in history.seen_ids
            and resumed_active + evaluated < self.budget.max_trials
        ):
            initial = run(Proposal(self.adapter.initial_config, "initial"))
            if not initial.successful:
                self.store.finish_session(
                    session_id,
                    {"status": "failed", "reason": "initial configuration failed"},
                )
                raise RuntimeError(
                    f"initial configuration failed: {initial.outcome.status}: {initial.outcome.error}"
                )

        names = tuple(self.strategies)
        empty_attempts = 0
        while resumed_active + evaluated < self.budget.max_trials:
            if time.monotonic() - started >= self.budget.time_budget_s:
                break
            trial_index = resumed_active + evaluated
            arm_name = self.scheduler.select(names, statistics, trial_index)
            if arm_name is None:
                break
            strategy = self.strategies[arm_name]
            snapshot = getattr(self.scheduler, "snapshot", None)
            self.store.record_event(
                session_id,
                "strategy_selected",
                {
                    "trial": trial_index,
                    "strategy": arm_name,
                    "statistics": {
                        name: arm.as_dict() for name, arm in statistics.items()
                    },
                    "scheduler_state": (
                        None
                        if snapshot is None
                        else snapshot(names, statistics, trial_index)
                    ),
                },
            )
            try:
                proposals = strategy.propose(self.adapter, history, rng, 1)
            except Exception as exc:
                statistics[arm_name].unavailable_until = trial_index + len(names)
                self.store.record_event(
                    session_id,
                    "strategy_error",
                    {
                        "trial": trial_index,
                        "strategy": arm_name,
                        "error": f"{type(exc).__name__}: {exc}"[:4000],
                    },
                )
                empty_attempts += 1
                if empty_attempts >= len(names) * 2:
                    break
                continue
            proposals = [
                proposal
                for proposal in proposals
                if self.adapter.config_id(proposal.config) not in history.seen_ids
            ]
            if not proposals:
                empty_attempts += 1
                statistics[arm_name].unavailable_until = trial_index + len(names)
                self.store.record_event(
                    session_id,
                    "strategy_empty",
                    {"trial": trial_index, "strategy": arm_name},
                )
                if empty_attempts >= len(names) * 2:
                    break
                continue
            empty_attempts = 0
            proposal = proposals[0]
            proposal.strategy = arm_name
            observation = run(proposal)
            strategy.observe(observation)

        best = history.best
        if best is None:
            self.store.finish_session(session_id, {"status": "failed", "reason": "no successful trials"})
            raise RuntimeError("autotuning produced no successful configurations")
        elapsed = time.monotonic() - started
        status = (
            "budget_exhausted"
            if resumed_active + evaluated >= self.budget.max_trials
            or elapsed >= self.budget.time_budget_s
            else "complete"
        )
        self.store.finish_session(
            session_id,
            {
                "status": status,
                "best_config_id": best.config_id,
                "best_median_ms": best.score,
                "evaluated_trials": evaluated,
                "elapsed_s": elapsed,
            },
        )
        return ComposableTuningResult(
            config=best.config,
            median_ms=best.score,
            session_id=session_id,
            context_id=self.adapter.context.identifier,
            evaluated_trials=evaluated,
            elapsed_s=elapsed,
            strategy_trials={name: arm.pulls for name, arm in statistics.items()},
            store_path=None if self.store.path is None else str(self.store.path),
        )


__all__ = [
    "AdaptiveBanditScheduler",
    "ArmStatistics",
    "AutotuneOrchestrator",
    "ConfirmationPolicy",
    "SequentialScheduler",
    "StrategyScheduler",
    "UCB1Scheduler",
]
