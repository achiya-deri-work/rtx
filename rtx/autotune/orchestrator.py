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

    @property
    def mean_reward(self) -> float:
        return 0.0 if self.pulls == 0 else self.reward_sum / self.pulls


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
                arm.pulls += 1
                arm.successes += int(observation.successful)
                arm.elapsed_s += observation.elapsed_s
                arm.reward_sum += self._reward(before, observation)
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
            self.store.record_event(
                session_id,
                "strategy_selected",
                {
                    "trial": trial_index,
                    "strategy": arm_name,
                    "statistics": {
                        name: {
                            "pulls": arm.pulls,
                            "successes": arm.successes,
                            "reward_sum": arm.reward_sum,
                            "elapsed_s": arm.elapsed_s,
                        }
                        for name, arm in statistics.items()
                    },
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
    "ArmStatistics",
    "AutotuneOrchestrator",
    "ConfirmationPolicy",
    "SequentialScheduler",
    "StrategyScheduler",
    "UCB1Scheduler",
]
