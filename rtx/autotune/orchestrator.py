"""Budgeted orchestration for interchangeable autotuning strategies."""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
import statistics as python_statistics
import time
from typing import Callable, Generic, Mapping, Protocol, Sequence

from .bandit import (
    AdaptiveBanditScheduler,
    ArmStatistics,
    UCB1Scheduler,
    observation_strategy_cost_s,
)
from .core import (
    ComposableTuningResult,
    ConfigT,
    KernelAdapter,
    Observation,
    Proposal,
    SearchHistory,
    TuningBudget,
    evaluate_proposal,
    observation_from_dict,
    stable_id,
    utc_now,
)
from .outcomes import TrialOutcome, raise_if_fatal_device_context_error
from .store import TuningStore
from .strategies import SearchStrategy


class NoEligibleTimingError(RuntimeError):
    """No latency winner; optionally retain an independently checked seed.

    Offline tuning still fails. A runtime caller may execute the checked seed
    without promoting the unstable observations to a persistent winner.
    """

    def __init__(self, fallback_config=None, *, fallback_validation=None):
        super().__init__("autotuning produced no successful configurations with eligible timings")
        self.fallback_config = fallback_config
        self.fallback_validation = fallback_validation


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
        coverage_proposals: Sequence[Proposal[ConfigT]] = (),
        started_at: float | None = None,
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
        self.coverage_proposals = tuple(coverage_proposals)
        self.started_at=started_at
        self.pool_audit = None
        self.checkpoint_key = None
        self.finalist_rounds = 0
        self.finalist_margin = 0.005
        self.finalist_reserve_s = 0.
        self.effort_contract = None

    def _log(self, started: float, message: str) -> None:
        if self.progress is not None:
            self.progress(f"[{time.monotonic() - started:8.2f}s] {message}")

    @staticmethod
    def _reward(before: float, observation: Observation[ConfigT]) -> float:
        if not observation.outcome.timing_eligible:
            return -0.01
        if not math.isfinite(before):
            return 0.05
        improvement = max(0.0, math.log(before / observation.score))
        return 0.002 + improvement

    def tune(self) -> ComposableTuningResult[ConfigT]:
        from .store import tuning_session_lock
        with tuning_session_lock(self.store, self.adapter.context):
            return self._tune_locked()

    def _tune_locked(self) -> ComposableTuningResult[ConfigT]:
        started = time.monotonic() if self.started_at is None else self.started_at
        search_seconds = max(0., self.budget.time_budget_s - self.finalist_reserve_s)
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
                try:
                    observation=observation_from_dict(self.adapter, record)
                    rejection=self.adapter.rejection(observation.config)
                    if rejection is not None:
                        # Current legality wins over a previously timed cache
                        # entry. Keep the historical trial count, but never
                        # restore its latency as an eligible winner.
                        observation.outcome=TrialOutcome(rejection[0],error=rejection[1])
                    restored.append(observation)
                except (KeyError, TypeError, ValueError):
                    continue
        history = SearchHistory(restored, self.adapter.context.identifier)
        by_config = {}
        for item in history.current:
            by_config.setdefault(item.config_id, []).append(item)
        history.retryable_config_ids.update(cid for cid,items in by_config.items()
            if all(item.successful and not item.outcome.timing_eligible for item in items))
        incomplete = getattr(self.store, "incomplete_candidates", None)
        abandoned = (
            []
            if incomplete is None
            else list(incomplete(self.adapter.context.identifier))
        )
        abandoned_ids = {
            str(item.get("config_id", "")) for item in abandoned
        } - {""}
        abandoned_ids -= {o.config_id for o in history.current}
        history.excluded_config_ids.update(abandoned_ids)
        resumed_active = (
            len(history.current) + len(abandoned_ids) if self.max_trials_includes_resumed else 0
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
                "candidate_pool": self.pool_audit,
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
        if self.checkpoint_key and history.current:
            checkpoint = history.current[-1].metadata.get('search_checkpoint')
            if checkpoint is not None:
                if (checkpoint.get('key') != self.checkpoint_key or
                        checkpoint.get('count') != len(history.current) or
                        checkpoint.get('history_digest') != stable_id([
                            (o.config_id, o.outcome.status) for o in history.current]) or
                        set(checkpoint.get('statistics', {})) != set(statistics)):
                    raise ValueError('progressive search checkpoint is incompatible with this policy/history')
                def tuples(value):
                    return tuple(tuples(v) for v in value) if isinstance(value, (list, tuple)) else value
                rng.setstate(tuples(checkpoint['rng']))
                history.excluded_config_ids.update(checkpoint.get('excluded_ids', []))
                statistics = {name: ArmStatistics(**value)
                              for name, value in checkpoint['statistics'].items()}
        evaluated = 0
        if abandoned:
            self.store.record_event(
                session_id,
                "abandoned_candidates_recovered",
                {
                    "context_id": self.adapter.context.identifier,
                    "count": len(abandoned),
                    "config_ids": sorted(abandoned_ids),
                    "attempt_ids": sorted(
                        str(item.get("attempt_id", "")) for item in abandoned
                    ),
                },
            )
            self._log(
                started,
                f"RECOVER excluded {len(abandoned_ids)} incomplete candidate(s)",
            )

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
                # One failed repeat invalidates the candidate. In particular,
                # do not launch again after a sticky device fault. Persist the
                # failed outcome below before run() propagates the fatal error.
                if not outcome.successful:
                    break
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
            failed = next((item for item in outcomes if not item.successful), None)
            if failed is not None:
                observation.outcome = failed
            elif median_ms is not None:
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
                    metadata=dict(screen.metadata),
                )
                if any("sampling" in item.metadata for item in outcomes):
                    from .measurement import runtime_timing_metadata
                    quality = runtime_timing_metadata(timings)
                    if any(not item.timing_eligible for item in outcomes):
                        quality["sampling"]["collection"]["stationary"] = False
                    observation.outcome.metadata.update(quality)
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
            config_id = self.adapter.config_id(proposal.config)
            attempt_id = stable_id(
                {
                    "session_id": session_id,
                    "sequence": evaluated,
                    "context_id": self.adapter.context.identifier,
                    "config_id": config_id,
                },
                32,
            )
            self.store.record_event(
                session_id,
                "candidate_started",
                {
                    "attempt_id": attempt_id,
                    "sequence": evaluated,
                    "context_id": self.adapter.context.identifier,
                    "family": self.adapter.context.family,
                    "kernel_revision": self.adapter.context.kernel_revision,
                    "config_id": config_id,
                    "config": dict(self.adapter.serialize(proposal.config)),
                    "features": dict(self.adapter.features(proposal.config)),
                    "strategy": proposal.strategy,
                    "parent_config_id": proposal.parent_config_id,
                    "coordinate": proposal.coordinate,
                    "coordinate_value": proposal.coordinate_value,
                    "metadata": dict(proposal.metadata),
                },
            )
            self._log(started, f"TRY  {proposal.strategy} {config_id}")
            observation = evaluate_proposal(
                self.adapter,
                proposal,
                session_id=session_id,
                sequence=evaluated,
            )
            observation.metadata = {
                **observation.metadata,
                "attempt_id": attempt_id,
            }
            confirm(observation, proposal, before)
            evaluated += 1
            history.observations.append(observation)
            history.retryable_config_ids.discard(observation.config_id)
            arm = statistics.get(proposal.strategy)
            try:
                if arm is not None:
                    reward_fn = getattr(self.scheduler, "reward", self._reward)
                    reward = float(reward_fn(before, observation))
                    update = getattr(self.scheduler, "update", None)
                    if update is not None:
                        update(statistics, proposal.strategy, reward, observation)
                    else:
                        arm.pulls += 1
                        arm.successes += int(observation.successful)
                        arm.elapsed_s += observation_strategy_cost_s(observation)
                        arm.reward_sum += reward
            except Exception:
                # A strategy/scheduler bug must not erase a completed GPU
                # measurement. Recovery can rebuild state from this evidence.
                self.store.record_observation(observation)
                raise
            if self.checkpoint_key:
                observation.metadata['search_checkpoint'] = dict(
                    key=self.checkpoint_key, count=len(history.current), rng=rng.getstate(),
                    history_digest=stable_id([(o.config_id, o.outcome.status) for o in history.current]),
                    excluded_ids=sorted(history.excluded_config_ids),
                    statistics={name: arm.as_dict() for name, arm in statistics.items()})
            self.store.record_observation(observation)
            self.store.record_event(session_id, 'candidate_completed', dict(
                attempt_id=attempt_id, sequence=observation.sequence,
                context_id=observation.context_id, config_id=observation.config_id,
                observation_id=observation.observation_id, status=observation.outcome.status))
            self._log(
                started,
                f"SAVE {proposal.strategy} {observation.config_id} "
                f"{observation.outcome.status}"
                + ("" if observation.outcome.median_ms is None else f" {observation.score * 1000:.3f}us")
                + (" timing_ineligible" if observation.outcome.successful and not observation.outcome.timing_eligible else "")
                + (
                    ""
                    if observation.outcome.error is None
                    else " "
                    + " ".join(observation.outcome.error.splitlines())[:240]
                ),
            )
            # Sticky CUDA launch faults poison all subsequent work in this
            # process. The observation is already durable; abort now so a
            # supervisor can resume past it in a fresh device context.
            if observation.outcome.error is not None:
                raise_if_fatal_device_context_error(observation.outcome.error)
            return observation

        initial_id = self.adapter.config_id(self.adapter.initial_config)
        if (
            initial_id not in history.seen_ids | abandoned_ids
            and resumed_active + evaluated < self.budget.max_trials
            and time.monotonic()-started<search_seconds
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

        # Deterministic, resumable structural coverage precedes exploitation.
        # These trials share the ordinary budget, journal, correctness gates,
        # confirmations and history; they are not unrecorded pilot timings.
        coverage_evaluated = 0
        for proposal in self.coverage_proposals:
            if (resumed_active + evaluated >= self.budget.max_trials
                    or time.monotonic() - started >= search_seconds):
                break
            if self.adapter.config_id(proposal.config) in history.seen_ids:
                continue
            if self.adapter.rejection(proposal.config) is not None:
                history.excluded_config_ids.add(self.adapter.config_id(proposal.config))
                continue
            run(proposal)
            coverage_evaluated += 1

        names = tuple(self.strategies)
        empty_attempts = 0
        while resumed_active + evaluated < self.budget.max_trials:
            if time.monotonic() - started >= search_seconds:
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
            proposal_started = time.monotonic()
            try:
                proposals = strategy.propose(self.adapter, history, rng, 1)
                take_audit = getattr(strategy, 'take_audit', None)
                if take_audit is not None:
                    audit = take_audit()
                    if audit is not None:
                        self.store.record_event(session_id, 'strategy_candidate_ranking',
                                                {'strategy': arm_name, 'trial': trial_index, **audit})
            except Exception as exc:
                proposal_elapsed_s = time.monotonic() - proposal_started
                statistics[arm_name].proposal_calls += 1
                statistics[arm_name].proposal_elapsed_s += proposal_elapsed_s
                statistics[arm_name].proposal_errors += 1
                statistics[arm_name].unavailable_until = trial_index + len(names)
                self.store.record_event(
                    session_id,
                    "strategy_error",
                    {
                        "trial": trial_index,
                        "strategy": arm_name,
                        "proposal_elapsed_s": proposal_elapsed_s,
                        "error": f"{type(exc).__name__}: {exc}"[:4000],
                    },
                )
                empty_attempts += 1
                if empty_attempts >= len(names) * 2:
                    break
                continue
            proposal_elapsed_s = time.monotonic() - proposal_started
            statistics[arm_name].proposal_calls += 1
            statistics[arm_name].proposal_elapsed_s += proposal_elapsed_s
            proposals = [
                proposal
                for proposal in proposals
                if self.adapter.config_id(proposal.config)
                not in history.seen_ids | abandoned_ids
            ]
            legal=[]
            for proposal in proposals:
                rejection=self.adapter.rejection(proposal.config)
                if rejection is None:legal.append(proposal)
                else:
                    cid=self.adapter.config_id(proposal.config)
                    history.excluded_config_ids.add(cid)
                    self.store.record_event(session_id,'static_rejected',
                        {'config_id':cid,'reason':rejection[1],'strategy':arm_name})
            proposals=legal
            if not proposals:
                statistics[arm_name].proposal_empty += 1
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
            if time.monotonic()-started>=search_seconds:
                break
            proposal.strategy = arm_name
            proposal.metadata = {
                **proposal.metadata,
                "proposal_elapsed_s": proposal_elapsed_s,
            }
            observation = run(proposal)
            strategy.observe(observation)

        best = history.best
        if best is None:
            self.store.finish_session(session_id, {"status": "failed", "reason": "no successful trials"})
            initial_id = self.adapter.config_id(self.adapter.initial_config)
            checked_seed = next((item for item in history.successful
                if item.config_id == initial_id and item.session_id == session_id
                and item.outcome.max_abs_error is not None
                and math.isfinite(item.outcome.max_abs_error)), None)
            raise NoEligibleTimingError(None if checked_seed is None else checked_seed.config)
        decision = None
        search_elapsed = time.monotonic() - started
        selected_score = best.score
        if self.finalist_rounds:
            from .paired_promotion import promote
            decision = promote(self.adapter, self.store, history, session_id,
                rounds=self.finalist_rounds, margin=self.finalist_margin,
                deadline=started+self.budget.time_budget_s, seed=self.seed)
            best = next(o for o in history.successful if o.config_id == decision['config_id'])
            selected_score = decision['median_ms']
        elapsed = time.monotonic() - started
        status = (
            "budget_exhausted"
            if resumed_active + evaluated >= self.budget.max_trials
            or search_elapsed >= search_seconds
            else "complete"
        )
        self.store.finish_session(
            session_id,
            {
                "status": status,
                "best_config_id": best.config_id,
                "best_median_ms": selected_score,
                "winner_decision": decision,
                "search_elapsed_s": search_elapsed,
                "finalist_evaluations": (0 if decision is None or decision.get('reused') or
                    decision.get('confirmation') is None else decision['confirmation']['evaluations']),
                "evaluated_trials": evaluated,
                "elapsed_s": elapsed,
                "resumed_trials": resumed_active,
                "stop_reason": ('trial_budget' if resumed_active+evaluated>=self.budget.max_trials
                                else 'confirmation_reserve' if search_elapsed>=search_seconds and self.finalist_reserve_s
                                else 'time_budget' if elapsed>=self.budget.time_budget_s else 'search_exhausted'),
                "confirmation_evaluations":sum(len(o.metadata.get('confirmation',{}).get('outcomes',[]))
                    for o in history.current if o.session_id==session_id),
                "proposal_elapsed_s":sum(a.proposal_elapsed_s for a in statistics.values()),
            },
        )
        return ComposableTuningResult(
            config=best.config,
            median_ms=selected_score,
            session_id=session_id,
            context_id=self.adapter.context.identifier,
            evaluated_trials=evaluated,
            elapsed_s=elapsed,
            strategy_trials={**{name: arm.pulls for name, arm in statistics.items()},
                             **({"coverage": coverage_evaluated} if self.coverage_proposals else {})},
            store_path=None if self.store.path is None else str(self.store.path),
            winner_decision=decision,
            tuning_evidence=(None if self.effort_contract is None else dict(
                schema=1, contract=self.effort_contract, attempted_trials=resumed_active+evaluated,
                context_id=self.adapter.context.identifier, session_id=session_id,
                backward_coordinate_revision=self.adapter.context.tags.get('runtime_backward_coordinate_revision'),
                finalist_rounds=(len(decision['confirmation']['rounds']) if decision and
                    decision.get('confirmation') and decision['confirmation']['status']=='complete' else 0),
                finalist_margin=self.finalist_margin, winner_decision=decision,
                promotion_revision=decision.get('promotion_revision') if decision else None)),
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
