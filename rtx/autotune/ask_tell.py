"""Serializable asynchronous ask/tell boundary for tuning workers."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import math
import random
import time
from typing import Generic, Mapping, Sequence, TypeVar
import uuid

from .bandit import ArmStatistics
from .bandit import observation_strategy_cost_s
from .core import (
    ComposableTuningResult,
    KernelAdapter,
    Observation,
    Proposal,
    SearchHistory,
    observation_from_dict,
    stable_id,
    utc_now,
)
from .outcomes import TrialOutcome, raise_if_fatal_device_context_error
from .orchestrator import StrategyScheduler
from .store import TuningStore
from .strategies import SearchStrategy


ConfigT = TypeVar("ConfigT")


def _tuple_tree(value):
    if isinstance(value, list):
        return tuple(_tuple_tree(item) for item in value)
    return value


@dataclass(slots=True)
class TrialRequest(Generic[ConfigT]):
    """Leaseable work item; ``as_dict`` is safe to send to another process."""

    request_id: str
    sequence: int
    context_id: str
    family: str
    kernel_revision: int
    config_id: str
    config: ConfigT
    serialized_config: Mapping[str, object]
    strategy: str
    fidelity: float
    issued_at: str
    lease_expires_at_monotonic: float
    parent_config_id: str | None = None
    coordinate: str | None = None
    coordinate_value: object = None
    metadata: dict[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "type": "rtx_autotune_trial_request",
            "request_id": self.request_id,
            "sequence": self.sequence,
            "context_id": self.context_id,
            "family": self.family,
            "kernel_revision": self.kernel_revision,
            "config_id": self.config_id,
            "config": dict(self.serialized_config),
            "strategy": self.strategy,
            "fidelity": self.fidelity,
            "issued_at": self.issued_at,
            # A monotonic deadline is local-only. Distributed transports should
            # enforce their own server-side lease from issued_at + lease_s.
            "lease_s": self.metadata.get("lease_s"),
            "parent_config_id": self.parent_config_id,
            "coordinate": self.coordinate,
            "coordinate_value": self.coordinate_value,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(
        cls, adapter: KernelAdapter[ConfigT], value: Mapping[str, object]
    ) -> "TrialRequest[ConfigT]":
        serialized = dict(value["config"])  # type: ignore[arg-type]
        metadata = dict(value.get("metadata", {}))  # type: ignore[arg-type]
        lease_s = float(value.get("lease_s") or metadata.get("lease_s") or 300.0)
        return cls(
            request_id=str(value["request_id"]),
            sequence=int(value["sequence"]),
            context_id=str(value["context_id"]),
            family=str(value["family"]),
            kernel_revision=int(value["kernel_revision"]),
            config_id=str(value["config_id"]),
            config=adapter.deserialize(serialized),
            serialized_config=serialized,
            strategy=str(value["strategy"]),
            fidelity=float(value.get("fidelity", 1.0)),
            issued_at=str(value["issued_at"]),
            lease_expires_at_monotonic=time.monotonic() + lease_s,
            parent_config_id=(
                None
                if value.get("parent_config_id") is None
                else str(value["parent_config_id"])
            ),
            coordinate=(
                None if value.get("coordinate") is None else str(value["coordinate"])
            ),
            coordinate_value=value.get("coordinate_value"),
            metadata=metadata,
        )


@dataclass(frozen=True, slots=True)
class TrialResponse:
    """Worker response independent of optimizer and storage implementation."""

    request_id: str
    outcome: TrialOutcome
    started_at: str
    finished_at: str
    elapsed_s: float
    worker: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "type": "rtx_autotune_trial_response",
            "request_id": self.request_id,
            "outcome": self.outcome.as_dict(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": self.elapsed_s,
            "worker": dict(self.worker),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "TrialResponse":
        return cls(
            request_id=str(value["request_id"]),
            outcome=TrialOutcome.from_dict(value["outcome"]),  # type: ignore[arg-type]
            started_at=str(value["started_at"]),
            finished_at=str(value["finished_at"]),
            elapsed_s=float(value["elapsed_s"]),
            worker=dict(value.get("worker", {})),  # type: ignore[arg-type]
        )


@dataclass(slots=True)
class LocalTrialWorker(Generic[ConfigT]):
    """Reference worker used locally; remote workers implement the same wire API."""

    adapter: KernelAdapter[ConfigT]
    metadata: Mapping[str, object] = field(default_factory=dict)

    def evaluate(self, request: TrialRequest[ConfigT]) -> TrialResponse:
        if request.context_id != self.adapter.context.identifier:
            raise ValueError("trial request belongs to a different task context")
        if request.config_id != self.adapter.config_id(request.config):
            raise ValueError("trial request config ID does not match its payload")
        started_at = utc_now()
        started = time.monotonic()
        rejection = self.adapter.rejection(request.config)
        if rejection is not None:
            status, reason = rejection
            outcome = TrialOutcome(status, error=reason)  # type: ignore[arg-type]
        else:
            try:
                evaluate_at_fidelity = getattr(
                    self.adapter, "evaluate_at_fidelity", None
                )
                outcome = (
                    self.adapter.evaluate(request.config)
                    if evaluate_at_fidelity is None
                    else evaluate_at_fidelity(request.config, request.fidelity)
                )
            except Exception as exc:
                outcome = TrialOutcome(
                    "runtime_error", error=f"{type(exc).__name__}: {exc}"[:4000]
                )
        return TrialResponse(
            request.request_id,
            outcome,
            started_at,
            utc_now(),
            time.monotonic() - started,
            self.metadata,
        )


class AskTellSession(Generic[ConfigT]):
    """Optimizer-side state machine supporting out-of-order worker responses."""

    def __init__(
        self,
        adapter: KernelAdapter[ConfigT],
        strategies: Sequence[SearchStrategy[ConfigT]],
        scheduler: StrategyScheduler,
        *,
        seed: int = 0,
        observations: Sequence[Observation[ConfigT]] = (),
        session_id: str = "ask-tell",
        excluded_config_ids: Sequence[str] = (),
    ) -> None:
        if not strategies:
            raise ValueError("at least one strategy is required")
        names = [strategy.name for strategy in strategies]
        if len(names) != len(set(names)):
            raise ValueError("strategy names must be unique")
        self.adapter = adapter
        self.strategies = {strategy.name: strategy for strategy in strategies}
        self.scheduler = scheduler
        self.rng = random.Random(seed)
        self.seed = seed
        self.session_id = session_id
        self.history = SearchHistory(
            list(observations), adapter.context.identifier
        )
        self.statistics = {name: ArmStatistics() for name in names}
        self.pending: dict[str, TrialRequest[ConfigT]] = {}
        self.excluded_config_ids = set(excluded_config_ids)
        self.history.excluded_config_ids.update(self.excluded_config_ids)
        self._next_sequence = 1 + max(
            (observation.sequence for observation in observations), default=-1
        )
        initialize = getattr(scheduler, "initialize", None)
        if initialize is not None:
            initialize(
                self.statistics,
                self.history,
                self.adapter.context.features(),
            )

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.strategies)

    def _shadow_statistics(self) -> dict[str, ArmStatistics]:
        """Count reservations so concurrent asks spread across empty arms."""

        shadow = {name: replace(value) for name, value in self.statistics.items()}
        for request in self.pending.values():
            if request.strategy not in shadow:
                continue
            arm = shadow[request.strategy]
            arm.pulls += 1
            arm.effective_pulls += 1.0
        return shadow

    def _request(
        self,
        proposal: Proposal[ConfigT],
        *,
        lease_s: float,
        fidelity: float,
    ) -> TrialRequest[ConfigT]:
        sequence = self._next_sequence
        self._next_sequence += 1
        config_id = self.adapter.config_id(proposal.config)
        request_id = uuid.uuid4().hex
        metadata = dict(proposal.metadata)
        metadata["lease_s"] = lease_s
        metadata["ask_sequence"] = sequence
        request = TrialRequest(
            request_id=request_id,
            sequence=sequence,
            context_id=self.adapter.context.identifier,
            family=self.adapter.context.family,
            kernel_revision=self.adapter.context.kernel_revision,
            config_id=config_id,
            config=proposal.config,
            serialized_config=self.adapter.serialize(proposal.config),
            strategy=proposal.strategy,
            fidelity=fidelity,
            issued_at=utc_now(),
            lease_expires_at_monotonic=time.monotonic() + lease_s,
            parent_config_id=proposal.parent_config_id,
            coordinate=proposal.coordinate,
            coordinate_value=proposal.coordinate_value,
            metadata=metadata,
        )
        self.pending[request_id] = request
        return request

    def reclaim_expired(self, *, now: float | None = None) -> list[TrialRequest[ConfigT]]:
        current = time.monotonic() if now is None else now
        expired = [
            request
            for request in self.pending.values()
            if request.lease_expires_at_monotonic <= current
        ]
        for request in expired:
            self.pending.pop(request.request_id, None)
        return expired

    def promote(
        self,
        config_id: str,
        fidelity: float,
        *,
        lease_s: float = 300.0,
    ) -> TrialRequest[ConfigT]:
        """Re-evaluate a completed configuration at a higher fidelity."""

        if not 0 < fidelity <= 1:
            raise ValueError("trial fidelity must be in (0, 1]")
        if lease_s <= 0:
            raise ValueError("trial leases must be positive")
        candidates = [
            observation
            for observation in self.history.current
            if observation.config_id == config_id
        ]
        if not candidates:
            raise KeyError(f"cannot promote unknown configuration {config_id}")
        source = max(
            candidates,
            key=lambda observation: float(
                observation.metadata.get(
                    "fidelity",
                    observation.outcome.metadata.get("completed_fidelity", 1.0),
                )
            ),
        )
        source_fidelity = float(
            source.metadata.get(
                "fidelity",
                source.outcome.metadata.get("completed_fidelity", 1.0),
            )
        )
        if fidelity <= source_fidelity:
            raise ValueError(
                f"promotion fidelity {fidelity} must exceed {source_fidelity}"
            )
        if any(
            request.config_id == config_id and request.fidelity == fidelity
            for request in self.pending.values()
        ):
            raise ValueError("that configuration/fidelity pair is already pending")
        return self._request(
            Proposal(
                source.config,
                "promotion",
                parent_config_id=source.config_id,
                metadata={
                    "promoted_from_observation": source.observation_id,
                    "promoted_from_fidelity": source_fidelity,
                },
            ),
            lease_s=lease_s,
            fidelity=fidelity,
        )

    def ask(
        self,
        count: int = 1,
        *,
        lease_s: float = 300.0,
        fidelity: float = 1.0,
    ) -> list[TrialRequest[ConfigT]]:
        if count <= 0:
            return []
        if lease_s <= 0:
            raise ValueError("trial leases must be positive")
        if not 0 < fidelity <= 1:
            raise ValueError("trial fidelity must be in (0, 1]")
        self.reclaim_expired()
        pending_ids = {request.config_id for request in self.pending.values()}
        initial_id = self.adapter.config_id(self.adapter.initial_config)
        seen_or_excluded = (
            self.history.seen_ids | pending_ids | self.excluded_config_ids
        )
        if initial_id not in seen_or_excluded:
            # Establish a valid baseline before strategies make incumbent-based
            # decisions. Subsequent asks can fan out across workers.
            return [
                self._request(
                    Proposal(
                        self.adapter.initial_config,
                        "initial",
                        metadata={"proposal_elapsed_s": 0.0},
                    ),
                    lease_s=lease_s,
                    fidelity=fidelity,
                )
            ]

        requests: list[TrialRequest[ConfigT]] = []
        empty_attempts = 0
        while len(requests) < count:
            trial_index = len(self.history.current) + len(self.pending)
            arm_name = self.scheduler.select(
                self.names, self._shadow_statistics(), trial_index
            )
            if arm_name is None:
                break
            strategy = self.strategies[arm_name]
            proposal_started = time.monotonic()
            proposals = strategy.propose(self.adapter, self.history, self.rng, 1)
            proposal_elapsed_s = time.monotonic() - proposal_started
            arm = self.statistics[arm_name]
            arm.proposal_calls += 1
            arm.proposal_elapsed_s += proposal_elapsed_s
            if not proposals:
                arm.proposal_empty += 1
            for item in proposals:
                item.metadata = {
                    **item.metadata,
                    "proposal_elapsed_s": proposal_elapsed_s / max(1, len(proposals)),
                }
            pending_ids = {request.config_id for request in self.pending.values()}
            proposal = next(
                (
                    item
                    for item in proposals
                    if self.adapter.config_id(item.config)
                    not in self.history.seen_ids
                    | pending_ids
                    | self.excluded_config_ids
                ),
                None,
            )
            if proposal is None:
                empty_attempts += 1
                self.statistics[arm_name].unavailable_until = (
                    trial_index + len(self.names)
                )
                if empty_attempts >= len(self.names) * 2:
                    break
                continue
            empty_attempts = 0
            proposal.strategy = arm_name
            requests.append(
                self._request(proposal, lease_s=lease_s, fidelity=fidelity)
            )
        return requests

    @staticmethod
    def _default_reward(before: float, observation: Observation[ConfigT]) -> float:
        if not observation.successful:
            return -0.01
        if not math.isfinite(before):
            return 0.05
        return 0.002 + max(0.0, math.log(before / observation.score))

    def tell(self, response: TrialResponse) -> Observation[ConfigT]:
        try:
            request = self.pending.pop(response.request_id)
        except KeyError as exc:
            raise KeyError(f"unknown or expired trial request {response.request_id}") from exc
        before = math.inf if self.history.best is None else self.history.best.score
        observation = Observation(
            observation_id=stable_id(
                {
                    "request_id": request.request_id,
                    "context_id": request.context_id,
                    "config_id": request.config_id,
                },
                32,
            ),
            session_id=self.session_id,
            sequence=request.sequence,
            context_id=request.context_id,
            family=request.family,
            kernel_revision=request.kernel_revision,
            config_id=request.config_id,
            config=request.config,
            serialized_config=request.serialized_config,
            features=self.adapter.features(request.config),
            strategy=request.strategy,
            outcome=response.outcome,
            started_at=response.started_at,
            finished_at=response.finished_at,
            elapsed_s=response.elapsed_s,
            parent_config_id=request.parent_config_id,
            coordinate=request.coordinate,
            coordinate_value=request.coordinate_value,
            metadata={
                **request.metadata,
                "request_id": request.request_id,
                "fidelity": request.fidelity,
                "worker": dict(response.worker),
            },
        )
        self.history.observations.append(observation)
        if request.strategy in self.strategies:
            self.strategies[request.strategy].observe(observation)
            reward_fn = getattr(self.scheduler, "reward", self._default_reward)
            reward = float(reward_fn(before, observation))
            update = getattr(self.scheduler, "update", None)
            if update is not None:
                update(self.statistics, request.strategy, reward, observation)
            else:
                arm = self.statistics[request.strategy]
                arm.pulls += 1
                arm.successes += int(observation.successful)
                arm.elapsed_s += observation_strategy_cost_s(observation)
                arm.reward_sum += reward
        return observation

    def state_dict(self) -> dict[str, object]:
        snapshot = getattr(self.scheduler, "snapshot", None)
        trial_index = len(self.history.current) + len(self.pending)
        return {
            "schema_version": 1,
            "type": "rtx_autotune_ask_tell_session",
            "context": self.adapter.context.as_dict(),
            "seed": self.seed,
            "session_id": self.session_id,
            "rng_state": self.rng.getstate(),
            "next_sequence": self._next_sequence,
            "scheduler": (
                {"name": self.scheduler.name}
                if snapshot is None
                else snapshot(self.names, self.statistics, trial_index)
            ),
            "statistics": {
                name: asdict(value) for name, value in self.statistics.items()
            },
            "pending": [request.as_dict() for request in self.pending.values()],
            "excluded_config_ids": sorted(self.excluded_config_ids),
            "observations": [
                observation.as_dict() for observation in self.history.observations
            ],
        }

    @classmethod
    def from_state_dict(
        cls,
        adapter: KernelAdapter[ConfigT],
        strategies: Sequence[SearchStrategy[ConfigT]],
        scheduler: StrategyScheduler,
        state: Mapping[str, object],
    ) -> "AskTellSession[ConfigT]":
        if state.get("type") != "rtx_autotune_ask_tell_session":
            raise ValueError("unsupported ask/tell session state")
        if stable_id(state["context"]) != adapter.context.identifier:
            raise ValueError("ask/tell state belongs to a different task context")
        observations = [
            observation_from_dict(adapter, value)
            for value in state.get("observations", [])  # type: ignore[union-attr]
        ]
        session = cls(
            adapter,
            strategies,
            scheduler,
            seed=int(state.get("seed", 0)),
            observations=observations,
            session_id=str(state.get("session_id", "ask-tell")),
            excluded_config_ids=tuple(
                str(value) for value in state.get("excluded_config_ids", [])
            ),
        )
        if state.get("rng_state") is not None:
            session.rng.setstate(_tuple_tree(state["rng_state"]))
        session._next_sequence = int(
            state.get("next_sequence", session._next_sequence)
        )
        restored_statistics = dict(state.get("statistics", {}))
        for name in session.statistics:
            if name in restored_statistics:
                session.statistics[name] = ArmStatistics(
                    **dict(restored_statistics[name])  # type: ignore[arg-type]
                )
        pending = [
            TrialRequest.from_dict(adapter, value)
            for value in state.get("pending", [])  # type: ignore[union-attr]
        ]
        session.pending = {request.request_id: request for request in pending}
        return session


class DurableLocalAskTellRunner(Generic[ConfigT]):
    """Run the portable optimizer locally with campaign-grade durability.

    This is deliberately a thin deployment wrapper: optimization remains in
    :class:`AskTellSession`, execution remains in ``LocalTrialWorker``, and all
    accepted responses are written through the ordinary ``TuningStore``. The
    same session can later be driven by remote workers without changing its
    optimizer state machine.
    """

    def __init__(
        self,
        adapter: KernelAdapter[ConfigT],
        store: TuningStore[ConfigT],
        strategies: Sequence[SearchStrategy[ConfigT]],
        scheduler: StrategyScheduler,
        *,
        max_trials: int = 256,
        time_budget_s: float = 1800.0,
        seed: int = 0,
        resume: bool = True,
        max_trials_includes_resumed: bool = True,
        transfer_history: bool = True,
        lease_s: float = 300.0,
        worker_metadata: Mapping[str, object] | None = None,
        progress=None,
    ) -> None:
        if max_trials <= 0 or time_budget_s <= 0 or lease_s <= 0:
            raise ValueError("ask/tell budgets and leases must be positive")
        self.adapter = adapter
        self.store = store
        self.strategies = tuple(strategies)
        self.scheduler = scheduler
        self.max_trials = int(max_trials)
        self.time_budget_s = float(time_budget_s)
        self.seed = int(seed)
        self.resume = bool(resume)
        self.max_trials_includes_resumed = bool(max_trials_includes_resumed)
        self.transfer_history = bool(transfer_history)
        self.lease_s = float(lease_s)
        self.worker_metadata = dict(worker_metadata or {})
        self.progress = progress

    def _restored(self) -> list[Observation[ConfigT]]:
        if not self.resume:
            return []
        source_context = None if self.transfer_history else self.adapter.context
        restored: list[Observation[ConfigT]] = []
        for record in self.store.records(source_context):
            if (
                record.get("family") != self.adapter.context.family
                or int(record.get("kernel_revision", -1))
                != self.adapter.context.kernel_revision
            ):
                continue
            try:
                restored.append(observation_from_dict(self.adapter, record))
            except (KeyError, TypeError, ValueError):
                continue
        return restored

    def tune(self) -> ComposableTuningResult[ConfigT]:
        started = time.monotonic()
        restored = self._restored()
        incomplete = getattr(self.store, "incomplete_candidates", None)
        abandoned = (
            []
            if incomplete is None
            else list(incomplete(self.adapter.context.identifier))
        )
        abandoned_ids = tuple(
            sorted(
                {
                    str(item.get("config_id", "")) for item in abandoned
                }
                - {""}
            )
        )
        active_restored = sum(
            item.context_id == self.adapter.context.identifier for item in restored
        )
        consumed = active_restored if self.max_trials_includes_resumed else 0
        session_id = self.store.start_session(
            self.adapter.context,
            {
                "engine": "durable_local_ask_tell",
                "scheduler": self.scheduler.name,
                "strategies": [strategy.name for strategy in self.strategies],
                "budget": {
                    "max_trials": self.max_trials,
                    "time_budget_s": self.time_budget_s,
                },
                "seed": self.seed,
                "resumed_observations": len(restored),
                "resumed_active_budget": consumed,
                "max_trials_includes_resumed": self.max_trials_includes_resumed,
            },
        )
        optimizer = AskTellSession(
            self.adapter,
            self.strategies,
            self.scheduler,
            seed=self.seed,
            observations=restored,
            session_id=session_id,
            excluded_config_ids=abandoned_ids,
        )
        if abandoned:
            self.store.record_event(
                session_id,
                "abandoned_candidates_recovered",
                {
                    "context_id": self.adapter.context.identifier,
                    "count": len(abandoned),
                    "config_ids": list(abandoned_ids),
                    "attempt_ids": sorted(
                        str(item.get("attempt_id", "")) for item in abandoned
                    ),
                },
            )
        worker = LocalTrialWorker(
            self.adapter,
            {"execution": "local", **self.worker_metadata},
        )
        evaluated = 0
        status = "complete"
        while consumed + evaluated < self.max_trials:
            if time.monotonic() - started >= self.time_budget_s:
                status = "budget_exhausted"
                break
            requests = optimizer.ask(1, lease_s=self.lease_s)
            if not requests:
                break
            request = requests[0]
            self.store.record_event(
                session_id,
                "trial_issued",
                {
                    "sequence": request.sequence,
                    "request_id": request.request_id,
                    "attempt_id": request.request_id,
                    "context_id": request.context_id,
                    "family": request.family,
                    "kernel_revision": request.kernel_revision,
                    "config_id": request.config_id,
                    "config": dict(request.serialized_config),
                    "strategy": request.strategy,
                    "fidelity": request.fidelity,
                },
            )
            response = worker.evaluate(request)
            observation = optimizer.tell(response)
            observation.metadata = {
                **observation.metadata,
                "attempt_id": request.request_id,
            }
            self.store.record_observation(observation)
            self.store.record_event(
                session_id,
                "trial_completed",
                {
                    "attempt_id": request.request_id,
                    "sequence": observation.sequence,
                    "context_id": observation.context_id,
                    "config_id": observation.config_id,
                    "observation_id": observation.observation_id,
                    "status": observation.outcome.status,
                },
            )
            evaluated += 1
            if self.progress is not None:
                suffix = (
                    ""
                    if observation.outcome.median_ms is None
                    else f" {observation.score * 1000:.3f}us"
                )
                self.progress(
                    f"[{time.monotonic() - started:8.2f}s] SAVE "
                    f"{observation.strategy} {observation.config_id} "
                    f"{observation.outcome.status}{suffix}"
                )
            if observation.outcome.error is not None:
                raise_if_fatal_device_context_error(observation.outcome.error)
            if request.strategy == "initial" and not observation.successful:
                self.store.finish_session(
                    session_id,
                    {
                        "status": "failed",
                        "reason": "initial configuration failed",
                    },
                )
                raise RuntimeError(
                    "initial configuration failed: "
                    f"{observation.outcome.status}: {observation.outcome.error}"
                )
        if consumed + evaluated >= self.max_trials:
            status = "budget_exhausted"
        best = optimizer.history.best
        if best is None:
            self.store.finish_session(
                session_id, {"status": "failed", "reason": "no successful trials"}
            )
            raise RuntimeError("autotuning produced no successful configurations")
        elapsed = time.monotonic() - started
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
            strategy_trials={
                name: arm.pulls for name, arm in optimizer.statistics.items()
            },
            store_path=None if self.store.path is None else str(self.store.path),
        )


__all__ = [
    "AskTellSession",
    "DurableLocalAskTellRunner",
    "LocalTrialWorker",
    "TrialRequest",
    "TrialResponse",
]
