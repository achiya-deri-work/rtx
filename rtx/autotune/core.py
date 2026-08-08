"""Generic contracts and records for composable kernel autotuning."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import random
import time
from typing import Callable, Generic, Iterable, Mapping, Protocol, Sequence, TypeVar

from .legacy import TrialOutcome


ConfigT = TypeVar("ConfigT")
FeatureMap = dict[str, float]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def stable_id(value: object, length: int = 24) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()[:length]


def _plain(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def flatten_features(value: object, prefix: str = "") -> FeatureMap:
    """Create a stable sparse numeric/one-hot representation of nested data."""

    result: FeatureMap = {}

    def visit(item: object, path: str) -> None:
        item = _plain(item)
        if isinstance(item, Mapping):
            for key in sorted(item):
                visit(item[key], f"{path}.{key}" if path else str(key))
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        elif isinstance(item, bool):
            result[path] = float(item)
        elif isinstance(item, (int, float)):
            numeric = float(item)
            result[path] = numeric if math.isfinite(numeric) else 0.0
        elif item is None:
            result[f"{path}=<none>"] = 1.0
        else:
            result[f"{path}={item}"] = 1.0

    visit(value, prefix)
    return result


@dataclass(frozen=True, slots=True)
class KernelContext:
    """Everything that makes one tuning result transferable or nontransferable."""

    family: str
    kernel_revision: int
    workload: Mapping[str, object]
    device: Mapping[str, object] = field(default_factory=dict)
    environment: Mapping[str, object] = field(default_factory=dict)
    regime: str = "hot"
    tags: Mapping[str, object] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "family": self.family,
            "kernel_revision": self.kernel_revision,
            "workload": _plain(self.workload),
            "device": _plain(self.device),
            "environment": _plain(self.environment),
            "regime": self.regime,
            "tags": _plain(self.tags),
        }

    @property
    def identifier(self) -> str:
        return stable_id(self.as_dict())

    def features(self) -> FeatureMap:
        return flatten_features(self.as_dict(), "context")


@dataclass(slots=True)
class Proposal(Generic[ConfigT]):
    config: ConfigT
    strategy: str
    parent_config_id: str | None = None
    coordinate: str | None = None
    coordinate_value: object = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class Observation(Generic[ConfigT]):
    observation_id: str
    session_id: str
    sequence: int
    context_id: str
    family: str
    kernel_revision: int
    config_id: str
    config: ConfigT
    serialized_config: Mapping[str, object]
    features: FeatureMap
    strategy: str
    outcome: TrialOutcome
    started_at: str
    finished_at: str
    elapsed_s: float
    parent_config_id: str | None = None
    coordinate: str | None = None
    coordinate_value: object = None
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def successful(self) -> bool:
        return self.outcome.successful

    @property
    def score(self) -> float:
        return math.inf if self.outcome.median_ms is None else self.outcome.median_ms

    def as_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "observation_id": self.observation_id,
            "session_id": self.session_id,
            "sequence": self.sequence,
            "context_id": self.context_id,
            "family": self.family,
            "kernel_revision": self.kernel_revision,
            "config_id": self.config_id,
            "config": _plain(self.serialized_config),
            "features": self.features,
            "strategy": self.strategy,
            "outcome": self.outcome.as_dict(),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": self.elapsed_s,
            "parent_config_id": self.parent_config_id,
            "coordinate": self.coordinate,
            "coordinate_value": _plain(self.coordinate_value),
            "metadata": _plain(self.metadata),
        }


class KernelAdapter(Protocol[ConfigT]):
    context: KernelContext
    initial_config: ConfigT

    def config_id(self, config: ConfigT) -> str: ...

    def serialize(self, config: ConfigT) -> Mapping[str, object]: ...

    def deserialize(self, value: Mapping[str, object]) -> ConfigT: ...

    def features(self, config: ConfigT) -> FeatureMap: ...

    def rejection(self, config: ConfigT) -> tuple[str, str] | None: ...

    def evaluate(self, config: ConfigT) -> TrialOutcome: ...

    def coordinates(self) -> Sequence[str]: ...

    def neighbors(self, config: ConfigT) -> Iterable[tuple[str, object, ConfigT]]: ...

    def sample(self, rng: random.Random, count: int, seeds: Sequence[ConfigT]) -> list[ConfigT]: ...


@dataclass(slots=True)
class DiscreteKernelAdapter(Generic[ConfigT]):
    """Adapter for coupled, discrete kernel configuration spaces."""

    context: KernelContext
    initial_config: ConfigT
    axes: Mapping[str, Sequence[object]]
    config_id_fn: Callable[[ConfigT], str]
    serialize_fn: Callable[[ConfigT], Mapping[str, object]]
    deserialize_fn: Callable[[Mapping[str, object]], ConfigT]
    update_fn: Callable[[ConfigT, str, object], ConfigT]
    evaluator: Callable[[ConfigT], TrialOutcome]
    rejection_fn: Callable[[ConfigT], tuple[str, str] | None]
    extra_features_fn: Callable[[ConfigT], Mapping[str, float]] | None = None

    def __post_init__(self) -> None:
        if not self.axes:
            raise ValueError("a discrete kernel adapter requires at least one axis")
        empty = [name for name, values in self.axes.items() if not values]
        if empty:
            raise ValueError(f"tuning axes have no values: {empty}")

    def config_id(self, config: ConfigT) -> str:
        return self.config_id_fn(config)

    def serialize(self, config: ConfigT) -> Mapping[str, object]:
        return self.serialize_fn(config)

    def deserialize(self, value: Mapping[str, object]) -> ConfigT:
        return self.deserialize_fn(value)

    def features(self, config: ConfigT) -> FeatureMap:
        values = self.context.features()
        values.update(flatten_features(self.serialize(config), "config"))
        if self.extra_features_fn is not None:
            values.update(
                {f"derived.{key}": float(value) for key, value in self.extra_features_fn(config).items()}
            )
        return values

    def rejection(self, config: ConfigT) -> tuple[str, str] | None:
        return self.rejection_fn(config)

    def evaluate(self, config: ConfigT) -> TrialOutcome:
        return self.evaluator(config)

    def coordinates(self) -> Sequence[str]:
        return tuple(self.axes)

    def neighbors(self, config: ConfigT) -> Iterable[tuple[str, object, ConfigT]]:
        seen: set[str] = set()
        for coordinate, values in self.axes.items():
            for value in values:
                candidate = self.update_fn(config, coordinate, value)
                identifier = self.config_id(candidate)
                if identifier == self.config_id(config) or identifier in seen:
                    continue
                seen.add(identifier)
                yield coordinate, value, candidate

    def sample(
        self,
        rng: random.Random,
        count: int,
        seeds: Sequence[ConfigT],
    ) -> list[ConfigT]:
        """Sample legal-ish points by random walks, preserving coupled invariants."""

        bases = list(seeds) or [self.initial_config]
        result: list[ConfigT] = []
        seen: set[str] = set()
        coordinates = tuple(self.axes)
        attempts = max(32, count * 20)
        for _ in range(attempts):
            if len(result) >= count:
                break
            candidate = rng.choice(bases)
            steps = rng.randint(1, max(1, min(8, len(coordinates))))
            for _step in range(steps):
                coordinate = rng.choice(coordinates)
                proposal = self.update_fn(
                    candidate, coordinate, rng.choice(tuple(self.axes[coordinate]))
                )
                # Coupled hardware spaces are overwhelmingly sparse. Preserve
                # legality after every random-walk step rather than applying a
                # sequence of independent mutations and hoping the final point
                # happens to repair every broken invariant.
                if self.rejection(proposal) is None:
                    candidate = proposal
            identifier = self.config_id(candidate)
            if identifier in seen or self.rejection(candidate) is not None:
                continue
            seen.add(identifier)
            result.append(candidate)
        return result


@dataclass(slots=True)
class SearchHistory(Generic[ConfigT]):
    observations: list[Observation[ConfigT]] = field(default_factory=list)
    active_context_id: str | None = None

    @property
    def current(self) -> list[Observation[ConfigT]]:
        if self.active_context_id is None:
            return self.observations
        return [
            item for item in self.observations if item.context_id == self.active_context_id
        ]

    @property
    def successful(self) -> list[Observation[ConfigT]]:
        return [item for item in self.current if item.successful]

    @property
    def training_successful(self) -> list[Observation[ConfigT]]:
        return [item for item in self.observations if item.successful]

    @property
    def best(self) -> Observation[ConfigT] | None:
        successes = self.successful
        return min(successes, key=lambda item: item.score) if successes else None

    @property
    def seen_ids(self) -> set[str]:
        return {item.config_id for item in self.current}


@dataclass(frozen=True, slots=True)
class TuningBudget:
    max_trials: int = 256
    time_budget_s: float = 1800.0

    def __post_init__(self) -> None:
        if self.max_trials <= 0 or self.time_budget_s <= 0:
            raise ValueError("tuning budgets must be positive")


@dataclass(slots=True)
class ComposableTuningResult(Generic[ConfigT]):
    config: ConfigT
    median_ms: float
    session_id: str
    context_id: str
    evaluated_trials: int
    elapsed_s: float
    strategy_trials: dict[str, int]
    store_path: str | None


def evaluate_proposal(
    adapter: KernelAdapter[ConfigT],
    proposal: Proposal[ConfigT],
    *,
    session_id: str,
    sequence: int,
) -> Observation[ConfigT]:
    started_wall = utc_now()
    started = time.monotonic()
    rejection = adapter.rejection(proposal.config)
    if rejection is not None:
        status, reason = rejection
        outcome = TrialOutcome(status, error=reason)  # type: ignore[arg-type]
    else:
        try:
            outcome = adapter.evaluate(proposal.config)
        except Exception as exc:
            outcome = TrialOutcome(
                "runtime_error", error=f"{type(exc).__name__}: {exc}"[:4000]
            )
    elapsed = time.monotonic() - started
    config_id = adapter.config_id(proposal.config)
    payload = {
        "session": session_id,
        "sequence": sequence,
        "config": config_id,
        "strategy": proposal.strategy,
        "started_at": started_wall,
    }
    return Observation(
        observation_id=stable_id(payload, 32),
        session_id=session_id,
        sequence=sequence,
        context_id=adapter.context.identifier,
        family=adapter.context.family,
        kernel_revision=adapter.context.kernel_revision,
        config_id=config_id,
        config=proposal.config,
        serialized_config=adapter.serialize(proposal.config),
        features=adapter.features(proposal.config),
        strategy=proposal.strategy,
        outcome=outcome,
        started_at=started_wall,
        finished_at=utc_now(),
        elapsed_s=elapsed,
        parent_config_id=proposal.parent_config_id,
        coordinate=proposal.coordinate,
        coordinate_value=proposal.coordinate_value,
        metadata=proposal.metadata,
    )


__all__ = [
    "ComposableTuningResult",
    "ConfigT",
    "DiscreteKernelAdapter",
    "FeatureMap",
    "KernelAdapter",
    "KernelContext",
    "Observation",
    "Proposal",
    "SearchHistory",
    "TuningBudget",
    "canonical_json",
    "evaluate_proposal",
    "flatten_features",
    "stable_id",
    "utc_now",
]
