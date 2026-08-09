"""Portable tasks and multi-fidelity evaluation plans."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum
import statistics
import time
from typing import Callable, Generic, Iterable, Mapping, Protocol, Sequence, TypeVar

from .core import FeatureMap, KernelAdapter, KernelContext, flatten_features
from .outcomes import TrialOutcome
from .space import SearchSpace


ConfigT = TypeVar("ConfigT")


class StageKind(str, Enum):
    STATIC = "static"
    COMPILE = "compile"
    CORRECTNESS = "correctness"
    BENCHMARK = "benchmark"
    APPLICATION = "application"


@dataclass(frozen=True, slots=True)
class EvaluationStage:
    """One independently schedulable evaluation fidelity."""

    name: str
    kind: StageKind
    fidelity: float
    timeout_s: float | None = None
    required: bool = True
    metadata: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("evaluation stage names cannot be empty")
        if not 0 < self.fidelity <= 1:
            raise ValueError("stage fidelity must be in (0, 1]")
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError("stage timeout must be positive")

    def as_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "kind": self.kind.value,
            "fidelity": self.fidelity,
            "timeout_s": self.timeout_s,
            "required": self.required,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class EvaluationPlan:
    stages: tuple[EvaluationStage, ...]
    objective: str = "latency_ms"
    minimize: bool = True
    revision: int = 1

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("an evaluation plan requires at least one stage")
        names = [stage.name for stage in self.stages]
        if len(names) != len(set(names)):
            raise ValueError("evaluation stage names must be unique")
        fidelities = [stage.fidelity for stage in self.stages]
        if fidelities != sorted(fidelities):
            raise ValueError("evaluation stages must have nondecreasing fidelity")

    def through(self, fidelity: float) -> tuple[EvaluationStage, ...]:
        eligible = tuple(stage for stage in self.stages if stage.fidelity <= fidelity)
        if eligible:
            return eligible
        return (self.stages[0],)

    def as_dict(self) -> dict[str, object]:
        return {
            "revision": self.revision,
            "objective": self.objective,
            "minimize": self.minimize,
            "stages": [stage.as_dict() for stage in self.stages],
        }


@dataclass(frozen=True, slots=True)
class StageResult:
    """Typed result retained even when it cannot become a latency sample."""

    status: str = "ok"
    metrics: Mapping[str, object] = field(default_factory=dict)
    elapsed_s: float = 0.0
    error: str | None = None
    artifacts: Mapping[str, object] = field(default_factory=dict)
    metadata: Mapping[str, object] = field(default_factory=dict)

    @property
    def successful(self) -> bool:
        return self.status == "ok"

    def as_dict(self) -> dict[str, object]:
        return {
            "status": self.status,
            "metrics": dict(self.metrics),
            "elapsed_s": self.elapsed_s,
            "error": self.error,
            "artifacts": dict(self.artifacts),
            "metadata": dict(self.metadata),
        }


class PortableKernelTask(Protocol[ConfigT]):
    """Project plugin boundary; contains no CUDA, Torch, or RTX assumptions."""

    context: KernelContext
    space: SearchSpace[ConfigT]
    plan: EvaluationPlan

    def features(self, config: ConfigT) -> Mapping[str, float]: ...

    def run_stage(self, config: ConfigT, stage: EvaluationStage) -> StageResult: ...


@dataclass(slots=True)
class FunctionKernelTask(Generic[ConfigT]):
    """Small task implementation for external projects and tests."""

    context: KernelContext
    space: SearchSpace[ConfigT]
    plan: EvaluationPlan
    stage_runner: Callable[[ConfigT, EvaluationStage], StageResult]
    feature_fn: Callable[[ConfigT], Mapping[str, float]] | None = None

    def features(self, config: ConfigT) -> Mapping[str, float]:
        return {} if self.feature_fn is None else self.feature_fn(config)

    def run_stage(self, config: ConfigT, stage: EvaluationStage) -> StageResult:
        return self.stage_runner(config, stage)


def _failure_status(stage: EvaluationStage, status: str) -> str:
    if status in {
        "architecture_rejected",
        "implementation_rejected",
        "compile_error",
        "correctness_error",
        "runtime_error",
    }:
        return status
    if status == "rejected":
        return "implementation_rejected"
    if stage.kind == StageKind.COMPILE:
        return "compile_error"
    if stage.kind == StageKind.CORRECTNESS:
        return "correctness_error"
    return "runtime_error"


@dataclass(slots=True)
class StagedTaskAdapter(Generic[ConfigT]):
    """Compile a portable staged task down to the established adapter API."""

    task: PortableKernelTask[ConfigT]
    maximum_fidelity: float = 1.0

    @property
    def context(self) -> KernelContext:
        return self.task.context

    @property
    def initial_config(self) -> ConfigT:
        return self.task.space.initial_config

    def config_id(self, config: ConfigT) -> str:
        return self.task.space.config_id(config)

    def serialize(self, config: ConfigT) -> Mapping[str, object]:
        return self.task.space.serialize(config)

    def deserialize(self, value: Mapping[str, object]) -> ConfigT:
        return self.task.space.deserialize(value)

    def features(self, config: ConfigT) -> FeatureMap:
        features = self.context.features()
        features.update(flatten_features(self.serialize(config), "config"))
        features.update(
            {f"derived.{name}": float(value) for name, value in self.task.features(config).items()}
        )
        features["evaluation.maximum_fidelity"] = self.maximum_fidelity
        return features

    def rejection(self, config: ConfigT) -> tuple[str, str] | None:
        return self.task.space.rejection(config)

    def evaluate(self, config: ConfigT) -> TrialOutcome:
        return self.evaluate_at_fidelity(config, self.maximum_fidelity)

    def evaluate_at_fidelity(
        self, config: ConfigT, fidelity: float
    ) -> TrialOutcome:
        if not 0 < fidelity <= 1:
            raise ValueError("evaluation fidelity must be in (0, 1]")
        stage_records: list[dict[str, object]] = []
        results: list[tuple[EvaluationStage, StageResult]] = []
        for stage in self.task.plan.through(fidelity):
            started = time.monotonic()
            try:
                result = self.task.run_stage(config, stage)
            except Exception as exc:
                result = StageResult(
                    status="exception",
                    error=f"{type(exc).__name__}: {exc}"[:4000],
                )
            measured_elapsed = time.monotonic() - started
            if result.elapsed_s <= 0:
                result = replace(result, elapsed_s=measured_elapsed)
            results.append((stage, result))
            stage_records.append({"stage": stage.as_dict(), "result": result.as_dict()})
            if not result.successful and stage.required:
                return TrialOutcome(
                    _failure_status(stage, result.status),  # type: ignore[arg-type]
                    error=result.error or f"{stage.name} failed with {result.status}",
                    metadata={
                        "failure_kind": result.status,
                        "failed_stage": stage.name,
                        "completed_fidelity": stage.fidelity,
                        "evaluation_plan": self.task.plan.as_dict(),
                        "stages": stage_records,
                    },
                )

        objective_values = [
            result.metrics[self.task.plan.objective]
            for _stage, result in results
            if result.successful and self.task.plan.objective in result.metrics
        ]
        if not objective_values:
            completed = max(stage.fidelity for stage, _result in results)
            final_fidelity = max(stage.fidelity for stage in self.task.plan.stages)
            if completed < final_fidelity:
                return TrialOutcome(
                    "ok",
                    metadata={
                        "partial": True,
                        "completed_fidelity": completed,
                        "evaluation_plan": self.task.plan.as_dict(),
                        "stages": stage_records,
                    },
                )
            return TrialOutcome(
                "runtime_error",
                error=f"no stage produced objective {self.task.plan.objective!r}",
                metadata={
                    "failure_kind": "missing_objective",
                    "evaluation_plan": self.task.plan.as_dict(),
                    "stages": stage_records,
                },
            )
        objective = float(objective_values[-1])
        timings: list[float] = []
        compile_ms = 0.0
        max_abs_error: float | None = None
        for stage, result in results:
            raw_timings = result.metrics.get("timings_ms", ())
            if isinstance(raw_timings, (list, tuple)):
                timings.extend(float(value) for value in raw_timings)
            if stage.kind == StageKind.COMPILE:
                compile_ms += float(result.metrics.get("compile_ms", result.elapsed_s * 1000))
            if result.metrics.get("max_abs_error") is not None:
                error = float(result.metrics["max_abs_error"])
                max_abs_error = error if max_abs_error is None else max(max_abs_error, error)
        if timings:
            objective = float(statistics.median(timings))
        if not self.task.plan.minimize:
            objective = -objective
        return TrialOutcome(
            "ok",
            median_ms=objective,
            timings_ms=timings,
            compile_ms=compile_ms or None,
            max_abs_error=max_abs_error,
            metadata={
                "completed_fidelity": max(stage.fidelity for stage, _result in results),
                "evaluation_plan": self.task.plan.as_dict(),
                "stages": stage_records,
            },
        )

    def coordinates(self) -> Sequence[str]:
        return self.task.space.coordinates()

    def neighbors(self, config: ConfigT) -> Iterable[tuple[str, object, ConfigT]]:
        return self.task.space.neighbors(config)

    def sample(
        self,
        rng,
        count: int,
        seeds: Sequence[ConfigT],
    ) -> list[ConfigT]:
        return self.task.space.sample(rng, count, seeds)


@dataclass(slots=True)
class AdapterSearchSpace(Generic[ConfigT]):
    """Expose an existing RTX or third-party adapter through SearchSpace."""

    adapter: KernelAdapter[ConfigT]

    @property
    def initial_config(self) -> ConfigT:
        return self.adapter.initial_config

    def config_id(self, config: ConfigT) -> str:
        return self.adapter.config_id(config)

    def serialize(self, config: ConfigT) -> Mapping[str, object]:
        return self.adapter.serialize(config)

    def deserialize(self, value: Mapping[str, object]) -> ConfigT:
        return self.adapter.deserialize(value)

    def rejection(self, config: ConfigT) -> tuple[str, str] | None:
        return self.adapter.rejection(config)

    def coordinates(self) -> Sequence[str]:
        return self.adapter.coordinates()

    def neighbors(self, config: ConfigT) -> Iterable[tuple[str, object, ConfigT]]:
        return self.adapter.neighbors(config)

    def sample(self, rng, count: int, seeds: Sequence[ConfigT]) -> list[ConfigT]:
        return self.adapter.sample(rng, count, seeds)


@dataclass(slots=True)
class AdapterKernelTask(Generic[ConfigT]):
    """Compatibility wrapper that makes any current adapter a portable task."""

    adapter: KernelAdapter[ConfigT]
    context: KernelContext = field(init=False)
    space: AdapterSearchSpace[ConfigT] = field(init=False)
    plan: EvaluationPlan = field(init=False)

    def __post_init__(self) -> None:
        self.context = self.adapter.context
        self.space = AdapterSearchSpace(self.adapter)
        self.plan = EvaluationPlan(
            (EvaluationStage("evaluate", StageKind.BENCHMARK, 1.0),)
        )

    def features(self, config: ConfigT) -> Mapping[str, float]:
        return {
            key.removeprefix("derived."): value
            for key, value in self.adapter.features(config).items()
            if key.startswith("derived.")
        }

    def run_stage(self, config: ConfigT, stage: EvaluationStage) -> StageResult:
        outcome = self.adapter.evaluate(config)
        metrics: dict[str, object] = dict(outcome.metadata)
        if outcome.median_ms is not None:
            metrics["latency_ms"] = outcome.median_ms
        if outcome.timings_ms:
            metrics["timings_ms"] = list(outcome.timings_ms)
        if outcome.compile_ms is not None:
            metrics["compile_ms"] = outcome.compile_ms
        if outcome.max_abs_error is not None:
            metrics["max_abs_error"] = outcome.max_abs_error
        return StageResult(
            status=outcome.status,
            metrics=metrics,
            error=outcome.error,
            metadata=outcome.metadata,
        )


__all__ = [
    "AdapterKernelTask",
    "AdapterSearchSpace",
    "EvaluationPlan",
    "EvaluationStage",
    "FunctionKernelTask",
    "PortableKernelTask",
    "StageKind",
    "StageResult",
    "StagedTaskAdapter",
]
