"""Rigorous, resumable experiment campaigns for dynamic MXFP8 linear.

This module complements :mod:`rtx.prequant_autotune`.  Coordinate descent is
useful for quickly finding a local winner, while this runner is intended to
produce training data which remains statistically and operationally useful:

* legal configurations are sampled conditionally and selected for pairwise
  coverage instead of drawing the impossible Cartesian product;
* every measurement is an append-only observation;
* timing batches are calibrated to a requested duration;
* hot and rotating-input cache regimes are explicit;
* finalists are remeasured and compared with interleaved paired races; and
* manifests, device metadata, telemetry, derived features, and raw timings are
  stored together for cross-device analysis.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import statistics
import subprocess
import tempfile
import time
from typing import Callable, Iterable, Iterator, Literal, Mapping, Sequence
import uuid

import torch

from .autotune import DeviceFingerprint
from .autotune.hardware import (
    compiled_resource_metadata,
    device_properties,
    static_device_profile,
)
from .fp8 import (
    DEFAULT_MXFP8_PREQUANT_CONFIG,
    MXFP8PrequantConfig,
    _build_prequant_runner,
    _intern_prequant_config,
    _set_l2_fetch_granularity,
)
from .kernels.mxfp8 import MXFP8Problem
from .prequant_autotune import (
    PREQUANT_SEARCH_SPACE,
    prequant_config_from_dict,
    prequant_config_id,
    prequant_config_to_dict,
    update_prequant_config,
)

try:
    import fcntl
except ImportError:  # pragma: no cover - Linux/CUDA is the target platform.
    fcntl = None


EXPERIMENT_SCHEMA_VERSION = 1
CacheRegime = Literal["hot", "rotate"]
Progress = Callable[[str], None]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(value: object, length: int = 20) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()[:length]


@dataclass(frozen=True, slots=True)
class ShapeSpec:
    m: int
    n: int
    k: int
    name: str | None = None
    weight: float = 1.0

    def __post_init__(self) -> None:
        if min(self.m, self.n, self.k) <= 0:
            raise ValueError("shape dimensions must be positive")
        if self.weight <= 0:
            raise ValueError("shape weight must be positive")

    @property
    def key(self) -> str:
        return f"m{self.m}_n{self.n}_k{self.k}"

    @property
    def problem(self) -> MXFP8Problem:
        return MXFP8Problem(self.m, self.n, self.k)

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ShapeSpec":
        return cls(
            m=int(value["m"]),
            n=int(value["n"]),
            k=int(value["k"]),
            name=None if value.get("name") is None else str(value["name"]),
            weight=float(value.get("weight", 1.0)),
        )


@dataclass(frozen=True, slots=True)
class BenchmarkProtocol:
    warmup_calls: int = 5
    samples: int = 5
    confirm_samples: int = 11
    race_rounds: int = 11
    target_batch_ms: float = 30.0
    stabilization_target_ms: float = 50.0
    stabilization_max_batches: int = 8
    measurement_retries: int = 1
    adaptive_sampling: bool = True
    screen_min_samples: int = 3
    confirm_min_samples: int = 5
    race_min_rounds: int = 7
    screen_stable_cv: float = 0.005
    confirm_stable_cv: float = 0.0025
    race_stable_cv: float = 0.005
    screen_max_relative_drift: float = 0.03
    confirm_max_relative_drift: float = 0.015
    screen_max_relative_range: float = 0.10
    confirm_max_relative_range: float = 0.05
    min_calls_per_sample: int = 1
    max_calls_per_sample: int = 4096
    correctness_rtol: float = 5e-2
    correctness_atol: float = 5e-1
    practical_threshold: float = 0.005
    bootstrap_resamples: int = 1000
    rotation_l2_multiple: float = 2.0
    max_rotation_buffers: int = 16
    max_rotation_bytes: int = 1 << 30
    telemetry: bool = True

    def __post_init__(self) -> None:
        if self.warmup_calls < 0:
            raise ValueError("warmup_calls must be nonnegative")
        if min(self.samples, self.confirm_samples, self.race_rounds) <= 0:
            raise ValueError("sample and race counts must be positive")
        if min(
            self.screen_min_samples,
            self.confirm_min_samples,
            self.race_min_rounds,
        ) <= 0:
            raise ValueError("adaptive sample floors must be positive")
        if min(
            self.screen_stable_cv,
            self.confirm_stable_cv,
            self.race_stable_cv,
        ) < 0:
            raise ValueError("adaptive stability thresholds must be nonnegative")
        if self.target_batch_ms <= 0:
            raise ValueError("target_batch_ms must be positive")
        if self.stabilization_target_ms < 0:
            raise ValueError("stabilization target must be nonnegative")
        if self.stabilization_max_batches <= 0:
            raise ValueError("stabilization batch limit must be positive")
        if self.measurement_retries < 0:
            raise ValueError("measurement retries must be nonnegative")
        if min(
            self.screen_max_relative_drift,
            self.confirm_max_relative_drift,
            self.screen_max_relative_range,
            self.confirm_max_relative_range,
        ) < 0:
            raise ValueError("timing stationarity thresholds must be nonnegative")
        if not 0 <= self.practical_threshold < 1:
            raise ValueError("practical_threshold must be in [0, 1)")
        if self.min_calls_per_sample <= 0:
            raise ValueError("min_calls_per_sample must be positive")
        if self.max_calls_per_sample < self.min_calls_per_sample:
            raise ValueError("max_calls_per_sample must be >= minimum")
        if self.bootstrap_resamples <= 0:
            raise ValueError("bootstrap_resamples must be positive")

    @staticmethod
    def relative_stdev(values: Sequence[float]) -> float:
        if len(values) < 2:
            return math.inf
        center = statistics.fmean(float(value) for value in values)
        if center <= 0:
            return math.inf
        return float(statistics.stdev(float(value) for value in values) / center)

    def timing_complete(
        self, values: Sequence[float], *, requested_samples: int
    ) -> bool:
        if len(values) >= requested_samples:
            return True
        if not self.adaptive_sampling:
            return False
        is_confirmation = requested_samples > self.samples
        floor = min(
            requested_samples,
            self.confirm_min_samples if is_confirmation else self.screen_min_samples,
        )
        threshold = (
            self.confirm_stable_cv if is_confirmation else self.screen_stable_cv
        )
        return (
            len(values) >= floor
            and len(values) % 2 == 1
            and self.relative_stdev(values) <= threshold
        )

    def timing_quality(
        self,
        values: Sequence[float],
        *,
        requested_samples: int,
    ) -> dict[str, object]:
        """Describe dispersion and within-attempt timing-plateau movement."""

        numeric = [float(value) for value in values]
        if not numeric:
            return {
                "stationary": False,
                "relative_stdev": math.inf,
                "relative_mad": math.inf,
                "relative_range": math.inf,
                "relative_split_drift": math.inf,
                "quality_score": math.inf,
            }
        center = float(statistics.median(numeric))
        denominator = max(abs(center), 1.0e-12)
        relative_stdev = self.relative_stdev(numeric)
        relative_mad = float(
            statistics.median(abs(value - center) for value in numeric)
            / denominator
        )
        relative_range = float((max(numeric) - min(numeric)) / denominator)
        relative_split_drift = 0.0
        if len(numeric) >= 5:
            half = len(numeric) // 2
            early = float(statistics.median(numeric[:half]))
            late = float(statistics.median(numeric[-half:]))
            relative_split_drift = abs(early - late) / denominator
        confirmation = requested_samples > self.samples
        drift_limit = (
            self.confirm_max_relative_drift
            if confirmation
            else self.screen_max_relative_drift
        )
        range_limit = (
            self.confirm_max_relative_range
            if confirmation
            else self.screen_max_relative_range
        )
        stationary = (
            relative_split_drift <= drift_limit
            and relative_range <= range_limit
        )
        return {
            "stationary": stationary,
            "relative_stdev": relative_stdev,
            "relative_mad": relative_mad,
            "relative_range": relative_range,
            "relative_split_drift": relative_split_drift,
            "drift_limit": drift_limit,
            "range_limit": range_limit,
            "quality_score": max(
                relative_split_drift / max(drift_limit, 1.0e-12),
                relative_range / max(range_limit, 1.0e-12),
            ),
        }

    def race_complete(
        self,
        incumbent: Sequence[float],
        challenger: Sequence[float],
    ) -> bool:
        if len(incumbent) >= self.race_rounds:
            return True
        if not self.adaptive_sampling:
            return False
        floor = min(self.race_rounds, self.race_min_rounds)
        return (
            len(incumbent) >= floor
            and len(incumbent) % 2 == 1
            and self.relative_stdev(incumbent) <= self.race_stable_cv
            and self.relative_stdev(challenger) <= self.race_stable_cv
        )

    def sampling_metadata(
        self, values: Sequence[float], *, requested_samples: int
    ) -> dict[str, object]:
        is_confirmation = requested_samples > self.samples
        floor = min(
            requested_samples,
            self.confirm_min_samples if is_confirmation else self.screen_min_samples,
        )
        threshold = (
            self.confirm_stable_cv if is_confirmation else self.screen_stable_cv
        )
        stopped_early = len(values) < requested_samples
        return {
            "adaptive": self.adaptive_sampling,
            "stage": "confirmation" if is_confirmation else "screen",
            "requested_samples": requested_samples,
            "actual_samples": len(values),
            "minimum_samples": floor,
            "stable_cv_threshold": threshold,
            "stopped_early": stopped_early,
            "stop_reason": "stable_dispersion" if stopped_early else "sample_budget",
            "relative_stdev": self.relative_stdev(values),
        }

    def race_sampling_metadata(
        self,
        incumbent: Sequence[float],
        challenger: Sequence[float],
    ) -> dict[str, object]:
        stopped_early = len(incumbent) < self.race_rounds
        return {
            "adaptive": self.adaptive_sampling,
            "stage": "paired_race",
            "requested_rounds": self.race_rounds,
            "actual_rounds": len(incumbent),
            "minimum_rounds": min(self.race_rounds, self.race_min_rounds),
            "stable_cv_threshold": self.race_stable_cv,
            "stopped_early": stopped_early,
            "stop_reason": "stable_dispersion" if stopped_early else "round_budget",
            "incumbent_relative_stdev": self.relative_stdev(incumbent),
            "challenger_relative_stdev": self.relative_stdev(challenger),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, object] | None) -> "BenchmarkProtocol":
        return cls() if value is None else cls(**dict(value))  # type: ignore[arg-type]


def collect_timing_samples(
    measure: Callable[[int], float],
    protocol: BenchmarkProtocol,
    *,
    requested_samples: int,
) -> list[float]:
    """Collect an odd, dispersion-gated number of calibrated timing batches."""

    timings: list[float] = []
    for sample in range(requested_samples):
        timings.append(float(measure(sample)))
        if protocol.timing_complete(timings, requested_samples=requested_samples):
            break
    return timings


def stabilize_timing_batches(
    measure_batch: Callable[[int], float],
    protocol: BenchmarkProtocol,
) -> dict[str, object]:
    """Burn calibrated GPU work until the requested stabilization duration."""

    batches: list[float] = []
    elapsed_ms = 0.0
    while (
        len(batches) < protocol.stabilization_max_batches
        and elapsed_ms < protocol.stabilization_target_ms
    ):
        duration_ms = max(0.0, float(measure_batch(len(batches))))
        batches.append(duration_ms)
        elapsed_ms += duration_ms
    return {
        "target_ms": protocol.stabilization_target_ms,
        "elapsed_ms": elapsed_ms,
        "batches": len(batches),
        "batch_timings_ms": batches,
        "target_reached": elapsed_ms >= protocol.stabilization_target_ms,
    }


def collect_stable_timing_samples(
    measure: Callable[[int], float],
    protocol: BenchmarkProtocol,
    *,
    requested_samples: int,
    stabilize: Callable[[int], Mapping[str, object]] | None = None,
    telemetry: Callable[[], Mapping[str, object]] | None = None,
) -> tuple[list[float], dict[str, object]]:
    """Retry a measurement when samples move between timing plateaus."""

    attempts: list[dict[str, object]] = []
    for attempt in range(protocol.measurement_retries + 1):
        stabilization = None if stabilize is None else dict(stabilize(attempt))
        telemetry_before = None if telemetry is None else dict(telemetry())
        offset = attempt * requested_samples
        timings = collect_timing_samples(
            lambda sample: measure(offset + sample),
            protocol,
            requested_samples=requested_samples,
        )
        quality = protocol.timing_quality(
            timings,
            requested_samples=requested_samples,
        )
        telemetry_after = None if telemetry is None else dict(telemetry())
        attempts.append(
            {
                "attempt": attempt,
                "timings_ms": timings,
                "quality": quality,
                "stabilization": stabilization,
                "telemetry_before": telemetry_before,
                "telemetry_after": telemetry_after,
            }
        )
        if bool(quality["stationary"]):
            break
    selected_index = min(
        range(len(attempts)),
        key=lambda index: (
            not bool(attempts[index]["quality"]["stationary"]),  # type: ignore[index]
            float(attempts[index]["quality"]["quality_score"]),  # type: ignore[index]
            index,
        ),
    )
    selected = attempts[selected_index]
    return list(selected["timings_ms"]), {  # type: ignore[arg-type]
        "attempts": attempts,
        "attempt_count": len(attempts),
        "retry_count": len(attempts) - 1,
        "selected_attempt": selected_index,
        "stationary": bool(selected["quality"]["stationary"]),  # type: ignore[index]
        "quality": selected["quality"],
    }


@dataclass(frozen=True, slots=True)
class ExperimentManifest:
    name: str
    shapes: tuple[ShapeSpec, ...]
    regimes: tuple[CacheRegime, ...] = ("hot", "rotate")
    candidates_per_shape: int = 64
    promote: int = 8
    seed: int = 0
    shard_index: int = 0
    shard_count: int = 1
    protocol: BenchmarkProtocol = BenchmarkProtocol()

    def __post_init__(self) -> None:
        if not self.name or any(part in self.name for part in ("/", "\\", "..")):
            raise ValueError("manifest name must be a safe path component")
        if not self.shapes:
            raise ValueError("manifest must contain at least one shape")
        if not self.regimes or any(value not in ("hot", "rotate") for value in self.regimes):
            raise ValueError("regimes must contain hot and/or rotate")
        if self.candidates_per_shape <= 0:
            raise ValueError("candidates_per_shape must be positive")
        if not 1 <= self.promote <= self.candidates_per_shape:
            raise ValueError("promote must be between 1 and candidates_per_shape")
        if self.shard_count <= 0 or not 0 <= self.shard_index < self.shard_count:
            raise ValueError("invalid shard index/count")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "ExperimentManifest":
        return cls(
            name=str(value["name"]),
            shapes=tuple(
                ShapeSpec.from_dict(item) for item in value["shapes"]  # type: ignore[union-attr]
            ),
            regimes=tuple(value.get("regimes", ("hot", "rotate"))),  # type: ignore[arg-type]
            candidates_per_shape=int(value.get("candidates_per_shape", 64)),
            promote=int(value.get("promote", 8)),
            seed=int(value.get("seed", 0)),
            shard_index=int(value.get("shard_index", 0)),
            shard_count=int(value.get("shard_count", 1)),
            protocol=BenchmarkProtocol.from_dict(value.get("protocol")),  # type: ignore[arg-type]
        )

    @classmethod
    def load(cls, path: Path | str) -> "ExperimentManifest":
        with Path(path).open(encoding="utf-8") as source:
            return cls.from_dict(json.load(source))

    def as_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def digest(self) -> str:
        return _digest(self.as_dict())


@dataclass(frozen=True, slots=True)
class RobustSummary:
    count: int
    median: float
    mean: float
    trimmed_mean: float
    stdev: float
    ci_low: float
    ci_high: float

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def robust_summary(
    values: Sequence[float],
    *,
    seed: int = 0,
    bootstrap_resamples: int = 1000,
    confidence: float = 0.95,
) -> RobustSummary:
    """Return robust descriptive statistics and a bootstrap median interval."""

    if not values:
        raise ValueError("cannot summarize an empty sample")
    data = [float(value) for value in values]
    ordered = sorted(data)
    trim = int(len(ordered) * 0.1)
    trimmed = ordered[trim : len(ordered) - trim] if trim else ordered
    median = float(statistics.median(ordered))
    rng = random.Random(seed)
    medians = sorted(
        statistics.median(rng.choices(ordered, k=len(ordered)))
        for _ in range(bootstrap_resamples)
    )
    tail = (1.0 - confidence) / 2.0
    low_index = max(0, min(len(medians) - 1, int(tail * len(medians))))
    high_index = max(
        0,
        min(len(medians) - 1, math.ceil((1.0 - tail) * len(medians)) - 1),
    )
    return RobustSummary(
        count=len(ordered),
        median=median,
        mean=float(statistics.fmean(ordered)),
        trimmed_mean=float(statistics.fmean(trimmed)),
        stdev=float(statistics.stdev(ordered)) if len(ordered) > 1 else 0.0,
        ci_low=float(medians[low_index]),
        ci_high=float(medians[high_index]),
    )


def _apply_conditional_variant(
    config: MXFP8PrequantConfig,
    coordinate: str,
    variant: Mapping[str, object],
) -> MXFP8PrequantConfig:
    """Apply one search coordinate while crossing required family boundaries."""

    update = dict(variant)
    if config.quant_launches == "dual" and coordinate.startswith("w_"):
        update["quant_launches"] = "separate"
    candidate = update_prequant_config(config, update)
    if coordinate == "gemm_geometry" and (
        candidate.gemm.tile_m > 128 or candidate.gemm.tile_n > 128
    ):
        candidate = update_prequant_config(
            candidate,
            {
                "gemm": {
                    "stages": 1,
                    "epilogue": "direct",
                    "epilogue_stages": 1,
                    "store_vec": 1,
                }
            },
        )
    return candidate


def _coverage_tokens(config: MXFP8PrequantConfig) -> tuple[str, ...]:
    qx = config.quant
    qw = config.resolved_weight_quant()
    gemm = config.gemm
    values: tuple[tuple[str, object], ...] = (
        ("family", (qx.scale_layout, qw.scale_layout, gemm.scale_layout, gemm.scale_role)),
        ("launches", config.quant_launches),
        (
            "x_io",
            (
                qx.quant_vec,
                qx.load_bits,
                qx.quant_store_bits,
                qx.native_scale_store,
            ),
        ),
        (
            "w_io",
            (
                qw.quant_vec,
                qw.load_bits,
                qw.quant_store_bits,
                qw.native_scale_store,
            ),
        ),
        ("x_math", (qx.quant_math, qx.quant_amax, qx.reduction)),
        ("w_math", (qw.quant_math, qw.quant_amax, qw.reduction)),
        ("x_launch", (qx.num_warps, qx.persistent_waves, qx.maxrregcount)),
        ("w_launch", (qw.num_warps, qw.persistent_waves, qw.maxrregcount)),
        ("tile", (gemm.tile_m, gemm.tile_n, gemm.tile_k)),
        ("mma", (gemm.atom_layout_m, gemm.atom_layout_n)),
        ("pipeline", (gemm.stages, gemm.a_ldmatrix_matrices, gemm.b_ldmatrix_matrices)),
        ("smem", (gemm.a_swizzle, gemm.b_swizzle)),
        ("scale", (gemm.scale_schedule, gemm.scale_load_vec, gemm.sfa_s2r_bits, gemm.sfb_s2r_bits)),
        ("registers", (gemm.producer_registers, gemm.consumer_registers, gemm.maxrregcount)),
        (
            "epilogue",
            (gemm.epilogue, gemm.epilogue_stages, gemm.store_vec),
        ),
        ("persistence", (gemm.tiles_per_cta, gemm.tile_locality)),
        ("raster", (gemm.raster, gemm.grid_swizzle)),
        ("l2", config.l2_fetch_granularity),
    )
    return tuple(f"{name}={_canonical_json(value)}" for name, value in values)


def _coverage_items(config: MXFP8PrequantConfig) -> set[tuple[str, ...]]:
    tokens = _coverage_tokens(config)
    singles = {(token,) for token in tokens}
    pairs = {tuple(pair) for pair in itertools.combinations(tokens, 2)}
    return singles | pairs


def generate_legal_catalog(
    problem: MXFP8Problem,
    count: int,
    *,
    seed: int,
    axes: Mapping[str, Iterable[Mapping[str, object]]] = PREQUANT_SEARCH_SPACE,
) -> list[MXFP8PrequantConfig]:
    """Generate a deterministic legal catalogue with broad pairwise coverage.

    Proposals are legal random walks.  This avoids constructing inactive or
    impossible Cartesian points.  A stochastic greedy pass then favours new
    single-coordinate and pairwise structural coverage.
    """

    if count <= 0:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    normalized_axes = {name: tuple(values) for name, values in axes.items()}
    unique: dict[str, MXFP8PrequantConfig] = {}
    structural: list[MXFP8PrequantConfig] = []

    def admit(config: MXFP8PrequantConfig) -> bool:
        config = config.normalized()
        if config.rejection(problem) is not None:
            return False
        unique.setdefault(prequant_config_id(config), config)
        return True

    admit(DEFAULT_MXFP8_PREQUANT_CONFIG)
    structural.append(DEFAULT_MXFP8_PREQUANT_CONFIG)
    # Reserve early budget for interpretable one-factor structural probes.
    # They preserve strong baseline schedules while exposing the shape-driven
    # effects (especially raster/reuse) that random compound points can bury.
    structural_order = (
        "raster_group",
        "layout_transport",
        "gemm_geometry",
        "gemm_stages",
        "quant_launches",
        "epilogue",
        "global_l2_fetch",
    )
    for coordinate in structural_order:
        for variant in normalized_axes.get(coordinate, ()):
            candidate = _apply_conditional_variant(
                DEFAULT_MXFP8_PREQUANT_CONFIG, coordinate, variant
            )
            if candidate.rejection(problem) is None:
                admit(candidate)
                if all(
                    prequant_config_id(existing) != prequant_config_id(candidate)
                    for existing in structural
                ):
                    structural.append(candidate)

    pool_target = max(count, min(max(count * 6, 256), 20_000))
    coordinates = tuple(normalized_axes)
    attempts = 0
    max_attempts = max(2_000, pool_target * 80)
    while len(unique) < pool_target and attempts < max_attempts:
        attempts += 1
        parent = rng.choice(tuple(unique.values()))
        current = parent
        mutation_count = rng.randint(1, min(8, len(coordinates)))
        for coordinate in rng.sample(coordinates, mutation_count):
            variants = normalized_axes[coordinate]
            if not variants:
                continue
            proposal = _apply_conditional_variant(
                current, coordinate, rng.choice(variants)
            )
            if proposal.rejection(problem) is None:
                current = proposal
        admit(current)

    pool = list(unique.values())
    default_id = prequant_config_id(DEFAULT_MXFP8_PREQUANT_CONFIG)
    pool.sort(key=lambda cfg: (prequant_config_id(cfg) != default_id, prequant_config_id(cfg)))
    if len(pool) <= count:
        return pool

    # If the requested catalogue is small, retain a coverage-balanced subset
    # of the structural basis.  The baseline and at least one M/N raster probe
    # are explicitly protected because those are the cheapest, most
    # interpretable locality hypotheses.
    protected: list[MXFP8PrequantConfig] = [structural[0]]
    for raster in ("m", "n"):
        match = next(
            (
                config
                for config in structural
                if config.gemm.raster == raster and config.gemm.grid_swizzle == 1
            ),
            None,
        )
        if match is not None and prequant_config_id(match) not in {
            prequant_config_id(config) for config in protected
        }:
            protected.append(match)
    remaining_structural = [
        config
        for config in structural
        if prequant_config_id(config)
        not in {prequant_config_id(item) for item in protected}
    ]
    selected = protected[:count]
    covered: set[tuple[str, ...]] = set()
    for config in selected:
        covered.update(_coverage_items(config))
    while remaining_structural and len(selected) < min(count, len(structural)):
        best_index = max(
            range(len(remaining_structural)),
            key=lambda index: (
                len(_coverage_items(remaining_structural[index]) - covered),
                prequant_config_id(remaining_structural[index]),
            ),
        )
        winner = remaining_structural.pop(best_index)
        selected.append(winner)
        covered.update(_coverage_items(winner))
    selected_ids = {prequant_config_id(config) for config in selected}
    pool = [config for config in pool if prequant_config_id(config) not in selected_ids]
    while pool and len(selected) < count:
        sample_size = min(128, len(pool))
        indices = rng.sample(range(len(pool)), sample_size)
        best_index = max(
            indices,
            key=lambda index: (
                len(_coverage_items(pool[index]) - covered),
                prequant_config_id(pool[index]),
            ),
        )
        winner = pool.pop(best_index)
        selected.append(winner)
        covered.update(_coverage_items(winner))
    return selected


def config_in_shard(config: MXFP8PrequantConfig, index: int, count: int) -> bool:
    if count <= 0 or not 0 <= index < count:
        raise ValueError("invalid shard")
    return int(prequant_config_id(config), 16) % count == index


def _nvidia_smi_snapshot(device_index: int) -> dict[str, object]:
    fields = (
        "timestamp,driver_version,pstate,clocks.sm,clocks.mem,temperature.gpu,"
        "power.draw,power.limit,utilization.gpu,utilization.memory"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                str(device_index),
                f"--query-gpu={fields}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError) as exc:
        return {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    values = [value.strip() for value in completed.stdout.strip().split(",")]
    names = fields.split(",")
    if len(values) != len(names):
        return {"available": False, "error": completed.stdout.strip()}
    result: dict[str, object] = {"available": True}
    result.update(zip(names, values))
    return result


def _device_properties(device: torch.device) -> dict[str, object]:
    return device_properties(device)


def probe_device(
    device: torch.device | str = "cuda",
    *,
    calibration: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Report campaign-relevant capabilities without compiling a kernel."""

    resolved = torch.device(device)
    fingerprint = DeviceFingerprint.current(resolved)
    free_memory, total_memory = torch.cuda.mem_get_info(resolved)
    index = resolved.index
    if index is None:
        index = torch.cuda.current_device()
    hardware = static_device_profile(resolved, calibration=calibration)
    hardware["software"] = fingerprint.as_dict()
    return {
        "fingerprint_id": fingerprint.identifier,
        "fingerprint": fingerprint.as_dict(),
        "properties": hardware["properties"],
        "hardware_profile": hardware,
        "free_memory": int(free_memory),
        "total_memory": int(total_memory),
        "native_rtx_mxfp8_campaign_supported": fingerprint.capability[0] == 12,
        "telemetry": _nvidia_smi_snapshot(index),
    }


def derived_features(
    shape: ShapeSpec,
    config: MXFP8PrequantConfig,
    fingerprint: DeviceFingerprint,
    *,
    l2_cache_size: int | None = None,
) -> dict[str, object]:
    gemm = config.gemm
    cta_m = math.ceil(shape.m / gemm.tile_m)
    cta_n = math.ceil(shape.n / gemm.tile_n)
    ctas = cta_m * cta_n
    sms = fingerprint.multiprocessor_count
    x_bytes = shape.m * shape.k * 2
    w_bytes = shape.n * shape.k * 2
    output_bytes = shape.m * shape.n * 2
    storage_k = shape.problem.storage_k
    quantized_bytes = (shape.m + shape.n) * storage_k
    scale_bytes = (shape.m + shape.n) * (storage_k // 32)
    result: dict[str, object] = {
        "m": shape.m,
        "n": shape.n,
        "k": shape.k,
        "storage_k": storage_k,
        "storage_k_tail": storage_k - shape.k,
        "storage_k_overhead": storage_k / shape.k - 1.0,
        "m_over_n": shape.m / shape.n,
        "k_over_mn_sqrt": shape.k / math.sqrt(shape.m * shape.n),
        "cta_m": cta_m,
        "cta_n": cta_n,
        "cta_count": ctas,
        "complete_waves": ctas // sms,
        "last_wave_fraction": (ctas % sms) / sms,
        "m_tail": shape.m % gemm.tile_m,
        "n_tail": shape.n % gemm.tile_n,
        "k_tail": shape.k % gemm.tile_k,
        "x_bytes": x_bytes,
        "w_bytes": w_bytes,
        "output_bytes": output_bytes,
        "quantized_operand_bytes": quantized_bytes,
        "scale_bytes": scale_bytes,
        "nominal_flops": 2 * shape.m * shape.n * shape.k,
        "x_reuse_ctas": cta_n,
        "w_reuse_ctas": cta_m,
        "raster": gemm.raster,
        "grid_swizzle": gemm.grid_swizzle,
        "tile_m": gemm.tile_m,
        "tile_n": gemm.tile_n,
        "tile_k": gemm.tile_k,
        "stages": gemm.stages,
    }
    if l2_cache_size:
        result.update(
            x_l2_ratio=x_bytes / l2_cache_size,
            w_l2_ratio=w_bytes / l2_cache_size,
            working_set_l2_ratio=(x_bytes + w_bytes + output_bytes + quantized_bytes)
            / l2_cache_size,
        )
    return result


@dataclass(slots=True)
class _PreparedCandidate:
    config: MXFP8PrequantConfig
    runner: object
    out: torch.Tensor
    compile_ms: float
    max_abs_error: float
    compiled_resources: Mapping[str, object]


class CandidateCompileError(RuntimeError):
    pass


class CandidateCorrectnessError(RuntimeError):
    pass


def _reference_prequant_config(problem: MXFP8Problem) -> MXFP8PrequantConfig:
    """Choose a stable legal reference family, including for 64-row shapes."""

    if DEFAULT_MXFP8_PREQUANT_CONFIG.rejection(problem) is None:
        return DEFAULT_MXFP8_PREQUANT_CONFIG
    for updates in PREQUANT_SEARCH_SPACE["layout_transport"]:
        candidate = update_prequant_config(DEFAULT_MXFP8_PREQUANT_CONFIG, updates)
        if (
            candidate.gemm.scale_layout == "row_major"
            and candidate.rejection(problem) is None
        ):
            return candidate
    raise RuntimeError(
        f"no legal prequant correctness-reference configuration for {problem}"
    )


class PrequantBenchmarkHarness:
    """Own tensors and perform calibrated measurements for one shape/regime."""

    def __init__(
        self,
        shape: ShapeSpec,
        regime: CacheRegime,
        protocol: BenchmarkProtocol,
        *,
        device: torch.device | str = "cuda",
        seed: int = 0,
    ) -> None:
        self.shape = shape
        self.problem = shape.problem
        self.regime = regime
        self.protocol = protocol
        self.device = torch.device(device)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        self.x = torch.randn(
            shape.m, shape.k, device=self.device, dtype=torch.bfloat16, generator=generator
        )
        self.weight = torch.randn(
            shape.n, shape.k, device=self.device, dtype=torch.bfloat16, generator=generator
        )
        self._inputs = self._make_input_ring()
        self._expected = self._make_reference()

    def _l2_bytes(self) -> int:
        props = torch.cuda.get_device_properties(self.device)
        return int(
            getattr(props, "L2_cache_size", getattr(props, "l2_cache_size", 0)) or 0
        )

    def _make_input_ring(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self.regime == "hot":
            return [(self.x, self.weight)]
        pair_bytes = self.x.numel() * self.x.element_size()
        pair_bytes += self.weight.numel() * self.weight.element_size()
        free_bytes, _total_bytes = torch.cuda.mem_get_info(self.device)
        budget = min(self.protocol.max_rotation_bytes, int(free_bytes * 0.35))
        l2_target = max(pair_bytes * 2, int(self._l2_bytes() * self.protocol.rotation_l2_multiple))
        requested = math.ceil(l2_target / pair_bytes)
        count = max(
            2,
            min(
                self.protocol.max_rotation_buffers,
                max(1, budget // pair_bytes),
                requested,
            ),
        )
        ring = [(self.x, self.weight)]
        for _ in range(1, count):
            ring.append((self.x.clone(), self.weight.clone()))
        torch.cuda.synchronize(self.device)
        return ring

    def _make_reference(self) -> torch.Tensor:
        # The global native mma128 default requires M/N/K multiples of 128.
        # Campaigns deliberately include M=64, where row-major transport is
        # used as the correctness reference instead.
        reference_config = _reference_prequant_config(self.problem)
        runner = _build_prequant_runner(
            self.x,
            self.weight,
            _intern_prequant_config(reference_config),
        )
        out = torch.empty(
            (self.shape.m, self.shape.n), device=self.device, dtype=torch.bfloat16
        )
        runner(self.x, self.weight, out)
        torch.cuda.synchronize(self.device)
        return out.clone()

    def prepare(self, config: MXFP8PrequantConfig) -> _PreparedCandidate:
        previous_l2: int | None = None
        if config.l2_fetch_granularity is not None:
            previous_l2 = _set_l2_fetch_granularity(config.l2_fetch_granularity)
        started = time.monotonic()
        try:
            try:
                runner = _build_prequant_runner(
                    self.x, self.weight, _intern_prequant_config(config)
                )
            except Exception as exc:
                raise CandidateCompileError(
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            compile_ms = (time.monotonic() - started) * 1000
            out = torch.empty(
                (self.shape.m, self.shape.n), device=self.device, dtype=torch.bfloat16
            )
            runner(self.x, self.weight, out)
            torch.cuda.synchronize(self.device)
            max_abs_error = float((out.float() - self._expected.float()).abs().max())
            if not torch.allclose(
                out,
                self._expected,
                rtol=self.protocol.correctness_rtol,
                atol=self.protocol.correctness_atol,
                equal_nan=True,
            ):
                raise CandidateCorrectnessError(
                    f"candidate differs from reference (max abs {max_abs_error})"
                )
            return _PreparedCandidate(
                config,
                runner,
                out,
                compile_ms,
                max_abs_error,
                compiled_resource_metadata(runner),
            )
        finally:
            if previous_l2 is not None:
                _set_l2_fetch_granularity(previous_l2)

    def _time_batch(
        self, prepared: _PreparedCandidate, calls: int, offset: int
    ) -> float:
        previous_l2: int | None = None
        if prepared.config.l2_fetch_granularity is not None:
            previous_l2 = _set_l2_fetch_granularity(
                prepared.config.l2_fetch_granularity
            )
        try:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for call in range(calls):
                x, weight = self._inputs[(offset + call) % len(self._inputs)]
                prepared.runner(x, weight, prepared.out)
            end.record()
            end.synchronize()
            return float(start.elapsed_time(end)) / calls
        finally:
            if previous_l2 is not None:
                _set_l2_fetch_granularity(previous_l2)

    def calibrate_calls(self, prepared: _PreparedCandidate) -> tuple[int, float]:
        pilot_calls = self.protocol.min_calls_per_sample
        pilot_ms = self._time_batch(prepared, pilot_calls, 0)
        calls = math.ceil(self.protocol.target_batch_ms / max(pilot_ms, 1e-6))
        calls = min(
            self.protocol.max_calls_per_sample,
            max(self.protocol.min_calls_per_sample, calls),
        )
        return calls, pilot_ms

    def _time_callable(self, function: Callable[[int], None], calls: int) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for index in range(calls):
            function(index)
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end)) / calls

    def _measure_components(
        self, prepared: _PreparedCandidate, calls: int, samples: int
    ) -> dict[str, object]:
        """Measure kernels separately as diagnostics; E2E remains authoritative."""

        runner = prepared.runner
        component_calls = max(1, min(calls, 1024))
        component_samples = max(3, min(samples, 5))
        components: dict[str, list[float]] = {}
        if runner.quant_launches == "dual":
            components["dual_quant"] = [
                self._time_callable(
                    lambda index: runner.quant_x(
                        *self._inputs[index % len(self._inputs)],
                        runner.qx,
                        runner.qw,
                        runner.sx,
                        runner.sw,
                    ),
                    component_calls,
                )
                for _ in range(component_samples)
            ]
        else:
            components["x_quant"] = [
                self._time_callable(
                    lambda index: runner.quant_x(
                        self._inputs[index % len(self._inputs)][0], runner.qx, runner.sx
                    ),
                    component_calls,
                )
                for _ in range(component_samples)
            ]
            assert runner.quant_w is not None
            components["w_quant"] = [
                self._time_callable(
                    lambda index: runner.quant_w(
                        self._inputs[index % len(self._inputs)][1], runner.qw, runner.sw
                    ),
                    component_calls,
                )
                for _ in range(component_samples)
            ]
        components["gemm_hot_materialized"] = [
            self._time_callable(
                lambda _index: runner.gemm(
                    runner.qx, runner.qw, runner.sx, runner.sw, prepared.out
                ),
                component_calls,
            )
            for _ in range(component_samples)
        ]
        return {
            name: {
                "timings_ms": timings,
                "summary_ms": robust_summary(
                    timings,
                    seed=len(name) ^ calls,
                    bootstrap_resamples=self.protocol.bootstrap_resamples,
                ).as_dict(),
            }
            for name, timings in components.items()
        }

    def measure(
        self,
        config: MXFP8PrequantConfig,
        *,
        samples: int,
        seed: int,
        components: bool = False,
    ) -> dict[str, object]:
        telemetry_before = (
            _nvidia_smi_snapshot(self.device.index or torch.cuda.current_device())
            if self.protocol.telemetry
            else {"available": False, "disabled": True}
        )
        started = time.monotonic()
        try:
            prepared = self.prepare(config)
        except Exception as exc:
            if isinstance(exc, CandidateCompileError):
                status = "compile_error"
            elif isinstance(exc, CandidateCorrectnessError):
                status = "correctness_error"
            else:
                status = "runtime_error"
            return {
                "status": status,
                "error": f"{type(exc).__name__}: {exc}"[:4000],
                "elapsed_s": time.monotonic() - started,
                "telemetry_before": telemetry_before,
            }
        for index in range(self.protocol.warmup_calls):
            x, weight = self._inputs[index % len(self._inputs)]
            prepared.runner(x, weight, prepared.out)
        torch.cuda.synchronize(self.device)
        calls, pilot_ms = self.calibrate_calls(prepared)
        timings, collection = collect_stable_timing_samples(
            lambda sample: self._time_batch(prepared, calls, sample * calls),
            self.protocol,
            requested_samples=samples,
            stabilize=lambda attempt: stabilize_timing_batches(
                lambda batch: self._time_batch(
                    prepared,
                    calls,
                    -((attempt + 1) * self.protocol.stabilization_max_batches + batch)
                    * calls,
                )
                * calls,
                self.protocol,
            ),
            telemetry=(
                lambda: _nvidia_smi_snapshot(
                    self.device.index or torch.cuda.current_device()
                )
            )
            if self.protocol.telemetry
            else None,
        )
        telemetry_after = (
            _nvidia_smi_snapshot(self.device.index or torch.cuda.current_device())
            if self.protocol.telemetry
            else {"available": False, "disabled": True}
        )
        summary = robust_summary(
            timings,
            seed=seed,
            bootstrap_resamples=self.protocol.bootstrap_resamples,
        )
        component_results = (
            self._measure_components(prepared, calls, samples) if components else None
        )
        return {
            "status": "ok",
            "compile_ms": prepared.compile_ms,
            "max_abs_error": prepared.max_abs_error,
            "compiled_resources": prepared.compiled_resources,
            "calls_per_sample": calls,
            "pilot_ms_per_call": pilot_ms,
            "rotation_buffers": len(self._inputs),
            "timings_ms": timings,
            "sampling": {
                **self.protocol.sampling_metadata(
                    timings, requested_samples=samples
                ),
                "collection": collection,
            },
            "summary_ms": summary.as_dict(),
            "components": component_results,
            "elapsed_s": time.monotonic() - started,
            "telemetry_before": telemetry_before,
            "telemetry_after": telemetry_after,
        }

    def race(
        self,
        incumbent: MXFP8PrequantConfig,
        challenger: MXFP8PrequantConfig,
        *,
        seed: int,
    ) -> dict[str, object]:
        try:
            a = self.prepare(incumbent)
            b = self.prepare(challenger)
        except Exception as exc:
            return {"status": "prepare_error", "error": f"{type(exc).__name__}: {exc}"[:4000]}
        calls_a, _pilot_a = self.calibrate_calls(a)
        calls_b, _pilot_b = self.calibrate_calls(b)
        stabilization = stabilize_timing_batches(
            lambda batch: (
                self._time_batch(a, calls_a, -(batch + 1) * calls_a) * calls_a
                + self._time_batch(b, calls_b, -(batch + 1) * calls_b) * calls_b
            ),
            self.protocol,
        )
        a_times: list[float] = []
        b_times: list[float] = []
        for round_index in range(self.protocol.race_rounds):
            if round_index % 2:
                b_time = self._time_batch(b, calls_b, round_index * calls_b)
                a_time = self._time_batch(a, calls_a, round_index * calls_a)
            else:
                a_time = self._time_batch(a, calls_a, round_index * calls_a)
                b_time = self._time_batch(b, calls_b, round_index * calls_b)
            a_times.append(a_time)
            b_times.append(b_time)
            if self.protocol.race_complete(a_times, b_times):
                break
        speedups = [(a_time - b_time) / a_time for a_time, b_time in zip(a_times, b_times)]
        summary = robust_summary(
            speedups,
            seed=seed,
            bootstrap_resamples=self.protocol.bootstrap_resamples,
        )
        threshold = self.protocol.practical_threshold
        if summary.ci_low > threshold:
            decision = "challenger"
        elif summary.ci_high < -threshold:
            decision = "incumbent"
        else:
            decision = "tie"
        return {
            "status": "ok",
            "decision": decision,
            "practical_threshold": threshold,
            "incumbent_timings_ms": a_times,
            "challenger_timings_ms": b_times,
            "paired_speedup": summary.as_dict(),
            "incumbent_calls_per_sample": calls_a,
            "challenger_calls_per_sample": calls_b,
            "rotation_buffers": len(self._inputs),
            "sampling": self.protocol.race_sampling_metadata(a_times, b_times),
            "stabilization": stabilization,
        }


class ExperimentJournal:
    """Append-only JSONL journal with lock-protected writes and resumable keys."""

    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock:
            if fcntl is not None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    def append(self, record: Mapping[str, object]) -> None:
        payload = _canonical_json(dict(record)) + "\n"
        with self._locked():
            with self.path.open("a", encoding="utf-8") as sink:
                sink.write(payload)
                sink.flush()
                os.fsync(sink.fileno())

    def records(self) -> list[dict[str, object]]:
        if not self.path.exists():
            return []
        with self._locked():
            with self.path.open(encoding="utf-8") as source:
                return [json.loads(line) for line in source if line.strip()]

    def completed_keys(self) -> set[str]:
        return {
            str(record["observation_key"])
            for record in self.records()
            if record.get("record_type")
            in ("measurement", "verification_measurement", "race")
        }


def _flatten_mapping(
    prefix: str, value: Mapping[str, object], row: dict[str, object]
) -> None:
    for key, item in value.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(item, dict):
            _flatten_mapping(name, item, row)
        elif isinstance(item, (list, tuple)):
            row[name] = _canonical_json(item)
        else:
            row[name] = item


def _journal_records(paths: Iterable[Path | str]) -> Iterator[dict[str, object]]:
    for path_value in paths:
        path = Path(path_value)
        if not path.exists():
            continue
        with path.open(encoding="utf-8") as source:
            for line in source:
                if line.strip():
                    record = json.loads(line)
                    record.setdefault("source_journal", str(path))
                    yield record


def merge_journals(
    journals: Iterable[Path | str], destination: Path | str
) -> dict[str, int]:
    """Merge copied device/shard journals without losing provenance."""

    unique: dict[tuple[object, ...], dict[str, object]] = {}
    sessions: list[dict[str, object]] = []
    duplicates = 0
    for record in _journal_records(journals):
        if record.get("record_type") in ("measurement", "race"):
            # Device ID is part of the identity for older journals whose
            # observation key predated cross-device merging.
            identity = (
                record.get("device_id"),
                record.get("observation_key"),
            )
            if identity in unique:
                duplicates += 1
                continue
            unique[identity] = record
        else:
            sessions.append(record)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=destination.name + ".",
            suffix=".tmp",
            delete=False,
        ) as sink:
            temporary = sink.name
            for record in sessions:
                sink.write(_canonical_json(record) + "\n")
            for identity in sorted(unique, key=lambda value: tuple(map(str, value))):
                sink.write(_canonical_json(unique[identity]) + "\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, destination)
        temporary = None
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)
    return {
        "sessions": len(sessions),
        "observations": len(unique),
        "duplicates": duplicates,
    }


def export_journals_csv(
    journals: Iterable[Path | str], destination: Path | str
) -> None:
    """Export one or more journals to a flat, model-training-friendly CSV."""

    records = list(_journal_records(journals))
    rows: list[dict[str, object]] = []
    for record in records:
        if record.get("record_type") not in ("measurement", "race"):
            continue
        outcome = record.get("outcome", {})
        summary = outcome.get("summary_ms", {}) if isinstance(outcome, dict) else {}
        row = {
            "record_type": record.get("record_type"),
            "observation_key": record.get("observation_key"),
            "recorded_at": record.get("recorded_at"),
            "device_id": record.get("device_id"),
            "shape": record.get("shape_key"),
            "regime": record.get("regime"),
            "stage": record.get("stage"),
            "config_id": record.get("config_id"),
            "incumbent_id": record.get("incumbent_id"),
            "challenger_id": record.get("challenger_id"),
            "status": outcome.get("status") if isinstance(outcome, dict) else None,
            "median_ms": summary.get("median") if isinstance(summary, dict) else None,
            "ci_low_ms": summary.get("ci_low") if isinstance(summary, dict) else None,
            "ci_high_ms": summary.get("ci_high") if isinstance(summary, dict) else None,
            "decision": outcome.get("decision") if isinstance(outcome, dict) else None,
            "manifest_digest": record.get("manifest_digest"),
            "run_id": record.get("run_id"),
            "source_journal": record.get("source_journal"),
        }
        for prefix, field in (
            ("feature", "features"),
            ("config", "config"),
            ("shape_value", "shape"),
            ("device", "device_properties"),
            ("protocol", "protocol"),
        ):
            values = record.get(field, {})
            if isinstance(values, dict):
                _flatten_mapping(prefix, values, row)
        if isinstance(outcome, dict):
            row["timings_ms_json"] = _canonical_json(outcome.get("timings_ms", []))
            for prefix, field in (
                ("telemetry_before", "telemetry_before"),
                ("telemetry_after", "telemetry_after"),
            ):
                values = outcome.get(field, {})
                if isinstance(values, dict):
                    _flatten_mapping(prefix, values, row)
        rows.append(row)
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with destination.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def export_journal_csv(journal: Path | str, destination: Path | str) -> None:
    export_journals_csv((journal,), destination)


def analyze_observations(
    records: Iterable[Mapping[str, object]], *, tolerance: float = 0.01
) -> dict[str, object]:
    """Build empirical winners and a greedy near-optimal config portfolio."""

    if not 0 <= tolerance < 1:
        raise ValueError("tolerance must be in [0, 1)")
    measurements = [
        dict(record)
        for record in records
        if record.get("record_type") == "measurement"
        and isinstance(record.get("outcome"), dict)
        and record["outcome"].get("status") == "ok"  # type: ignore[union-attr]
    ]
    context_records: dict[tuple[str, str, str], list[dict[str, object]]] = {}
    for record in measurements:
        context = (
            str(record.get("device_id")),
            str(record.get("shape_key")),
            str(record.get("regime")),
        )
        context_records.setdefault(context, []).append(record)
    contexts: dict[tuple[str, str, str], dict[str, object]] = {}
    performance: dict[str, dict[tuple[str, str, str], float]] = {}
    configs: dict[str, object] = {}
    for context, values in context_records.items():
        confirmed = [value for value in values if value.get("stage") == "confirm"]
        selected = confirmed or [value for value in values if value.get("stage") == "screen"]
        by_config: dict[str, list[float]] = {}
        weight = 1.0
        for record in selected:
            config_id = str(record.get("config_id"))
            outcome = record["outcome"]
            summary = outcome.get("summary_ms", {})  # type: ignore[union-attr]
            if not isinstance(summary, dict) or summary.get("median") is None:
                continue
            by_config.setdefault(config_id, []).append(float(summary["median"]))
            configs[config_id] = record.get("config")
            shape = record.get("shape", {})
            if isinstance(shape, dict):
                weight = float(shape.get("weight", 1.0))
        if not by_config:
            continue
        medians = {
            config_id: float(statistics.median(samples))
            for config_id, samples in by_config.items()
        }
        best_id = min(medians, key=medians.__getitem__)
        best_ms = medians[best_id]
        contexts[context] = {
            "best_config_id": best_id,
            "best_ms": best_ms,
            "weight": weight,
            "near_optimal": sorted(
                config_id
                for config_id, latency in medians.items()
                if latency <= best_ms * (1 + tolerance)
            ),
        }
        for config_id, latency in medians.items():
            performance.setdefault(config_id, {})[context] = latency

    uncovered = set(contexts)
    portfolio: list[dict[str, object]] = []
    while uncovered:
        candidates = []
        for config_id, measured in performance.items():
            covered = {
                context
                for context in uncovered
                if context in measured
                and measured[context]
                <= float(contexts[context]["best_ms"]) * (1 + tolerance)
            }
            if not covered:
                continue
            weight = sum(float(contexts[context]["weight"]) for context in covered)
            mean_regret = statistics.fmean(
                measured[context] / float(contexts[context]["best_ms"]) - 1
                for context in covered
            )
            candidates.append((weight, -mean_regret, config_id, covered))
        if not candidates:
            break
        weight, negative_regret, config_id, covered = max(candidates)
        portfolio.append(
            {
                "config_id": config_id,
                "config": configs.get(config_id),
                "new_contexts": len(covered),
                "new_weight": weight,
                "mean_regret": -negative_regret,
                "contexts": ["/".join(context) for context in sorted(covered)],
            }
        )
        uncovered.difference_update(covered)

    return {
        "tolerance": tolerance,
        "context_count": len(contexts),
        "portfolio_size": len(portfolio),
        "uncovered_contexts": ["/".join(context) for context in sorted(uncovered)],
        "portfolio": portfolio,
        "contexts": {
            "/".join(context): value for context, value in sorted(contexts.items())
        },
    }


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=path.name + ".", suffix=".tmp", delete=False
        ) as sink:
            temporary = sink.name
            json.dump(value, sink, indent=2, sort_keys=True)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None and os.path.exists(temporary):
            os.unlink(temporary)


class PrequantExperimentCampaign:
    def __init__(
        self,
        manifest: ExperimentManifest,
        output_dir: Path | str,
        *,
        device: torch.device | str = "cuda",
        progress: Progress | None = print,
    ) -> None:
        self.manifest = manifest
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)
        self.progress = progress
        self.fingerprint = DeviceFingerprint.current(self.device)
        if self.fingerprint.capability[0] != 12:
            raise RuntimeError(
                "this native MXFP8 campaign currently requires SM120/SM121; "
                f"got capability {self.fingerprint.capability} on "
                f"{self.fingerprint.name}"
            )
        self.device_properties = _device_properties(self.device)
        directory = self.output_dir / manifest.name / self.fingerprint.identifier
        self.directory = directory
        self.journal = ExperimentJournal(
            directory
            / f"shard-{manifest.shard_index:03d}-of-{manifest.shard_count:03d}.jsonl"
        )
        self.run_id = uuid.uuid4().hex

    def _log(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def _base_record(self, shape: ShapeSpec, regime: CacheRegime) -> dict[str, object]:
        return {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "manifest_digest": self.manifest.digest,
            "run_id": self.run_id,
            "device_id": self.fingerprint.identifier,
            "device": self.fingerprint.as_dict(),
            "device_properties": self.device_properties,
            "shape_key": shape.key,
            "shape": asdict(shape),
            "regime": regime,
            "recorded_at": _utc_now(),
        }

    def _measurement_key(
        self, shape: ShapeSpec, regime: CacheRegime, stage: str, config_id: str
    ) -> str:
        return _digest(
            {
                "manifest": self.manifest.digest,
                "shape": shape.key,
                "regime": regime,
                "stage": stage,
                "config": config_id,
                "shard": self.manifest.shard_index,
                "device": self.fingerprint.identifier,
            }
        )

    def _record_measurement(
        self,
        harness_factory: Callable[[], PrequantBenchmarkHarness],
        shape: ShapeSpec,
        regime: CacheRegime,
        stage: str,
        config: MXFP8PrequantConfig,
        samples: int,
        completed: set[str],
    ) -> dict[str, object] | None:
        config_id = prequant_config_id(config)
        key = self._measurement_key(shape, regime, stage, config_id)
        if key in completed:
            return None
        harness = harness_factory()
        self._log(f"MEASURE {shape.key} {regime} {stage} {config_id}")
        outcome = harness.measure(
            config,
            samples=samples,
            seed=self.manifest.seed ^ int(config_id[:8], 16),
            components=stage == "confirm",
        )
        record = {
            **self._base_record(shape, regime),
            "record_type": "measurement",
            "observation_key": key,
            "stage": stage,
            "config_id": config_id,
            "config": prequant_config_to_dict(config),
            "features": derived_features(
                shape,
                config,
                self.fingerprint,
                l2_cache_size=self.device_properties.get("l2_cache_size"),  # type: ignore[arg-type]
            ),
            "protocol": asdict(self.manifest.protocol),
            "outcome": outcome,
        }
        self.journal.append(record)
        completed.add(key)
        return record

    def _records_for(
        self, shape: ShapeSpec, regime: CacheRegime, stage: str
    ) -> list[dict[str, object]]:
        return [
            record
            for record in self.journal.records()
            if record.get("record_type") == "measurement"
            and record.get("shape_key") == shape.key
            and record.get("regime") == regime
            and record.get("stage") == stage
            and record.get("manifest_digest") == self.manifest.digest
        ]

    @staticmethod
    def _successful_rank(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
        successful = []
        for record in records:
            outcome = record.get("outcome")
            if not isinstance(outcome, dict) or outcome.get("status") != "ok":
                continue
            summary = outcome.get("summary_ms")
            if not isinstance(summary, dict) or summary.get("median") is None:
                continue
            successful.append(dict(record))
        return sorted(
            successful,
            key=lambda record: float(record["outcome"]["summary_ms"]["median"]),  # type: ignore[index]
        )

    def _race_key(
        self,
        shape: ShapeSpec,
        regime: CacheRegime,
        incumbent_id: str,
        challenger_id: str,
    ) -> str:
        return _digest(
            {
                "manifest": self.manifest.digest,
                "shape": shape.key,
                "regime": regime,
                "stage": "race",
                "incumbent": incumbent_id,
                "challenger": challenger_id,
                "shard": self.manifest.shard_index,
                "device": self.fingerprint.identifier,
            }
        )

    def _run_shape_regime(
        self,
        shape: ShapeSpec,
        regime: CacheRegime,
        completed: set[str],
    ) -> dict[str, object]:
        catalog = generate_legal_catalog(
            shape.problem,
            self.manifest.candidates_per_shape,
            seed=self.manifest.seed ^ int(_digest(shape.key, 8), 16),
        )
        catalog = [
            config
            for config in catalog
            if config_in_shard(
                config, self.manifest.shard_index, self.manifest.shard_count
            )
        ]
        if not catalog:
            return {"shape": shape.key, "regime": regime, "status": "empty_shard"}
        harness: PrequantBenchmarkHarness | None = None

        def get_harness() -> PrequantBenchmarkHarness:
            nonlocal harness
            if harness is None:
                harness = PrequantBenchmarkHarness(
                    shape,
                    regime,
                    self.manifest.protocol,
                    device=self.device,
                    seed=self.manifest.seed
                    ^ int(_digest((shape.key, regime), 8), 16),
                )
            return harness

        for config in catalog:
            self._record_measurement(
                get_harness,
                shape,
                regime,
                "screen",
                config,
                self.manifest.protocol.samples,
                completed,
            )
        screen = self._successful_rank(self._records_for(shape, regime, "screen"))
        finalists = screen[: self.manifest.promote]
        for record in finalists:
            config = prequant_config_from_dict(record["config"])  # type: ignore[arg-type]
            self._record_measurement(
                get_harness,
                shape,
                regime,
                "confirm",
                config,
                self.manifest.protocol.confirm_samples,
                completed,
            )
        confirmed = self._successful_rank(self._records_for(shape, regime, "confirm"))
        if not confirmed:
            return {"shape": shape.key, "regime": regime, "status": "no_success"}
        incumbent = prequant_config_from_dict(confirmed[0]["config"])  # type: ignore[arg-type]
        races: list[dict[str, object]] = []
        prior_races = {
            str(record.get("observation_key")): record
            for record in self.journal.records()
            if record.get("record_type") == "race"
        }
        for record in confirmed[1:]:
            challenger = prequant_config_from_dict(record["config"])  # type: ignore[arg-type]
            incumbent_id = prequant_config_id(incumbent)
            challenger_id = prequant_config_id(challenger)
            key = self._race_key(shape, regime, incumbent_id, challenger_id)
            if key in completed:
                previous = prior_races.get(key, {})
                previous_outcome = previous.get("outcome", {})
                if (
                    isinstance(previous_outcome, dict)
                    and previous_outcome.get("decision") == "challenger"
                ):
                    incumbent = challenger
                continue
            self._log(
                f"RACE    {shape.key} {regime} {incumbent_id} vs {challenger_id}"
            )
            outcome = get_harness().race(
                incumbent,
                challenger,
                seed=self.manifest.seed ^ int(key[:8], 16),
            )
            race_record = {
                **self._base_record(shape, regime),
                "record_type": "race",
                "observation_key": key,
                "stage": "race",
                "incumbent_id": incumbent_id,
                "challenger_id": challenger_id,
                "incumbent_config": prequant_config_to_dict(incumbent),
                "challenger_config": prequant_config_to_dict(challenger),
                "outcome": outcome,
            }
            self.journal.append(race_record)
            completed.add(key)
            races.append(race_record)
            if outcome.get("decision") == "challenger":
                incumbent = challenger
        winner_id = prequant_config_id(incumbent)
        return {
            "shape": shape.key,
            "regime": regime,
            "status": "ok",
            "catalog_size": len(catalog),
            "screen_successes": len(screen),
            "confirmed": len(confirmed),
            "new_races": len(races),
            "winner_id": winner_id,
            "winner_config": prequant_config_to_dict(incumbent),
        }

    def run(self) -> Path:
        self.directory.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.directory / "manifest.json", self.manifest.as_dict())
        completed = self.journal.completed_keys()
        self.journal.append(
            {
                "schema_version": EXPERIMENT_SCHEMA_VERSION,
                "record_type": "session",
                "event": "started",
                "recorded_at": _utc_now(),
                "run_id": self.run_id,
                "manifest_digest": self.manifest.digest,
                "device_id": self.fingerprint.identifier,
                "device": self.fingerprint.as_dict(),
                "device_properties": self.device_properties,
            }
        )
        results = []
        status = "complete"
        try:
            for shape in self.manifest.shapes:
                for regime in self.manifest.regimes:
                    results.append(
                        self._run_shape_regime(shape, regime, completed)
                    )
        except Exception:
            status = "failed"
            raise
        finally:
            self.journal.append(
                {
                    "schema_version": EXPERIMENT_SCHEMA_VERSION,
                    "record_type": "session",
                    "event": "finished",
                    "status": status,
                    "recorded_at": _utc_now(),
                    "run_id": self.run_id,
                    "manifest_digest": self.manifest.digest,
                    "device_id": self.fingerprint.identifier,
                }
            )
            _atomic_json(
                self.directory
                / f"summary-shard-{self.manifest.shard_index:03d}.json",
                {
                    "schema_version": EXPERIMENT_SCHEMA_VERSION,
                    "run_id": self.run_id,
                    "manifest_digest": self.manifest.digest,
                    "status": status,
                    "device": self.fingerprint.as_dict(),
                    "device_properties": self.device_properties,
                    "journal": str(self.journal.path),
                    "results": results,
                },
            )
        return self.journal.path


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("autotune_results/experiments")
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--export-csv", type=Path)
    parser.add_argument(
        "--merge-root",
        type=Path,
        help="merge all JSONL journals recursively under this directory",
    )
    parser.add_argument("--merged-jsonl", type=Path)
    parser.add_argument("--analysis-json", type=Path)
    parser.add_argument("--portfolio-tolerance", type=float, default=0.01)
    parser.add_argument(
        "--probe",
        action="store_true",
        help="print device/campaign capabilities without running kernels",
    )
    args = parser.parse_args()
    if args.merge_root is not None:
        journals = sorted(args.merge_root.rglob("*.jsonl"))
        if not journals:
            parser.error(f"no JSONL journals under {args.merge_root}")
        merged = args.merged_jsonl or args.output_dir / "merged.jsonl"
        report = merge_journals(journals, merged)
        if args.export_csv is not None:
            export_journals_csv((merged,), args.export_csv)
        if args.analysis_json is not None:
            analysis = analyze_observations(
                _journal_records((merged,)), tolerance=args.portfolio_tolerance
            )
            _atomic_json(args.analysis_json, analysis)
        print(
            json.dumps(
                {**report, "journals": len(journals), "merged_jsonl": str(merged)},
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.probe:
        if not torch.cuda.is_available():
            parser.error("CUDA is not available")
        print(json.dumps(probe_device(args.device), indent=2, sort_keys=True))
        return
    if args.manifest is None:
        parser.error("--manifest is required unless --merge-root is used")
    manifest = ExperimentManifest.load(args.manifest)
    if args.dry_run:
        report = []
        for shape in manifest.shapes:
            catalog = generate_legal_catalog(
                shape.problem,
                manifest.candidates_per_shape,
                seed=manifest.seed ^ int(_digest(shape.key, 8), 16),
            )
            shard = [
                config
                for config in catalog
                if config_in_shard(config, manifest.shard_index, manifest.shard_count)
            ]
            report.append(
                {
                    "shape": shape.key,
                    "legal_catalog": len(catalog),
                    "this_shard": len(shard),
                }
            )
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if not torch.cuda.is_available():
        parser.error("CUDA is not available")
    campaign = PrequantExperimentCampaign(
        manifest,
        args.output_dir,
        device=args.device,
        progress=None if args.quiet else print,
    )
    journal = campaign.run()
    if args.export_csv is not None:
        export_journal_csv(journal, args.export_csv)
    print(json.dumps({"journal": str(journal)}, indent=2))


__all__ = [
    "BenchmarkProtocol",
    "CandidateCompileError",
    "CandidateCorrectnessError",
    "ExperimentJournal",
    "ExperimentManifest",
    "PrequantBenchmarkHarness",
    "PrequantExperimentCampaign",
    "RobustSummary",
    "ShapeSpec",
    "config_in_shard",
    "collect_stable_timing_samples",
    "collect_timing_samples",
    "analyze_observations",
    "derived_features",
    "export_journal_csv",
    "export_journals_csv",
    "generate_legal_catalog",
    "merge_journals",
    "probe_device",
    "robust_summary",
    "stabilize_timing_batches",
]


if __name__ == "__main__":
    _cli()
