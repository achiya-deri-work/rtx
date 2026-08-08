"""Portable, resumable cross-device autotuning dataset campaigns.

The JSONL journals in a campaign bundle are the durable source of truth. CSV
and Parquet files are deterministic, normalized exports that can be rebuilt
after copying bundles from any number of machines.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import socket
import statistics
import subprocess
import sys
import time
from typing import Callable, Iterable, Literal, Mapping, Sequence

import torch

from .adapters import (
    make_mxfp8_bwd_adapter,
    make_mxfp8_fully_prequant_adapter,
    make_mxfp8_fwd_adapter,
    make_mxfp8_prequant_adapter,
    make_mxfp8_weight_prequant_adapter,
)
from .core import KernelAdapter, canonical_json, stable_id
from .evaluators import CalibratedBwdEvaluator, CalibratedPrequantEvaluator
from .hardware import compiled_resource_metadata
from .legacy import DeviceFingerprint
from .recipes import HybridTuningPolicy, make_hybrid_autotuner
from .store import JsonlTuningStore
from .winners import runtime_winner_key, save_runtime_winner
from ..bwd_experiments import BwdBenchmarkHarness
from ..kernels.mxfp8 import (
    DEFAULT_MXFP8_FWD_CONFIG,
    MXFP8FwdConfig,
    MXFP8Problem,
)
from ..runtime import load_kernel_symbol
from ..prequant_experiments import (
    BenchmarkProtocol,
    CacheRegime,
    ExperimentJournal,
    PrequantBenchmarkHarness,
    ShapeSpec,
    _atomic_json,
    _device_properties,
    _nvidia_smi_snapshot,
    _reference_prequant_config,
    probe_device,
    robust_summary,
)
from ..inference_experiments import (
    FullyPrequantBenchmarkHarness,
    WeightPrequantBenchmarkHarness,
)


DATASET_SCHEMA_VERSION = 2
KernelFamily = str
ExportFormat = Literal["csv", "parquet", "both", "none"]


def compile_mxfp8_fwd(*args, **kwargs):
    return load_kernel_symbol("mxfp8_fwd", "compile_mxfp8_fwd")(*args, **kwargs)


@dataclass(frozen=True, slots=True)
class AnytimeRunPolicy:
    """Breadth-first campaign scheduling for useful preemptible runs."""

    wall_time_s: float
    context_slice_s: float = 120.0
    trial_milestones: tuple[int, ...] = (32, 96, 192, 384, 512)
    initial_promote: int = 2

    def __post_init__(self) -> None:
        if self.wall_time_s <= 0 or self.context_slice_s <= 0:
            raise ValueError("anytime wall time and context slice must be positive")
        if not self.trial_milestones or any(value <= 0 for value in self.trial_milestones):
            raise ValueError("anytime trial milestones must be positive")
        if tuple(sorted(set(self.trial_milestones))) != self.trial_milestones:
            raise ValueError("anytime trial milestones must be unique and increasing")
        if self.initial_promote <= 0:
            raise ValueError("anytime initial promotion count must be positive")


@dataclass(frozen=True, slots=True)
class DatasetBackend:
    """Pluggable bridge from a manifest family to its harness and adapter."""

    make_harness: Callable[["DatasetCampaign", "DatasetJob", ShapeSpec, CacheRegime], object]
    make_adapter: Callable[
        ["DatasetCampaign", "DatasetJob", ShapeSpec, CacheRegime, object, Mapping[str, object]],
        KernelAdapter,
    ]


_BACKENDS: dict[str, DatasetBackend] = {}


def register_dataset_backend(
    family: str,
    backend: DatasetBackend,
    *,
    replace_existing: bool = False,
) -> None:
    """Register another kernel family without changing campaign orchestration."""

    if not family or family in _BACKENDS and not replace_existing:
        raise ValueError(f"dataset backend already registered or invalid: {family!r}")
    _BACKENDS[family] = backend


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _digest(value: object, length: int = 24) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()[:length]


def _policy_from_dict(value: Mapping[str, object] | None) -> HybridTuningPolicy:
    return HybridTuningPolicy() if value is None else replace(
        HybridTuningPolicy(), **dict(value)
    )


@dataclass(frozen=True, slots=True)
class DatasetJob:
    """One kernel family evaluated over shapes and cache regimes."""

    family: KernelFamily
    shapes: tuple[ShapeSpec, ...]
    regimes: tuple[CacheRegime, ...] = ("hot", "rotate")
    tuning: HybridTuningPolicy = HybridTuningPolicy()
    protocol: BenchmarkProtocol = BenchmarkProtocol()
    promote: int = 4
    tags: Mapping[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.family not in _BACKENDS:
            raise ValueError(f"unsupported kernel family: {self.family}")
        if not self.shapes:
            raise ValueError("dataset job requires at least one shape")
        if not self.regimes or any(regime not in ("hot", "rotate") for regime in self.regimes):
            raise ValueError("regimes must contain hot and/or rotate")
        if self.promote <= 0:
            raise ValueError("promote must be positive")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "DatasetJob":
        return cls(
            family=str(value["family"]),  # type: ignore[arg-type]
            shapes=tuple(
                ShapeSpec.from_dict(item)
                for item in value["shapes"]  # type: ignore[union-attr]
            ),
            regimes=tuple(value.get("regimes", ("hot", "rotate"))),  # type: ignore[arg-type]
            tuning=_policy_from_dict(value.get("tuning")),  # type: ignore[arg-type]
            protocol=BenchmarkProtocol.from_dict(value.get("protocol")),  # type: ignore[arg-type]
            promote=int(value.get("promote", 4)),
            tags=dict(value.get("tags", {})),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class DatasetManifest:
    name: str
    jobs: tuple[DatasetJob, ...]
    seed: int = 0
    shard_index: int = 0
    shard_count: int = 1

    def __post_init__(self) -> None:
        if not self.name or any(part in self.name for part in ("/", "\\", "..")):
            raise ValueError("manifest name must be a safe path component")
        if not self.jobs:
            raise ValueError("dataset manifest requires at least one job")
        if self.shard_count <= 0 or not 0 <= self.shard_index < self.shard_count:
            raise ValueError("invalid shard index/count")

    @classmethod
    def from_dict(cls, value: Mapping[str, object]) -> "DatasetManifest":
        version = int(value.get("schema_version", DATASET_SCHEMA_VERSION))
        if version not in (1, DATASET_SCHEMA_VERSION):
            raise ValueError(f"unsupported dataset manifest schema {version}")
        return cls(
            name=str(value["name"]),
            jobs=tuple(
                DatasetJob.from_dict(item)
                for item in value["jobs"]  # type: ignore[union-attr]
            ),
            seed=int(value.get("seed", 0)),
            shard_index=int(value.get("shard_index", 0)),
            shard_count=int(value.get("shard_count", 1)),
        )

    @classmethod
    def load(cls, path: Path | str) -> "DatasetManifest":
        with Path(path).open(encoding="utf-8") as source:
            return cls.from_dict(json.load(source))

    def as_dict(self) -> dict[str, object]:
        return {"schema_version": DATASET_SCHEMA_VERSION, **asdict(self)}

    @property
    def digest(self) -> str:
        return _digest(self.as_dict())

    def with_shard(self, index: int, count: int) -> "DatasetManifest":
        return replace(self, shard_index=index, shard_count=count)


@dataclass(slots=True)
class _PreparedFused:
    config: MXFP8FwdConfig
    launcher: object
    out: torch.Tensor
    compile_ms: float
    max_abs_error: float
    compiled_resources: Mapping[str, object]


class FusedCandidateCompileError(RuntimeError):
    pass


class FusedCandidateCorrectnessError(RuntimeError):
    pass


class FusedFwdBenchmarkHarness:
    """Calibrated benchmark and AB/BA race harness for fused BF16->MXFP8."""

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
            shape.m, shape.k, dtype=torch.bfloat16, device=self.device, generator=generator
        )
        self.weight = torch.randn(
            shape.n, shape.k, dtype=torch.bfloat16, device=self.device, generator=generator
        )
        self._inputs = self._make_input_ring()
        baseline = compile_mxfp8_fwd(self.problem, DEFAULT_MXFP8_FWD_CONFIG)
        expected = torch.empty(
            (shape.m, shape.n), dtype=torch.bfloat16, device=self.device
        )
        baseline(self.x, self.weight, expected)
        torch.cuda.synchronize(self.device)
        self._expected = expected.clone()

    def _l2_bytes(self) -> int:
        props = torch.cuda.get_device_properties(self.device)
        return int(
            getattr(props, "L2_cache_size", getattr(props, "l2_cache_size", 0)) or 0
        )

    def _make_input_ring(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self.regime == "hot":
            return [(self.x, self.weight)]
        input_bytes = sum(
            tensor.numel() * tensor.element_size() for tensor in (self.x, self.weight)
        )
        free_bytes, _ = torch.cuda.mem_get_info(self.device)
        budget = min(self.protocol.max_rotation_bytes, int(free_bytes * 0.35))
        target = max(
            input_bytes * 2,
            int(self._l2_bytes() * self.protocol.rotation_l2_multiple),
        )
        count = max(
            2,
            min(
                self.protocol.max_rotation_buffers,
                max(1, budget // input_bytes),
                math.ceil(target / input_bytes),
            ),
        )
        ring = [(self.x, self.weight)]
        ring.extend((self.x.clone(), self.weight.clone()) for _ in range(1, count))
        torch.cuda.synchronize(self.device)
        return ring

    def prepare(self, config: MXFP8FwdConfig) -> _PreparedFused:
        reason = config.architecture_rejection(self.problem)
        if reason is None:
            reason = config.implementation_rejection(self.problem)
        if reason is not None:
            raise RuntimeError(reason)
        started = time.monotonic()
        try:
            launcher = compile_mxfp8_fwd(self.problem, config)
        except Exception as exc:
            raise FusedCandidateCompileError(f"{type(exc).__name__}: {exc}") from exc
        compile_ms = (time.monotonic() - started) * 1000
        out = torch.empty_like(self._expected)
        launcher(self.x, self.weight, out)
        torch.cuda.synchronize(self.device)
        max_error = float((out.float() - self._expected.float()).abs().max())
        if not torch.allclose(
            out,
            self._expected,
            rtol=self.protocol.correctness_rtol,
            atol=self.protocol.correctness_atol,
            equal_nan=True,
        ):
            raise FusedCandidateCorrectnessError(
                f"candidate differs from baseline (max abs {max_error})"
            )
        return _PreparedFused(
            config,
            launcher,
            out,
            compile_ms,
            max_error,
            compiled_resource_metadata(launcher),
        )

    def _time_batch(self, prepared: _PreparedFused, calls: int, offset: int) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for call in range(calls):
            x, weight = self._inputs[(offset + call) % len(self._inputs)]
            prepared.launcher(x, weight, prepared.out)
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end)) / calls

    def calibrate_calls(self, prepared: _PreparedFused) -> tuple[int, float]:
        pilot = self._time_batch(prepared, self.protocol.min_calls_per_sample, 0)
        calls = math.ceil(self.protocol.target_batch_ms / max(pilot, 1e-6))
        return (
            min(
                self.protocol.max_calls_per_sample,
                max(self.protocol.min_calls_per_sample, calls),
            ),
            pilot,
        )

    def measure(
        self,
        config: MXFP8FwdConfig,
        *,
        samples: int,
        seed: int,
        components: bool = False,
    ) -> dict[str, object]:
        del components
        telemetry_before = (
            _nvidia_smi_snapshot(self.device.index or torch.cuda.current_device())
            if self.protocol.telemetry
            else {"available": False, "disabled": True}
        )
        started = time.monotonic()
        try:
            prepared = self.prepare(config)
        except Exception as exc:
            if isinstance(exc, FusedCandidateCompileError):
                status = "compile_error"
            elif isinstance(exc, FusedCandidateCorrectnessError):
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
            prepared.launcher(x, weight, prepared.out)
        torch.cuda.synchronize(self.device)
        calls, pilot = self.calibrate_calls(prepared)
        timings = [
            self._time_batch(prepared, calls, sample * calls)
            for sample in range(samples)
        ]
        telemetry_after = (
            _nvidia_smi_snapshot(self.device.index or torch.cuda.current_device())
            if self.protocol.telemetry
            else {"available": False, "disabled": True}
        )
        return {
            "status": "ok",
            "compile_ms": prepared.compile_ms,
            "max_abs_error": prepared.max_abs_error,
            "compiled_resources": prepared.compiled_resources,
            "calls_per_sample": calls,
            "pilot_ms_per_call": pilot,
            "rotation_buffers": len(self._inputs),
            "timings_ms": timings,
            "summary_ms": robust_summary(
                timings,
                seed=seed,
                bootstrap_resamples=self.protocol.bootstrap_resamples,
            ).as_dict(),
            "elapsed_s": time.monotonic() - started,
            "telemetry_before": telemetry_before,
            "telemetry_after": telemetry_after,
        }

    def race(
        self, incumbent: MXFP8FwdConfig, challenger: MXFP8FwdConfig, *, seed: int
    ) -> dict[str, object]:
        try:
            a, b = self.prepare(incumbent), self.prepare(challenger)
        except Exception as exc:
            return {"status": "prepare_error", "error": f"{type(exc).__name__}: {exc}"[:4000]}
        calls_a, _ = self.calibrate_calls(a)
        calls_b, _ = self.calibrate_calls(b)
        a_times: list[float] = []
        b_times: list[float] = []
        for index in range(self.protocol.race_rounds):
            if index % 2:
                bt = self._time_batch(b, calls_b, index * calls_b)
                at = self._time_batch(a, calls_a, index * calls_a)
            else:
                at = self._time_batch(a, calls_a, index * calls_a)
                bt = self._time_batch(b, calls_b, index * calls_b)
            a_times.append(at)
            b_times.append(bt)
        speedups = [(a - b) / a for a, b in zip(a_times, b_times)]
        summary = robust_summary(
            speedups,
            seed=seed,
            bootstrap_resamples=self.protocol.bootstrap_resamples,
        )
        threshold = self.protocol.practical_threshold
        decision = (
            "challenger"
            if summary.ci_low > threshold
            else "incumbent"
            if summary.ci_high < -threshold
            else "tie"
        )
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
        }


def _source_snapshot() -> dict[str, object]:
    root = Path(__file__).resolve().parents[2]
    digest = hashlib.sha256()
    for path in sorted((root / "rtx").rglob("*.py")):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    result: dict[str, object] = {"python_source_sha256": digest.hexdigest()}
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        result["git_commit"] = completed.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
        result["git_dirty"] = bool(dirty.stdout.strip())
    except (FileNotFoundError, subprocess.SubprocessError):
        result["git_commit"] = None
        result["git_dirty"] = None
    return result


def machine_snapshot(
    device: torch.device | str = "cuda",
    *,
    calibration: Mapping[str, object] | None = None,
) -> dict[str, object]:
    device_report = probe_device(device, calibration=calibration)
    source = _source_snapshot()
    identity = {
        "hostname": socket.gethostname(),
        "device_id": device_report["fingerprint_id"],
        "platform": platform.platform(),
        "dataset_schema_version": DATASET_SCHEMA_VERSION,
        "hardware_profile": device_report.get("hardware_profile"),
    }
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "machine_id": _digest(identity),
        "recorded_at": _utc_now(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "device": device_report,
        "source": source,
    }


def _assigned(manifest: DatasetManifest, family: str, shape: ShapeSpec, regime: str) -> bool:
    value = int(_digest((family, shape.key, regime), 16), 16)
    return value % manifest.shard_count == manifest.shard_index


def _backend_seed(
    campaign: "DatasetCampaign", job: DatasetJob, shape: ShapeSpec, regime: str
) -> int:
    return campaign.manifest.seed ^ int(
        _digest((job.family, shape.key, regime), 8), 16
    )


def _fused_harness(
    campaign: "DatasetCampaign", job: DatasetJob, shape: ShapeSpec, regime: CacheRegime
):
    return FusedFwdBenchmarkHarness(
        shape,
        regime,
        job.protocol,
        device=campaign.device,
        seed=_backend_seed(campaign, job, shape, regime),
    )


def _prequant_harness(
    campaign: "DatasetCampaign", job: DatasetJob, shape: ShapeSpec, regime: CacheRegime
):
    return PrequantBenchmarkHarness(
        shape,
        regime,
        job.protocol,
        device=campaign.device,
        seed=_backend_seed(campaign, job, shape, regime),
    )


def _bwd_harness(
    campaign: "DatasetCampaign", job: DatasetJob, shape: ShapeSpec, regime: CacheRegime
):
    return BwdBenchmarkHarness(
        shape,
        job.protocol,
        regime=regime,
        device=campaign.device,
        seed=_backend_seed(campaign, job, shape, regime),
    )


def _weight_prequant_harness(
    campaign: "DatasetCampaign", job: DatasetJob, shape: ShapeSpec, regime: CacheRegime
):
    return WeightPrequantBenchmarkHarness(
        shape,
        regime,
        job.protocol,
        device=campaign.device,
        seed=_backend_seed(campaign, job, shape, regime),
    )


def _fully_prequant_harness(
    campaign: "DatasetCampaign", job: DatasetJob, shape: ShapeSpec, regime: CacheRegime
):
    return FullyPrequantBenchmarkHarness(
        shape,
        regime,
        job.protocol,
        device=campaign.device,
        seed=_backend_seed(campaign, job, shape, regime),
    )


def _fused_adapter(
    campaign: "DatasetCampaign",
    job: DatasetJob,
    shape: ShapeSpec,
    regime: CacheRegime,
    harness,
    tags: Mapping[str, object],
) -> KernelAdapter:
    evaluator = CalibratedPrequantEvaluator(
        harness, samples=job.protocol.samples, seed=campaign.manifest.seed
    )
    return make_mxfp8_fwd_adapter(
        shape.problem,
        evaluator,
        device=campaign.hardware_profile,
        regime=regime,
        tags=tags,
    )


def _prequant_adapter(
    campaign: "DatasetCampaign",
    job: DatasetJob,
    shape: ShapeSpec,
    regime: CacheRegime,
    harness,
    tags: Mapping[str, object],
) -> KernelAdapter:
    evaluator = CalibratedPrequantEvaluator(
        harness, samples=job.protocol.samples, seed=campaign.manifest.seed
    )
    return make_mxfp8_prequant_adapter(
        shape.problem,
        evaluator,
        initial=_reference_prequant_config(shape.problem),
        device=campaign.hardware_profile,
        regime=regime,
        tags=tags,
    )


def _bwd_adapter(
    campaign: "DatasetCampaign",
    job: DatasetJob,
    shape: ShapeSpec,
    regime: CacheRegime,
    harness,
    tags: Mapping[str, object],
) -> KernelAdapter:
    evaluator = CalibratedBwdEvaluator(
        harness, samples=job.protocol.samples, seed=campaign.manifest.seed
    )
    return make_mxfp8_bwd_adapter(
        shape.problem,
        evaluator,
        device=campaign.hardware_profile,
        regime=regime,
        tags=tags,
    )


def _weight_prequant_adapter(
    campaign: "DatasetCampaign",
    job: DatasetJob,
    shape: ShapeSpec,
    regime: CacheRegime,
    harness,
    tags: Mapping[str, object],
) -> KernelAdapter:
    evaluator = CalibratedPrequantEvaluator(
        harness, samples=job.protocol.samples, seed=campaign.manifest.seed
    )
    return make_mxfp8_weight_prequant_adapter(
        shape.problem,
        evaluator,
        device=campaign.hardware_profile,
        regime=regime,
        tags=tags,
    )


def _fully_prequant_adapter(
    campaign: "DatasetCampaign",
    job: DatasetJob,
    shape: ShapeSpec,
    regime: CacheRegime,
    harness,
    tags: Mapping[str, object],
) -> KernelAdapter:
    evaluator = CalibratedPrequantEvaluator(
        harness, samples=job.protocol.samples, seed=campaign.manifest.seed
    )
    return make_mxfp8_fully_prequant_adapter(
        shape.problem,
        evaluator,
        device=campaign.hardware_profile,
        regime=regime,
        tags=tags,
    )


register_dataset_backend(
    "mxfp8_fused_fwd", DatasetBackend(_fused_harness, _fused_adapter)
)
register_dataset_backend(
    "mxfp8_prequant_fwd", DatasetBackend(_prequant_harness, _prequant_adapter)
)
register_dataset_backend(
    "mxfp8_weight_prequant_fwd",
    DatasetBackend(_weight_prequant_harness, _weight_prequant_adapter),
)
register_dataset_backend(
    "mxfp8_fully_prequant_fwd",
    DatasetBackend(_fully_prequant_harness, _fully_prequant_adapter),
)
register_dataset_backend("mxfp8_bwd", DatasetBackend(_bwd_harness, _bwd_adapter))


class DatasetCampaign:
    """Run all assigned manifest contexts and produce a copyable bundle."""

    def __init__(
        self,
        manifest: DatasetManifest,
        output_dir: Path | str,
        *,
        device: torch.device | str = "cuda",
        calibration: Mapping[str, object] | None = None,
        anytime: AnytimeRunPolicy | None = None,
        adopt_existing_context_identity: bool = False,
        progress=print,
    ) -> None:
        self.manifest = manifest
        self.output_dir = Path(output_dir)
        self.device = torch.device(device)
        self.progress = progress
        self.anytime = anytime
        self.adopt_existing_context_identity = adopt_existing_context_identity
        self.fingerprint = DeviceFingerprint.current(self.device)
        if self.fingerprint.capability[0] != 12:
            raise RuntimeError(
                "native RTX MXFP8 campaigns require SM120/SM121; got "
                f"{self.fingerprint.capability} on {self.fingerprint.name}"
            )
        self.machine = machine_snapshot(self.device, calibration=calibration)
        self.hardware_profile = self.machine["device"]["hardware_profile"]  # type: ignore[index]
        self.bundle = (
            self.output_dir
            / manifest.name
            / str(self.machine["machine_id"])
            / f"shard-{manifest.shard_index:03d}-of-{manifest.shard_count:03d}"
        )
        self.context_source = self.machine["source"]
        if adopt_existing_context_identity:
            self.context_source = self._existing_context_source()
        self.verification = ExperimentJournal(self.bundle / "verification.jsonl")

    def _existing_context_source(self) -> Mapping[str, object]:
        """Adopt v2 context tags after runner-only source changes.

        This is deliberately opt-in: kernel changes should normally create new
        context identifiers, while scheduler/CLI changes may safely continue an
        existing append-only v2 bundle.
        """

        machine_path = self.bundle / "machine.json"
        manifest_path = self.bundle / "manifest.json"
        if not machine_path.exists() or not manifest_path.exists():
            raise RuntimeError(
                "--adopt-existing-context-identity requires an existing bundle at "
                f"{self.bundle}"
            )
        try:
            prior_machine = json.loads(machine_path.read_text(encoding="utf-8"))
            prior_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot read existing v2 bundle identity: {exc}") from exc
        if prior_machine.get("machine_id") != self.machine.get("machine_id"):
            raise RuntimeError("existing bundle machine identity does not match this device")
        if _digest(prior_manifest) != self.manifest.digest:
            raise RuntimeError("existing bundle manifest does not match the requested manifest")
        source = prior_machine.get("source")
        if not isinstance(source, dict) or not source.get("python_source_sha256"):
            raise RuntimeError("existing bundle has no reusable source identity")
        return source

    def _log(self, message: str) -> None:
        if self.progress is not None:
            self.progress(message)

    def _make_harness(self, job: DatasetJob, shape: ShapeSpec, regime: CacheRegime):
        return _BACKENDS[job.family].make_harness(self, job, shape, regime)

    def _make_adapter(
        self, job: DatasetJob, shape: ShapeSpec, regime: CacheRegime, harness
    ) -> KernelAdapter:
        tags = {
            **dict(job.tags),
            "campaign": self.manifest.name,
            "manifest_digest": self.manifest.digest,
            "machine_id": self.machine["machine_id"],
            "source_sha256": self.context_source["python_source_sha256"],
            "git_commit": self.context_source.get("git_commit"),
            "git_dirty": self.context_source.get("git_dirty"),
        }
        return _BACKENDS[job.family].make_adapter(
            self, job, shape, regime, harness, tags
        )

    def _verification_base(
        self, adapter: KernelAdapter, job: DatasetJob, shape: ShapeSpec, regime: str
    ) -> dict[str, object]:
        return {
            "schema_version": DATASET_SCHEMA_VERSION,
            "recorded_at": _utc_now(),
            "manifest_digest": self.manifest.digest,
            "machine_id": self.machine["machine_id"],
            "device_id": self.fingerprint.identifier,
            "context_id": adapter.context.identifier,
            "context": adapter.context.as_dict(),
            "family": job.family,
            "kernel_revision": adapter.context.kernel_revision,
            "shape": asdict(shape),
            "shape_key": shape.key,
            "regime": regime,
            "protocol": asdict(job.protocol),
        }

    def _verify(
        self,
        adapter: KernelAdapter,
        harness,
        job: DatasetJob,
        shape: ShapeSpec,
        regime: CacheRegime,
        store: JsonlTuningStore,
    ) -> dict[str, object]:
        observations = [
            record
            for record in store.records(adapter.context)
            if isinstance(record.get("outcome"), dict)
            and record["outcome"].get("status") == "ok"  # type: ignore[union-attr]
            and record["outcome"].get("median_ms") is not None  # type: ignore[union-attr]
        ]
        best_by_config: dict[str, Mapping[str, object]] = {}
        for record in observations:
            config_id = str(record["config_id"])
            previous = best_by_config.get(config_id)
            score = float(record["outcome"]["median_ms"])  # type: ignore[index]
            if previous is None or score < float(previous["outcome"]["median_ms"]):  # type: ignore[index]
                best_by_config[config_id] = record
        finalists = sorted(
            best_by_config.values(),
            key=lambda record: float(record["outcome"]["median_ms"]),  # type: ignore[index]
        )[: job.promote]
        completed = self.verification.completed_keys()
        base = self._verification_base(adapter, job, shape, regime)
        for record in finalists:
            config_id = str(record["config_id"])
            key = _digest((self.manifest.digest, adapter.context.identifier, "confirm", config_id))
            if key in completed:
                continue
            self._log(f"CONFIRM {job.family} {shape.key} {regime} {config_id}")
            config = adapter.deserialize(record["config"])  # type: ignore[arg-type]
            outcome = harness.measure(
                config,
                samples=job.protocol.confirm_samples,
                seed=self.manifest.seed ^ int(config_id[:8], 16),
                components=True,
            )
            self.verification.append(
                {
                    **base,
                    "record_type": "verification_measurement",
                    "observation_key": key,
                    "stage": "confirm",
                    "config_id": config_id,
                    "config": adapter.serialize(config),
                    "features": adapter.features(config),
                    "outcome": outcome,
                }
            )
            completed.add(key)

        records = [
            record
            for record in self.verification.records()
            if record.get("record_type") == "verification_measurement"
            and record.get("context_id") == adapter.context.identifier
            and isinstance(record.get("outcome"), dict)
            and record["outcome"].get("status") == "ok"  # type: ignore[union-attr]
        ]
        confirmed = sorted(
            records,
            key=lambda record: float(record["outcome"]["summary_ms"]["median"]),  # type: ignore[index]
        )
        if not confirmed:
            return {"status": "no_verified_candidate"}
        incumbent = adapter.deserialize(confirmed[0]["config"])  # type: ignore[arg-type]
        prior_races = {
            str(record.get("observation_key")): record
            for record in self.verification.records()
            if record.get("record_type") == "race"
        }
        for record in confirmed[1:]:
            challenger = adapter.deserialize(record["config"])  # type: ignore[arg-type]
            incumbent_id = adapter.config_id(incumbent)
            challenger_id = adapter.config_id(challenger)
            key = _digest(
                (
                    self.manifest.digest,
                    adapter.context.identifier,
                    "race",
                    incumbent_id,
                    challenger_id,
                )
            )
            if key in completed:
                old_outcome = prior_races.get(key, {}).get("outcome", {})
                if isinstance(old_outcome, dict) and old_outcome.get("decision") == "challenger":
                    incumbent = challenger
                continue
            self._log(
                f"RACE    {job.family} {shape.key} {regime} "
                f"{incumbent_id} vs {challenger_id}"
            )
            outcome = harness.race(
                incumbent,
                challenger,
                seed=self.manifest.seed ^ int(key[:8], 16),
            )
            self.verification.append(
                {
                    **base,
                    "record_type": "race",
                    "observation_key": key,
                    "stage": "race",
                    "incumbent_id": incumbent_id,
                    "challenger_id": challenger_id,
                    "incumbent_config": adapter.serialize(incumbent),
                    "challenger_config": adapter.serialize(challenger),
                    "outcome": outcome,
                }
            )
            completed.add(key)
            if outcome.get("decision") == "challenger":
                incumbent = challenger
        winner_id = adapter.config_id(incumbent)
        winner_measurement = next(
            (
                record
                for record in confirmed
                if str(record.get("config_id")) == winner_id
            ),
            None,
        )
        winner_median = None
        if winner_measurement is not None:
            winner_median = float(
                winner_measurement["outcome"]["summary_ms"]["median"]  # type: ignore[index]
            )
        cache_path = save_runtime_winner(
            runtime_winner_key(
                job.family,
                shape.problem,
                regime=regime,
                fingerprint=self.fingerprint,
                variant=(
                    f"w-{incumbent.operand_scale_layouts[1]}"
                    if job.family == "mxfp8_weight_prequant_fwd"
                    else "x-{}_w-{}".format(*incumbent.operand_scale_layouts)
                    if job.family == "mxfp8_fully_prequant_fwd"
                    else "default"
                ),
            ),
            adapter.serialize(incumbent),
            config_id=winner_id,
            root=self.bundle,
            median_ms=winner_median,
            metadata={
                "context_id": adapter.context.identifier,
                "manifest_digest": self.manifest.digest,
            },
        )
        return {
            "status": "ok",
            "screened": len(observations),
            "confirmed": len(confirmed),
            "winner_id": winner_id,
            "winner_config": adapter.serialize(incumbent),
            "runtime_cache": str(cache_path.relative_to(self.bundle)),
        }

    def _assigned_contexts(
        self,
    ) -> list[tuple[DatasetJob, ShapeSpec, CacheRegime]]:
        contexts = [
            (job, shape, regime)
            for job in self.manifest.jobs
            for shape in job.shapes
            for regime in job.regimes
            if _assigned(self.manifest, job.family, shape, regime)
        ]
        if self.anytime is None:
            return contexts

        family_order = {
            "mxfp8_fused_fwd": 0,
            "mxfp8_prequant_fwd": 1,
            "mxfp8_weight_prequant_fwd": 2,
            "mxfp8_fully_prequant_fwd": 3,
            "mxfp8_bwd": 4,
        }

        def shape_priority(shape: ShapeSpec) -> int:
            name = (shape.name or "").lower()
            if "balanced" in name:
                return 0
            if "underfill" in name:
                return 1
            if any(token in name for token in ("one_m", "tiny", "short")):
                return 2
            if "wide" in name:
                return 3
            if any(token in name for token in ("long", "tall")):
                return 4
            if "deep" in name:
                return 5
            if not name:
                return 6
            if "large_cube" in name:
                return 7
            if "mlp" in name:
                return 8
            return 6

        return sorted(
            contexts,
            key=lambda item: (
                shape_priority(item[1]),
                0 if item[2] == "hot" else 1,
                family_order.get(item[0].family, 99),
                item[1].m,
                item[1].n,
                item[1].k,
            ),
        )

    @staticmethod
    def _context_trials(store: JsonlTuningStore, adapter: KernelAdapter) -> int:
        return sum(1 for _record in store.records(adapter.context))

    def _run_context(
        self,
        job: DatasetJob,
        shape: ShapeSpec,
        regime: CacheRegime,
        *,
        target_trials: int | None = None,
        time_budget_s: float | None = None,
        promote: int | None = None,
        verify: bool = True,
    ) -> tuple[dict[str, object], int, int]:
        harness = self._make_harness(job, shape, regime)
        adapter = self._make_adapter(job, shape, regime, harness)
        store = JsonlTuningStore(self.bundle / "stores" / job.family / regime)
        before = self._context_trials(store, adapter)
        effective_job = job
        if target_trials is not None or time_budget_s is not None:
            effective_job = replace(
                effective_job,
                tuning=replace(
                    effective_job.tuning,
                    max_trials=(
                        effective_job.tuning.max_trials
                        if target_trials is None
                        else min(target_trials, effective_job.tuning.max_trials)
                    ),
                    time_budget_s=(
                        effective_job.tuning.time_budget_s
                        if time_budget_s is None
                        else min(time_budget_s, effective_job.tuning.time_budget_s)
                    ),
                ),
            )
        if promote is not None:
            effective_job = replace(effective_job, promote=min(promote, job.promote))
        tuner = make_hybrid_autotuner(
            adapter,
            store,
            effective_job.tuning,
            progress=self.progress,
        )
        tuned = tuner.tune()
        after = self._context_trials(store, adapter)
        verification = (
            self._verify(adapter, harness, effective_job, shape, regime, store)
            if verify
            else {"status": "deferred"}
        )
        return (
            {
                "family": job.family,
                "shape": shape.key,
                "regime": regime,
                "context_id": adapter.context.identifier,
                "target_trials": target_trials,
                "trials_before": before,
                "trials_after": after,
                "tuning_best_ms": tuned.median_ms,
                "evaluated_trials": tuned.evaluated_trials,
                "elapsed_s": tuned.elapsed_s,
                "verification": verification,
            },
            before,
            after,
        )

    def _run_sequential(self, results: list[dict[str, object]]) -> str:
        for job, shape, regime in self._assigned_contexts():
            self._log(f"START   {job.family} {shape.key} {regime}")
            result, _before, _after = self._run_context(job, shape, regime)
            results.append(result)
        return "complete"

    def _run_anytime(
        self,
        results: list[dict[str, object]],
        *,
        started: float,
    ) -> str:
        assert self.anytime is not None
        policy = self.anytime
        deadline = started + policy.wall_time_s
        contexts = self._assigned_contexts()
        milestones = tuple(
            sorted(
                set(policy.trial_milestones)
                | {job.tuning.max_trials for job, _shape, _regime in contexts}
            )
        )
        first_milestone = milestones[0]
        total = len(contexts)
        self._log(
            f"ANYTIME contexts={total} wall={policy.wall_time_s:.0f}s "
            f"slice={policy.context_slice_s:.0f}s milestones={milestones}"
        )
        for milestone in milestones:
            phase_targets = {
                index: min(milestone, job.tuning.max_trials)
                for index, (job, _shape, _regime) in enumerate(contexts)
            }
            self._log(f"PHASE   target={milestone} trials/context")
            blocked: set[int] = set()
            while True:
                phase_pending = False
                sweep_progress = False
                for index, (job, shape, regime) in enumerate(contexts):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._log("DEADLINE global anytime wall-time exhausted")
                        return "time_budget_exhausted"
                    if index in blocked:
                        continue
                    target = phase_targets[index]
                    self._log(
                        f"CONTEXT {index + 1}/{total} {job.family} "
                        f"{shape.key} {regime} target={target}"
                    )
                    try:
                        harness = self._make_harness(job, shape, regime)
                        adapter = self._make_adapter(job, shape, regime, harness)
                        store = JsonlTuningStore(
                            self.bundle / "stores" / job.family / regime
                        )
                        current = self._context_trials(store, adapter)
                    except Exception as exc:
                        blocked.add(index)
                        self._log(
                            f"SKIP    {job.family} {shape.key} {regime} "
                            f"setup_error {type(exc).__name__}: {exc}"
                        )
                        results.append(
                            {
                                "family": job.family,
                                "shape": shape.key,
                                "regime": regime,
                                "target_trials": target,
                                "status": "setup_error",
                                "error": f"{type(exc).__name__}: {exc}"[:4000],
                            }
                        )
                        continue
                    if current >= target:
                        continue
                    phase_pending = True
                    budget = min(policy.context_slice_s, remaining)
                    verify = target == job.tuning.max_trials
                    promote = job.promote if verify else policy.initial_promote
                    # The actual run builds its own harness. Drop this probe
                    # harness first so large rotating-input pools cannot overlap.
                    del harness, adapter, store
                    try:
                        result, before, after = self._run_context(
                            job,
                            shape,
                            regime,
                            target_trials=target,
                            time_budget_s=budget,
                            promote=promote,
                            verify=False,
                        )
                        if after > before:
                            sweep_progress = True
                        reached = after >= target
                        if reached and (milestone == first_milestone or verify):
                            # Rebuilds are cached; verification is intentionally
                            # done only at broad coverage and final depth.
                            harness = self._make_harness(job, shape, regime)
                            adapter = self._make_adapter(job, shape, regime, harness)
                            store = JsonlTuningStore(
                                self.bundle / "stores" / job.family / regime
                            )
                            verify_job = replace(job, promote=promote)
                            result["verification"] = self._verify(
                                adapter, harness, verify_job, shape, regime, store
                            )
                        results.append(result)
                    except Exception as exc:
                        blocked.add(index)
                        self._log(
                            f"SKIP    {job.family} {shape.key} {regime} "
                            f"tuning_error {type(exc).__name__}: {exc}"
                        )
                        results.append(
                            {
                                "family": job.family,
                                "shape": shape.key,
                                "regime": regime,
                                "target_trials": target,
                                "status": "tuning_error",
                                "error": f"{type(exc).__name__}: {exc}"[:4000],
                            }
                        )
                if not phase_pending:
                    break
                if not sweep_progress:
                    self._log(f"PHASE   target={milestone} stalled; advancing")
                    break
        return "complete"

    def run(self, *, export_format: ExportFormat = "csv") -> Path:
        self.bundle.mkdir(parents=True, exist_ok=True)
        _atomic_json(self.bundle / "manifest.json", self.manifest.as_dict())
        if self.adopt_existing_context_identity:
            self._log(
                "RESUME  adopting existing v2 context identity "
                f"source={str(self.context_source['python_source_sha256'])[:12]}"
            )
            _atomic_json(
                self.bundle / "runner.json",
                {
                    "schema_version": DATASET_SCHEMA_VERSION,
                    "recorded_at": _utc_now(),
                    "runner_source": self.machine["source"],
                    "adopted_context_source": self.context_source,
                },
            )
        else:
            _atomic_json(self.bundle / "machine.json", self.machine)
        results: list[dict[str, object]] = []
        status = "complete"
        started = time.monotonic()
        try:
            status = (
                self._run_sequential(results)
                if self.anytime is None
                else self._run_anytime(results, started=started)
            )
        except KeyboardInterrupt:
            status = "interrupted"
            raise
        except BaseException:
            status = "failed"
            raise
        finally:
            _atomic_json(
                self.bundle / "summary.json",
                {
                    "schema_version": DATASET_SCHEMA_VERSION,
                    "status": status,
                    "recorded_at": _utc_now(),
                    "elapsed_s": time.monotonic() - started,
                    "manifest_digest": self.manifest.digest,
                    "machine_id": self.machine["machine_id"],
                    "run_mode": "sequential" if self.anytime is None else "anytime",
                    "anytime_policy": (
                        None if self.anytime is None else asdict(self.anytime)
                    ),
                    "adopted_existing_context_identity": (
                        self.adopt_existing_context_identity
                    ),
                    "results": results,
                },
            )
            if export_format != "none":
                export_bundle(
                    (self.bundle,),
                    self.bundle / "dataset",
                    export_format=export_format,
                )
        return self.bundle


def _iter_jsonl(paths: Iterable[Path | str], filename: str):
    seen_paths: set[Path] = set()
    for root_value in paths:
        root = Path(root_value)
        candidates = [root] if root.is_file() and root.name == filename else root.rglob(filename)
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen_paths:
                continue
            seen_paths.add(resolved)
            with path.open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    try:
                        yield path, json.loads(line)
                    except json.JSONDecodeError:
                        continue


def _json_documents(paths: Iterable[Path | str], filename: str):
    seen: set[Path] = set()
    for root_value in paths:
        root = Path(root_value)
        candidates = [root] if root.is_file() and root.name == filename else root.rglob(filename)
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                with path.open(encoding="utf-8") as source:
                    yield path, json.load(source)
            except (OSError, json.JSONDecodeError):
                continue


def _flatten(prefix: str, value: object, row: dict[str, object]) -> None:
    if isinstance(value, Mapping):
        for key in sorted(value):
            _flatten(f"{prefix}__{key}" if prefix else str(key), value[key], row)
    elif isinstance(value, (list, tuple)):
        row[prefix] = canonical_json(value)
    elif value is None or isinstance(value, (bool, int, float, str)):
        row[prefix] = value
    else:
        row[prefix] = str(value)


def normalized_rows(paths: Iterable[Path | str]) -> list[dict[str, object]]:
    """Read arbitrary copied bundles and return deduplicated flat records."""

    sources = tuple(paths)
    machines = {
        str(document.get("machine_id")): document
        for _path, document in _json_documents(sources, "machine.json")
        if isinstance(document, dict) and document.get("machine_id") is not None
    }
    contexts: dict[str, Mapping[str, object]] = {}
    for _path, record in _iter_jsonl(sources, "sessions.jsonl"):
        if record.get("event") == "start" and isinstance(record.get("context"), dict):
            contexts[str(record.get("context_id"))] = record["context"]

    unique: dict[tuple[object, ...], dict[str, object]] = {}
    for path, record in _iter_jsonl(sources, "observations.jsonl"):
        context = contexts.get(str(record.get("context_id")), {})
        outcome = record.get("outcome", {})
        context_tags = context.get("tags", {}) if isinstance(context, Mapping) else {}
        machine_id = (
            context_tags.get("machine_id")
            if isinstance(context_tags, Mapping)
            else None
        )
        row: dict[str, object] = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "record_type": "measurement",
            "source_path": str(path),
            "observation_id": record.get("observation_id"),
            "session_id": record.get("session_id"),
            "sequence": record.get("sequence"),
            "context_id": record.get("context_id"),
            "family": record.get("family"),
            "kernel_revision": record.get("kernel_revision"),
            "manifest_digest": (
                context_tags.get("manifest_digest")
                if isinstance(context_tags, Mapping)
                else None
            ),
            "machine_id": machine_id,
            "config_id": record.get("config_id"),
            "strategy": record.get("strategy"),
            "parent_config_id": record.get("parent_config_id"),
            "coordinate": record.get("coordinate"),
            "started_at": record.get("started_at"),
            "finished_at": record.get("finished_at"),
            "elapsed_s": record.get("elapsed_s"),
        }
        for prefix, value in (
            ("context", context),
            ("machine", machines.get(str(machine_id), {})),
            ("config", record.get("config", {})),
            ("feature", record.get("features", {})),
            ("proposal", record.get("metadata", {})),
            ("outcome", outcome),
        ):
            _flatten(prefix, value, row)
        identity = (
            "measurement",
            record.get("context_id"),
            record.get("observation_id"),
        )
        unique.setdefault(identity, row)

    for path, record in _iter_jsonl(sources, "verification.jsonl"):
        outcome = record.get("outcome", {})
        row = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "record_type": record.get("record_type"),
            "source_path": str(path),
            "observation_id": record.get("observation_key"),
            "context_id": record.get("context_id"),
            "family": record.get("family"),
            "kernel_revision": record.get("kernel_revision"),
            "config_id": record.get("config_id"),
            "incumbent_id": record.get("incumbent_id"),
            "challenger_id": record.get("challenger_id"),
            "stage": record.get("stage"),
            "recorded_at": record.get("recorded_at"),
            "manifest_digest": record.get("manifest_digest"),
            "machine_id": record.get("machine_id"),
            "device_id": record.get("device_id"),
        }
        for prefix, value in (
            ("context", record.get("context", {})),
            ("machine", machines.get(str(record.get("machine_id")), {})),
            ("shape", record.get("shape", {})),
            ("protocol", record.get("protocol", {})),
            ("config", record.get("config", {})),
            ("feature", record.get("features", {})),
            ("incumbent_config", record.get("incumbent_config", {})),
            ("challenger_config", record.get("challenger_config", {})),
            ("outcome", outcome),
        ):
            _flatten(prefix, value, row)
        identity = (record.get("record_type"), record.get("observation_key"))
        unique.setdefault(identity, row)
    return [unique[key] for key in sorted(unique, key=lambda item: tuple(map(str, item)))]


def export_csv(rows: Sequence[Mapping[str, object]], destination: Path | str) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_parquet(rows: Sequence[Mapping[str, object]], destination: Path | str) -> Path:
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - broken/incomplete install.
        raise RuntimeError(
            "Parquet export requires the core pyarrow dependency; reinstall rtx "
            "or run `python -m pip install pyarrow`."
        ) from exc
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist([dict(row) for row in rows])
    pq.write_table(table, path, compression="zstd")
    return path


def export_bundle(
    paths: Iterable[Path | str],
    output_prefix: Path | str,
    *,
    export_format: ExportFormat = "both",
) -> dict[str, object]:
    sources = tuple(paths)
    machine_documents = [
        document
        for _path, document in _json_documents(sources, "machine.json")
        if isinstance(document, dict)
    ]
    source_schema_versions = sorted(
        {
            int(document.get("schema_version", 1))
            for document in machine_documents
        }
    )
    rows = normalized_rows(sources)
    prefix = Path(output_prefix)
    written: dict[str, str] = {}
    if export_format in ("csv", "both"):
        written["csv"] = str(export_csv(rows, prefix.with_suffix(".csv")))
    if export_format in ("parquet", "both"):
        written["parquet"] = str(export_parquet(rows, prefix.with_suffix(".parquet")))
    report: dict[str, object] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "source_dataset_schema_versions": source_schema_versions,
        "mixed_dataset_schemas": len(source_schema_versions) > 1,
        "recorded_at": _utc_now(),
        "rows": len(rows),
        "measurements": sum(row.get("record_type") == "measurement" for row in rows),
        "verification_measurements": sum(
            row.get("record_type") == "verification_measurement" for row in rows
        ),
        "races": sum(row.get("record_type") == "race" for row in rows),
        "successful_rows": sum(row.get("outcome__status") == "ok" for row in rows),
        "families": sorted(
            {str(row["family"]) for row in rows if row.get("family") is not None}
        ),
        "contexts": len(
            {str(row["context_id"]) for row in rows if row.get("context_id") is not None}
        ),
        "machines": sorted(
            {
                str(machine)
                for row in rows
                for machine in (
                    row.get("machine_id"),
                    row.get("context__tags__machine_id"),
                )
                if machine is not None
            }
        ),
        "files": written,
    }
    _atomic_json(prefix.with_suffix(".export.json"), report)
    return report


def _parse_duration(value: str) -> float:
    text = value.strip().lower()
    multipliers = {"s": 1.0, "m": 60.0, "h": 3600.0}
    suffix = text[-1:] if text else ""
    if suffix in multipliers:
        text = text[:-1]
        multiplier = multipliers[suffix]
    else:
        multiplier = 1.0
    try:
        seconds = float(text) * multiplier
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "duration must be seconds or a number suffixed by s, m, or h"
        ) from exc
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("duration must be positive and finite")
    return seconds


def _parse_milestones(value: str) -> tuple[int, ...]:
    try:
        milestones = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError("milestones must be comma-separated integers") from exc
    if not milestones or any(item <= 0 for item in milestones):
        raise argparse.ArgumentTypeError("milestones must be positive")
    if tuple(sorted(set(milestones))) != milestones:
        raise argparse.ArgumentTypeError("milestones must be unique and increasing")
    return milestones


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run", help="run/resume a campaign on this GPU")
    run.add_argument("manifest", type=Path)
    run.add_argument("--output-dir", type=Path, default=Path("autotune_datasets"))
    run.add_argument("--device", default="cuda")
    run.add_argument("--format", choices=("csv", "parquet", "both", "none"), default="csv")
    run.add_argument("--shard-index", type=int)
    run.add_argument("--shard-count", type=int)
    run.add_argument("--quiet", action="store_true")
    run.add_argument(
        "--calibration",
        type=Path,
        help="JSON emitted by 'rtx-autotune calibrate' on this machine",
    )
    run.add_argument(
        "--wall-time",
        type=_parse_duration,
        help="enable anytime scheduling with this global budget (e.g. 2h or 90m)",
    )
    run.add_argument(
        "--context-slice",
        type=_parse_duration,
        default=120.0,
        help="maximum time per context visit in anytime mode (default: 2m)",
    )
    run.add_argument(
        "--trial-milestones",
        type=_parse_milestones,
        default=(32, 96, 192, 384, 512),
        help="absolute breadth-first trial targets (default: 32,96,192,384,512)",
    )
    run.add_argument(
        "--initial-promote",
        type=int,
        default=2,
        help="finalists verified after the first coverage milestone (default: 2)",
    )
    run.add_argument(
        "--adopt-existing-context-identity",
        action="store_true",
        help="resume a v2 bundle across runner-only source changes",
    )

    collect = subparsers.add_parser("collect", help="merge copied campaign bundles")
    collect.add_argument("paths", type=Path, nargs="+")
    collect.add_argument("--output", type=Path, required=True, help="output path without extension")
    collect.add_argument("--format", choices=("csv", "parquet", "both"), default="csv")

    validate = subparsers.add_parser("validate", help="validate and summarize a manifest")
    validate.add_argument("manifest", type=Path)
    probe = subparsers.add_parser("probe", help="print machine and GPU capability metadata")
    probe.add_argument("--device", default="cuda")
    probe.add_argument("--calibration", type=Path)
    calibrate = subparsers.add_parser(
        "calibrate", help="measure device rooflines for architecture-aware search"
    )
    calibrate.add_argument("--device", default="cuda")
    calibrate.add_argument("--output", type=Path, required=True)
    calibrate.add_argument("--samples", type=int, default=5)
    calibrate.add_argument("--target-ms", type=float, default=30.0)
    calibrate.add_argument("--skip-native-mxfp8", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "validate":
        manifest = DatasetManifest.load(args.manifest)
        contexts = sum(len(job.shapes) * len(job.regimes) for job in manifest.jobs)
        print(json.dumps({"manifest": manifest.as_dict(), "digest": manifest.digest, "contexts": contexts}, indent=2))
        return
    if args.command == "probe":
        if not torch.cuda.is_available():
            parser.error("CUDA is not available")
        calibration = None
        if args.calibration is not None:
            calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
        print(
            json.dumps(
                machine_snapshot(args.device, calibration=calibration),
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.command == "calibrate":
        if not torch.cuda.is_available():
            parser.error("CUDA is not available")
        from .calibration import calibrate_device

        result = calibrate_device(
            args.device,
            samples=args.samples,
            target_ms=args.target_ms,
            include_native_mxfp8=not args.skip_native_mxfp8,
        )
        _atomic_json(args.output, result)
        print(json.dumps({"calibration": str(args.output)}, indent=2))
        return
    if args.command == "collect":
        report = export_bundle(args.paths, args.output, export_format=args.format)
        print(json.dumps(report, indent=2, sort_keys=True))
        return
    if not torch.cuda.is_available():
        parser.error("CUDA is not available")
    manifest = DatasetManifest.load(args.manifest)
    if (args.shard_index is None) != (args.shard_count is None):
        parser.error("--shard-index and --shard-count must be provided together")
    if args.shard_index is not None:
        manifest = manifest.with_shard(args.shard_index, args.shard_count)
    calibration = None
    if args.calibration is not None:
        calibration = json.loads(args.calibration.read_text(encoding="utf-8"))
    campaign = DatasetCampaign(
        manifest,
        args.output_dir,
        device=args.device,
        calibration=calibration,
        anytime=(
            None
            if args.wall_time is None
            else AnytimeRunPolicy(
                wall_time_s=args.wall_time,
                context_slice_s=args.context_slice,
                trial_milestones=args.trial_milestones,
                initial_promote=args.initial_promote,
            )
        ),
        adopt_existing_context_identity=args.adopt_existing_context_identity,
        progress=None if args.quiet else print,
    )
    bundle = campaign.run(export_format=args.format)
    print(json.dumps({"bundle": str(bundle)}, indent=2))


__all__ = [
    "AnytimeRunPolicy",
    "DATASET_SCHEMA_VERSION",
    "DatasetBackend",
    "DatasetCampaign",
    "DatasetJob",
    "DatasetManifest",
    "FusedFwdBenchmarkHarness",
    "export_bundle",
    "export_csv",
    "export_parquet",
    "machine_snapshot",
    "main",
    "normalized_rows",
    "register_dataset_backend",
]


if __name__ == "__main__":
    main()
