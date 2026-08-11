"""Persistent joint autotuning for dynamic-quantized MXFP8 forward.

Every candidate is timed end to end: X quantization, W quantization, and GEMM.
The search coordinates include implementation-family changes which cannot be
reached by changing one leaf field at a time (native scale layouts, M64, and
combined versus independent quantizers).
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import tempfile
import time
from typing import Iterable, Iterator, Mapping
import uuid

import torch

from .autotune import (
    CoordinateDescentPolicy,
    DeviceFingerprint,
    ProgressCallback,
    TrialOutcome,
    default_cache_dir,
)
from .fp8 import (
    DEFAULT_MXFP8_PREQUANT_CONFIG,
    MXFP8PrequantConfig,
    _build_prequant_runner,
    _set_l2_fetch_granularity,
)
from .kernels.mxfp8 import MXFP8Problem
from .configs import MXFP8GemmConfig, MXFP8QuantConfig

try:
    import fcntl
except ImportError:  # pragma: no cover - Linux/CUDA is the supported target.
    fcntl = None


SCHEMA_VERSION = 1
KERNEL_NAME = "mxfp8_prequant_e2e"
KERNEL_REVISION = 7


def _quant_vector_variants() -> tuple[dict[str, object], ...]:
    return tuple(
        {
            "quant_vec": vector,
            "load_bits": bits,
            "quant_store_bits": store_bits,
        }
        for vector in (1, 2, 4, 8)
        for bits in (16, 32, 64, 128)
        for store_bits in (8, 16, 32)
        if bits <= vector * 16 and (vector * 16) % bits == 0
        and store_bits <= vector * 8
        and (vector * 8) % store_bits == 0
    )


def _quant_arithmetic_variants() -> tuple[dict[str, object], ...]:
    return tuple(
        {"quant_math": math, "quant_amax": amax, "reduction": reduction}
        for math in ("fp32", "bf16x2")
        for amax in ("fp32", "bf16_bits")
        for reduction in ("shuffle", "redux")
    )


def _quant_launch_variants() -> tuple[dict[str, object], ...]:
    return tuple(
        {"num_warps": warps, "persistent_waves": waves}
        for warps in (4, 8, 16)
        for waves in (1, 2, 3, 4, 6, 8)
    )


def _layout_variant(
    x_layout: str,
    w_layout: str,
    gemm_layout: str,
    scale_role: str,
    tile_m: int,
    atom_m: int,
) -> dict[str, object]:
    return {
        "quant": {"scale_layout": x_layout},
        "weight_quant": {"scale_layout": w_layout},
        "gemm": {
            "scale_layout": gemm_layout,
            "scale_role": scale_role,
            "tile_m": tile_m,
            "tile_n": 128,
            "tile_k": 128,
            "atom_layout_m": atom_m,
            "atom_layout_n": 2,
            # Hybrid native scale TMA currently has a one-stage physical
            # layout. Other families retain the incumbent stage count.
            **({"stages": 1} if gemm_layout == "mma64x128" or tile_m == 256 else {}),
            **(
                {"epilogue": "direct", "epilogue_stages": 1, "store_vec": 1}
                if tile_m == 256
                else {}
            ),
        },
    }


# Values are nested replacements, not labels or no-op metadata.  Coordinates
# are deliberately compound where all single-field intermediate states would
# be illegal (for example row-major -> tensor-core-native scales).
PREQUANT_SEARCH_SPACE: dict[str, tuple[dict[str, object], ...]] = {
    "layout_transport": (
        _layout_variant("mma128", "mma128", "mma128", "tma", 128, 4),
        _layout_variant("mma64", "mma128", "mma64x128", "tma", 64, 2),
        _layout_variant("row_major", "row_major", "row_major", "consumers", 64, 2),
        _layout_variant("row_major", "row_major", "row_major", "producer", 64, 2),
        _layout_variant("row_major", "row_major", "row_major", "consumers", 128, 4),
        _layout_variant("row_major", "row_major", "row_major", "producer", 128, 4),
        _layout_variant("row_major", "row_major", "row_major", "consumers", 256, 8),
        _layout_variant("row_major", "row_major", "row_major", "producer", 256, 8),
    ),
    "quant_launches": (
        {"quant_launches": "dual"},
        {"quant_launches": "separate"},
    ),
    "x_vector_load": tuple({"quant": value} for value in _quant_vector_variants()),
    "w_vector_load": tuple(
        {"weight_quant": value} for value in _quant_vector_variants()
    ),
    "x_arithmetic": tuple(
        {"quant": value} for value in _quant_arithmetic_variants()
    ),
    "w_arithmetic": tuple(
        {"weight_quant": value} for value in _quant_arithmetic_variants()
    ),
    "x_launch": tuple({"quant": value} for value in _quant_launch_variants()),
    "w_launch": tuple(
        {"weight_quant": value} for value in _quant_launch_variants()
    ),
    "x_registers": tuple(
        {"quant": {"maxrregcount": value}}
        for value in (64, 80, 96, 112, 128, 160, 192, 224, 255)
    ),
    "w_registers": tuple(
        {"weight_quant": {"maxrregcount": value}}
        for value in (64, 80, 96, 112, 128, 160, 192, 224, 255)
    ),
    "x_scale_store": (
        {"quant": {"native_scale_store": "scalar"}},
        {"quant": {"native_scale_store": "packed"}},
    ),
    "w_scale_store": (
        {"weight_quant": {"native_scale_store": "scalar"}},
        {"weight_quant": {"native_scale_store": "packed"}},
    ),
    "gemm_geometry": tuple(
        {
            "gemm": {
                "tile_m": tile_m,
                "tile_n": tile_n,
                "tile_k": tile_k,
                "atom_layout_m": atom_m,
                "atom_layout_n": atom_n,
            }
        }
        for tile_m, atom_m in ((64, 2), (128, 2), (128, 4), (128, 8), (256, 8))
        for tile_n, atom_n in ((128, 2), (256, 4))
        for tile_k in (128,)
    ),
    "gemm_stages": tuple(
        {"gemm": {"stages": value}} for value in (1, 2, 3, 4)
    ),
    "ldmatrix": tuple(
        {"gemm": {"a_ldmatrix_matrices": a, "b_ldmatrix_matrices": b}}
        for a in (1, 2, 4)
        for b in (1, 2, 4)
    ),
    "mma_schedule": tuple(
        {"gemm": {"mma_schedule": value}}
        for value in ("interleaved", "preload")
    ),
    "smem_swizzle": tuple(
        {"gemm": {"a_swizzle": a, "b_swizzle": b}}
        for a in ("none", "32b", "64b", "128b")
        for b in ("none", "32b", "64b", "128b")
    ),
    "scale_s2r": tuple(
        {"gemm": {"sfa_s2r_bits": a, "sfb_s2r_bits": b}}
        for a in (0, 8)
        for b in (0, 8)
    ),
    "scale_schedule": tuple(
        {"gemm": {"scale_schedule": schedule, "scale_load_vec": width}}
        for schedule in ("before_wait", "after_wait")
        for width in (1, 2, 4, 8)
    ),
    "scale_recycle": (
        {"gemm": {"scale_recycle": "barrier"}},
        {"gemm": {"scale_recycle": "staged", "stages": 2}},
        {"gemm": {"scale_recycle": "staged", "stages": 3}},
    ),
    "scale_smem_store": (
        {"gemm": {"scale_smem_store": "scalar"}},
        *(
            {
                "gemm": {
                    "scale_smem_store": "packed",
                    "scale_role": "consumers",
                    "scale_load_vec": width,
                }
            }
            for width in (2, 4, 8)
        ),
    ),
    "scale_cache": tuple(
        {"gemm": value}
        for value in (
            *(
                {
                    "scale_l2_prefetch": prefetch,
                    "scale_l1_evict": evict,
                    "scale_cache": "default",
                }
                for prefetch in ("none", "64b", "128b", "256b")
                for evict in ("default", "normal", "first", "last", "noallocate")
            ),
            *(
                {
                    "scale_l2_prefetch": prefetch,
                    "scale_l1_evict": "default",
                    "scale_cache": cache,
                }
                for prefetch in ("none", "64b", "128b", "256b")
                for cache in ("ca", "cg", "cs")
            ),
        )
    ),
    "gemm_registers": tuple(
        {
            "gemm": {
                "producer_registers": producer,
                "consumer_registers": consumer,
                "maxrregcount": maximum,
            }
        }
        for producer in (24, 32, 40, 48, 64, 80)
        for consumer, maximum in (
            (96, 160),
            (128, 192),
            (160, 224),
            (192, 255),
            (224, 255),
            (232, 255),
        )
    ),
    "producer_registers": tuple(
        {"gemm": {"producer_registers": value}}
        for value in (24, 32, 40, 48, 56, 64, 72, 80, 96)
    ),
    "consumer_registers": tuple(
        {"gemm": {"consumer_registers": value}}
        for value in (96, 112, 128, 144, 160, 176, 192, 208, 224, 232)
    ),
    "gemm_maxrregcount": tuple(
        {"gemm": {"maxrregcount": value}}
        for value in (128, 144, 160, 176, 192, 208, 224, 240, 255)
    ),
    "epilogue": (
        {"gemm": {"epilogue": "direct", "epilogue_stages": 1, "store_vec": 1}},
        *(
            {
                "gemm": {
                    "epilogue": "tma",
                    "epilogue_stages": stages,
                    "store_vec": vector,
                }
            }
            for stages in (1, 2, 3, 4)
            for vector in (1, 2, 4)
        ),
    ),
    "raster_group": tuple(
        {"gemm": {"raster": raster, "grid_swizzle": group}}
        for raster in ("m", "n")
        for group in (1, 2, 4, 8)
    ),
    "gemm_persistence": (
        *(
            {
                "gemm": {
                    "tiles_per_cta": tiles,
                    "tile_locality": locality,
                }
            }
            for tiles in (1, 2, 4, 8)
            for locality in (
                "raster",
                "same_a",
                "same_b",
                "serpentine_a",
                "serpentine_b",
            )
        ),
        *(
            {
                "gemm": {
                    "stages": 1,
                    "epilogue": "tma",
                    "epilogue_stages": epilogue_stages,
                    "store_vec": 4,
                    "tiles_per_cta": tiles,
                    "tile_locality": locality,
                }
            }
            for tiles in (2, 4, 8)
            for epilogue_stages in range(2, min(4, tiles) + 1)
            for locality in (
                "raster",
                "same_a",
                "same_b",
                "serpentine_a",
                "serpentine_b",
            )
        ),
    ),
    "global_l2_fetch": tuple(
        {"l2_fetch_granularity": value} for value in (None, 0, 32, 64, 128)
    ),
}

PREQUANT_COORDINATE_ORDER = tuple(PREQUANT_SEARCH_SPACE)


def prequant_config_to_dict(config: MXFP8PrequantConfig) -> dict[str, object]:
    return asdict(config.normalized())


def prequant_config_from_dict(value: Mapping[str, object]) -> MXFP8PrequantConfig:
    quant = MXFP8QuantConfig(**dict(value["quant"]))  # type: ignore[arg-type]
    gemm = MXFP8GemmConfig(**dict(value["gemm"]))  # type: ignore[arg-type]
    weight_value = value.get("weight_quant")
    weight_quant = (
        None
        if weight_value is None
        else MXFP8QuantConfig(**dict(weight_value))  # type: ignore[arg-type]
    )
    return MXFP8PrequantConfig(
        quant=quant,
        gemm=gemm,
        quant_launches=str(value.get("quant_launches", "dual")),
        weight_quant=weight_quant,
        weight_scale_layout=(
            None
            if value.get("weight_scale_layout") is None
            else str(value["weight_scale_layout"])
        ),
        l2_fetch_granularity=(
            None
            if value.get("l2_fetch_granularity") is None
            else int(value["l2_fetch_granularity"])
        ),
    ).normalized()


def update_prequant_config(
    config: MXFP8PrequantConfig,
    updates: Mapping[str, object],
) -> MXFP8PrequantConfig:
    quant_values = asdict(config.quant)
    weight_values = asdict(config.resolved_weight_quant())
    gemm_values = asdict(config.gemm)
    quant_updates = dict(updates.get("quant", {}))  # type: ignore[arg-type]
    weight_updates = dict(updates.get("weight_quant", {}))  # type: ignore[arg-type]
    quant_values.update(quant_updates)
    weight_values.update(weight_updates)
    if config.quant_launches == "dual" and quant_updates and not weight_updates:
        weight_layout = weight_values["scale_layout"]
        weight_values.update(quant_updates)
        weight_values["scale_layout"] = weight_layout
    gemm_values.update(dict(updates.get("gemm", {})))  # type: ignore[arg-type]
    return MXFP8PrequantConfig(
        quant=MXFP8QuantConfig(**quant_values),
        weight_quant=MXFP8QuantConfig(**weight_values),
        gemm=MXFP8GemmConfig(**gemm_values),
        quant_launches=str(updates.get("quant_launches", config.quant_launches)),
        weight_scale_layout=None,
        l2_fetch_granularity=updates.get(  # type: ignore[arg-type]
            "l2_fetch_granularity", config.l2_fetch_granularity
        ),
    ).normalized()


def prequant_config_id(config: MXFP8PrequantConfig) -> str:
    payload = json.dumps(prequant_config_to_dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _search_digest(axes: Mapping[str, Iterable[object]]) -> str:
    serializable = {name: list(values) for name, values in axes.items()}
    return hashlib.sha256(
        json.dumps(serializable, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class PrequantTuningResult:
    config: MXFP8PrequantConfig
    median_ms: float
    database_path: Path
    session_id: str
    evaluated_trials: int
    reused_trials: int
    elapsed_s: float


class PrequantJsonTuningDatabase:
    """Atomic, lock-protected per-shape/device joint tuning database."""

    def __init__(
        self,
        root: Path | str | None,
        fingerprint: DeviceFingerprint,
        problem: MXFP8Problem,
        axes: Mapping[str, Iterable[object]],
    ) -> None:
        self.root = default_cache_dir() if root is None else Path(root).expanduser()
        self.fingerprint = fingerprint
        self.problem = problem
        self.axes = {name: tuple(values) for name, values in axes.items()}
        self.search_digest = _search_digest(self.axes)
        self.path = (
            self.root
            / KERNEL_NAME
            / fingerprint.identifier
            / f"m{problem.m}_n{problem.n}_k{problem.k}_{self.search_digest[:12]}.json"
        )
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def _new(self) -> dict[str, object]:
        now = _utc_now()
        return {
            "schema_version": SCHEMA_VERSION,
            "kernel": KERNEL_NAME,
            "kernel_revision": KERNEL_REVISION,
            "created_at": now,
            "updated_at": now,
            "fingerprint": self.fingerprint.as_dict(),
            "problem": asdict(self.problem),
            "search_space_digest": self.search_digest,
            "search_space": {name: list(values) for name, values in self.axes.items()},
            "coordinate_order": list(self.axes),
            "best": None,
            "trials": {},
            "sessions": [],
        }

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

    def _read(self) -> dict[str, object]:
        if not self.path.exists():
            return self._new()
        with self.path.open(encoding="utf-8") as source:
            document = json.load(source)
        valid = (
            document.get("schema_version") == SCHEMA_VERSION
            and document.get("kernel") == KERNEL_NAME
            and document.get("kernel_revision") == KERNEL_REVISION
            and document.get("fingerprint") == self.fingerprint.as_dict()
            and document.get("problem") == asdict(self.problem)
            and document.get("search_space_digest") == self.search_digest
        )
        if not valid:
            raise RuntimeError(f"incompatible prequant tuning database: {self.path}")
        return document

    def _write(self, document: dict[str, object]) -> None:
        document["updated_at"] = _utc_now()
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=self.path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as sink:
                temp_name = sink.name
                json.dump(document, sink, indent=2, sort_keys=True)
                sink.write("\n")
                sink.flush()
                os.fsync(sink.fileno())
            os.replace(temp_name, self.path)
            temp_name = None
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temp_name is not None and os.path.exists(temp_name):
                os.unlink(temp_name)

    def read(self) -> dict[str, object]:
        with self._locked():
            return self._read()

    def start_session(self, policy: CoordinateDescentPolicy) -> str:
        session_id = uuid.uuid4().hex
        with self._locked():
            document = self._read()
            document["sessions"].append(  # type: ignore[union-attr]
                {
                    "id": session_id,
                    "started_at": _utc_now(),
                    "finished_at": None,
                    "status": "running",
                    "policy": asdict(policy),
                }
            )
            self._write(document)
        return session_id

    def finish_session(self, session_id: str, status: str, elapsed_s: float) -> None:
        with self._locked():
            document = self._read()
            sessions = document["sessions"]
            best = document.get("best")
            for session in reversed(sessions):  # type: ignore[arg-type]
                if session.get("id") == session_id:
                    session.update(
                        finished_at=_utc_now(),
                        status=status,
                        elapsed_s=elapsed_s,
                        best_config_id=(
                            None if best is None else best.get("config_id")
                        ),
                    )
                    break
            self._write(document)

    def get_trial(self, config: MXFP8PrequantConfig) -> TrialOutcome | None:
        trial = self.read()["trials"].get(prequant_config_id(config))  # type: ignore[union-attr]
        return None if trial is None else TrialOutcome.from_dict(trial)

    def record(
        self,
        config: MXFP8PrequantConfig,
        outcome: TrialOutcome,
        *,
        session_id: str,
        pass_index: int,
        coordinate: str,
        coordinate_value: object,
    ) -> None:
        config_id = prequant_config_id(config)
        with self._locked():
            document = self._read()
            trials = document["trials"]
            previous = trials.get(config_id)  # type: ignore[union-attr]
            history: list[dict[str, object]] = []
            if previous is not None:
                history.extend(previous.get("history", []))
                history.append(
                    {
                        key: value
                        for key, value in previous.items()
                        if key not in {"config", "history"}
                    }
                )
            trial = {
                "config_id": config_id,
                "config": prequant_config_to_dict(config),
                "attempt": 1 if previous is None else int(previous.get("attempt", 0)) + 1,
                "recorded_at": _utc_now(),
                "session_id": session_id,
                "pass_index": pass_index,
                "coordinate": coordinate,
                "coordinate_value": coordinate_value,
                "history": history,
                **outcome.as_dict(),
            }
            trials[config_id] = trial  # type: ignore[index]
            successful = [
                item
                for item in trials.values()  # type: ignore[union-attr]
                if item.get("status") == "ok" and item.get("median_ms") is not None
            ]
            if successful:
                winner = min(successful, key=lambda item: float(item["median_ms"]))
                document["best"] = {
                    "config_id": winner["config_id"],
                    "config": winner["config"],
                    "median_ms": winner["median_ms"],
                    "selected_at": winner["recorded_at"],
                    "selected_by_session": winner["session_id"],
                    "provisional": True,
                }
            self._write(document)

    def select(self, config: MXFP8PrequantConfig, median_ms: float, session_id: str) -> None:
        with self._locked():
            document = self._read()
            document["best"] = {
                "config_id": prequant_config_id(config),
                "config": prequant_config_to_dict(config),
                "median_ms": median_ms,
                "selected_at": _utc_now(),
                "selected_by_session": session_id,
            }
            self._write(document)

    def best(self) -> tuple[MXFP8PrequantConfig, float] | None:
        best = self.read().get("best")
        if best is None:
            return None
        return prequant_config_from_dict(best["config"]), float(best["median_ms"])


class MXFP8PrequantEvaluator:
    def __init__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        policy: CoordinateDescentPolicy,
    ) -> None:
        self.x = x.contiguous()
        self.weight = weight.contiguous()
        self.policy = policy
        self.problem = MXFP8Problem(x.shape[0], weight.shape[0], x.shape[1])
        self._expected: torch.Tensor | None = None

    def _reference(self) -> torch.Tensor:
        if self._expected is None:
            if self.policy.correctness == "torch":
                from .autotune import _torch_mxfp8_reference

                self._expected = _torch_mxfp8_reference(self.x, self.weight)
                return self._expected
            runner = _build_prequant_runner(
                self.x, self.weight, _intern_for_tuning(DEFAULT_MXFP8_PREQUANT_CONFIG)
            )
            out = torch.empty(
                (self.problem.m, self.problem.n),
                device=self.x.device,
                dtype=torch.bfloat16,
            )
            runner(self.x, self.weight, out)
            torch.cuda.synchronize(self.x.device)
            self._expected = out.clone()
        return self._expected

    def __call__(self, config: MXFP8PrequantConfig) -> TrialOutcome:
        previous_l2: int | None = None
        if config.l2_fetch_granularity is not None:
            previous_l2 = _set_l2_fetch_granularity(config.l2_fetch_granularity)
        compile_start = time.monotonic()
        try:
            runner = _build_prequant_runner(
                self.x, self.weight, _intern_for_tuning(config)
            )
        except Exception as exc:
            if previous_l2 is not None:
                _set_l2_fetch_granularity(previous_l2)
            return TrialOutcome(
                "compile_error",
                compile_ms=(time.monotonic() - compile_start) * 1000,
                error=f"{type(exc).__name__}: {exc}"[:4000],
            )
        compile_ms = (time.monotonic() - compile_start) * 1000
        out = torch.empty(
            (self.problem.m, self.problem.n),
            device=self.x.device,
            dtype=torch.bfloat16,
        )
        max_abs_error: float | None = None
        try:
            runner(self.x, self.weight, out)
            torch.cuda.synchronize(self.x.device)
            if self.policy.correctness != "none":
                expected = self._reference()
                max_abs_error = float((out.float() - expected.float()).abs().max())
                if not torch.allclose(
                    out,
                    expected,
                    rtol=self.policy.correctness_rtol,
                    atol=self.policy.correctness_atol,
                    equal_nan=True,
                ):
                    return TrialOutcome(
                        "correctness_error",
                        compile_ms=compile_ms,
                        max_abs_error=max_abs_error,
                        error="candidate differs from the reference prequant pipeline",
                    )
            for _ in range(self.policy.warmup):
                runner(self.x, self.weight, out)
            torch.cuda.synchronize(self.x.device)
            timings: list[float] = []
            for _ in range(self.policy.samples):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _call in range(self.policy.calls_per_sample):
                    runner(self.x, self.weight, out)
                end.record()
                end.synchronize()
                timings.append(
                    float(start.elapsed_time(end)) / self.policy.calls_per_sample
                )
            return TrialOutcome(
                "ok",
                median_ms=float(statistics.median(timings)),
                timings_ms=timings,
                compile_ms=compile_ms,
                max_abs_error=max_abs_error,
            )
        except Exception as exc:
            return TrialOutcome(
                "runtime_error",
                compile_ms=compile_ms,
                max_abs_error=max_abs_error,
                error=f"{type(exc).__name__}: {exc}"[:4000],
            )
        finally:
            if previous_l2 is not None:
                _set_l2_fetch_granularity(previous_l2)


def _intern_for_tuning(config: MXFP8PrequantConfig) -> str:
    # Lazy import avoids exporting frontend registries as public API.
    from .fp8 import _intern_prequant_config

    return _intern_prequant_config(config)


class PrequantCoordinateDescentTuner:
    def __init__(
        self,
        problem: MXFP8Problem,
        evaluator: MXFP8PrequantEvaluator,
        database: PrequantJsonTuningDatabase,
        policy: CoordinateDescentPolicy,
        *,
        axes: Mapping[str, Iterable[dict[str, object]]] = PREQUANT_SEARCH_SPACE,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.problem = problem
        self.evaluator = evaluator
        self.database = database
        self.policy = policy
        self.axes = {name: tuple(values) for name, values in axes.items()}
        self.progress = progress
        self.evaluated_trials = 0
        self.reused_trials = 0
        self._memo: dict[str, TrialOutcome] = {}
        self._start = 0.0
        self._session = ""
        missing = set(policy.coordinate_order).difference(self.axes)
        if missing:
            raise ValueError(f"prequant coordinate order has missing axes: {sorted(missing)}")

    def _log(self, message: str) -> None:
        if self.progress is not None:
            self.progress(f"[{time.monotonic() - self._start:8.2f}s] {message}")

    def _evaluate(
        self,
        config: MXFP8PrequantConfig,
        pass_index: int,
        coordinate: str,
        value: object,
    ) -> TrialOutcome:
        config_id = prequant_config_id(config)
        if config_id in self._memo:
            self.reused_trials += 1
            return self._memo[config_id]
        if self.policy.resume and not self.policy.force:
            cached = self.database.get_trial(config)
            if cached is not None:
                self.reused_trials += 1
                self._memo[config_id] = cached
                self._log(f"REUSE {config_id} {coordinate} status={cached.status}")
                return cached
        rejection = config.rejection(self.problem)
        if rejection is None:
            self._log(f"RUN   {config_id} {coordinate} compile -> check -> E2E benchmark")
            try:
                outcome = self.evaluator(config)
            except Exception as exc:
                outcome = TrialOutcome(
                    "runtime_error",
                    error=f"{type(exc).__name__}: {exc}"[:4000],
                )
        else:
            outcome = TrialOutcome("implementation_rejected", error=rejection)
        self.evaluated_trials += 1
        self._memo[config_id] = outcome
        self.database.record(
            config,
            outcome,
            session_id=self._session,
            pass_index=pass_index,
            coordinate=coordinate,
            coordinate_value=value,
        )
        score = "" if outcome.median_ms is None else f" {outcome.median_ms * 1000:.4f}us"
        reason = "" if outcome.error is None else f" reason={outcome.error}"
        self._log(f"SAVE  {config_id} {outcome.status}{score}{reason}")
        return outcome

    def _restart_seed(
        self, initial: MXFP8PrequantConfig, rng: random.Random
    ) -> MXFP8PrequantConfig:
        current = initial
        coordinates = list(self.policy.coordinate_order)
        rng.shuffle(coordinates)
        for coordinate in coordinates:
            variants = list(self.axes[coordinate])
            rng.shuffle(variants)
            for variant in variants:
                candidate = self._candidate(current, coordinate, variant)
                if candidate.rejection(self.problem) is None:
                    current = candidate
                    break
        return current

    def _candidate(
        self,
        current: MXFP8PrequantConfig,
        coordinate: str,
        variant: Mapping[str, object],
    ) -> MXFP8PrequantConfig:
        # Cross the launch-topology boundary together with a W-only schedule
        # move. Otherwise the slower same-schedule separate launch can trap
        # ordinary coordinate descent in the dual basin.
        if current.quant_launches == "dual" and coordinate.startswith("w_"):
            variant = {**variant, "quant_launches": "separate"}
        candidate = update_prequant_config(current, variant)
        # A 256-row/column output tile is only a viable structural candidate
        # with the low-SMEM direct epilogue and one operand stage.
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

    def tune(
        self, initial: MXFP8PrequantConfig = DEFAULT_MXFP8_PREQUANT_CONFIG
    ) -> PrequantTuningResult:
        self._start = time.monotonic()
        self._session = self.database.start_session(self.policy)
        status = "failed"
        try:
            cached = self.database.best() if self.policy.resume else None
            global_config = cached[0] if cached is not None else initial
            first = self._evaluate(global_config, -1, "initial", None)
            if not first.successful:
                raise RuntimeError(
                    f"initial prequant configuration failed: {first.status}: {first.error}"
                )
            global_score = float(first.median_ms)
            rng = random.Random(self.policy.seed)
            budget_stop = False
            for restart in range(self.policy.restarts):
                current = (
                    global_config
                    if restart == 0
                    else self._restart_seed(initial, rng)
                )
                seed = self._evaluate(current, -(restart + 1), "restart", restart)
                if not seed.successful:
                    continue
                current_score = float(seed.median_ms)
                self._log(
                    f"BASIN {restart + 1}/{self.policy.restarts} "
                    f"seed={prequant_config_id(current)} {current_score * 1000:.4f}us"
                )
                for local_pass in range(self.policy.max_passes):
                    changed = False
                    coordinates = list(self.policy.coordinate_order)
                    if self.policy.randomize_coordinates:
                        rng.shuffle(coordinates)
                    for coordinate in coordinates:
                        if time.monotonic() - self._start >= self.policy.time_budget_s:
                            budget_stop = True
                            break
                        axis_config, axis_score = current, current_score
                        self._log(
                            f"AXIS  {coordinate} variants={len(self.axes[coordinate])}"
                        )
                        for variant in self.axes[coordinate]:
                            if time.monotonic() - self._start >= self.policy.time_budget_s:
                                budget_stop = True
                                break
                            candidate = self._candidate(current, coordinate, variant)
                            outcome = self._evaluate(
                                candidate,
                                restart * self.policy.max_passes + local_pass,
                                coordinate,
                                variant,
                            )
                            if not outcome.successful:
                                continue
                            score = float(outcome.median_ms)
                            if score < axis_score * (1 - self.policy.min_improvement):
                                axis_config, axis_score = candidate, score
                        if axis_config != current:
                            current, current_score = axis_config, axis_score
                            changed = True
                            self._log(
                                f"BEST  {coordinate} {current_score * 1000:.4f}us "
                                f"config={prequant_config_id(current)}"
                            )
                    if budget_stop or not changed:
                        break
                if current_score < global_score * (1 - self.policy.min_improvement):
                    global_config, global_score = current, current_score
                if budget_stop:
                    break
            status = "budget_exhausted" if budget_stop else "complete"
            self.database.select(global_config, global_score, self._session)
            return PrequantTuningResult(
                config=global_config,
                median_ms=global_score,
                database_path=self.database.path,
                session_id=self._session,
                evaluated_trials=self.evaluated_trials,
                reused_trials=self.reused_trials,
                elapsed_s=time.monotonic() - self._start,
            )
        finally:
            self.database.finish_session(
                self._session, status, time.monotonic() - self._start
            )


def _policy_for_prequant(policy: CoordinateDescentPolicy | None) -> CoordinateDescentPolicy:
    if policy is None:
        return CoordinateDescentPolicy(
            time_budget_s=float(os.getenv("RTX_MXFP8_AUTOTUNE_SECONDS", "1800")),
            max_passes=int(os.getenv("RTX_MXFP8_AUTOTUNE_PASSES", "4")),
            restarts=int(os.getenv("RTX_MXFP8_AUTOTUNE_RESTARTS", "2")),
            warmup=int(os.getenv("RTX_MXFP8_AUTOTUNE_WARMUP", "10")),
            samples=int(os.getenv("RTX_MXFP8_AUTOTUNE_SAMPLES", "11")),
            calls_per_sample=int(
                os.getenv("RTX_MXFP8_AUTOTUNE_CALLS_PER_SAMPLE", "50")
            ),
            correctness_rtol=5e-2,
            correctness_atol=5e-1,
            coordinate_order=PREQUANT_COORDINATE_ORDER,
        )
    # CoordinateDescentPolicy predates the joint tuner. Treat its fused default
    # order as "unspecified" while preserving every other user setting.
    from .kernels.mxfp8 import FWD_COORDINATE_ORDER

    if policy.coordinate_order == FWD_COORDINATE_ORDER:
        return replace(policy, coordinate_order=PREQUANT_COORDINATE_ORDER)
    return policy


def load_cached_mxfp8_prequant_config(
    problem: MXFP8Problem,
    *,
    device: torch.device | str | int | None = None,
    cache_dir: Path | str | None = None,
    axes: Mapping[str, Iterable[dict[str, object]]] = PREQUANT_SEARCH_SPACE,
) -> MXFP8PrequantConfig | None:
    database = PrequantJsonTuningDatabase(
        cache_dir, DeviceFingerprint.current(device), problem, axes
    )
    best = database.best()
    if best is None or best[0].rejection(problem) is not None:
        return None
    return best[0]


def tune_mxfp8_prequant(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    policy: CoordinateDescentPolicy | None = None,
    initial: MXFP8PrequantConfig = DEFAULT_MXFP8_PREQUANT_CONFIG,
    cache_dir: Path | str | None = None,
    axes: Mapping[str, Iterable[dict[str, object]]] = PREQUANT_SEARCH_SPACE,
    progress: ProgressCallback | None = print,
) -> PrequantTuningResult:
    selected_policy = _policy_for_prequant(policy)
    x_2d = x.reshape(-1, x.shape[-1])
    evaluator = MXFP8PrequantEvaluator(x_2d, weight, selected_policy)
    database = PrequantJsonTuningDatabase(
        cache_dir, DeviceFingerprint.current(x.device), evaluator.problem, axes
    )
    return PrequantCoordinateDescentTuner(
        evaluator.problem,
        evaluator,
        database,
        selected_policy,
        axes=axes,
        progress=progress,
    ).tune(initial)


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=1800.0)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--calls-per-sample", type=int, default=50)
    parser.add_argument("--min-improvement", type=float, default=0.002)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--randomize-coordinates", action="store_true")
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument(
        "--correctness", choices=("baseline", "torch", "none"), default="baseline"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is not available")
    if torch.cuda.get_device_capability()[0] != 12:
        parser.error("joint MXFP8 tuning requires an SM120/SM121 GPU")
    torch.manual_seed(args.seed)
    x = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(args.n, args.k, device="cuda", dtype=torch.bfloat16)
    policy = CoordinateDescentPolicy(
        time_budget_s=args.seconds,
        max_passes=args.passes,
        restarts=args.restarts,
        warmup=args.warmup,
        samples=args.samples,
        calls_per_sample=args.calls_per_sample,
        min_improvement=args.min_improvement,
        correctness_rtol=5e-2,
        correctness_atol=5e-1,
        correctness=args.correctness,
        force=args.force,
        randomize_coordinates=args.randomize_coordinates,
        seed=args.seed,
        coordinate_order=PREQUANT_COORDINATE_ORDER,
    )

    def progress(message: str) -> None:
        if not args.quiet:
            print(message, flush=True)
        if args.log_file is not None:
            args.log_file.parent.mkdir(parents=True, exist_ok=True)
            with args.log_file.open("a", encoding="utf-8") as sink:
                sink.write(f"{_utc_now()} {message}\n")

    result = tune_mxfp8_prequant(
        x,
        weight,
        policy=policy,
        cache_dir=args.cache_dir,
        progress=progress,
    )
    print(
        json.dumps(
            {
                "config": prequant_config_to_dict(result.config),
                "median_ms": result.median_ms,
                "database_path": str(result.database_path),
                "session_id": result.session_id,
                "evaluated_trials": result.evaluated_trials,
                "reused_trials": result.reused_trials,
                "elapsed_s": result.elapsed_s,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


__all__ = [
    "PREQUANT_COORDINATE_ORDER",
    "PREQUANT_SEARCH_SPACE",
    "PrequantCoordinateDescentTuner",
    "PrequantJsonTuningDatabase",
    "PrequantTuningResult",
    "load_cached_mxfp8_prequant_config",
    "prequant_config_from_dict",
    "prequant_config_id",
    "prequant_config_to_dict",
    "tune_mxfp8_prequant",
    "update_prequant_config",
]


if __name__ == "__main__":
    _cli()
