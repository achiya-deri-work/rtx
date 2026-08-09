"""Persistent end-to-end coordinate tuner for MXFP8 linear backward.

Every timing covers four logical-layout operand quantizations, the dX GEMM and
the long-reduction dW GEMM. Logical transpose layouts are metadata, never
physical kernels or buffers. dX and dW remain independently tunable because
their aspect ratios and reduction lengths are unrelated.
"""

from __future__ import annotations

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
from .fp8_bwd import _build_bwd_runner
from .kernels.mxfp8 import (
    DEFAULT_MXFP8_FWD_CONFIG,
    FWD_SEARCH_SPACE,
    MXFP8FwdConfig,
    MXFP8Problem,
    fwd_config_from_dict,
    normalize_fwd_config,
)
from .kernels.mxfp8_bwd import (
    DEFAULT_MXFP8_BWD_CONFIG,
    MXFP8BwdConfig,
    MXFP8BwdMatmulConfig,
)
from .configs import MXFP8GemmConfig, MXFP8QuantConfig

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None


SCHEMA_VERSION = 1
KERNEL_NAME = "mxfp8_bwd_e2e"
KERNEL_REVISION = 10


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


def _layout_variants() -> tuple[dict[str, object], ...]:
    def variant(
        a_layout: str,
        b_layout: str,
        gemm_layout: str,
        scale_role: str,
        tile_m: int,
        atom_m: int,
    ) -> dict[str, object]:
        return {
            "quant_a": {"scale_layout": a_layout},
            "quant_b": {"scale_layout": b_layout},
            "gemm": {
                "scale_layout": gemm_layout,
                "scale_role": scale_role,
                "tile_m": tile_m,
                "tile_n": 128,
                "tile_k": 128,
                "atom_layout_m": atom_m,
                "atom_layout_n": 2,
                **({"stages": 1} if gemm_layout == "mma64x128" else {}),
            },
        }

    return (
        variant("mma128", "mma128", "mma128", "tma", 128, 4),
        variant("mma64", "mma128", "mma64x128", "tma", 64, 2),
        variant("row_major", "row_major", "row_major", "consumers", 64, 2),
        variant("row_major", "row_major", "row_major", "producer", 64, 2),
        variant("row_major", "row_major", "row_major", "consumers", 128, 4),
        variant("row_major", "row_major", "row_major", "producer", 128, 4),
    )


def _matmul_axes(prefix: str) -> dict[str, tuple[dict[str, object], ...]]:
    def wrap(value: Mapping[str, object]) -> dict[str, object]:
        return {prefix: dict(value)}

    axes: dict[str, tuple[dict[str, object], ...]] = {
        f"{prefix}_backend": tuple(
            wrap({"backend": value}) for value in ("fused", "decomposed")
        ),
        f"{prefix}_quant_launches": tuple(
            wrap({"quant_launches": value}) for value in ("dual", "separate")
        ),
        f"{prefix}_layout_transport": tuple(wrap(value) for value in _layout_variants()),
        f"{prefix}_a_vector": tuple(
            wrap({"quant_a": value}) for value in _quant_vector_variants()
        ),
        f"{prefix}_b_vector": tuple(
            wrap({"quant_b": value}) for value in _quant_vector_variants()
        ),
        f"{prefix}_a_arithmetic": tuple(
            wrap({"quant_a": value}) for value in _quant_arithmetic_variants()
        ),
        f"{prefix}_b_arithmetic": tuple(
            wrap({"quant_b": value}) for value in _quant_arithmetic_variants()
        ),
        f"{prefix}_a_launch": tuple(
            wrap({"quant_a": value}) for value in _quant_launch_variants()
        ),
        f"{prefix}_b_launch": tuple(
            wrap({"quant_b": value}) for value in _quant_launch_variants()
        ),
        f"{prefix}_a_registers": tuple(
            wrap({"quant_a": {"maxrregcount": value}})
            for value in (64, 80, 96, 112, 128, 160, 192, 224, 255)
        ),
        f"{prefix}_b_registers": tuple(
            wrap({"quant_b": {"maxrregcount": value}})
            for value in (64, 80, 96, 112, 128, 160, 192, 224, 255)
        ),
        f"{prefix}_a_scale_store": tuple(
            wrap({"quant_a": {"native_scale_store": value}})
            for value in ("scalar", "packed")
        ),
        f"{prefix}_b_scale_store": tuple(
            wrap({"quant_b": {"native_scale_store": value}})
            for value in ("scalar", "packed")
        ),
        f"{prefix}_gemm_geometry": tuple(
            wrap(
                {
                    "gemm": {
                        "tile_m": tile_m,
                        "tile_n": tile_n,
                        "tile_k": tile_k,
                        "atom_layout_m": atom_m,
                        "atom_layout_n": atom_n,
                    }
                }
            )
            for tile_m, atom_m in ((64, 2), (128, 2), (128, 4), (128, 8), (256, 8))
            for tile_n, atom_n in ((128, 2), (256, 4))
            for tile_k in (128,)
        ),
        f"{prefix}_gemm_stages": tuple(
            wrap({"gemm": {"stages": value}}) for value in (1, 2, 3, 4)
        ),
        f"{prefix}_ldmatrix": tuple(
            wrap({"gemm": {"a_ldmatrix_matrices": a, "b_ldmatrix_matrices": b}})
            for a in (1, 2, 4)
            for b in (1, 2, 4)
        ),
        f"{prefix}_smem_swizzle": tuple(
            wrap({"gemm": {"a_swizzle": a, "b_swizzle": b}})
            for a in ("none", "32b", "64b", "128b")
            for b in ("none", "32b", "64b", "128b")
        ),
        f"{prefix}_scale_s2r": tuple(
            wrap({"gemm": {"sfa_s2r_bits": a, "sfb_s2r_bits": b}})
            for a in (0, 8)
            for b in (0, 8)
        ),
        f"{prefix}_scale_schedule": tuple(
            wrap(
                {
                    "gemm": {
                        "scale_schedule": schedule,
                        "scale_load_vec": width,
                    }
                }
            )
            for schedule in ("before_wait", "after_wait")
            for width in (1, 2, 4, 8)
        ),
        f"{prefix}_scale_cache": tuple(
            wrap({"gemm": value})
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
        f"{prefix}_gemm_registers": tuple(
            wrap(
                {
                    "gemm": {
                        "producer_registers": producer,
                        "consumer_registers": consumer,
                        "maxrregcount": maximum,
                    }
                }
            )
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
        f"{prefix}_epilogue": tuple(
            wrap({"gemm": {"epilogue": epilogue, "store_vec": vector}})
            for epilogue, vector in (
                ("direct", 1),
                ("tma", 1),
                ("tma", 2),
                ("tma", 4),
            )
        ),
        f"{prefix}_raster_group": tuple(
            wrap({"gemm": {"raster": raster, "grid_swizzle": group}})
            for raster in ("m", "n")
            for group in (1, 2, 4, 8)
        ),
        f"{prefix}_reduction": tuple(
            wrap(value)
            for value in (
                {
                    "reduction": "full_fp32",
                    "split_reduction": 1,
                    "reduction_tile": 0,
                    "workspace_epilogue": "none",
                },
                *(
                    {
                        "reduction": "split_fp32_workspace",
                        "split_reduction": split,
                        "reduction_tile": tile,
                        "workspace_epilogue": epilogue,
                    }
                    for split in (2, 4, 8, 16, 32)
                    for tile in (128, 256, 512, 1024, 2048, 4096)
                    for epilogue in ("serial", "tree", "persistent_tree")
                ),
                *(
                    {
                        "reduction": family,
                        "split_reduction": split,
                        "reduction_tile": tile,
                        "workspace_epilogue": "none",
                    }
                    for family in ("split_fp32_atomic",)
                    for split in (2, 4, 8, 16)
                    for tile in (128, 256, 512, 1024, 2048)
                ),
            )
        ),
        f"{prefix}_reduction_epilogue_launch": tuple(
            wrap(
                {
                    "reduction_threads": threads,
                    "reduction_vector": vector,
                    "reduction_waves": waves,
                }
            )
            for threads in (64, 128, 256, 512, 1024)
            for vector in (1, 2, 4, 8)
            for waves in (1, 2, 3, 4, 6, 8)
        ),
    }
    # Backward's fused family is the dynamic-forward implementation itself,
    # specialized only by the two logical GMEM layouts. Keep its entire real
    # code-generation space available independently for dX and dW.
    axes.update(
        {
            f"{prefix}_fused_{coordinate}": tuple(
                wrap({"fused": {coordinate: value}}) for value in variants
            )
            for coordinate, variants in FWD_SEARCH_SPACE.items()
        }
    )
    transposed_tiles = tuple(
        {
            "transposed_tile_rows": rows,
            "transposed_tile_k": tile_k,
            "transposed_smem_padding": padding,
        }
        for rows in (32, 64, 128, 256)
        for tile_k in (32, 64, 128)
        for padding in (0, 1, 2, 4, 8)
    )
    if prefix == "dx":
        axes["dx_b_logical_tile"] = tuple(
            wrap({"quant_b": value}) for value in transposed_tiles
        )
        axes["dx_b_scale_store"] = tuple(
            wrap(
                {
                    "quant_b": {
                        "native_scale_store": value,
                        "transposed_tile_k": 128 if value == "packed" else 32,
                    }
                }
            )
            for value in ("scalar", "packed")
        )
    else:
        axes["dw_a_logical_tile"] = tuple(
            wrap({"quant_a": value}) for value in transposed_tiles
        )
        axes["dw_b_logical_tile"] = tuple(
            wrap({"quant_b": value}) for value in transposed_tiles
        )
        for operand in ("a", "b"):
            axes[f"dw_{operand}_scale_store"] = tuple(
                wrap(
                    {
                        f"quant_{operand}": {
                            "native_scale_store": value,
                            "transposed_tile_k": (
                                128 if value == "packed" else 32
                            ),
                        }
                    }
                )
                for value in ("scalar", "packed")
            )
    if prefix == "dx":
        axes["dx_b_logical_transport"] = tuple(
            wrap(
                {
                    "quant_b": {
                        "transposed_load_engine": value,
                        "transposed_smem_padding": 0 if value == "cp_async" else 1,
                    }
                }
            )
            for value in ("register", "cp_async")
        )
    else:
        axes["dw_a_logical_transport"] = tuple(
            wrap(
                {
                    "quant_a": {
                        "transposed_load_engine": value,
                        "transposed_smem_padding": 0 if value == "cp_async" else 1,
                    }
                }
            )
            for value in ("register", "cp_async")
        )
        axes["dw_b_logical_transport"] = tuple(
            wrap(
                {
                    "quant_b": {
                        "transposed_load_engine": value,
                        "transposed_smem_padding": 0 if value == "cp_async" else 1,
                    }
                }
            )
            for value in ("register", "cp_async")
        )
    return axes


BWD_SEARCH_SPACE: dict[str, tuple[dict[str, object], ...]] = {
    "execution_order": (
        {"execution_order": "dx_first"},
        {"execution_order": "dw_first"},
        {"execution_order": "interleaved"},
    ),
    "stream_schedule": tuple(
        {"stream_schedule": value} for value in ("single", "dual_stream")
    ),
    "quant_schedule": tuple(
        {"quant_schedule": value} for value in ("per_matmul", "quad")
    ),
    # Quad quantization shares one transposed schedule across dX.B, dW.A and
    # dW.B. This compound coordinate lets local search cross that legality
    # boundary atomically instead of rejecting every single-leaf mutation.
    "quad_logical_transport": tuple(
        {
            "dx": {
                "quant_b": {
                    "transposed_load_engine": value,
                    "transposed_smem_padding": 0 if value == "cp_async" else 1,
                }
            },
            "dw": {
                "quant_a": {
                    "transposed_load_engine": value,
                    "transposed_smem_padding": 0 if value == "cp_async" else 1,
                },
                "quant_b": {
                    "transposed_load_engine": value,
                    "transposed_smem_padding": 0 if value == "cp_async" else 1,
                },
            },
        }
        for value in ("register", "cp_async")
    ),
    "quad_native_scale_store": tuple(
        {
            "dx": {
                "quant_b": {
                    "native_scale_store": value,
                    "transposed_tile_k": 128 if value == "packed" else 32,
                }
            },
            "dw": {
                "quant_a": {
                    "native_scale_store": value,
                    "transposed_tile_k": 128 if value == "packed" else 32,
                },
                "quant_b": {
                    "native_scale_store": value,
                    "transposed_tile_k": 128 if value == "packed" else 32,
                },
            },
        }
        for value in ("scalar", "packed")
    ),
    **_matmul_axes("dx"),
    **_matmul_axes("dw"),
}
BWD_COORDINATE_ORDER = tuple(BWD_SEARCH_SPACE)


def _matmul_from_dict(value: Mapping[str, object]) -> MXFP8BwdMatmulConfig:
    quant_b_value = value.get("quant_b")
    fused_value = value.get("fused")
    return MXFP8BwdMatmulConfig(
        a_orientation=str(value.get("a_orientation", "row")),
        b_orientation=str(value.get("b_orientation", "transpose")),
        # Revision <=3 records described only the decomposed implementation.
        backend=str(value.get("backend", "decomposed")),
        fused=(
            DEFAULT_MXFP8_FWD_CONFIG
            if fused_value is None
            else fwd_config_from_dict(dict(fused_value))  # type: ignore[arg-type]
        ),
        quant_launches=str(value.get("quant_launches", "separate")),
        quant_a=MXFP8QuantConfig(**dict(value["quant_a"])),  # type: ignore[arg-type]
        quant_b=(
            None
            if quant_b_value is None
            else MXFP8QuantConfig(**dict(quant_b_value))  # type: ignore[arg-type]
        ),
        gemm=MXFP8GemmConfig(**dict(value["gemm"])),  # type: ignore[arg-type]
        reduction=str(value.get("reduction", "full_fp32")),
        split_reduction=int(value.get("split_reduction", 1)),
        reduction_tile=int(value.get("reduction_tile", 0)),
        workspace_epilogue=str(value.get("workspace_epilogue", "none")),
        reduction_threads=int(value.get("reduction_threads", 256)),
        reduction_vector=int(value.get("reduction_vector", 4)),
        reduction_waves=int(value.get("reduction_waves", 1)),
        tile_scheduler=str(value.get("tile_scheduler", "static")),
        persistent_waves=int(value.get("persistent_waves", 1)),
        tiles_per_cta=int(value.get("tiles_per_cta", 1)),
        reuse_operand=str(value.get("reuse_operand", "none")),
        tile_locality=str(value.get("tile_locality", "raster")),
    ).normalized()


def bwd_config_to_dict(config: MXFP8BwdConfig) -> dict[str, object]:
    return asdict(config.normalized())


def bwd_config_from_dict(value: Mapping[str, object]) -> MXFP8BwdConfig:
    return MXFP8BwdConfig(
        dx=_matmul_from_dict(dict(value["dx"])),  # type: ignore[arg-type]
        dw=_matmul_from_dict(dict(value["dw"])),  # type: ignore[arg-type]
        execution_order=str(value.get("execution_order", "dx_first")),
        stream_schedule=str(value.get("stream_schedule", "single")),
        quant_schedule=str(value.get("quant_schedule", "per_matmul")),
    ).normalized()


def _update_matmul(
    config: MXFP8BwdMatmulConfig,
    updates: Mapping[str, object],
) -> MXFP8BwdMatmulConfig:
    qa = asdict(config.quant_a)
    qb = asdict(config.resolved_quant_b())
    gemm = asdict(config.gemm)
    fused_updates = dict(updates.get("fused", {}))  # type: ignore[arg-type]
    qa_updates = dict(updates.get("quant_a", {}))  # type: ignore[arg-type]
    qb_updates = dict(updates.get("quant_b", {}))  # type: ignore[arg-type]
    qa.update(qa_updates)
    qb.update(qb_updates)
    next_launches = str(updates.get("quant_launches", config.quant_launches))
    if next_launches == "dual":
        same_orientation = config.a_orientation == config.b_orientation
        if (
            same_orientation
            and config.quant_launches != "dual"
            and not qa_updates
            and not qb_updates
        ):
            # Crossing from independently tuned schedules to one kernel must
            # also pick one executable shared schedule.  Subsequent A/B axes
            # can cross back to separate launches as usual.
            qb = dict(qa)
        elif same_orientation and qa_updates and not qb_updates:
            qb.update(qa_updates)
        elif same_orientation and qb_updates and not qa_updates:
            qa.update(qb_updates)
    gemm.update(dict(updates.get("gemm", {})))  # type: ignore[arg-type]
    return MXFP8BwdMatmulConfig(
        a_orientation=config.a_orientation,
        b_orientation=config.b_orientation,
        backend=str(updates.get("backend", config.backend)),
        fused=(
            normalize_fwd_config(config.fused, **fused_updates)
            if fused_updates
            else config.fused
        ),
        quant_launches=next_launches,
        quant_a=MXFP8QuantConfig(**qa),
        quant_b=MXFP8QuantConfig(**qb),
        gemm=MXFP8GemmConfig(**gemm),
        reduction=str(updates.get("reduction", config.reduction)),
        split_reduction=int(updates.get("split_reduction", config.split_reduction)),
        reduction_tile=int(updates.get("reduction_tile", config.reduction_tile)),
        workspace_epilogue=str(
            updates.get("workspace_epilogue", config.workspace_epilogue)
        ),
        reduction_threads=int(
            updates.get("reduction_threads", config.reduction_threads)
        ),
        reduction_vector=int(
            updates.get("reduction_vector", config.reduction_vector)
        ),
        reduction_waves=int(updates.get("reduction_waves", config.reduction_waves)),
        tile_scheduler=str(updates.get("tile_scheduler", config.tile_scheduler)),
        persistent_waves=int(updates.get("persistent_waves", config.persistent_waves)),
        tiles_per_cta=int(updates.get("tiles_per_cta", config.tiles_per_cta)),
        reuse_operand=str(updates.get("reuse_operand", config.reuse_operand)),
        tile_locality=str(updates.get("tile_locality", config.tile_locality)),
    ).normalized()


def update_bwd_config(
    config: MXFP8BwdConfig,
    updates: Mapping[str, object],
) -> MXFP8BwdConfig:
    return MXFP8BwdConfig(
        dx=_update_matmul(config.dx, dict(updates.get("dx", {}))),  # type: ignore[arg-type]
        dw=_update_matmul(config.dw, dict(updates.get("dw", {}))),  # type: ignore[arg-type]
        execution_order=str(updates.get("execution_order", config.execution_order)),
        stream_schedule=str(updates.get("stream_schedule", config.stream_schedule)),
        quant_schedule=str(updates.get("quant_schedule", config.quant_schedule)),
    ).normalized()


def bwd_config_id(config: MXFP8BwdConfig) -> str:
    payload = json.dumps(bwd_config_to_dict(config), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:20]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _search_digest(axes: Mapping[str, Iterable[object]]) -> str:
    return hashlib.sha256(
        json.dumps(
            {name: list(values) for name, values in axes.items()},
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class BwdTuningResult:
    config: MXFP8BwdConfig
    median_ms: float
    database_path: Path
    session_id: str
    evaluated_trials: int
    reused_trials: int
    elapsed_s: float


class BwdJsonTuningDatabase:
    """Atomic and lock-protected database scoped by device, shape and axes."""

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
        if not (
            document.get("schema_version") == SCHEMA_VERSION
            and document.get("kernel_revision") == KERNEL_REVISION
            and document.get("fingerprint") == self.fingerprint.as_dict()
            and document.get("problem") == asdict(self.problem)
            and document.get("search_space_digest") == self.search_digest
        ):
            raise RuntimeError(f"incompatible MXFP8 backward database: {self.path}")
        return document

    def _write(self, document: dict[str, object]) -> None:
        document["updated_at"] = _utc_now()
        temporary: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=self.path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as sink:
                temporary = sink.name
                json.dump(document, sink, indent=2, sort_keys=True)
                sink.write("\n")
                sink.flush()
                os.fsync(sink.fileno())
            os.replace(temporary, self.path)
            temporary = None
        finally:
            if temporary is not None and os.path.exists(temporary):
                os.unlink(temporary)

    def _mutate(self, callback) -> object:
        with self._locked():
            document = self._read()
            result = callback(document)
            self._write(document)
            return result

    def start_session(self, policy: CoordinateDescentPolicy) -> str:
        session = uuid.uuid4().hex
        self._mutate(
            lambda document: document["sessions"].append(  # type: ignore[union-attr]
                {
                    "id": session,
                    "started_at": _utc_now(),
                    "status": "running",
                    "policy": asdict(policy),
                }
            )
        )
        return session

    def finish_session(self, session: str, status: str, elapsed_s: float) -> None:
        def finish(document: dict[str, object]) -> None:
            for item in document["sessions"]:  # type: ignore[union-attr]
                if item["id"] == session:
                    item.update(
                        status=status,
                        finished_at=_utc_now(),
                        elapsed_s=elapsed_s,
                    )
                    return

        self._mutate(finish)

    def get_trial(self, config: MXFP8BwdConfig) -> TrialOutcome | None:
        with self._locked():
            trial = self._read()["trials"].get(bwd_config_id(config))  # type: ignore[union-attr]
        return None if trial is None else TrialOutcome.from_dict(trial["outcome"])

    def record(
        self,
        config: MXFP8BwdConfig,
        outcome: TrialOutcome,
        *,
        session: str,
        pass_index: int,
        coordinate: str,
        coordinate_value: object,
    ) -> None:
        key = bwd_config_id(config)

        def add(document: dict[str, object]) -> None:
            document["trials"][key] = {  # type: ignore[index]
                "config": bwd_config_to_dict(config),
                "outcome": outcome.as_dict(),
                "session_id": session,
                "pass_index": pass_index,
                "coordinate": coordinate,
                "coordinate_value": coordinate_value,
                "recorded_at": _utc_now(),
            }

        self._mutate(add)

    def select(self, config: MXFP8BwdConfig, median_ms: float, session: str) -> None:
        self._mutate(
            lambda document: document.update(
                best={
                    "config": bwd_config_to_dict(config),
                    "median_ms": median_ms,
                    "session_id": session,
                    "selected_at": _utc_now(),
                }
            )
        )

    def best(self) -> tuple[MXFP8BwdConfig, float] | None:
        with self._locked():
            best = self._read().get("best")
        if best is None:
            return None
        return bwd_config_from_dict(best["config"]), float(best["median_ms"])


class MXFP8BwdEvaluator:
    def __init__(
        self,
        grad_output: torch.Tensor,
        x: torch.Tensor,
        weight: torch.Tensor,
        policy: CoordinateDescentPolicy,
    ) -> None:
        self.grad_output = grad_output.contiguous()
        self.x = x.contiguous()
        self.weight = weight.contiguous()
        self.policy = policy
        self.problem = MXFP8Problem(x.shape[0], weight.shape[0], x.shape[1])
        self._reference: tuple[torch.Tensor, torch.Tensor] | None = None

    def reference(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._reference is None:
            if self.policy.correctness == "torch":
                dx = self.grad_output.float() @ self.weight.float()
                dw = self.grad_output.float().T @ self.x.float()
                self._reference = (dx.to(torch.bfloat16), dw.to(torch.bfloat16))
            else:
                runner = _build_bwd_runner(
                    self.problem, DEFAULT_MXFP8_BWD_CONFIG, self.x.device
                )
                dx = torch.empty_like(self.x)
                dw = torch.empty_like(self.weight)
                runner(self.grad_output, self.x, self.weight, dx, dw)
                torch.cuda.synchronize(self.x.device)
                self._reference = (dx.clone(), dw.clone())
        return self._reference

    def __call__(self, config: MXFP8BwdConfig) -> TrialOutcome:
        start_compile = time.monotonic()
        try:
            runner = _build_bwd_runner(self.problem, config, self.x.device)
        except Exception as exc:
            return TrialOutcome(
                "compile_error",
                compile_ms=(time.monotonic() - start_compile) * 1000,
                error=f"{type(exc).__name__}: {exc}"[:4000],
            )
        compile_ms = (time.monotonic() - start_compile) * 1000
        dx = torch.empty_like(self.x)
        dw = torch.empty_like(self.weight)
        try:
            runner(self.grad_output, self.x, self.weight, dx, dw)
            torch.cuda.synchronize(self.x.device)
            max_error: float | None = None
            if self.policy.correctness != "none":
                ref_dx, ref_dw = self.reference()
                max_error = max(
                    float((dx.float() - ref_dx.float()).abs().max()),
                    float((dw.float() - ref_dw.float()).abs().max()),
                )
                if not (
                    torch.allclose(
                        dx,
                        ref_dx,
                        rtol=self.policy.correctness_rtol,
                        atol=self.policy.correctness_atol,
                        equal_nan=True,
                    )
                    and torch.allclose(
                        dw,
                        ref_dw,
                        rtol=self.policy.correctness_rtol,
                        atol=self.policy.correctness_atol,
                        equal_nan=True,
                    )
                ):
                    return TrialOutcome(
                        "correctness_error",
                        compile_ms=compile_ms,
                        max_abs_error=max_error,
                        error="candidate dX/dW differ from the backward reference",
                    )
            for _ in range(self.policy.warmup):
                runner(self.grad_output, self.x, self.weight, dx, dw)
            torch.cuda.synchronize(self.x.device)
            timings: list[float] = []
            for _ in range(self.policy.samples):
                begin = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                begin.record()
                for _call in range(self.policy.calls_per_sample):
                    runner(self.grad_output, self.x, self.weight, dx, dw)
                end.record()
                end.synchronize()
                timings.append(
                    float(begin.elapsed_time(end)) / self.policy.calls_per_sample
                )
            return TrialOutcome(
                "ok",
                median_ms=float(statistics.median(timings)),
                timings_ms=timings,
                compile_ms=compile_ms,
                max_abs_error=max_error,
            )
        except Exception as exc:
            return TrialOutcome(
                "runtime_error",
                compile_ms=compile_ms,
                error=f"{type(exc).__name__}: {exc}"[:4000],
            )


class BwdCoordinateDescentTuner:
    def __init__(
        self,
        evaluator: MXFP8BwdEvaluator,
        database: BwdJsonTuningDatabase,
        policy: CoordinateDescentPolicy,
        *,
        axes: Mapping[str, Iterable[dict[str, object]]],
        progress: ProgressCallback | None,
    ) -> None:
        self.evaluator = evaluator
        self.database = database
        self.policy = policy
        self.axes = {name: tuple(values) for name, values in axes.items()}
        self.progress = progress
        self.evaluated = 0
        self.reused = 0
        self.memo: dict[str, TrialOutcome] = {}
        self.started = 0.0
        self.session = ""

    def _log(self, message: str) -> None:
        if self.progress is not None:
            self.progress(f"[{time.monotonic() - self.started:8.2f}s] {message}")

    def _evaluate(
        self,
        config: MXFP8BwdConfig,
        pass_index: int,
        coordinate: str,
        value: object,
    ) -> TrialOutcome:
        key = bwd_config_id(config)
        if key in self.memo:
            self.reused += 1
            return self.memo[key]
        if self.policy.resume and not self.policy.force:
            cached = self.database.get_trial(config)
            if cached is not None:
                self.memo[key] = cached
                self.reused += 1
                self._log(f"REUSE {key} {coordinate} status={cached.status}")
                return cached
        reason = config.implementation_rejection(self.evaluator.problem)
        if reason is None:
            self._log(f"RUN   {key} {coordinate} compile -> check -> BWD E2E")
            outcome = self.evaluator(config)
        else:
            outcome = TrialOutcome("implementation_rejected", error=reason)
        self.evaluated += 1
        self.memo[key] = outcome
        self.database.record(
            config,
            outcome,
            session=self.session,
            pass_index=pass_index,
            coordinate=coordinate,
            coordinate_value=value,
        )
        score = "" if outcome.median_ms is None else f" {outcome.median_ms * 1000:.3f}us"
        reason_text = "" if outcome.error is None else f" reason={outcome.error}"
        self._log(f"SAVE  {key} {outcome.status}{score}{reason_text}")
        return outcome

    def _candidate(
        self,
        current: MXFP8BwdConfig,
        coordinate: str,
        variant: Mapping[str, object],
    ) -> MXFP8BwdConfig:
        # B-only quantizer coordinates are meaningful only after crossing from
        # the shared dual launch to independent launches.  Make that a compound
        # move so coordinate descent is not trapped by an illegal intermediate.
        path = coordinate.split("_")
        if len(path) >= 3 and path[0] in ("dx", "dw") and path[1] == "b":
            matmul = current.dx if path[0] == "dx" else current.dw
            if (
                matmul.quant_launches == "dual"
                and matmul.a_orientation == matmul.b_orientation
            ):
                variant = {
                    **variant,
                    path[0]: {
                        **dict(variant.get(path[0], {})),  # type: ignore[arg-type]
                        "quant_launches": "separate",
                    },
                }
        return update_bwd_config(current, variant)

    def _restart_seed(
        self,
        initial: MXFP8BwdConfig,
        rng: random.Random,
    ) -> MXFP8BwdConfig:
        current = initial
        coordinates = list(self.policy.coordinate_order)
        rng.shuffle(coordinates)
        for coordinate in coordinates:
            variants = list(self.axes[coordinate])
            rng.shuffle(variants)
            for variant in variants:
                candidate = self._candidate(current, coordinate, variant)
                if (
                    candidate.implementation_rejection(self.evaluator.problem)
                    is None
                ):
                    current = candidate
                    break
        return current

    def tune(
        self,
        initial: MXFP8BwdConfig = DEFAULT_MXFP8_BWD_CONFIG,
    ) -> BwdTuningResult:
        self.started = time.monotonic()
        self.session = self.database.start_session(self.policy)
        status = "failed"
        try:
            cached = self.database.best() if self.policy.resume else None
            best_config = cached[0] if cached is not None else initial
            first = self._evaluate(best_config, -1, "initial", None)
            if not first.successful:
                raise RuntimeError(
                    f"initial backward configuration failed: {first.status}: {first.error}"
                )
            best_score = float(first.median_ms)
            rng = random.Random(self.policy.seed)
            budget_stop = False
            for restart in range(self.policy.restarts):
                current = (
                    best_config
                    if restart == 0
                    else self._restart_seed(initial, rng)
                )
                seed = self._evaluate(
                    current, -(restart + 1), "restart", restart
                )
                if not seed.successful:
                    continue
                current_score = float(seed.median_ms)
                self._log(
                    f"BASIN {restart + 1}/{self.policy.restarts} "
                    f"seed={bwd_config_id(current)} {current_score * 1000:.3f}us"
                )
                for local_pass in range(self.policy.max_passes):
                    changed = False
                    coordinates = list(self.policy.coordinate_order)
                    if self.policy.randomize_coordinates:
                        rng.shuffle(coordinates)
                    for coordinate in coordinates:
                        if time.monotonic() - self.started >= self.policy.time_budget_s:
                            budget_stop = True
                            break
                        axis_config, axis_score = current, current_score
                        variants = self.axes[coordinate]
                        self._log(f"AXIS  {coordinate} variants={len(variants)}")
                        for variant in variants:
                            if time.monotonic() - self.started >= self.policy.time_budget_s:
                                budget_stop = True
                                break
                            candidate = self._candidate(current, coordinate, variant)
                            outcome = self._evaluate(
                                candidate,
                                restart * self.policy.max_passes + local_pass,
                                coordinate,
                                variant,
                            )
                            if outcome.successful and float(
                                outcome.median_ms
                            ) < axis_score * (1 - self.policy.min_improvement):
                                axis_config = candidate
                                axis_score = float(outcome.median_ms)
                        if axis_config != current:
                            current, current_score = axis_config, axis_score
                            changed = True
                            self._log(
                                f"BEST  {coordinate} {current_score * 1000:.3f}us "
                                f"config={bwd_config_id(current)}"
                            )
                    if budget_stop or not changed:
                        break
                if current_score < best_score * (1 - self.policy.min_improvement):
                    best_config, best_score = current, current_score
                if budget_stop:
                    break
            status = "budget_exhausted" if budget_stop else "complete"
            self.database.select(best_config, best_score, self.session)
            return BwdTuningResult(
                config=best_config,
                median_ms=best_score,
                database_path=self.database.path,
                session_id=self.session,
                evaluated_trials=self.evaluated,
                reused_trials=self.reused,
                elapsed_s=time.monotonic() - self.started,
            )
        finally:
            self.database.finish_session(
                self.session, status, time.monotonic() - self.started
            )


def _policy_for_bwd(policy: CoordinateDescentPolicy | None) -> CoordinateDescentPolicy:
    if policy is None:
        return CoordinateDescentPolicy(
            time_budget_s=float(os.getenv("RTX_MXFP8_BWD_AUTOTUNE_SECONDS", "1800")),
            max_passes=int(os.getenv("RTX_MXFP8_BWD_AUTOTUNE_PASSES", "4")),
            warmup=int(os.getenv("RTX_MXFP8_BWD_AUTOTUNE_WARMUP", "10")),
            samples=int(os.getenv("RTX_MXFP8_BWD_AUTOTUNE_SAMPLES", "11")),
            calls_per_sample=int(
                os.getenv("RTX_MXFP8_BWD_AUTOTUNE_CALLS_PER_SAMPLE", "20")
            ),
            correctness_rtol=7e-2,
            correctness_atol=1.0,
            coordinate_order=BWD_COORDINATE_ORDER,
        )
    from .kernels.mxfp8 import FWD_COORDINATE_ORDER

    if policy.coordinate_order == FWD_COORDINATE_ORDER:
        return replace(policy, coordinate_order=BWD_COORDINATE_ORDER)
    return policy


def tune_mxfp8_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    policy: CoordinateDescentPolicy | None = None,
    initial: MXFP8BwdConfig = DEFAULT_MXFP8_BWD_CONFIG,
    cache_dir: Path | str | None = None,
    axes: Mapping[str, Iterable[dict[str, object]]] = BWD_SEARCH_SPACE,
    progress: ProgressCallback | None = print,
) -> BwdTuningResult:
    selected_policy = _policy_for_bwd(policy)
    x_2d = x.reshape(-1, x.shape[-1])
    grad_2d = grad_output.reshape(-1, grad_output.shape[-1])
    evaluator = MXFP8BwdEvaluator(grad_2d, x_2d, weight, selected_policy)
    missing = set(selected_policy.coordinate_order).difference(axes)
    if missing:
        raise ValueError(f"backward coordinate order has missing axes: {sorted(missing)}")
    database = BwdJsonTuningDatabase(
        cache_dir,
        DeviceFingerprint.current(x.device),
        evaluator.problem,
        axes,
    )
    return BwdCoordinateDescentTuner(
        evaluator,
        database,
        selected_policy,
        axes=axes,
        progress=progress,
    ).tune(initial)


def load_cached_mxfp8_bwd_config(
    problem: MXFP8Problem,
    *,
    device: torch.device | str | int | None = None,
    cache_dir: Path | str | None = None,
    axes: Mapping[str, Iterable[dict[str, object]]] = BWD_SEARCH_SPACE,
) -> MXFP8BwdConfig | None:
    database = BwdJsonTuningDatabase(
        cache_dir, DeviceFingerprint.current(device), problem, axes
    )
    best = database.best()
    if best is None or best[0].implementation_rejection(problem) is not None:
        return None
    return best[0]


def load_mxfp8_bwd_config(path: Path | str) -> MXFP8BwdConfig:
    """Load a raw or tuner-result backward configuration JSON file."""

    document = json.loads(Path(path).read_text(encoding="utf-8"))
    value = document.get("config", document)
    if not isinstance(value, Mapping):
        raise ValueError(f"backward configuration file has no config mapping: {path}")
    return bwd_config_from_dict(value)


__all__ = [
    "BWD_COORDINATE_ORDER",
    "BWD_SEARCH_SPACE",
    "BwdCoordinateDescentTuner",
    "BwdJsonTuningDatabase",
    "BwdTuningResult",
    "MXFP8BwdEvaluator",
    "bwd_config_from_dict",
    "bwd_config_id",
    "bwd_config_to_dict",
    "load_cached_mxfp8_bwd_config",
    "load_mxfp8_bwd_config",
    "tune_mxfp8_backward",
    "update_bwd_config",
]
