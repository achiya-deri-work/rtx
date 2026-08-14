"""Small, deterministic runtime-winner cache shared by tuning frontends."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Iterable, Mapping, TypeVar

from .legacy import DeviceFingerprint, default_cache_dir
from ..kernels.mxfp8 import MXFP8Problem


ConfigT = TypeVar("ConfigT")
RUNTIME_WINNER_SCHEMA_VERSION = 2


@dataclass(frozen=True, slots=True)
class RuntimeWinnerKey:
    family: str
    problem: MXFP8Problem
    regime: str
    device_id: str
    kernel_revision: int
    variant: str = "default"

    @property
    def relative_path(self) -> Path:
        return (
            Path("runtime_winners")
            / self.family
            / self.device_id
            / (
                f"m{self.problem.m}_n{self.problem.n}_k{self.problem.k}_"
                f"{self.regime}_{self.variant}_rev{self.kernel_revision}.json"
            )
        )


def runtime_winner_key(
    family: str,
    problem: MXFP8Problem,
    *,
    device=None,
    regime: str = "hot",
    fingerprint: DeviceFingerprint | None = None,
    variant: str = "default",
    kernel_revision: int | None = None,
) -> RuntimeWinnerKey:
    selected = fingerprint or DeviceFingerprint.current(device)
    if not variant or any(value in variant for value in ("/", "\\", "..")):
        raise ValueError("runtime winner variant must be a safe path component")
    revision = (
        current_kernel_revision(family)
        if kernel_revision is None
        else int(kernel_revision)
    )
    if revision < 0:
        raise ValueError("runtime winner kernel revision must be non-negative")
    return RuntimeWinnerKey(
        family, problem, regime, selected.identifier, revision, variant
    )


def current_kernel_revision(family: str) -> int:
    """Return the cache ABI revision for a production kernel family."""

    if family == "mxfp8_fused_fwd":
        from ..kernels.mxfp8 import MXFP8_FWD_KERNEL_REVISION

        return MXFP8_FWD_KERNEL_REVISION
    if family == "nvfp4_fused_fwd":
        from ..configs.nvfp4 import NVFP4_KERNEL_REVISION

        return NVFP4_KERNEL_REVISION
    if family == "mxfp8_prequant_fwd":
        from ..prequant_autotune import KERNEL_REVISION

        return KERNEL_REVISION
    if family == "mxfp8_bwd":
        from ..bwd_autotune import KERNEL_REVISION

        return KERNEL_REVISION
    if family in ("mxfp8_weight_prequant_fwd", "mxfp8_fully_prequant_fwd"):
        from ..inference_autotune import INFERENCE_KERNEL_REVISION

        return INFERENCE_KERNEL_REVISION
    if family in ("nvfp4_weight_prequant_fwd", "nvfp4_fully_prequant_fwd"):
        from ..nvfp4_inference_autotune import NVFP4_INFERENCE_KERNEL_REVISION

        return NVFP4_INFERENCE_KERNEL_REVISION
    if family in (
        "nvfp4_dynamic_fwd",
        "nvfp4_delayed_fwd",
        "nvfp4_jit_row_region_fwd",
        "nvfp4_region_delayed_fwd",
    ):
        from ..nvfp4_inference_autotune import (
            NVFP4_DELAYED_KERNEL_REVISION,
            NVFP4_DYNAMIC_KERNEL_REVISION,
            NVFP4_JIT_ROW_REGION_KERNEL_REVISION,
            NVFP4_REGION_DELAYED_KERNEL_REVISION,
        )

        return (
            NVFP4_DELAYED_KERNEL_REVISION
            if family == "nvfp4_delayed_fwd"
            else NVFP4_JIT_ROW_REGION_KERNEL_REVISION
            if family == "nvfp4_jit_row_region_fwd"
            else NVFP4_REGION_DELAYED_KERNEL_REVISION
            if family == "nvfp4_region_delayed_fwd"
            else NVFP4_DYNAMIC_KERNEL_REVISION
        )
    raise ValueError(f"unsupported runtime winner family {family!r}")


def save_runtime_winner(
    key: RuntimeWinnerKey,
    config: Mapping[str, object],
    *,
    config_id: str,
    root: Path | str | None = None,
    median_ms: float | None = None,
    metadata: Mapping[str, object] | None = None,
) -> Path:
    base = default_cache_dir() if root is None else Path(root).expanduser()
    path = base / key.relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "schema_version": RUNTIME_WINNER_SCHEMA_VERSION,
        "family": key.family,
        "problem": {
            "m": key.problem.m,
            "n": key.problem.n,
            "k": key.problem.k,
        },
        "regime": key.regime,
        "device_id": key.device_id,
        "kernel_revision": key.kernel_revision,
        "variant": key.variant,
        "config_id": config_id,
        "config": dict(config),
        "median_ms": median_ms,
        "metadata": dict(metadata or {}),
    }
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=path.parent,
            prefix=path.name + ".",
            suffix=".tmp",
            delete=False,
        ) as sink:
            temporary = sink.name
            json.dump(document, sink, sort_keys=True, indent=2)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    return path


def load_runtime_winner(
    key: RuntimeWinnerKey,
    deserialize: Callable[[Mapping[str, object]], ConfigT],
    *,
    root: Path | str | None = None,
    rejection: Callable[[ConfigT], str | None] | None = None,
) -> ConfigT | None:
    base = default_cache_dir() if root is None else Path(root).expanduser()
    path = base / key.relative_path
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return None
    expected_problem = {
        "m": key.problem.m,
        "n": key.problem.n,
        "k": key.problem.k,
    }
    if (
        document.get("schema_version") != RUNTIME_WINNER_SCHEMA_VERSION
        or document.get("family") != key.family
        or document.get("problem") != expected_problem
        or document.get("regime") != key.regime
        or document.get("device_id") != key.device_id
        or document.get("kernel_revision") != key.kernel_revision
        or document.get("variant", "default") != key.variant
        or not isinstance(document.get("config"), dict)
    ):
        return None
    try:
        config = deserialize(document["config"])
    except (KeyError, TypeError, ValueError):
        return None
    if rejection is not None and rejection(config) is not None:
        return None
    return config


def list_runtime_winners(
    *,
    root: Path | str | None = None,
    families: Iterable[str] | None = None,
    device_ids: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    """Return installed winner documents without importing kernel backends."""

    base = default_cache_dir() if root is None else Path(root).expanduser()
    winner_root = base / "runtime_winners"
    family_filter = None if families is None else set(families)
    device_filter = None if device_ids is None else set(device_ids)
    rows: list[dict[str, object]] = []
    if not winner_root.exists():
        return rows
    for path in sorted(winner_root.glob("*/*/*.json")):
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            rows.append(
                {
                    "path": str(path),
                    "valid": False,
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            continue
        family = str(document.get("family", ""))
        device_id = str(document.get("device_id", ""))
        if family_filter is not None and family not in family_filter:
            continue
        if device_filter is not None and device_id not in device_filter:
            continue
        schema_version = document.get("schema_version")
        compatible = (
            schema_version == RUNTIME_WINNER_SCHEMA_VERSION
            and isinstance(document.get("kernel_revision"), int)
            and isinstance(document.get("config"), dict)
        )
        rows.append(
            {
                "path": str(path),
                "valid": compatible,
                "compatibility": (
                    "compatible"
                    if compatible
                    else "schema_v1_invalidated"
                    if schema_version == 1
                    else "incompatible_schema"
                ),
                **document,
            }
        )
    return rows


__all__ = [
    "RUNTIME_WINNER_SCHEMA_VERSION",
    "RuntimeWinnerKey",
    "current_kernel_revision",
    "load_runtime_winner",
    "list_runtime_winners",
    "runtime_winner_key",
    "save_runtime_winner",
]
