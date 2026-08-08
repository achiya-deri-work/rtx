"""Small, deterministic runtime-winner cache shared by tuning frontends."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Callable, Mapping, TypeVar

from .legacy import DeviceFingerprint, default_cache_dir
from ..kernels.mxfp8 import MXFP8Problem


ConfigT = TypeVar("ConfigT")
RUNTIME_WINNER_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class RuntimeWinnerKey:
    family: str
    problem: MXFP8Problem
    regime: str
    device_id: str
    variant: str = "default"

    @property
    def relative_path(self) -> Path:
        return (
            Path("runtime_winners")
            / self.family
            / self.device_id
            / (
                f"m{self.problem.m}_n{self.problem.n}_k{self.problem.k}_"
                f"{self.regime}_{self.variant}.json"
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
) -> RuntimeWinnerKey:
    selected = fingerprint or DeviceFingerprint.current(device)
    if not variant or any(value in variant for value in ("/", "\\", "..")):
        raise ValueError("runtime winner variant must be a safe path component")
    return RuntimeWinnerKey(
        family, problem, regime, selected.identifier, variant
    )


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


__all__ = [
    "RUNTIME_WINNER_SCHEMA_VERSION",
    "RuntimeWinnerKey",
    "load_runtime_winner",
    "runtime_winner_key",
    "save_runtime_winner",
]
