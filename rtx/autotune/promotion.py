"""Promotion of verified dataset winners into the runtime cache."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
import statistics
from typing import Iterable, Mapping

from .audit import _bundle_readers, audit_bundles
from .legacy import default_cache_dir
from .winners import RuntimeWinnerKey, save_runtime_winner
from ..configs.inference import gemm_operand_scale_layouts
from ..kernels.mxfp8 import MXFP8Problem


_SHAPE = re.compile(r"^m(?P<m>\d+)_n(?P<n>\d+)_k(?P<k>\d+)$")


def _variant(family: str, config: Mapping[str, object]) -> str:
    if family not in ("mxfp8_weight_prequant_fwd", "mxfp8_fully_prequant_fwd"):
        return "default"
    gemm = config.get("gemm")
    if not isinstance(gemm, Mapping) or gemm.get("scale_layout") is None:
        raise ValueError(f"{family} winner lacks GEMM scale layout")
    x_layout, w_layout = gemm_operand_scale_layouts(str(gemm["scale_layout"]))
    return f"w-{w_layout}" if family == "mxfp8_weight_prequant_fwd" else f"x-{x_layout}_w-{w_layout}"


def _problem(shape: str) -> MXFP8Problem:
    match = _SHAPE.fullmatch(shape)
    if match is None:
        raise ValueError(f"invalid dataset shape key {shape!r}")
    return MXFP8Problem(*(int(match.group(name)) for name in ("m", "n", "k")))


def _current_revision(family: str) -> int:
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
    raise ValueError(f"unsupported runtime winner family {family!r}")


def _config_rejection(
    family: str, config: Mapping[str, object], problem: MXFP8Problem
) -> str | None:
    try:
        if family == "mxfp8_fused_fwd":
            from ..kernels.mxfp8 import fwd_config_from_dict

            return fwd_config_from_dict(config).implementation_rejection(problem)
        if family == "nvfp4_fused_fwd":
            from ..configs.nvfp4 import (
                NVFP4Problem,
                normalize_nvfp4_fwd_config,
            )

            nv_problem = NVFP4Problem(problem.m, problem.n, problem.k)
            return normalize_nvfp4_fwd_config(
                **dict(config)
            ).implementation_rejection(nv_problem)
        if family == "mxfp8_prequant_fwd":
            from ..prequant_autotune import prequant_config_from_dict

            return prequant_config_from_dict(config).rejection(problem)
        if family == "mxfp8_bwd":
            from ..bwd_autotune import bwd_config_from_dict

            return bwd_config_from_dict(config).implementation_rejection(problem)
        if family == "mxfp8_weight_prequant_fwd":
            from ..inference_autotune import weight_prequant_config_from_dict

            return weight_prequant_config_from_dict(config).rejection(problem)
        if family == "mxfp8_fully_prequant_fwd":
            from ..inference_autotune import fully_prequant_config_from_dict

            return fully_prequant_config_from_dict(config).rejection(problem)
    except (KeyError, TypeError, ValueError) as exc:
        return f"cannot deserialize current config schema: {exc}"
    return f"unsupported runtime winner family {family!r}"


def install_verified_winners(
    paths: Iterable[Path | str],
    *,
    cache_dir: Path | str | None = None,
    families: Iterable[str] | None = None,
    treatments: Iterable[str] | None = None,
    device_ids: Iterable[str] | None = None,
    minimum_support: int = 1,
    dry_run: bool = False,
    force: bool = False,
) -> dict[str, object]:
    """Select repeated verified winners and atomically publish runtime entries."""

    if minimum_support <= 0:
        raise ValueError("minimum winner support must be positive")
    sources = tuple(paths)
    audit = audit_bundles(sources)
    if not audit["ok"]:
        raise ValueError(
            "refusing to promote from a bundle that failed rtx-autotune audit"
        )
    family_filter = None if families is None else set(families)
    treatment_filter = None if treatments is None else set(treatments)
    device_filter = None if device_ids is None else set(device_ids)
    candidates: dict[tuple[str, str, str, str, str], list[dict[str, object]]] = defaultdict(list)
    rejected_candidates: list[dict[str, object]] = []
    readers = _bundle_readers(sources)
    try:
        for reader in readers:
            prefix = reader.prefix + "/" if reader.prefix else ""
            names = reader.names()
            try:
                machine = json.loads(reader.read_text(prefix + "machine.json"))
                summary = json.loads(reader.read_text(prefix + "summary.json"))
            except (KeyError, OSError, json.JSONDecodeError):
                continue
            result_documents = list(summary.get("results", []))
            try:
                posthoc = json.loads(
                    reader.read_text(prefix + "verification_summary.json")
                )
            except (KeyError, OSError, json.JSONDecodeError):
                posthoc = None
            if (
                isinstance(posthoc, Mapping)
                and posthoc.get("type") == "rtx_autotune_posthoc_verification"
            ):
                result_documents.extend(posthoc.get("results", []))
            device_id = str(machine.get("device", {}).get("fingerprint_id", ""))
            if not device_id or (device_filter is not None and device_id not in device_filter):
                continue
            verified_measurements: dict[tuple[str, str], dict[str, object]] = {}
            for name in names:
                if not name.endswith("verification.jsonl"):
                    continue
                for line in reader.read_text(name).splitlines():
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if record.get("record_type") != "verification_measurement":
                        continue
                    outcome = record.get("outcome")
                    if not isinstance(outcome, Mapping) or outcome.get("status") != "ok":
                        continue
                    summary_ms = outcome.get("summary_ms")
                    if not isinstance(summary_ms, Mapping) or summary_ms.get("median") is None:
                        continue
                    verified_measurements[
                        (str(record.get("context_id")), str(record.get("config_id")))
                    ] = {
                        "median_ms": float(summary_ms["median"]),
                        "kernel_revision": record.get("kernel_revision"),
                    }
            latest: dict[str, Mapping[str, object]] = {}
            for result in result_documents:
                if not isinstance(result, Mapping):
                    continue
                verification = result.get("verification")
                if not isinstance(verification, Mapping) or verification.get("status") != "ok":
                    continue
                context_id = str(result.get("context_id", ""))
                prior = latest.get(context_id)
                if prior is None or int(result.get("target_trials", 0)) >= int(prior.get("target_trials", 0)):
                    latest[context_id] = result
            for result in latest.values():
                family = str(result.get("family", ""))
                treatment = str(result.get("treatment", ""))
                if family_filter is not None and family not in family_filter:
                    continue
                if treatment_filter is not None and treatment not in treatment_filter:
                    continue
                verification = result["verification"]
                assert isinstance(verification, Mapping)
                config = verification.get("winner_config")
                config_id = str(verification.get("winner_id", ""))
                context_id = str(result.get("context_id", ""))
                if not isinstance(config, Mapping) or not config_id:
                    continue
                measurement = verified_measurements.get((context_id, config_id))
                if measurement is None:
                    continue
                revision = int(
                    result.get(
                        "kernel_revision",
                        measurement.get("kernel_revision", -1),
                    )
                )
                try:
                    current_revision = _current_revision(family)
                except ValueError as exc:
                    rejected_candidates.append(
                        {"context_id": context_id, "family": family, "reason": str(exc)}
                    )
                    continue
                if revision != current_revision:
                    rejected_candidates.append(
                        {
                            "context_id": context_id,
                            "family": family,
                            "reason": (
                                f"kernel revision {revision} is not current "
                                f"revision {current_revision}"
                            ),
                        }
                    )
                    continue
                median_ms = float(measurement["median_ms"])
                shape = str(result.get("shape", ""))
                regime = str(result.get("regime", "hot"))
                problem = _problem(shape)
                rejection = _config_rejection(family, config, problem)
                if rejection is not None:
                    rejected_candidates.append(
                        {
                            "context_id": context_id,
                            "family": family,
                            "reason": rejection,
                        }
                    )
                    continue
                variant = _variant(family, config)
                key = (family, device_id, shape, regime, variant)
                candidates[key].append(
                    {
                        "config_id": config_id,
                        "config": dict(config),
                        "median_ms": median_ms,
                        "context_id": context_id,
                        "treatment": treatment,
                        "replicate": result.get("replicate"),
                        "bundle": reader.label,
                    }
                )
    finally:
        for reader in readers:
            reader.close()

    root = default_cache_dir() if cache_dir is None else Path(cache_dir).expanduser()
    installed = []
    skipped = []
    for raw_key, rows in sorted(candidates.items()):
        family, device_id, shape, regime, variant = raw_key
        by_config: dict[str, list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            by_config[str(row["config_id"])].append(row)
        eligible = [values for values in by_config.values() if len(values) >= minimum_support]
        if not eligible:
            skipped.append({"key": raw_key, "reason": "minimum_support", "candidates": len(rows)})
            continue
        selected = min(
            eligible,
            key=lambda values: (
                statistics.median(float(row["median_ms"]) for row in values),
                -len(values),
                str(values[0]["config_id"]),
            ),
        )
        median_ms = float(statistics.median(float(row["median_ms"]) for row in selected))
        winner_key = RuntimeWinnerKey(family, _problem(shape), regime, device_id, variant)
        destination = root / winner_key.relative_path
        if destination.exists() and not force:
            skipped.append({"key": raw_key, "reason": "exists", "path": str(destination)})
            continue
        if not dry_run:
            save_runtime_winner(
                winner_key,
                selected[0]["config"],  # type: ignore[arg-type]
                config_id=str(selected[0]["config_id"]),
                root=root,
                median_ms=median_ms,
                metadata={
                    "promotion": "verified_dataset_consensus",
                    "support": len(selected),
                    "contexts": [row["context_id"] for row in selected],
                    "treatments": sorted({str(row["treatment"]) for row in selected}),
                    "sources": sorted({str(row["bundle"]) for row in selected}),
                },
            )
        installed.append(
            {
                "family": family,
                "device_id": device_id,
                "shape": shape,
                "regime": regime,
                "variant": variant,
                "config_id": selected[0]["config_id"],
                "median_ms": median_ms,
                "support": len(selected),
                "path": str(destination),
                "dry_run": dry_run,
            }
        )
    return {
        "schema_version": 1,
        "type": "rtx_runtime_winner_install",
        "audit": audit["summary"],
        "cache_dir": str(root),
        "dry_run": dry_run,
        "installed": installed,
        "skipped": skipped,
        "rejected_candidates": rejected_candidates,
    }


__all__ = ["install_verified_winners"]
