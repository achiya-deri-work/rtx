"""Read-only integrity and coverage audits for copied autotune bundles."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path, PurePosixPath
import statistics
from typing import Iterable, Mapping
import zipfile


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _digest(value: object, length: int = 24) -> str:
    return hashlib.sha256(_canonical_json(value).encode()).hexdigest()[:length]


def _record_identity(record: Mapping[str, object]) -> str:
    return _canonical_json(
        {
            key: record.get(key)
            for key in (
                "record_type",
                "context_id",
                "observation_id",
                "observation_key",
                "config_id",
                "config",
                "incumbent_id",
                "challenger_id",
                "incumbent_config",
                "challenger_config",
            )
        }
    )


def _expected_contexts(manifest: Mapping[str, object]) -> int | None:
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        return None
    base = 0
    for job in jobs:
        if not isinstance(job, Mapping):
            continue
        shapes = job.get("shapes", [])
        regimes = job.get("regimes", [])
        if isinstance(shapes, list) and isinstance(regimes, list):
            base += len(shapes) * len(regimes)
    treatments = manifest.get("treatments", [])
    treatment_count = len(treatments) if isinstance(treatments, list) and treatments else 1
    replicates = max(1, int(manifest.get("replicates", 1)))
    shard_count = max(1, int(manifest.get("shard_count", 1)))
    # Hash sharding can differ by one context, so report a range rather than
    # claiming an exact per-shard count.
    total = base * treatment_count * replicates
    return (total + shard_count - 1) // shard_count


class _BundleReader:
    def __init__(self, source: Path, prefix: str = "") -> None:
        self.source = source
        self.prefix = prefix.rstrip("/")
        self._zip = zipfile.ZipFile(source) if source.is_file() else None

    @property
    def label(self) -> str:
        return f"{self.source}:{self.prefix}" if self.prefix else str(self.source)

    def names(self) -> list[str]:
        if self._zip is not None:
            prefix = self.prefix + "/" if self.prefix else ""
            return [name for name in self._zip.namelist() if name.startswith(prefix)]
        root = self.source / self.prefix if self.prefix else self.source
        return [path.relative_to(self.source).as_posix() for path in root.rglob("*") if path.is_file()]

    def read_text(self, name: str) -> str:
        if self._zip is not None:
            return self._zip.read(name).decode("utf-8")
        return (self.source / name).read_text(encoding="utf-8")

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()


def _bundle_readers(paths: Iterable[Path | str]) -> list[_BundleReader]:
    readers: list[_BundleReader] = []
    for raw in paths:
        path = Path(raw)
        if path.is_file() and zipfile.is_zipfile(path):
            with zipfile.ZipFile(path) as archive:
                prefixes = sorted(
                    str(PurePosixPath(name).parent)
                    for name in archive.namelist()
                    if name.endswith("/machine.json") or name == "machine.json"
                )
            readers.extend(_BundleReader(path, prefix) for prefix in prefixes)
            continue
        machine_files = sorted(path.rglob("machine.json")) if path.is_dir() else []
        if machine_files:
            readers.extend(
                _BundleReader(path, file.parent.relative_to(path).as_posix())
                for file in machine_files
            )
        else:
            readers.append(_BundleReader(path))
    return readers


def _json_document(reader: _BundleReader, name: str) -> tuple[dict[str, object] | None, str | None]:
    try:
        value = json.loads(reader.read_text(name))
    except (OSError, KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return None, f"{name}: {type(exc).__name__}: {exc}"
    return (value, None) if isinstance(value, dict) else (None, f"{name}: expected object")


def _audit_bundle(reader: _BundleReader) -> dict[str, object]:
    names = reader.names()
    prefix = reader.prefix + "/" if reader.prefix else ""
    errors: list[str] = []
    warnings: list[str] = []
    machine, error = _json_document(reader, prefix + "machine.json")
    if error:
        errors.append(error)
    manifest, error = _json_document(reader, prefix + "manifest.json")
    if error:
        errors.append(error)
    machine_id = None if machine is None else machine.get("machine_id")
    manifest_digest = None if manifest is None else _digest(manifest)

    units: dict[str, Mapping[str, object]] = {}
    for name in names:
        if not name.endswith("/unit.json"):
            continue
        unit, unit_error = _json_document(reader, name)
        if unit_error:
            errors.append(unit_error)
            continue
        assert unit is not None
        context_id = str(unit.get("context_id", ""))
        if not context_id:
            errors.append(f"{name}: missing context_id")
        elif context_id in units:
            errors.append(f"duplicate unit context_id {context_id}")
        else:
            units[context_id] = unit
        recorded_digest = unit.get("manifest_digest")
        if recorded_digest is not None and manifest_digest is not None and recorded_digest != manifest_digest:
            # Old bundles may intentionally adopt an earlier source identity;
            # this is an identity warning, not data corruption.
            warnings.append(
                f"{name}: manifest digest {recorded_digest} differs from raw digest {manifest_digest}"
            )

    jsonl = [name for name in names if name.endswith(".jsonl")]
    malformed = 0
    malformed_tails = 0
    record_counts: Counter[str] = Counter()
    statuses: Counter[str] = Counter()
    observations_per_context: Counter[str] = Counter()
    observation_ids: dict[str, str] = {}
    duplicate_observation_ids = 0
    conflicting_observation_ids = 0
    verification_keys: dict[str, str] = {}
    duplicate_verification_keys = 0
    conflicting_verification_keys = 0
    context_config: dict[tuple[str, str], str] = {}
    duplicate_context_configs = 0
    conflicting_context_configs = 0
    observed_contexts: set[str] = set()
    candidate_starts: dict[str, str] = {}
    candidate_completions: set[str] = set()
    observed_attempts: set[str] = set()

    for name in jsonl:
        text = reader.read_text(name)
        lines = text.splitlines()
        nonblank = [index for index, line in enumerate(lines) if line.strip()]
        last_nonblank = nonblank[-1] if nonblank else -1
        for index in nonblank:
            try:
                record = json.loads(lines[index])
            except json.JSONDecodeError as exc:
                malformed += 1
                if index == last_nonblank:
                    malformed_tails += 1
                    warnings.append(f"{name}:{index + 1}: malformed crash tail: {exc.msg}")
                else:
                    errors.append(f"{name}:{index + 1}: interior JSON corruption: {exc.msg}")
                continue
            if not isinstance(record, Mapping):
                errors.append(f"{name}:{index + 1}: record is not an object")
                continue
            kind = str(record.get("record_type", PurePosixPath(name).stem))
            record_identity = _record_identity(record)
            record_counts[kind] += 1
            context_id = str(record.get("context_id", ""))
            if context_id:
                observed_contexts.add(context_id)
            if name.endswith("/observations.jsonl"):
                observations_per_context[context_id or "missing"] += 1
                outcome = record.get("outcome")
                if isinstance(outcome, Mapping):
                    statuses[str(outcome.get("status", "missing"))] += 1
                observation_id = str(record.get("observation_id", ""))
                if observation_id:
                    if observation_id in observation_ids:
                        duplicate_observation_ids += 1
                        if observation_ids[observation_id] != record_identity:
                            conflicting_observation_ids += 1
                    else:
                        observation_ids[observation_id] = record_identity
                config_id = str(record.get("config_id", ""))
                pair = (context_id, config_id)
                if all(pair):
                    if pair in context_config:
                        duplicate_context_configs += 1
                        if context_config[pair] != _canonical_json(
                            record.get("config")
                        ):
                            conflicting_context_configs += 1
                    else:
                        context_config[pair] = _canonical_json(record.get("config"))
                unit = units.get(context_id)
                if unit is not None and record.get("family") not in (None, unit.get("family")):
                    errors.append(f"{name}:{index + 1}: family differs from unit")
                metadata = record.get("metadata", {})
                if isinstance(metadata, Mapping):
                    attempt_id = str(metadata.get("attempt_id", ""))
                    if attempt_id:
                        observed_attempts.add(attempt_id)
            if name.endswith("/events.jsonl"):
                event_kind = str(record.get("kind", ""))
                payload = record.get("payload", {})
                if isinstance(payload, Mapping):
                    attempt_id = str(payload.get("attempt_id", ""))
                    if attempt_id and event_kind in (
                        "candidate_started",
                        "trial_issued",
                    ):
                        candidate_starts[attempt_id] = str(
                            payload.get("config_id", "")
                        )
                    elif attempt_id and event_kind in (
                        "candidate_completed",
                        "trial_completed",
                    ):
                        candidate_completions.add(attempt_id)
            if name.endswith("/verification.jsonl") or name == prefix + "verification.jsonl":
                key = str(record.get("observation_key", ""))
                if key:
                    if key in verification_keys:
                        duplicate_verification_keys += 1
                        if verification_keys[key] != record_identity:
                            conflicting_verification_keys += 1
                    else:
                        verification_keys[key] = record_identity
            record_machine = record.get("machine_id")
            if record_machine is not None and machine_id is not None and record_machine != machine_id:
                errors.append(f"{name}:{index + 1}: machine_id differs from bundle")

    if conflicting_observation_ids:
        errors.append(f"{conflicting_observation_ids} conflicting duplicate observation_id values")
    elif duplicate_observation_ids:
        warnings.append(f"{duplicate_observation_ids} repeated observation_id values")
    if conflicting_context_configs:
        errors.append(
            f"{conflicting_context_configs} context/config IDs map to conflicting configs"
        )
    elif duplicate_context_configs:
        warnings.append(
            f"{duplicate_context_configs} repeated context/config measurements"
        )
    if conflicting_verification_keys:
        errors.append(f"{conflicting_verification_keys} conflicting duplicate verification keys")
    elif duplicate_verification_keys:
        warnings.append(f"{duplicate_verification_keys} repeated verification keys")
    actual_contexts = len(units or {key: {} for key in observed_contexts})
    expected = _expected_contexts(manifest or {})
    if expected is not None and actual_contexts < expected:
        warnings.append(f"context coverage is partial: {actual_contexts}/{expected}")
    orphaned_attempts = sorted(
        set(candidate_starts) - candidate_completions - observed_attempts
    )
    if orphaned_attempts:
        warnings.append(
            f"{len(orphaned_attempts)} candidate attempt(s) were interrupted before completion"
        )
    trial_counts = list(observations_per_context.values())
    return {
        "bundle": reader.label,
        "ok": not errors,
        "machine_id": machine_id,
        "device": (
            None
            if machine is None
            else machine.get("device", {}).get("fingerprint", {}).get("name")  # type: ignore[union-attr]
        ),
        "manifest_digest": manifest_digest,
        "expected_contexts_upper_bound": expected,
        "actual_contexts": actual_contexts,
        "jsonl_files": len(jsonl),
        "records": sum(record_counts.values()),
        "record_types": dict(record_counts),
        "outcome_statuses": dict(statuses),
        "coverage": {
            "minimum_trials": min(trial_counts, default=0),
            "median_trials": statistics.median(trial_counts) if trial_counts else 0,
            "maximum_trials": max(trial_counts, default=0),
        },
        "malformed_records": malformed,
        "malformed_tails": malformed_tails,
        "duplicate_observation_ids": duplicate_observation_ids,
        "conflicting_observation_ids": conflicting_observation_ids,
        "duplicate_context_configs": duplicate_context_configs,
        "conflicting_context_configs": conflicting_context_configs,
        "candidate_attempts": {
            "started": len(candidate_starts),
            "completed": len(candidate_completions | observed_attempts),
            "orphaned": len(orphaned_attempts),
            "orphaned_attempt_ids": orphaned_attempts,
            "orphaned_config_ids": [
                candidate_starts[attempt_id] for attempt_id in orphaned_attempts
            ],
        },
        "duplicate_verification_keys": duplicate_verification_keys,
        "conflicting_verification_keys": conflicting_verification_keys,
        "errors": errors,
        "warnings": warnings,
    }


def audit_bundles(paths: Iterable[Path | str]) -> dict[str, object]:
    """Audit directories or zip archives without mutating their journals."""

    reports = []
    readers = _bundle_readers(paths)
    try:
        reports = [_audit_bundle(reader) for reader in readers]
    finally:
        for reader in readers:
            reader.close()
    return {
        "schema_version": 1,
        "type": "rtx_autotune_audit",
        "ok": bool(reports) and all(report["ok"] for report in reports),
        "bundles": reports,
        "summary": {
            "bundles": len(reports),
            "contexts": sum(int(report["actual_contexts"]) for report in reports),
            "records": sum(int(report["records"]) for report in reports),
            "errors": sum(len(report["errors"]) for report in reports),
            "warnings": sum(len(report["warnings"]) for report in reports),
            "malformed_tails": sum(int(report["malformed_tails"]) for report in reports),
        },
    }


__all__ = ["audit_bundles"]
