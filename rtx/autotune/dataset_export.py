"""CPU-only normalization and export for portable autotuning bundles."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Literal, Mapping, Sequence

from .core import canonical_json


DATASET_SCHEMA_VERSION = 2
ExportFormat = Literal["csv", "parquet", "both", "none"]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(dict(value), indent=2, sort_keys=True) + "\n"
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        text=True,
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            sink.write(encoded)
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _iter_jsonl(paths: Iterable[Path | str], filename: str):
    seen_paths: set[Path] = set()
    for root_value in paths:
        root = Path(root_value)
        candidates = (
            [root]
            if root.is_file() and root.name == filename
            else root.rglob(filename)
        )
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
        candidates = (
            [root]
            if root.is_file() and root.name == filename
            else root.rglob(filename)
        )
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

    for path, record in _iter_jsonl(sources, "context_allocations.jsonl"):
        row = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "record_type": "context_allocation",
            "source_path": str(path),
            "observation_id": record.get("observation_key"),
            "context_id": record.get("context_id"),
            "family": record.get("family"),
            "manifest_digest": record.get("manifest_digest"),
            "machine_id": record.get("machine_id"),
            "device_id": record.get("device_id"),
            "recorded_at": record.get("recorded_at"),
            "sequence": record.get("sequence"),
        }
        for prefix, value in (
            ("machine", machines.get(str(record.get("machine_id")), {})),
            ("shape", record.get("shape", {})),
            ("allocation", record),
        ):
            _flatten(prefix, value, row)
        identity = ("context_allocation", record.get("observation_key"))
        unique.setdefault(identity, row)
    return [
        unique[key]
        for key in sorted(unique, key=lambda item: tuple(map(str, item)))
    ]


def export_csv(
    rows: Sequence[Mapping[str, object]], destination: Path | str
) -> Path:
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({str(key) for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as sink:
        writer = csv.DictWriter(sink, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def export_parquet(
    rows: Sequence[Mapping[str, object]], destination: Path | str
) -> Path:
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
        written["parquet"] = str(
            export_parquet(rows, prefix.with_suffix(".parquet"))
        )
    report: dict[str, object] = {
        "schema_version": DATASET_SCHEMA_VERSION,
        "source_dataset_schema_versions": source_schema_versions,
        "mixed_dataset_schemas": len(source_schema_versions) > 1,
        "recorded_at": _utc_now(),
        "rows": len(rows),
        "measurements": sum(
            row.get("record_type") == "measurement" for row in rows
        ),
        "verification_measurements": sum(
            row.get("record_type") == "verification_measurement" for row in rows
        ),
        "races": sum(row.get("record_type") == "race" for row in rows),
        "context_allocations": sum(
            row.get("record_type") == "context_allocation" for row in rows
        ),
        "successful_rows": sum(
            row.get("outcome__status") == "ok" for row in rows
        ),
        "families": sorted(
            {str(row["family"]) for row in rows if row.get("family") is not None}
        ),
        "contexts": len(
            {
                str(row["context_id"])
                for row in rows
                if row.get("context_id") is not None
            }
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


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "ExportFormat",
    "export_bundle",
    "export_csv",
    "export_parquet",
    "normalized_rows",
]
