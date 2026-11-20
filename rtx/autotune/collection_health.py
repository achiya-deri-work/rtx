"""Fail-closed evidence markers for collection, audit and offline training.

Raw journals are never rewritten to hide a fault. A quarantined bundle stays
readable for diagnosis but cannot be consumed as training/timing evidence.
This module deliberately has no CUDA or Torch dependency.
"""
from __future__ import annotations

import json
from pathlib import Path

from .outcomes import is_fatal_device_context_error


HEALTH_FILENAME = 'collection_health.json'
COLLECTION_GUARD_REVISION = 1


def first_device_fault(path: Path):
    if not path.exists():
        return None
    tail = ''
    with path.open(errors='replace') as source:
        for number, line in enumerate(source, 1):
            text = tail + line
            if is_fatal_device_context_error(text):
                return dict(path=str(path), line=number, diagnostic=text[-2000:].strip())
            tail = text[-8192:]
    return None


def check_health_document(text: str, label: str) -> None:
    try:
        document = json.loads(text)
        eligible = document['evidence_eligible'] is True
    except (ValueError, KeyError, TypeError) as error:
        raise ValueError(f'Invalid collection health marker: {label}') from error
    if not eligible:
        raise ValueError(f'Quarantined autotune evidence: {label}; preserve raw data and requalify')


def assert_healthy_path(path: Path) -> None:
    """Honor markers even when the caller points directly at a nested journal."""
    resolved = path.resolve()
    directory = resolved if resolved.is_dir() else resolved.parent
    for parent in (directory, *directory.parents):
        marker = parent / HEALTH_FILENAME
        if marker.exists():
            check_health_document(marker.read_text(), str(marker))
