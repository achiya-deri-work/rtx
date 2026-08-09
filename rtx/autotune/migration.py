"""Import historical tuner databases as transferable cost-model observations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from .core import KernelAdapter, Observation, stable_id
from .outcomes import TrialOutcome
from .store import TuningStore


def import_legacy_json_database(
    path: Path | str,
    adapter: KernelAdapter[object],
    store: TuningStore[object],
    *,
    source_name: str | None = None,
) -> int:
    """Import one old atomic-JSON database without suppressing local remeasurement.

    Imported records use a distinct context ID, so they train the cost model but
    do not count as measured on the active run. Repeated imports are idempotent.
    """

    source_path = Path(path)
    document = json.loads(source_path.read_text(encoding="utf-8"))
    trials = document.get("trials")
    if not isinstance(trials, Mapping):
        raise ValueError(f"legacy database {source_path} has no trial mapping")
    label = source_name or source_path.name
    transfer_context_id = stable_id(
        {
            "active_context": adapter.context.identifier,
            "legacy_source": label,
            "transfer_only": True,
        }
    )
    prior_ids = {
        str(record.get("observation_id")) for record in store.records(None)
    }
    session_id = store.start_session(
        adapter.context,
        {
            "kind": "legacy_import",
            "source": str(source_path),
            "transfer_context_id": transfer_context_id,
        },
    )
    imported = 0
    for sequence, (legacy_key, raw) in enumerate(sorted(trials.items())):
        if not isinstance(raw, Mapping) or not isinstance(raw.get("config"), Mapping):
            continue
        observation_id = stable_id(
            {
                "legacy_source": label,
                "legacy_key": legacy_key,
                "recorded_at": raw.get("recorded_at"),
            },
            32,
        )
        if observation_id in prior_ids:
            continue
        try:
            serialized = dict(raw["config"])
            config = adapter.deserialize(serialized)
        except (KeyError, TypeError, ValueError):
            continue
        # Forward databases stored outcome fields directly on a trial, while
        # the backward database stored the same schema under ``outcome``.
        outcome_raw = raw.get("outcome", raw)
        if not isinstance(outcome_raw, Mapping):
            continue
        status = str(outcome_raw.get("status", "runtime_error"))
        outcome = TrialOutcome(
            status,  # type: ignore[arg-type]
            median_ms=(
                None
                if outcome_raw.get("median_ms") is None
                else float(outcome_raw["median_ms"])
            ),
            timings_ms=[
                float(value) for value in outcome_raw.get("timings_ms", [])
            ],
            compile_ms=(
                None
                if outcome_raw.get("compile_ms") is None
                else float(outcome_raw["compile_ms"])
            ),
            max_abs_error=(
                None
                if outcome_raw.get("max_abs_error") is None
                else float(outcome_raw["max_abs_error"])
            ),
            error=(
                None
                if outcome_raw.get("error") is None
                else str(outcome_raw["error"])
            ),
        )
        timestamp = str(raw.get("recorded_at", "legacy"))
        observation = Observation(
            observation_id=observation_id,
            session_id=session_id,
            sequence=sequence,
            context_id=transfer_context_id,
            family=adapter.context.family,
            kernel_revision=adapter.context.kernel_revision,
            config_id=adapter.config_id(config),
            config=config,
            serialized_config=adapter.serialize(config),
            features=adapter.features(config),
            strategy="legacy_coordinate",
            outcome=outcome,
            started_at=timestamp,
            finished_at=timestamp,
            elapsed_s=0.0,
            coordinate=(
                None if raw.get("coordinate") is None else str(raw["coordinate"])
            ),
            coordinate_value=raw.get("coordinate_value"),
            metadata={
                "legacy_source": str(source_path),
                "legacy_session_id": raw.get("session_id"),
                "legacy_pass_index": raw.get("pass_index"),
                "legacy_attempt": raw.get("attempt"),
            },
        )
        store.record_observation(observation)
        prior_ids.add(observation_id)
        imported += 1
    store.finish_session(
        session_id,
        {
            "status": "complete",
            "kind": "legacy_import",
            "source": str(source_path),
            "imported_observations": imported,
        },
    )
    return imported


__all__ = ["import_legacy_json_database"]
