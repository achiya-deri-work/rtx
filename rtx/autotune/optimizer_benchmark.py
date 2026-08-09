"""Prospective optimizer-study summaries from residual campaign journals."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
import math
from pathlib import Path
import statistics
from typing import Iterable, Mapping

from .dataset_export import _atomic_json, export_csv, export_parquet, normalized_rows


_BUDGETS = (1, 4, 8, 16, 32, 64)


def _number(value: object) -> float | None:
    try:
        numeric = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _median(values) -> float | None:
    finite = [float(value) for value in values if value is not None]
    return None if not finite else float(statistics.median(finite))


def optimizer_study_rows(
    paths: Iterable[Path | str],
) -> list[dict[str, object]]:
    measurements = [
        row
        for row in normalized_rows(paths)
        if row.get("record_type") == "measurement"
    ]
    by_context: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in measurements:
        context_id = row.get("context_id")
        if context_id is not None:
            by_context[str(context_id)].append(row)

    summaries: list[dict[str, object]] = []
    for context_id, rows in sorted(by_context.items()):
        ordered = sorted(
            rows,
            key=lambda row: (
                int(row.get("sequence") or 0),
                str(row.get("finished_at") or ""),
            ),
        )
        first = ordered[0]
        incumbent = math.inf
        cumulative_s = 0.0
        first_valid_s = None
        budget_best: dict[int, float | None] = {budget: None for budget in _BUDGETS}
        statuses: Counter[str] = Counter()
        compile_ms = 0.0
        failed_compile_ms = 0.0
        for index, row in enumerate(ordered, start=1):
            elapsed = _number(row.get("elapsed_s")) or 0.0
            cumulative_s += elapsed
            status = str(row.get("outcome__status") or "unknown")
            statuses[status] += 1
            candidate_compile = _number(row.get("outcome__compile_ms")) or 0.0
            compile_ms += candidate_compile
            if status != "ok":
                failed_compile_ms += candidate_compile
            latency = _number(row.get("outcome__median_ms"))
            if status == "ok" and latency is not None:
                incumbent = min(incumbent, latency)
                if first_valid_s is None:
                    first_valid_s = cumulative_s
            if index in budget_best:
                budget_best[index] = None if not math.isfinite(incumbent) else incumbent

        summary: dict[str, object] = {
            "context_id": context_id,
            "machine_id": first.get("machine_id")
            or first.get("context__tags__machine_id"),
            "family": first.get("family"),
            "kernel_revision": first.get("kernel_revision"),
            "task_category": first.get("context__tags__task_category")
            or first.get("context__workload__name")
            or "unknown",
            "treatment": first.get("context__tags__treatment") or "default",
            "replicate": first.get("context__tags__replicate") or 0,
            "regime": first.get("context__regime"),
            "m": first.get("context__workload__m"),
            "n": first.get("context__workload__n"),
            "k": first.get("context__workload__k"),
            "trials": len(ordered),
            "successful_trials": statuses.get("ok", 0),
            "failure_rate": 1.0 - statuses.get("ok", 0) / len(ordered),
            "statuses": json.dumps(dict(statuses), sort_keys=True),
            "elapsed_evaluator_s": cumulative_s,
            "time_to_first_valid_s": first_valid_s,
            "compile_ms": compile_ms,
            "failed_compile_ms": failed_compile_ms,
            "best_ms": None if not math.isfinite(incumbent) else incumbent,
        }
        for budget, value in budget_best.items():
            summary[f"best_ms_at_{budget}"] = value
        summaries.append(summary)

    oracle: dict[tuple[object, ...], float] = {}
    for row in summaries:
        task = (
            row["machine_id"],
            row["family"],
            row["regime"],
            row["m"],
            row["n"],
            row["k"],
        )
        best = _number(row["best_ms"])
        if best is not None:
            oracle[task] = min(oracle.get(task, math.inf), best)
    for row in summaries:
        task = (
            row["machine_id"],
            row["family"],
            row["regime"],
            row["m"],
            row["n"],
            row["k"],
        )
        best = _number(row["best_ms"])
        reference = oracle.get(task)
        row["observed_oracle_ms"] = reference
        row["final_regret"] = (
            None
            if best is None or reference is None
            else best / reference - 1.0
        )
        for budget in _BUDGETS:
            value = _number(row[f"best_ms_at_{budget}"])
            row[f"regret_at_{budget}"] = (
                None
                if value is None or reference is None
                else value / reference - 1.0
            )
    return summaries


def summarize_optimizer_study(
    paths: Iterable[Path | str],
    output: Path | str,
    *,
    export_format: str = "both",
) -> dict[str, object]:
    rows = optimizer_study_rows(paths)
    prefix = Path(output)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    files = {}
    if export_format in ("csv", "both"):
        files["csv"] = str(export_csv(rows, prefix.with_suffix(".csv")))
    if export_format in ("parquet", "both"):
        files["parquet"] = str(export_parquet(rows, prefix.with_suffix(".parquet")))

    groups: dict[tuple[str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row.get("machine_id")),
                str(row.get("family")),
                str(row.get("treatment")),
            )
        ].append(row)
    aggregates = []
    for (machine, family, treatment), values in sorted(groups.items()):
        aggregate: dict[str, object] = {
            "machine_id": machine,
            "family": family,
            "treatment": treatment,
            "contexts": len(values),
            "median_final_regret": _median(
                value.get("final_regret") for value in values
            ),
            "median_time_to_first_valid_s": _median(
                value.get("time_to_first_valid_s") for value in values
            ),
            "mean_failure_rate": sum(
                float(value.get("failure_rate", 0.0)) for value in values
            )
            / len(values),
        }
        for budget in _BUDGETS:
            aggregate[f"median_regret_at_{budget}"] = _median(
                value.get(f"regret_at_{budget}") for value in values
            )
        aggregates.append(aggregate)
    report = {
        "schema_version": 1,
        "type": "rtx_optimizer_prospective_summary",
        "units": len(rows),
        "files": files,
        "aggregates": aggregates,
    }
    report_path = prefix.with_suffix(".summary.json")
    _atomic_json(report_path, report)
    report["files"]["summary"] = str(report_path)
    return report


__all__ = ["optimizer_study_rows", "summarize_optimizer_study"]
