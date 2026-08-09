"""Prospective optimizer-study summaries from residual campaign journals."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
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


def _bootstrap_median_ci(
    values: Iterable[float], *, seed_key: object, samples: int = 2000
) -> tuple[float | None, float | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None, None
    if len(finite) == 1:
        return finite[0], finite[0]
    seed = int(
        hashlib.sha256(repr(seed_key).encode()).hexdigest()[:16], 16
    )
    rng = random.Random(seed)
    medians = sorted(
        statistics.median(rng.choices(finite, k=len(finite)))
        for _ in range(samples)
    )
    return (
        float(medians[int(0.025 * (samples - 1))]),
        float(medians[int(0.975 * (samples - 1))]),
    )


def _bootstrap_mean_ci(
    values: Iterable[float], *, seed_key: object, samples: int = 2000
) -> tuple[float | None, float | None]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return None, None
    if len(finite) == 1:
        return finite[0], finite[0]
    seed = int(hashlib.sha256(repr(seed_key).encode()).hexdigest()[:16], 16)
    rng = random.Random(seed)
    means = sorted(
        statistics.fmean(rng.choices(finite, k=len(finite)))
        for _ in range(samples)
    )
    return (
        float(means[int(0.025 * (samples - 1))]),
        float(means[int(0.975 * (samples - 1))]),
    )


def _matched_treatment_comparisons(
    rows: Iterable[Mapping[str, object]], *, baseline: str = "random"
) -> list[dict[str, object]]:
    by_task: dict[tuple[object, ...], dict[str, Mapping[str, object]]] = defaultdict(dict)
    for row in rows:
        task = (
            row.get("machine_id"),
            row.get("family"),
            row.get("regime"),
            row.get("m"),
            row.get("n"),
            row.get("k"),
            row.get("replicate"),
        )
        by_task[task][str(row.get("treatment"))] = row
    deltas: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for task, treatments in by_task.items():
        control = treatments.get(baseline)
        if control is None:
            continue
        machine, family = str(task[0]), str(task[1])
        for treatment, candidate in treatments.items():
            if treatment == baseline:
                continue
            for metric in ("final_regret",) + tuple(
                f"regret_at_{budget}" for budget in _BUDGETS
            ):
                candidate_value = _number(candidate.get(metric))
                control_value = _number(control.get(metric))
                if candidate_value is not None and control_value is not None:
                    deltas[(machine, family, treatment)][metric].append(
                        candidate_value - control_value
                    )

    comparisons = []
    for (machine, family, treatment), metrics in sorted(deltas.items()):
        record: dict[str, object] = {
            "machine_id": machine,
            "family": family,
            "treatment": treatment,
            "baseline": baseline,
        }
        for metric, values in sorted(metrics.items()):
            low, high = _bootstrap_median_ci(
                values,
                seed_key=(machine, family, treatment, baseline, metric),
            )
            suffix = "final" if metric == "final_regret" else metric.removeprefix("regret_")
            record[f"matched_{suffix}"] = len(values)
            record[f"median_delta_{suffix}"] = float(statistics.median(values))
            record[f"ci_low_delta_{suffix}"] = low
            record[f"ci_high_delta_{suffix}"] = high
            wins = [float(value < 0) for value in values]
            win_low, win_high = _bootstrap_mean_ci(
                wins,
                seed_key=(machine, family, treatment, baseline, metric, "win"),
            )
            record[f"win_rate_{suffix}"] = statistics.fmean(wins)
            record[f"probability_beating_{baseline}_{suffix}"] = statistics.fmean(wins)
            record[f"ci_low_probability_beating_{baseline}_{suffix}"] = win_low
            record[f"ci_high_probability_beating_{baseline}_{suffix}"] = win_high
        comparisons.append(record)
    return comparisons


def _conditional_aggregates(
    rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[str, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                str(row.get("machine_id")),
                str(row.get("family")),
                str(row.get("task_category")),
                str(row.get("regime")),
                str(row.get("treatment")),
            )
        ].append(row)
    result = []
    for key, values in sorted(groups.items()):
        machine, family, category, regime, treatment = key
        result.append(
            {
                "machine_id": machine,
                "family": family,
                "task_category": category,
                "regime": regime,
                "treatment": treatment,
                "contexts": len(values),
                "median_trials": _median(value.get("trials") for value in values),
                "median_final_regret": _median(
                    value.get("final_regret") for value in values
                ),
                **{
                    f"median_regret_at_{budget}": _median(
                        value.get(f"regret_at_{budget}") for value in values
                    )
                    for budget in _BUDGETS
                },
            }
        )
    return result


def _shape_aggregates(
    rows: Iterable[Mapping[str, object]],
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        groups[
            (
                row.get("machine_id"),
                row.get("family"),
                row.get("m"),
                row.get("n"),
                row.get("k"),
                row.get("regime"),
                row.get("treatment"),
            )
        ].append(row)
    result = []
    for key, values in sorted(groups.items(), key=lambda item: repr(item[0])):
        machine, family, m, n, k, regime, treatment = key
        result.append(
            {
                "machine_id": machine,
                "family": family,
                "m": m,
                "n": n,
                "k": k,
                "regime": regime,
                "treatment": treatment,
                "replicates": len(values),
                "median_time_to_first_valid_s": _median(
                    value.get("time_to_first_valid_s") for value in values
                ),
                "mean_compile_failure_rate": statistics.fmean(
                    float(value.get("compile_failure_rate", 0.0)) for value in values
                ),
                "median_wasted_compile_time_rate": _median(
                    value.get("wasted_compile_time_rate") for value in values
                ),
                **{
                    f"median_regret_at_{budget}": _median(
                        value.get(f"regret_at_{budget}") for value in values
                    )
                    for budget in _BUDGETS
                },
            }
        )
    return result


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
        compile_error_elapsed_s = 0.0
        for index, row in enumerate(ordered, start=1):
            elapsed = _number(row.get("elapsed_s")) or 0.0
            cumulative_s += elapsed
            status = str(row.get("outcome__status") or "unknown")
            statuses[status] += 1
            candidate_compile = _number(row.get("outcome__compile_ms")) or 0.0
            compile_ms += candidate_compile
            if status != "ok":
                failed_compile_ms += candidate_compile
            if status == "compile_error":
                compile_error_elapsed_s += elapsed
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
            "compile_failure_rate": statuses.get("compile_error", 0) / len(ordered),
            "statuses": json.dumps(dict(statuses), sort_keys=True),
            "elapsed_evaluator_s": cumulative_s,
            "time_to_first_valid_s": first_valid_s,
            "compile_ms": compile_ms,
            "failed_compile_ms": failed_compile_ms,
            "compile_error_elapsed_s": compile_error_elapsed_s,
            "wasted_compile_time_rate": (
                compile_error_elapsed_s / cumulative_s if cumulative_s > 0 else 0.0
            ),
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
            "mean_compile_failure_rate": sum(
                float(value.get("compile_failure_rate", 0.0)) for value in values
            )
            / len(values),
            "median_wasted_compile_time_rate": _median(
                value.get("wasted_compile_time_rate") for value in values
            ),
        }
        for budget in _BUDGETS:
            aggregate[f"median_regret_at_{budget}"] = _median(
                value.get(f"regret_at_{budget}") for value in values
            )
        for metric in (
            "final_regret",
            "time_to_first_valid_s",
            "compile_failure_rate",
            "wasted_compile_time_rate",
        ):
            finite = [
                numeric
                for value in values
                if (numeric := _number(value.get(metric))) is not None
            ]
            low, high = _bootstrap_median_ci(
                finite,
                seed_key=(machine, family, treatment, metric),
            )
            aggregate[f"ci_low_median_{metric}"] = low
            aggregate[f"ci_high_median_{metric}"] = high
        aggregates.append(aggregate)
    report = {
        "schema_version": 3,
        "type": "rtx_optimizer_prospective_summary",
        "units": len(rows),
        "files": files,
        "aggregates": aggregates,
        "conditional_aggregates": _conditional_aggregates(rows),
        "shape_aggregates": _shape_aggregates(rows),
        "matched_comparisons": _matched_treatment_comparisons(rows),
        "coverage": {
            "machines": len({str(row.get("machine_id")) for row in rows}),
            "families": len({str(row.get("family")) for row in rows}),
            "treatments": len({str(row.get("treatment")) for row in rows}),
            "minimum_trials": min((int(row.get("trials", 0)) for row in rows), default=0),
            "median_trials": _median(row.get("trials") for row in rows),
            "maximum_trials": max((int(row.get("trials", 0)) for row in rows), default=0),
        },
    }
    report_path = prefix.with_suffix(".summary.json")
    _atomic_json(report_path, report)
    report["files"]["summary"] = str(report_path)
    return report


__all__ = ["optimizer_study_rows", "summarize_optimizer_study"]
