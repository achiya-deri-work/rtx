"""Interpretable cross-SKU analysis for residual autotuning datasets.

This complements the deployment-gated pretrained models.  It deliberately
separates within-context schedule effects, model split usage, feasibility, and
cross-SKU winner transfer instead of correlating raw latency with GPU size.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .core import Observation, canonical_json
from .pretrained import load_offline_observations


def _category(features: Mapping[str, float], prefix: str) -> str | None:
    for key, value in features.items():
        if key.startswith(prefix) and float(value) != 0.0:
            return key[len(prefix) :]
    return None


def _number(features: Mapping[str, float], key: str) -> float | None:
    value = features.get(key)
    return None if value is None else float(value)


def _flatten(value: object, prefix: str = "config") -> dict[str, object]:
    result: dict[str, object] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            result.update(_flatten(child, f"{prefix}.{key}"))
    elif isinstance(value, (tuple, list)):
        for index, child in enumerate(value):
            result.update(_flatten(child, f"{prefix}[{index}]"))
    else:
        result[prefix] = value
    return result


def _encoded(value: object) -> str:
    return canonical_json(value)


def _decoded(value: str) -> object:
    return json.loads(value)


def _sku(item: Observation[object]) -> str:
    return _category(item.features, "context.device.sku.sku_family=") or "unknown"


def _regime(item: Observation[object]) -> str:
    return _category(item.features, "context.regime=") or "unknown"


def _workload(item: Observation[object]) -> tuple[object, ...]:
    features = item.features
    return (
        item.family,
        item.kernel_revision,
        int(features.get("context.workload.m", 0)),
        int(features.get("context.workload.n", 0)),
        int(features.get("context.workload.k", 0)),
        _regime(item),
        _category(features, "context.tags.scale_policy=") or "unknown",
        _category(features, "context.tags.operand_state=") or "unknown",
    )


def _m_region(item: Observation[object]) -> str:
    m = int(item.features.get("context.workload.m", 0))
    if m <= 128:
        return "launch_floor"
    if m <= 1024:
        return "small"
    if m <= 8192:
        return "medium"
    if m <= 32768:
        return "large"
    return "very_long"


def _bootstrap_ci(
    values: Sequence[float], *, seed: int, samples: int = 400
) -> tuple[float, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 1:
        value = float(array[0])
        return value, value
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, array.size, size=(samples, array.size))
    medians = np.median(array[draws], axis=1)
    return float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def _device_profiles(rows: Sequence[Observation[object]]) -> list[dict[str, object]]:
    keys = (
        "context.device.multiprocessor_count",
        "context.device.sku.cuda_core_count",
        "context.device.sku.memory_bus_width_bits",
        "context.device.sku.theoretical_memory_bandwidth_gbps",
        "context.device.properties.l2_cache_size",
        "context.device.properties.regs_per_multiprocessor",
        "context.device.properties.shared_memory_per_multiprocessor",
        "context.device.properties.clock_rate_khz",
        "context.device.total_memory",
        "context.device.calibration.measured_dram_bandwidth_gbps",
        "context.device.calibration.measured_l2_copy_bandwidth_gbps",
        "context.device.calibration.measured_mxfp8_quant_bandwidth_gbps",
        "context.device.calibration.measured_native_mxfp8_gemm_tflops",
        "context.device.calibration.measured_bf16_matmul_tflops",
    )
    grouped: dict[str, list[Observation[object]]] = defaultdict(list)
    for item in rows:
        grouped[_sku(item)].append(item)
    profiles = []
    for sku, items in sorted(grouped.items()):
        features = items[0].features
        profile: dict[str, object] = {
            "sku": sku,
            "observations": len(items),
            "contexts": len({item.context_id for item in items}),
        }
        for key in keys:
            value = _number(features, key)
            if value is not None:
                profile[key.removeprefix("context.device.")] = value
        profiles.append(profile)
    return profiles


def _coordinate_effects(
    rows: Sequence[Observation[object]],
    *,
    seed: int,
    minimum_contexts: int,
) -> list[dict[str, object]]:
    successful = [item for item in rows if item.successful and item.score > 0]
    context_logs: dict[str, list[float]] = defaultdict(list)
    context_best: dict[str, float] = {}
    for item in successful:
        context_logs[item.context_id].append(math.log(item.score))
        context_best[item.context_id] = min(
            context_best.get(item.context_id, math.inf), item.score
        )
    centers = {
        context: float(np.median(values)) for context, values in context_logs.items()
    }
    buckets: dict[
        tuple[str, str, str, str, str], dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    wins: Counter[tuple[str, str, str, str, str]] = Counter()
    counts: Counter[tuple[str, str, str, str, str]] = Counter()
    for item in successful:
        effect = math.log(item.score) - centers[item.context_id]
        for coordinate, value in _flatten(item.serialized_config).items():
            encoded = _encoded(value)
            for region in ("all", _m_region(item)):
                key = (item.family, _sku(item), region, coordinate, encoded)
                buckets[key][item.context_id].append(effect)
                counts[key] += 1
                if item.score <= context_best[item.context_id] * 1.02:
                    wins[key] += 1
    effects = []
    for key, by_context in buckets.items():
        if len(by_context) < minimum_contexts:
            continue
        family, sku, region, coordinate, encoded = key
        context_effects = [float(np.median(values)) for values in by_context.values()]
        effect = float(np.median(context_effects))
        digest = hashlib.sha256("|".join(key).encode()).digest()
        local_seed = seed ^ int.from_bytes(digest[:8], "little")
        ci_low, ci_high = _bootstrap_ci(context_effects, seed=local_seed)
        material = abs(math.exp(effect) - 1.0) >= 0.01
        classification = (
            "good"
            if material and ci_high < 0.0
            else "bad"
            if material and ci_low > 0.0
            else "uncertain"
        )
        effects.append(
            {
                "family": family,
                "sku": sku,
                "m_region": region,
                "coordinate": coordinate,
                "value": _decoded(encoded),
                "effect_log_latency": effect,
                "median_speedup_vs_context_median": math.exp(-effect),
                "ci_low": ci_low,
                "ci_high": ci_high,
                "rows": counts[key],
                "contexts": len(by_context),
                "within_2pct_win_rate": wins[key] / counts[key],
                "classification": classification,
            }
        )
    return effects


def _portable_effects(effects: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str, str], list[Mapping[str, object]]] = defaultdict(list)
    for item in effects:
        if item["m_region"] != "all":
            continue
        key = (
            str(item["family"]),
            str(item["coordinate"]),
            _encoded(item["value"]),
            str(item["m_region"]),
        )
        grouped[key].append(item)
    result = []
    for (family, coordinate, encoded, region), items in grouped.items():
        if len(items) < 2:
            continue
        effects_by_sku = {
            str(item["sku"]): float(item["effect_log_latency"]) for item in items
        }
        values = list(effects_by_sku.values())
        signs = {0 if abs(value) < 0.005 else 1 if value > 0 else -1 for value in values}
        classes = {str(item["classification"]) for item in items}
        classification = (
            "portable_good"
            if classes == {"good"}
            else "portable_bad"
            if classes == {"bad"}
            else "sku_sensitive"
            if 1 in signs and -1 in signs
            else "mixed_or_uncertain"
        )
        result.append(
            {
                "family": family,
                "coordinate": coordinate,
                "value": _decoded(encoded),
                "m_region": region,
                "devices": len(items),
                "classification": classification,
                "median_effect_log_latency": float(np.median(values)),
                "effect_spread": max(values) - min(values),
                "effects_by_sku": effects_by_sku,
            }
        )
    return result


def _winner_transfer(rows: Sequence[Observation[object]]) -> list[dict[str, object]]:
    catalogs: dict[
        tuple[str, tuple[object, ...]], dict[str, dict[str, float]]
    ] = defaultdict(lambda: defaultdict(dict))
    for item in rows:
        if item.successful and item.score > 0:
            bucket = catalogs[(_sku(item), _workload(item))][item.context_id]
            bucket[item.config_id] = min(bucket.get(item.config_id, math.inf), item.score)

    collapsed: dict[tuple[str, tuple[object, ...]], dict[str, float]] = {}
    for key, contexts in catalogs.items():
        per_config: dict[str, list[float]] = defaultdict(list)
        for catalog in contexts.values():
            for config_id, score in catalog.items():
                per_config[config_id].append(score)
        collapsed[key] = {
            config_id: float(np.median(scores)) for config_id, scores in per_config.items()
        }
    devices = sorted({sku for sku, _ in collapsed})
    result = []
    for source in devices:
        for target in devices:
            if source == target:
                continue
            regrets = []
            matched = 0
            workloads = 0
            for (sku, workload), source_catalog in collapsed.items():
                if sku != source or (target, workload) not in collapsed:
                    continue
                workloads += 1
                target_catalog = collapsed[(target, workload)]
                source_winner = min(source_catalog, key=source_catalog.get)
                if source_winner not in target_catalog:
                    continue
                matched += 1
                target_best = min(target_catalog.values())
                regrets.append(target_catalog[source_winner] / target_best - 1.0)
            result.append(
                {
                    "source": source,
                    "target": target,
                    "matched_workloads": matched,
                    "shared_workloads": workloads,
                    "catalog_coverage": 0.0 if workloads == 0 else matched / workloads,
                    "median_regret": None if not regrets else float(np.median(regrets)),
                    "p90_regret": None if not regrets else float(np.quantile(regrets, 0.9)),
                    "within_2pct": None
                    if not regrets
                    else sum(value <= 0.02 for value in regrets) / len(regrets),
                }
            )
    return result


def _tree_depths(nodes: Sequence[Mapping[str, object]]) -> dict[int, int]:
    depths: dict[int, int] = {}
    stack = [(0, 0)] if nodes else []
    while stack:
        index, depth = stack.pop()
        if index < 0 or index in depths:
            continue
        depths[index] = depth
        node = nodes[index]
        if int(node.get("feature", -1)) >= 0:
            stack.append((int(node["left"]), depth + 1))
            stack.append((int(node["right"]), depth + 1))
    return depths


def _model_structure(model_path: Path) -> dict[str, object]:
    state = json.loads(model_path.read_text(encoding="utf-8"))
    while "model" in state and isinstance(state["model"], Mapping):
        state = state["model"]
    names = [str(value) for value in state.get("vectorizer", {}).get("names", [])]
    usage: Counter[str] = Counter()
    interactions: Counter[tuple[str, str]] = Counter()
    for ensemble in state.get("models", []):
        for tree in ensemble.get("trees", []):
            nodes = tree.get("nodes", [])
            depths = _tree_depths(nodes)
            for index, node in enumerate(nodes):
                feature = int(node.get("feature", -1))
                if feature < 0 or feature >= len(names):
                    continue
                name = names[feature]
                weight = 1.0 / (1.0 + depths.get(index, 0))
                usage[name] += weight
            stack: list[tuple[int, tuple[str, ...]]] = [(0, ())] if nodes else []
            while stack:
                index, ancestors = stack.pop()
                node = nodes[index]
                feature = int(node.get("feature", -1))
                if feature < 0 or feature >= len(names):
                    continue
                name = names[feature]
                for ancestor in set(ancestors):
                    pair = tuple(sorted((ancestor, name)))
                    if pair[0] != pair[1]:
                        interactions[pair] += 1
                next_ancestors = ancestors + (name,)
                stack.append((int(node["left"]), next_ancestors))
                stack.append((int(node["right"]), next_ancestors))
    return {
        "split_usage": [
            {"feature": key, "weighted_splits": value}
            for key, value in usage.most_common(40)
        ],
        "interactions": [
            {"features": list(key), "path_cooccurrences": value}
            for key, value in interactions.most_common(40)
        ],
    }


def _artifact_summary(artifact: Path | None) -> dict[str, object]:
    if artifact is None:
        return {}
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    families = {}
    for key, item in manifest.get("families", {}).items():
        value = {
            field: item.get(field)
            for field in (
                "rows",
                "successful_rows",
                "statuses",
                "contexts",
                "devices",
                "conditional_rules",
                "deployment",
                "leave_one_device_out",
                "device_models",
            )
        }
        value["ranking_structure"] = _model_structure(
            artifact / str(item["ranking_model"])
        )
        value["feasibility_structure"] = _model_structure(
            artifact / str(item["feasibility_model"])
        )
        rules = json.loads((artifact / str(item["rules"])).read_text(encoding="utf-8"))
        value["conditional_effect_rules"] = rules.get("rules", [])
        families[key] = value
    return {
        "artifact_id": manifest.get("artifact_id"),
        "deployment_models": families,
    }


def _markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Cross-SKU kernel search-space study",
        "",
        "Targets are normalized within each device/workload/cache context. Raw latency is never used to infer schedule quality across SKUs.",
        "",
        "## Dataset",
        "",
        f"- Observations: {report['input']['observations']}",
        f"- Dataset digest: `{report['input']['dataset_sha256']}`",
        "",
        "## Device profiles",
        "",
        "| SKU | SMs | Bus | Theoretical BW | L2 | Measured DRAM | Native MXFP8 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["device_profiles"]:
        lines.append(
            "| {sku} | {sm:.0f} | {bus:.0f} | {tbw:.1f} | {l2:.0f} | {dram:.1f} | {mx:.1f} |".format(
                sku=item["sku"],
                sm=item.get("multiprocessor_count", 0),
                bus=item.get("sku.memory_bus_width_bits", 0),
                tbw=item.get("sku.theoretical_memory_bandwidth_gbps", 0),
                l2=item.get("properties.l2_cache_size", 0),
                dram=item.get("calibration.measured_dram_bandwidth_gbps", 0),
                mx=item.get("calibration.measured_native_mxfp8_gemm_tflops", 0),
            )
        )
    lines.extend(["", "## Winner transfer", "", "| Source | Target | Coverage | Median regret | P90 regret | Within 2% |", "|---|---|---:|---:|---:|---:|"])
    for item in report["winner_transfer"]:
        def pct(value):
            return "n/a" if value is None else f"{100 * float(value):.2f}%"
        lines.append(
            f"| {item['source']} | {item['target']} | {pct(item['catalog_coverage'])} | {pct(item['median_regret'])} | {pct(item['p90_regret'])} | {pct(item['within_2pct'])} |"
        )
    deployment_models = report.get("deployment_models", {})
    if deployment_models:
        lines.extend(
            [
                "",
                "## Held-out model gate",
                "",
                "| Family | Portable head | Feasibility | Conditional rules |",
                "|---|---|---:|---:|",
            ]
        )
        for family, item in deployment_models.items():
            deployment = item.get("deployment", {})
            lines.append(
                f"| {family} | {deployment.get('selected_cost_head', 'none')} | {deployment.get('feasibility_enabled', False)} | {item.get('conditional_rules', 0)} |"
            )
        lines.extend(
            [
                "",
                "No latency/ranking head should guide runtime search when the portable head is `none`; the feasibility and conditional-rule gates are independent.",
            ]
        )
    portable = report["portable_effects"]
    for title, label, reverse in (
        ("Context-adjusted portable good associations", "portable_good", True),
        ("Context-adjusted portable bad associations", "portable_bad", True),
        ("Context-adjusted SKU-sensitive associations", "sku_sensitive", True),
    ):
        selected = [item for item in portable if item["classification"] == label]
        selected.sort(
            key=lambda item: abs(float(item["median_effect_log_latency"])),
            reverse=reverse,
        )
        lines.extend(["", f"## {title}", "", "| Family | Coordinate | Value | Median effect | Spread |", "|---|---|---|---:|---:|"])
        for item in selected[:30]:
            lines.append(
                f"| {item['family']} | `{item['coordinate']}` | `{json.dumps(item['value'])}` | {100 * (math.exp(float(item['median_effect_log_latency'])) - 1):+.2f}% | {100 * float(item['effect_spread']):.2f} log-points |"
            )
    if deployment_models:
        lines.extend(["", "## Ranking-model split usage", ""])
        for family, item in deployment_models.items():
            lines.extend([f"### {family}", "", "| Feature | Weighted splits |", "|---|---:|"])
            for split in item["ranking_structure"]["split_usage"][:15]:
                lines.append(
                    f"| `{split['feature']}` | {float(split['weighted_splits']):.2f} |"
                )
            lines.append("")
    lines.extend(
        [
            "",
            "## Interpretation limits",
            "",
            "Only three SKUs are present. Continuous correlations with SM count, bus width, or bandwidth are therefore hypotheses, not identified hardware laws. Coordinate effects are context-adjusted observational associations; the parent-linked conditional rules and held-out winner-transfer replay are stronger evidence. Adaptive search also over-samples promising neighborhoods, so support and confidence are reported rather than treating frequency as causality.",
            "",
        ]
    )
    return "\n".join(lines)


def study_sku_relationships(
    paths: Sequence[Path | str],
    output: Path | str,
    *,
    artifact: Path | str | None = None,
    seed: int = 20260817,
    minimum_contexts: int = 5,
) -> dict[str, object]:
    rows, load_report = load_offline_observations(paths)
    destination = Path(output).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    effects = _coordinate_effects(
        rows, seed=seed, minimum_contexts=minimum_contexts
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "type": "rtx_cross_sku_search_space_study",
        "input": load_report,
        "device_profiles": _device_profiles(rows),
        "statuses_by_sku": {
            sku: dict(Counter(item.outcome.status for item in rows if _sku(item) == sku))
            for sku in sorted({_sku(item) for item in rows})
        },
        "coordinate_effects": effects,
        "portable_effects": _portable_effects(effects),
        "winner_transfer": _winner_transfer(rows),
        **_artifact_summary(None if artifact is None else Path(artifact).expanduser()),
    }
    (destination / "study.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "study.md").write_text(_markdown(report), encoding="utf-8")
    return report


__all__ = ["study_sku_relationships"]
