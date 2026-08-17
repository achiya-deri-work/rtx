"""Evidence-oriented offline analysis for portable kernel autotuning.

The routines in this module avoid treating adaptively sampled observations as
IID rows.  Comparisons are made within exact contexts, parent/child moves are
kept paired, and model evaluation holds out complete shape groups.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from .core import FeatureMap, Observation, canonical_json
from .cost_model import GradientBoostedFeasibilityModel
from .outcomes import TrialOutcome
from .pretrained import load_offline_observations, portable_features


def _category(features: Mapping[str, float], prefix: str) -> str | None:
    for key, value in features.items():
        if key.startswith(prefix) and float(value) != 0.0:
            return key[len(prefix) :]
    return None


def _sku(item: Observation[object]) -> str:
    return _category(item.features, "context.device.sku.sku_family=") or "unknown"


def _regime(item: Observation[object]) -> str:
    return _category(item.features, "context.regime=") or "unknown"


def _shape_key(item: Observation[object]) -> tuple[object, ...]:
    return (
        item.family,
        item.kernel_revision,
        int(item.features.get("context.workload.m", 0)),
        int(item.features.get("context.workload.n", 0)),
        int(item.features.get("context.workload.k", 0)),
    )


def _flatten(value: object, prefix: str = "config") -> dict[str, object]:
    result: dict[str, object] = {}
    if isinstance(value, Mapping):
        for key, child in value.items():
            result.update(_flatten(child, f"{prefix}.{key}"))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            result.update(_flatten(child, f"{prefix}[{index}]"))
    else:
        result[prefix] = value
    return result


def _best_by_config(
    rows: Sequence[Observation[object]],
) -> dict[tuple[str, str], Observation[object]]:
    result: dict[tuple[str, str], Observation[object]] = {}
    for item in rows:
        if not item.successful or item.score <= 0:
            continue
        key = (item.context_id, item.config_id)
        prior = result.get(key)
        if prior is None or item.score < prior.score:
            result[key] = item
    return result


def _bootstrap_median(
    values: Sequence[float], *, seed: int, samples: int = 400
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    if len(array) < 2:
        return median, median, median
    rng = np.random.default_rng(seed)
    draw = rng.integers(0, len(array), size=(samples, len(array)))
    medians = np.median(array[draw], axis=1)
    return median, float(np.quantile(medians, 0.025)), float(
        np.quantile(medians, 0.975)
    )


def parent_move_analysis(
    rows: Sequence[Observation[object]], *, seed: int = 0, minimum_pairs: int = 8
) -> dict[str, object]:
    """Summarize exact parent→child mutations, including old and new values."""

    catalog = _best_by_config(rows)
    groups: dict[
        tuple[str, str, str, str, str], list[tuple[float, str]]
    ] = defaultdict(list)
    paired = 0
    multi_coordinate = 0
    for child in rows:
        if (
            not child.successful
            or child.parent_config_id is None
            or child.coordinate is None
            or child.score <= 0
        ):
            continue
        parent = catalog.get((child.context_id, child.parent_config_id))
        if parent is None or parent.score <= 0:
            continue
        before = _flatten(parent.serialized_config)
        after = _flatten(child.serialized_config)
        changed = sorted(
            key
            for key in before.keys() | after.keys()
            if canonical_json(before.get(key)) != canonical_json(after.get(key))
        )
        if len(changed) != 1:
            multi_coordinate += 1
        old_value = (
            before.get(changed[0]) if len(changed) == 1 else "<coupled-or-composite>"
        )
        new_value = (
            after.get(changed[0]) if len(changed) == 1 else child.coordinate_value
        )
        effect = math.log(child.score / parent.score)
        key = (
            child.family,
            _sku(child),
            child.coordinate,
            canonical_json(old_value),
            canonical_json(new_value),
        )
        groups[key].append((effect, child.context_id))
        paired += 1

    effects = []
    for index, (key, values) in enumerate(sorted(groups.items())):
        if len(values) < minimum_pairs:
            continue
        family, sku, coordinate, old_json, new_json = key
        context_values: dict[str, list[float]] = defaultdict(list)
        for effect, context_id in values:
            context_values[context_id].append(effect)
        centered = [float(np.median(value)) for value in context_values.values()]
        if len(centered) < 2:
            continue
        median, low, high = _bootstrap_median(
            centered, seed=seed ^ (index + 1) * 104729
        )
        effects.append(
            {
                "family": family,
                "sku": sku,
                "coordinate": coordinate,
                "old_value": json.loads(old_json),
                "new_value": json.loads(new_json),
                "pairs": len(values),
                "contexts": len(context_values),
                "effect_log_latency": median,
                "relative_latency": math.exp(median),
                "ci_low": low,
                "ci_high": high,
                "improvement_probability": sum(
                    effect < 0 for effect, _context in values
                )
                / len(values),
                "qualified": low > 0 or high < 0,
            }
        )
    effects.sort(
        key=lambda item: math.sqrt(int(item["pairs"]))
        * abs(float(item["effect_log_latency"])),
        reverse=True,
    )
    return {
        "paired_moves": paired,
        "coupled_or_composite_moves": multi_coordinate,
        "effects": effects,
    }


def _pair_features(
    left: Observation[object], right: Observation[object]
) -> FeatureMap:
    return pair_features(left.features, right.features)


def pair_features(
    left: Mapping[str, float], right: Mapping[str, float]
) -> FeatureMap:
    """Encode one oriented candidate preference comparison."""

    left_features = portable_features(left)
    right_features = portable_features(right)
    result: FeatureMap = {
        key: value for key, value in left_features.items() if key.startswith("context.")
    }
    names = sorted(
        key
        for key in left_features.keys() | right_features.keys()
        if key.startswith(("config.", "derived."))
    )
    for name in names:
        left_value = float(left_features.get(name, 0.0))
        right_value = float(right_features.get(name, 0.0))
        result[f"pair.left.{name}"] = left_value
        result[f"pair.right.{name}"] = right_value
        result[f"pair.delta.{name}"] = left_value - right_value
    return result


class PairwisePreferenceModel:
    """Parent-move preference classifier with a stable serializable ABI."""

    def __init__(
        self, model: GradientBoostedFeasibilityModel | None = None
    ) -> None:
        self.model = model or GradientBoostedFeasibilityModel(
            n_estimators=16,
            max_depth=4,
            min_leaf=8,
            ensembles=2,
            max_features=128,
            max_training_rows=40_000,
            positive_statuses=("ok",),
            negative_statuses=("correctness_error",),
        )

    @property
    def fitted(self) -> bool:
        return self.model.fitted

    def fit(self, observations: Sequence[Observation[object]]) -> None:
        self.model.fit(_parent_pairs(observations))

    def predict(
        self,
        left: Sequence[Mapping[str, float]],
        right: Sequence[Mapping[str, float]],
    ) -> tuple[np.ndarray, np.ndarray]:
        if len(left) != len(right):
            raise ValueError("pairwise prediction operands must have equal length")
        return self.model.predict(
            [pair_features(a, b) for a, b in zip(left, right, strict=True)]
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "type": "rtx_pairwise_preference_model",
            "target": "probability(left_latency <= right_latency)",
            "model": self.model.state_dict(),
        }

    def save(self, path: Path | str) -> None:
        Path(path).write_text(
            json.dumps(self.state_dict(), sort_keys=True), encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path | str) -> "PairwisePreferenceModel":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        if state.get("schema_version") != 1 or state.get("type") != "rtx_pairwise_preference_model":
            raise ValueError("unsupported pairwise preference model")
        return cls(
            GradientBoostedFeasibilityModel.from_state_dict(state["model"])
        )


def _pair_observation(
    left: Observation[object],
    right: Observation[object],
    *,
    sequence: int,
) -> Observation[object]:
    left_wins = left.score <= right.score
    return Observation(
        observation_id=f"pair-{left.observation_id}-{right.observation_id}",
        session_id="offline-pairwise",
        sequence=sequence,
        context_id=left.context_id,
        family=left.family,
        kernel_revision=left.kernel_revision,
        config_id=f"{left.config_id}>{right.config_id}",
        config={},
        serialized_config={},
        features=_pair_features(left, right),
        strategy="paired_parent_move",
        outcome=TrialOutcome("ok" if left_wins else "correctness_error"),
        started_at="",
        finished_at="",
        elapsed_s=0.0,
    )


def _parent_pairs(
    rows: Sequence[Observation[object]],
    *,
    symmetric: bool = True,
    tie_threshold: float = 0.002,
) -> list[Observation[object]]:
    catalog = _best_by_config(rows)
    result = []
    seen: set[tuple[str, str, str]] = set()
    for child in rows:
        if not child.successful or child.parent_config_id is None:
            continue
        parent = catalog.get((child.context_id, child.parent_config_id))
        if parent is None or parent.config_id == child.config_id:
            continue
        # Measurements inside the practical-equivalence band do not provide a
        # stable ordering label. Teaching either orientation as the winner
        # makes an otherwise symmetric model learn timer noise.
        if abs(child.score / parent.score - 1.0) < tie_threshold:
            continue
        identity = (child.context_id, parent.config_id, child.config_id)
        if identity in seen:
            continue
        seen.add(identity)
        result.append(_pair_observation(child, parent, sequence=len(result)))
        if symmetric:
            result.append(_pair_observation(parent, child, sequence=len(result)))
    return result


def _pair_auc(
    model: GradientBoostedFeasibilityModel,
    pairs: Sequence[Observation[object]],
) -> dict[str, float]:
    if not pairs:
        return {"rows": 0.0}
    probability, _ = model.predict([item.features for item in pairs])
    labels = np.asarray(
        [1.0 if item.outcome.status == "ok" else 0.0 for item in pairs],
        dtype=np.float64,
    )
    order = np.argsort(probability, kind="stable")
    ranks = np.empty(len(probability), dtype=np.float64)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and probability[order[stop]] == probability[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop - 1) + 1.0
        start = stop
    positive_count = int(np.count_nonzero(labels == 1))
    negative_count = len(labels) - positive_count
    auc = float(
        (
            np.sum(ranks[labels == 1])
            - positive_count * (positive_count + 1) / 2
        )
        / max(1, positive_count * negative_count)
    )
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return {
        "rows": float(len(pairs)),
        "auc": auc,
        "accuracy": float(np.mean((probability >= 0.5) == labels)),
        "log_loss": float(
            -np.mean(labels * np.log(clipped) + (1.0 - labels) * np.log(1.0 - clipped))
        ),
    }


def _catalog_replay(
    model: GradientBoostedFeasibilityModel,
    rows: Sequence[Observation[object]],
    *,
    seed: int,
) -> dict[str, object]:
    by_context: dict[str, list[Observation[object]]] = defaultdict(list)
    for item in _best_by_config(rows).values():
        by_context[item.context_id].append(item)
    budgets = (4, 8, 16, 32)
    model_regrets: dict[int, list[float]] = defaultdict(list)
    random_regrets: dict[int, list[float]] = defaultdict(list)
    wall_regrets: dict[float, list[float]] = defaultdict(list)
    random_wall_regrets: dict[float, list[float]] = defaultdict(list)
    contexts = 0
    for context_id, catalog in by_context.items():
        if len(catalog) < 4:
            continue
        contexts += 1
        references = sorted(
            catalog,
            key=lambda item: hashlib.sha256(
                f"reference:{context_id}:{item.config_id}".encode()
            ).hexdigest(),
        )[: min(8, len(catalog))]
        pair_rows = []
        owners = []
        for candidate in catalog:
            for reference in references:
                if candidate.config_id == reference.config_id:
                    continue
                pair_rows.append(_pair_features(candidate, reference))
                owners.append(candidate.config_id)
        probability, _ = model.predict(pair_rows)
        scores: dict[str, list[float]] = defaultdict(list)
        for owner, value in zip(owners, probability, strict=True):
            scores[owner].append(float(value))
        order = sorted(
            catalog,
            key=lambda item: (-float(np.mean(scores[item.config_id])), item.config_id),
        )
        best = min(item.score for item in catalog)
        rng = np.random.default_rng(
            seed
            ^ int(hashlib.sha256(context_id.encode()).hexdigest()[:16], 16)
        )
        for budget in budgets:
            count = min(budget, len(order))
            model_regrets[budget].append(
                min(item.score for item in order[:count]) / best - 1.0
            )
            draws = []
            for _ in range(100):
                indices = rng.choice(len(catalog), size=count, replace=False)
                draws.append(
                    min(catalog[int(index)].score for index in indices) / best - 1.0
                )
            random_regrets[budget].append(float(np.median(draws)))
        elapsed = 0.0
        best_seen = math.inf
        deadline_values: dict[float, float] = {}
        for candidate in order:
            elapsed += max(0.0, candidate.elapsed_s)
            best_seen = min(best_seen, candidate.score)
            for deadline in (1.0, 5.0, 30.0, 60.0):
                if deadline not in deadline_values and elapsed >= deadline:
                    deadline_values[deadline] = best_seen / best - 1.0
        for deadline in (1.0, 5.0, 30.0, 60.0):
            wall_regrets[deadline].append(
                deadline_values.get(deadline, best_seen / best - 1.0)
            )
        random_deadline_draws: dict[float, list[float]] = defaultdict(list)
        for _ in range(100):
            random_order = [catalog[int(index)] for index in rng.permutation(len(catalog))]
            random_elapsed = 0.0
            random_best = math.inf
            random_values: dict[float, float] = {}
            for candidate in random_order:
                random_elapsed += max(0.0, candidate.elapsed_s)
                random_best = min(random_best, candidate.score)
                for deadline in (1.0, 5.0, 30.0, 60.0):
                    if deadline not in random_values and random_elapsed >= deadline:
                        random_values[deadline] = random_best / best - 1.0
            for deadline in (1.0, 5.0, 30.0, 60.0):
                random_deadline_draws[deadline].append(
                    random_values.get(deadline, random_best / best - 1.0)
                )
        for deadline, values in random_deadline_draws.items():
            random_wall_regrets[deadline].append(float(np.median(values)))

    def summarize(values: Sequence[float]) -> dict[str, float]:
        return {
            "median": float(np.median(values)),
            "p90": float(np.quantile(values, 0.9)),
            "within_2pct": float(np.mean(np.asarray(values) <= 0.02)),
        }

    return {
        "contexts": contexts,
        "model": {str(key): summarize(value) for key, value in model_regrets.items()},
        "random": {
            str(key): summarize(value) for key, value in random_regrets.items()
        },
        "model_wall_clock_s": {
            str(key): summarize(value) for key, value in wall_regrets.items()
        },
        "random_wall_clock_s": {
            str(key): summarize(value) for key, value in random_wall_regrets.items()
        },
    }


def pairwise_shape_heldout_study(
    rows: Sequence[Observation[object]],
    *,
    seed: int = 0,
    folds: int = 4,
    n_estimators: int = 16,
) -> dict[str, object]:
    """Evaluate portable and exact-SKU pair ranking on unseen M/N/K groups."""

    def evaluate_scope(
        scope_rows: Sequence[Observation[object]], *, scope_seed: int
    ) -> dict[str, object] | None:
        shapes = sorted({_shape_key(item) for item in scope_rows})
        if len(shapes) < 2:
            return None
        fold_count = min(folds, len(shapes))
        shape_fold = {
            shape: int(
                hashlib.sha256(f"{scope_seed}:{shape}".encode()).hexdigest()[:8],
                16,
            )
            % fold_count
            for shape in shapes
        }
        fold_reports = []
        for fold in range(fold_count):
            train_rows = [
                item for item in scope_rows if shape_fold[_shape_key(item)] != fold
            ]
            test_rows = [
                item for item in scope_rows if shape_fold[_shape_key(item)] == fold
            ]
            train_pairs = _parent_pairs(train_rows)
            test_pairs = _parent_pairs(test_rows)
            model = GradientBoostedFeasibilityModel(
                n_estimators=n_estimators,
                max_depth=4,
                min_leaf=8,
                ensembles=2,
                max_features=128,
                max_training_rows=40_000,
                positive_statuses=("ok",),
                negative_statuses=("correctness_error",),
                seed=scope_seed ^ fold,
            )
            model.fit(train_pairs)
            if not model.fitted or not test_pairs:
                continue
            fold_reports.append(
                {
                    "fold": fold,
                    "train_shapes": len({_shape_key(item) for item in train_rows}),
                    "test_shapes": len({_shape_key(item) for item in test_rows}),
                    "train_pairs": len(train_pairs),
                    "test_pairs": len(test_pairs),
                    "pair_classification": _pair_auc(model, test_pairs),
                    "catalog_replay": _catalog_replay(
                        model, test_rows, seed=scope_seed ^ fold * 65537
                    ),
                }
            )
        if not fold_reports:
            return None
        gates = []
        for fold in fold_reports:
            replay = fold["catalog_replay"]
            model_budget = replay["model"].get("8")
            random_budget = replay["random"].get("8")
            if model_budget and random_budget:
                gates.append(
                    model_budget["median"] <= random_budget["median"]
                    and model_budget["p90"] <= random_budget["p90"]
                )
        required = math.ceil(0.75 * len(gates))
        return {
            "folds": fold_reports,
            "deployment_gate": {
                "budget": 8,
                "passed_folds": sum(gates),
                "required_folds": required,
                "enabled": bool(gates) and sum(gates) >= required,
            },
        }

    result: dict[str, object] = {}
    families = sorted({(item.family, item.kernel_revision) for item in rows})
    for family_index, (family, revision) in enumerate(families):
        family_rows = [
            item
            for item in rows
            if item.family == family and item.kernel_revision == revision
        ]
        family_seed = seed ^ (family_index + 1) * 104729
        portable = evaluate_scope(family_rows, scope_seed=family_seed)
        if portable is None:
            continue
        device_models = {}
        for device_index, device in enumerate(sorted({_sku(item) for item in family_rows})):
            device_rows = [item for item in family_rows if _sku(item) == device]
            evaluation = evaluate_scope(
                device_rows,
                scope_seed=family_seed ^ (device_index + 1) * 524287,
            )
            if evaluation is not None:
                device_models[device] = evaluation
        result[f"{family}@{revision}"] = {
            **portable,
            "scope": "portable_cross_sku",
            "device_models": device_models,
        }
    return result


def save_deployable_pairwise_models(
    rows: Sequence[Observation[object]],
    evaluation: Mapping[str, object],
    output: Path | str,
    *,
    seed: int = 0,
    n_estimators: int = 16,
) -> dict[str, object]:
    """Fit final models only for scopes which cleared shape-held-out replay."""

    destination = Path(output)
    model_root = destination / "models"
    entries = {}
    for family_index, (key, raw_entry) in enumerate(sorted(evaluation.items())):
        entry = raw_entry
        family, revision_text = key.rsplit("@", 1)
        revision = int(revision_text)
        family_rows = [
            item
            for item in rows
            if item.family == family and item.kernel_revision == revision
        ]
        scopes: list[tuple[str, Sequence[Observation[object]], Mapping[str, object]]] = []
        if entry["deployment_gate"]["enabled"]:
            scopes.append(("portable", family_rows, entry))
        for device, device_entry in entry.get("device_models", {}).items():
            if device_entry["deployment_gate"]["enabled"]:
                scopes.append(
                    (
                        f"device-{device}",
                        [item for item in family_rows if _sku(item) == device],
                        device_entry,
                    )
                )
        for scope_index, (scope, scope_rows, scope_evaluation) in enumerate(scopes):
            underlying = GradientBoostedFeasibilityModel(
                n_estimators=n_estimators,
                max_depth=4,
                min_leaf=8,
                ensembles=2,
                max_features=128,
                max_training_rows=40_000,
                positive_statuses=("ok",),
                negative_statuses=("correctness_error",),
                seed=seed ^ (family_index + 1) * 104729 ^ scope_index * 524287,
            )
            model = PairwisePreferenceModel(underlying)
            model.fit(scope_rows)
            if not model.fitted:
                continue
            relative = Path("models") / f"{family}-r{revision}-{scope}.json"
            path = destination / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            model.save(path)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            entries[f"{key}:{scope}"] = {
                "family": family,
                "kernel_revision": revision,
                "scope": scope,
                "rows": len(scope_rows),
                "pairs": len(_parent_pairs(scope_rows)),
                "model": str(relative),
                "sha256": digest,
                "deployment_gate": scope_evaluation["deployment_gate"],
            }
    artifact = {
        "schema_version": 1,
        "type": "rtx_pairwise_preference_bundle",
        "models": entries,
    }
    artifact["artifact_id"] = hashlib.sha256(
        canonical_json(artifact).encode()
    ).hexdigest()[:24]
    # Always replace the manifest, including with an empty model set. Reusing
    # an output directory after a stricter evaluation must fail closed instead
    # of leaving a previously qualified manifest deployable by accident.
    (destination / "pairwise_manifest.json").write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


def _feature_value(item: Observation[object], suffix: str, default: float = 0.0) -> float:
    matches = [float(value) for key, value in item.features.items() if key.endswith(suffix)]
    return matches[0] if matches else default


def bottleneck_class(item: Observation[object]) -> str:
    """A transparent roofline/resource heuristic, not a learned ground truth."""

    waves = _feature_value(item, "effective_cta_waves")
    smem = _feature_value(item, "smem_fraction_per_cta")
    registers = _feature_value(item, "register_fraction_per_cta")
    latency = item.score if item.successful else math.inf
    if waves < 1.0 or (waves < 2.0 and latency < 0.06):
        return "launch_or_grid_underfill"
    if smem >= 0.75 or registers >= 0.9:
        return "resource_or_occupancy_limited"
    memory_ms = _feature_value(item, "memory_roofline_ms")
    flops = _feature_value(item, "nominal_flops")
    throughput = _feature_value(
        item, "calibration.measured_native_mxfp8_gemm_tflops"
    )
    compute_ms = flops / (throughput * 1e9) if flops > 0 and throughput > 0 else 0
    if memory_ms > compute_ms * 1.25:
        return "memory_or_scale_traffic_limited"
    if compute_ms > memory_ms * 1.25:
        return "tensor_core_or_compute_limited"
    return "balanced_roofline"


def _archetype_dimensions(item: Observation[object]) -> dict[str, str]:
    config = _flatten(item.serialized_config)
    stages = int(config.get("config.gemm.stages", 0) or 0)
    smem = _feature_value(item, "smem_fraction_per_cta")
    registers = _feature_value(item, "register_fraction_per_cta")
    waves = int(config.get("config.gemm.persistent_waves", 0) or 0)
    x_rows = int(config.get("config.x_scale_region_rows", 0) or 0)
    w_rows = int(config.get("config.weight_scale_region_rows", 0) or 0)
    return {
        "pipeline": "shallow" if stages <= 2 else "balanced" if stages == 3 else "deep",
        "resource": (
            "high_smem"
            if smem >= 0.75
            else "high_register"
            if registers >= 0.9
            else "occupancy_friendly"
        ),
        "persistence": "single_wave" if waves <= 1 else "few_waves" if waves <= 3 else "many_waves",
        "epilogue": str(config.get("config.gemm.regional_scale_epilogue", config.get("config.gemm.epilogue", "default"))),
        "scale_transport": str(config.get("config.gemm.scale_role", "default")),
        "region_geometry": (
            "not_regional"
            if x_rows <= 0 or w_rows <= 0
            else "fine"
            if x_rows <= 2 or w_rows <= 2
            else "mid"
            if x_rows <= 8 and w_rows <= 8
            else "coarse"
        ),
    }


def archetype_analysis(
    rows: Sequence[Observation[object]], *, minimum_contexts: int = 5
) -> list[dict[str, object]]:
    successful = [item for item in rows if item.successful and item.score > 0]
    centers: dict[str, float] = {}
    grouped_context: dict[str, list[float]] = defaultdict(list)
    for item in successful:
        grouped_context[item.context_id].append(math.log(item.score))
    centers = {
        context: float(np.median(values)) for context, values in grouped_context.items()
    }
    buckets: dict[
        tuple[str, str, str, str, str], dict[str, list[float]]
    ] = defaultdict(lambda: defaultdict(list))
    for item in successful:
        effect = math.log(item.score) - centers[item.context_id]
        bottleneck = bottleneck_class(item)
        for dimension, value in _archetype_dimensions(item).items():
            buckets[(item.family, _sku(item), bottleneck, dimension, value)][
                item.context_id
            ].append(effect)
    result = []
    for key, contexts in buckets.items():
        if len(contexts) < minimum_contexts:
            continue
        family, sku, bottleneck, dimension, value = key
        effects = [float(np.median(samples)) for samples in contexts.values()]
        result.append(
            {
                "family": family,
                "sku": sku,
                "bottleneck": bottleneck,
                "dimension": dimension,
                "value": value,
                "contexts": len(contexts),
                "effect_log_latency": float(np.median(effects)),
                "relative_latency": math.exp(float(np.median(effects))),
            }
        )
    result.sort(
        key=lambda item: abs(float(item["effect_log_latency"])), reverse=True
    )
    return result


def _failure_kind(item: Observation[object]) -> str:
    status = item.outcome.status
    message = (item.outcome.error or "").lower()
    if status == "correctness_error":
        if "cosine" in message or "nrmse" in message or "numerical" in message:
            return "numerical_contract"
        return "correctness_other"
    if status == "compile_error":
        if "nvvm" in message:
            return "compile_nvvm"
        if "resource" in message or "register" in message or "shared" in message:
            return "compile_resource"
        return "compile_other"
    if status == "runtime_error":
        return "runtime_device" if "cuda" in message else "runtime_other"
    return status


def _failure_tokens(item: Observation[object]) -> set[str]:
    tokens = {
        f"{key}={canonical_json(value)}"
        for key, value in _flatten(item.serialized_config).items()
    }
    for axis in ("m", "n", "k"):
        tail = _feature_value(item, f"{axis}_tail_fraction")
        tokens.add(f"derived.{axis}_has_tail={int(tail > 0)}")
    for name, thresholds in (
        ("smem_fraction_per_cta", (0.5, 0.75, 0.9)),
        ("register_fraction_per_cta", (0.5, 0.75, 0.9, 1.0)),
        ("effective_cta_waves", (1.0, 2.0, 4.0)),
    ):
        value = _feature_value(item, name)
        bucket = sum(value >= threshold for threshold in thresholds)
        tokens.add(f"derived.{name}.bucket={bucket}")
    return tokens


def failure_analysis(
    rows: Sequence[Observation[object]], *, minimum_support: int = 8
) -> dict[str, object]:
    statuses = Counter(item.outcome.status for item in rows)
    kinds = Counter(_failure_kind(item) for item in rows if not item.successful)
    by_family: dict[tuple[str, str], list[Observation[object]]] = defaultdict(list)
    for item in rows:
        by_family[(item.family, _sku(item))].append(item)
    enrichments = []
    for (family, sku), items in by_family.items():
        failures = [item for item in items if not item.successful]
        successes = [item for item in items if item.successful]
        if not failures or not successes:
            continue
        failed_counts = Counter(token for item in failures for token in _failure_tokens(item))
        success_counts = Counter(token for item in successes for token in _failure_tokens(item))
        for token, failed in failed_counts.items():
            successful = success_counts[token]
            if failed + successful < minimum_support:
                continue
            failure_rate = (failed + 0.5) / (failed + successful + 1.0)
            background = (len(failures) + 0.5) / (len(items) + 1.0)
            enrichments.append(
                {
                    "family": family,
                    "sku": sku,
                    "feature": token,
                    "failed": failed,
                    "successful": successful,
                    "failure_rate": failure_rate,
                    "background_failure_rate": background,
                    "risk_ratio": failure_rate / background,
                }
            )
    enrichments.sort(
        key=lambda item: int(item["failed"]) * math.log(max(1.0, float(item["risk_ratio"]))),
        reverse=True,
    )
    return {
        "statuses": dict(statuses),
        "failure_kinds": dict(kinds),
        "enrichments": enrichments,
    }


def strategy_efficiency_analysis(
    rows: Sequence[Observation[object]],
) -> list[dict[str, object]]:
    """Measure useful configurations per observed optimizer/compile second."""

    context_best: dict[str, float] = {}
    for item in rows:
        if item.successful and item.score > 0:
            context_best[item.context_id] = min(
                context_best.get(item.context_id, math.inf), item.score
            )
    groups: dict[tuple[str, str, str], list[Observation[object]]] = defaultdict(list)
    for item in rows:
        groups[(item.family, _sku(item), item.strategy)].append(item)
    result = []
    for (family, sku, strategy), items in sorted(groups.items()):
        wall_s = sum(max(0.0, item.elapsed_s) for item in items)
        proposal_s = sum(
            max(0.0, float(item.metadata.get("proposal_elapsed_s", 0.0) or 0.0))
            for item in items
        )
        compile_s = sum(
            max(0.0, float(item.outcome.compile_ms or 0.0)) / 1000.0
            for item in items
        )
        useful = sum(
            item.successful
            and item.context_id in context_best
            and item.score <= context_best[item.context_id] * 1.02
            for item in items
        )
        result.append(
            {
                "family": family,
                "sku": sku,
                "strategy": strategy,
                "trials": len(items),
                "successful": sum(item.successful for item in items),
                "failures": len(items) - sum(item.successful for item in items),
                "within_2pct_of_context_best": useful,
                "evaluator_seconds": wall_s,
                "proposal_seconds": proposal_s,
                "compile_seconds": compile_s,
                "total_attributed_seconds": wall_s + proposal_s,
                "useful_per_minute": 60.0 * useful / max(1e-9, wall_s + proposal_s),
            }
        )
    return result


def timing_convergence_analysis(
    rows: Sequence[Observation[object]],
) -> dict[str, object]:
    usable = [
        item
        for item in rows
        if item.successful and len(item.outcome.timings_ms) >= 3
    ]
    sample_counts = (1, 2, 3, 5, 9, 12, 15, 20)
    result = []
    for family in sorted({item.family for item in usable}):
        for sku in sorted({_sku(item) for item in usable if item.family == family}):
            selected = [
                item for item in usable if item.family == family and _sku(item) == sku
            ]
            common = [
                item for item in selected if len(item.outcome.timings_ms) >= 15
            ]
            common_20 = [
                item for item in selected if len(item.outcome.timings_ms) >= 20
            ]
            for cohort, cohort_rows in (
                ("available", selected),
                ("common_15", common),
                ("common_20", common_20),
            ):
                for count in sample_counts:
                    eligible = [
                        item
                        for item in cohort_rows
                        if len(item.outcome.timings_ms) >= count
                    ]
                    if not eligible:
                        continue
                    errors = []
                    for item in eligible:
                        timings = np.asarray(item.outcome.timings_ms, dtype=np.float64)
                        full = float(np.median(timings))
                        prefix = float(np.median(timings[:count]))
                        errors.append(abs(prefix / full - 1.0))
                    by_context: dict[str, list[Observation[object]]] = defaultdict(list)
                    for item in eligible:
                        by_context[item.context_id].append(item)
                    regrets = []
                    for catalog in by_context.values():
                        if len(catalog) < 2:
                            continue
                        chosen = min(
                            catalog,
                            key=lambda item: float(
                                np.median(item.outcome.timings_ms[:count])
                            ),
                        )
                        full_scores = {
                            item.config_id: float(np.median(item.outcome.timings_ms))
                            for item in catalog
                        }
                        regrets.append(
                            full_scores[chosen.config_id] / min(full_scores.values())
                            - 1.0
                        )
                    result.append(
                        {
                            "family": family,
                            "sku": sku,
                            "cohort": cohort,
                            "samples": count,
                            "rows": len(eligible),
                            "contexts": len(regrets),
                            "median_absolute_relative_error": float(np.median(errors)),
                            "p90_absolute_relative_error": float(
                                np.quantile(errors, 0.9)
                            ),
                            "winner_median_regret": None
                            if not regrets
                            else float(np.median(regrets)),
                            "winner_p90_regret": None
                            if not regrets
                            else float(np.quantile(regrets, 0.9)),
                            "winner_within_2pct": None
                            if not regrets
                            else float(np.mean(np.asarray(regrets) <= 0.02)),
                        }
                    )
    recommendations = []
    for family in sorted({item["family"] for item in result}):
        for sku in sorted(
            {item["sku"] for item in result if item["family"] == family}
        ):
            preferred_cohort = (
                "common_20"
                if any(
                    item["family"] == family
                    and item["sku"] == sku
                    and item["cohort"] == "common_20"
                    and int(item["contexts"]) >= 5
                    for item in result
                )
                else "common_15"
            )
            candidates = [
                item
                for item in result
                if item["family"] == family
                and item["sku"] == sku
                and item["cohort"] == preferred_cohort
                and item["samples"] in (3, 5, 9, 15, 20)
            ]
            qualified = [
                item
                for item in candidates
                if float(item["p90_absolute_relative_error"]) <= 0.01
                and item["winner_p90_regret"] is not None
                and float(item["winner_p90_regret"]) <= 0.02
                and item["winner_within_2pct"] is not None
                and float(item["winner_within_2pct"]) >= 0.95
            ]
            selected = (
                min(qualified, key=lambda item: int(item["samples"]))
                if qualified
                else max(
                    candidates,
                    key=lambda item: int(item["samples"]),
                    default=None,
                )
            )
            if selected is not None:
                recommendations.append(
                    {
                        "family": family,
                        "sku": sku,
                        "samples": selected["samples"],
                        "cohort": preferred_cohort,
                        "qualified": selected in qualified,
                        "criteria": {
                            "p90_median_error_max": 0.01,
                            "p90_winner_regret_max": 0.02,
                            "winner_within_2pct_min": 0.95,
                        },
                    }
                )
    return {
        "rows_with_raw_timings": len(usable),
        "sample_count_distribution": dict(
            Counter(len(item.outcome.timings_ms) for item in usable)
        ),
        "convergence": result,
        "screening_recommendations": recommendations,
    }


def _markdown(report: Mapping[str, object]) -> str:
    lines = [
        "# Autotuning evidence study",
        "",
        "All latency effects are paired or normalized within exact contexts. Shape-held-out models never train on the same M/N/K group they evaluate.",
        "",
        "## Parent-linked mutations",
        "",
        f"- Paired moves: {report['parent_moves']['paired_moves']}",
        f"- Coupled/composite moves: {report['parent_moves']['coupled_or_composite_moves']}",
        "",
        "| Family | SKU | Coordinate | Old → new | Relative latency | 95% CI | P(improve) | Pairs |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    shown = 0
    for item in report["parent_moves"]["effects"]:
        if (
            not item["qualified"]
            or item["old_value"] == "<coupled-or-composite>"
        ):
            continue
        lines.append(
            f"| {item['family']} | {item['sku']} | `{item['coordinate']}` | `{json.dumps(item['old_value'])}` → `{json.dumps(item['new_value'])}` | {float(item['relative_latency']):.3f}x | [{math.exp(float(item['ci_low'])):.3f}, {math.exp(float(item['ci_high'])):.3f}] | {float(item['improvement_probability']):.1%} | {item['pairs']} |"
        )
        shown += 1
        if shown >= 40:
            break
    lines.extend(["", "## Shape-held-out pairwise model", ""])
    for family, item in report["pairwise_shape_heldout"].items():
        gate = item["deployment_gate"]
        lines.append(
            f"- `{family}`: gate **{'PASS' if gate['enabled'] else 'FAIL'}**, {gate['passed_folds']}/{len(item['folds'])} folds beat random at 8 trials."
        )
        for fold in item["folds"]:
            classification = fold["pair_classification"]
            replay = fold["catalog_replay"]
            model = replay["model"].get("8", {})
            random = replay["random"].get("8", {})
            lines.append(
                f"  - fold {fold['fold']}: pair AUC {classification['auc']:.3f}; model regret {model.get('median', math.nan):.2%}/{model.get('p90', math.nan):.2%} median/p90; random {random.get('median', math.nan):.2%}/{random.get('p90', math.nan):.2%}."
            )
        for device, device_item in item.get("device_models", {}).items():
            device_gate = device_item["deployment_gate"]
            lines.append(
                f"  - exact SKU `{device}`: gate **{'PASS' if device_gate['enabled'] else 'FAIL'}**, {device_gate['passed_folds']}/{len(device_item['folds'])} folds."
            )
    failure = report["failures"]
    lines.extend(
        [
            "",
            "## Failure taxonomy",
            "",
            f"- Statuses: `{json.dumps(failure['statuses'], sort_keys=True)}`",
            f"- Failure kinds: `{json.dumps(failure['failure_kinds'], sort_keys=True)}`",
            "",
            "| Family | SKU | Feature | Failures | Successes | Risk ratio |",
            "|---|---|---|---:|---:|---:|",
        ]
    )
    for item in failure["enrichments"][:30]:
        lines.append(
            f"| {item['family']} | {item['sku']} | `{item['feature']}` | {item['failed']} | {item['successful']} | {float(item['risk_ratio']):.2f}x |"
        )
    lines.extend(
        [
            "",
            "## Strategy efficiency",
            "",
            "| Family | SKU | Strategy | Trials | Failures | Within 2% | Useful/min | Proposal time |",
            "|---|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    for item in sorted(
        report["strategy_efficiency"],
        key=lambda value: float(value["useful_per_minute"]),
        reverse=True,
    ):
        lines.append(
            f"| {item['family']} | {item['sku']} | {item['strategy']} | {item['trials']} | {item['failures']} | {item['within_2pct_of_context_best']} | {float(item['useful_per_minute']):.2f} | {float(item['proposal_seconds']):.1f}s |"
        )
    timing = report["timing_convergence"]
    lines.extend(
        [
            "",
            "## Timing convergence",
            "",
            "The `common_15` and `common_20` cohorts hold the candidate population fixed, avoiding the selection bias caused by comparing configurations which happened to receive different sample counts.",
            "",
            "| Family | SKU | Samples | P90 median error | P90 winner regret | Within 2% |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for item in timing["convergence"]:
        if item["cohort"] != "common_15" or item["samples"] not in (1, 3, 5, 9, 15):
            continue
        regret = item["winner_p90_regret"]
        within = item["winner_within_2pct"]
        lines.append(
            f"| {item['family']} | {item['sku']} | {item['samples']} | {float(item['p90_absolute_relative_error']):.2%} | {'n/a' if regret is None else f'{float(regret):.2%}'} | {'n/a' if within is None else f'{float(within):.1%}'} |"
        )
    lines.extend(
        [
            "",
            "Empirical screening floors (≤1% p90 median error, ≤2% p90 winner regret, and ≥95% winners within 2% on the fixed cohort):",
            "",
        ]
    )
    for item in timing["screening_recommendations"]:
        lines.append(
            f"- `{item['family']}` / `{item['sku']}`: {item['samples']} samples on `{item['cohort']}` ({'qualified' if item['qualified'] else 'no tested count met every threshold'})."
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "Pairwise deployment remains gated by complete-shape held-out replay. Failure enrichments and bottleneck classes are diagnostic hypotheses; they become legality or runtime policy only after deterministic boundary tests or prospective validation.",
            "",
        ]
    )
    return "\n".join(lines)


def build_evidence_study(
    paths: Sequence[Path | str],
    output: Path | str,
    *,
    seed: int = 20260817,
    minimum_pairs: int = 8,
    minimum_contexts: int = 5,
    pairwise_estimators: int = 16,
) -> dict[str, object]:
    rows, load_report = load_offline_observations(paths)
    if not rows:
        raise ValueError("no autotuning observations were found")
    destination = Path(output).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    pairwise = pairwise_shape_heldout_study(
        rows, seed=seed, n_estimators=pairwise_estimators
    )
    pairwise_artifact = save_deployable_pairwise_models(
        rows,
        pairwise,
        destination,
        seed=seed,
        n_estimators=pairwise_estimators,
    )
    report: dict[str, object] = {
        "schema_version": 1,
        "type": "rtx_autotuning_evidence_study",
        "input": load_report,
        "parent_moves": parent_move_analysis(
            rows, seed=seed, minimum_pairs=minimum_pairs
        ),
        "pairwise_shape_heldout": pairwise,
        "pairwise_artifact": pairwise_artifact,
        "archetypes": archetype_analysis(
            rows, minimum_contexts=minimum_contexts
        ),
        "failures": failure_analysis(rows, minimum_support=minimum_pairs),
        "strategy_efficiency": strategy_efficiency_analysis(rows),
        "timing_convergence": timing_convergence_analysis(rows),
    }
    (destination / "evidence.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (destination / "evidence.md").write_text(_markdown(report), encoding="utf-8")
    return report


__all__ = [
    "archetype_analysis",
    "bottleneck_class",
    "build_evidence_study",
    "failure_analysis",
    "PairwisePreferenceModel",
    "pair_features",
    "pairwise_shape_heldout_study",
    "parent_move_analysis",
    "strategy_efficiency_analysis",
    "save_deployable_pairwise_models",
    "timing_convergence_analysis",
]
