"""Offline pretraining and conditional-effect priors for autotuning.

Artifacts are deliberately kernel-revision scoped.  They are proposal priors,
not runtime winners: every candidate still has to compile, run, pass numerical
validation, and win on the local device.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Iterable, Mapping, Sequence
import zipfile

import numpy as np

from .core import FeatureMap, Observation, canonical_json
from .cost_model import GradientBoostedCostModel, GradientBoostedFeasibilityModel
from .outcomes import TrialOutcome


PRETRAINED_SCHEMA_VERSION = 1
# Bump this whenever training, validation, or deployment semantics change in a
# way that can alter how an otherwise identical set of model files is used.
# Schema version describes readability; trainer revision describes behavior.
PRETRAINED_TRAINER_REVISION = 5
_ALL_FAILURES = (
    "compile_error",
    "runtime_error",
    "correctness_error",
    "implementation_rejected",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        dir=path.parent, prefix=f".{path.name}.", suffix=".tmp", text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as sink:
            json.dump(dict(value), sink, indent=2, sort_keys=True)
            sink.write("\n")
            sink.flush()
            os.fsync(sink.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _observation_documents(
    paths: Iterable[Path | str], *, campaigns: tuple[str, ...] = ()
):
    """Yield raw observation records from directories, JSONL files, or ZIPs."""

    seen_files: set[str] = set()
    for raw in paths:
        path = Path(raw).expanduser()
        if path.is_file() and path.suffix.lower() == ".zip":
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    if member.is_dir() or not member.filename.endswith(
                        "/observations.jsonl"
                    ):
                        continue
                    identity = f"{path.resolve()}!{member.filename}"
                    if campaigns and not any(
                        f"/{campaign}/" in f"/{member.filename}"
                        for campaign in campaigns
                    ):
                        continue
                    if identity in seen_files:
                        continue
                    seen_files.add(identity)
                    with archive.open(member) as source:
                        for encoded in source:
                            if not encoded.strip():
                                continue
                            try:
                                yield identity, json.loads(encoded)
                            except (UnicodeDecodeError, json.JSONDecodeError):
                                continue
            continue
        candidates = (
            [path]
            if path.is_file() and path.name == "observations.jsonl"
            else path.rglob("observations.jsonl")
        )
        for candidate in candidates:
            identity = str(candidate.resolve())
            if campaigns and not any(
                campaign in candidate.parts for campaign in campaigns
            ):
                continue
            if identity in seen_files:
                continue
            seen_files.add(identity)
            with candidate.open(encoding="utf-8") as source:
                for line in source:
                    if not line.strip():
                        continue
                    try:
                        yield identity, json.loads(line)
                    except json.JSONDecodeError:
                        continue


def portable_features(features: Mapping[str, object]) -> FeatureMap:
    """Remove provenance and high-cardinality timing noise from model inputs."""

    rejected_prefixes = (
        "context.tags.",
        "context.device.software.",
        "context.device.calibration.raw_",
        "context.device.calibration.native_mxfp8_outcome.timings_ms",
        "context.device.calibration.native_mxfp8_outcome.components.",
        "context.device.calibration.native_mxfp8_outcome.telemetry_",
    )
    rejected_fragments = (
        ".uuid=",
        ".platform=",
        ".name=",
        ".recorded_at",
        ".timestamp",
    )
    result: FeatureMap = {}
    for key, value in features.items():
        name = str(key)
        if name.startswith(rejected_prefixes) or any(
            fragment in name for fragment in rejected_fragments
        ):
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if math.isfinite(numeric):
            result[name] = numeric
    baseline = analytical_baseline_ms(result)
    result["offline.analytical_baseline_ms"] = baseline
    for axis in ("m", "n", "k"):
        value = result.get(f"context.workload.{axis}")
        if value is not None and value > 0:
            result[f"offline.log2_{axis}"] = math.log2(value)
    return result


def analytical_baseline_ms(features: Mapping[str, float]) -> float:
    """Portable lower-bound scale used to residualize cross-device latency."""

    exact_memory = float(features.get("derived.memory_roofline_ms", 0.0))
    component_memory = [
        float(value)
        for key, value in features.items()
        if key.startswith("derived.")
        and key.endswith("_memory_roofline_ms")
        and key != "derived.memory_roofline_ms"
        and float(value) > 0
    ]
    memory_ms = exact_memory if exact_memory > 0 else sum(component_memory)
    flops = float(
        features.get(
            "derived.combined_nominal_flops",
            features.get("derived.nominal_flops", 0.0),
        )
    )
    throughput = float(
        features.get(
            "context.device.calibration.measured_native_mxfp8_gemm_tflops",
            features.get(
                "context.device.calibration.measured_bf16_matmul_tflops", 0.0
            ),
        )
    )
    compute_ms = flops / (throughput * 1.0e9) if flops > 0 and throughput > 0 else 0.0
    return max(1.0e-6, memory_ms, compute_ms)


def _explicitly_nonstationary(outcome: TrialOutcome) -> bool:
    sampling = outcome.metadata.get("sampling")
    if not isinstance(sampling, Mapping):
        return False
    collection = sampling.get("collection")
    return (
        isinstance(collection, Mapping)
        and collection.get("stationary") is False
    )


def load_offline_observations(
    paths: Iterable[Path | str],
    *,
    campaign: str | Sequence[str] | None = None,
) -> tuple[list[Observation[dict[str, object]]], dict[str, object]]:
    """Load and deduplicate portable observations without importing CUDA."""

    campaigns = (
        ()
        if campaign is None
        else (campaign,)
        if isinstance(campaign, str)
        else tuple(dict.fromkeys(campaign))
    )
    unique: dict[tuple[str, str], Observation[dict[str, object]]] = {}
    source_files: set[str] = set()
    malformed = 0
    for source, record in _observation_documents(paths, campaigns=campaigns):
        source_files.add(source)
        try:
            features = portable_features(dict(record["features"]))
            serialized = dict(record["config"])
            outcome = TrialOutcome.from_dict(record["outcome"])
            observation = Observation(
                observation_id=str(record["observation_id"]),
                session_id=str(record.get("session_id", "offline")),
                sequence=int(record.get("sequence", 0)),
                context_id=str(record["context_id"]),
                family=str(record["family"]),
                kernel_revision=int(record["kernel_revision"]),
                config_id=str(record["config_id"]),
                config=serialized,
                serialized_config=serialized,
                features=features,
                strategy=str(record.get("strategy", "unknown")),
                outcome=outcome,
                started_at=str(record.get("started_at", "")),
                finished_at=str(record.get("finished_at", "")),
                elapsed_s=float(record.get("elapsed_s", 0.0)),
                parent_config_id=(
                    None
                    if record.get("parent_config_id") is None
                    else str(record["parent_config_id"])
                ),
                coordinate=(
                    None if record.get("coordinate") is None else str(record["coordinate"])
                ),
                coordinate_value=record.get("coordinate_value"),
                metadata=dict(record.get("metadata", {})),
            )
        except (KeyError, TypeError, ValueError):
            malformed += 1
            continue
        unique.setdefault((observation.context_id, observation.observation_id), observation)
    observations = list(unique.values())
    dataset_digest = hashlib.sha256()
    for item in sorted(
        observations, key=lambda value: (value.context_id, value.observation_id)
    ):
        dataset_digest.update(
            canonical_json(
                {
                    "context_id": item.context_id,
                    "observation_id": item.observation_id,
                    "family": item.family,
                    "kernel_revision": item.kernel_revision,
                    "config_id": item.config_id,
                    "status": item.outcome.status,
                    "median_ms": item.outcome.median_ms,
                }
            ).encode()
        )
    nonstationary_successes = sum(
        item.successful and _explicitly_nonstationary(item.outcome)
        for item in observations
    )
    report = {
        "source_files": sorted(source_files),
        "observations": len(observations),
        "dataset_sha256": dataset_digest.hexdigest(),
        "observation_identity_hashes": sorted(
            hashlib.sha256(
                f"{item.context_id}:{item.observation_id}".encode()
            ).hexdigest()[:24]
            for item in observations
        ),
        "malformed": malformed,
        "latency_view": {
            "policy": "exclude_explicitly_nonstationary_successes",
            "excluded_nonstationary_successes": nonstationary_successes,
            "rows": len(observations) - nonstationary_successes,
        },
        "feasibility_view": {
            "policy": "retain_all_statuses_including_nonstationary_successes",
            "rows": len(observations),
        },
        "campaign_filter": list(campaigns),
        "families": dict(Counter(item.family for item in observations)),
        "devices": sorted(
            {
                key.split("=", 1)[1]
                for item in observations
                for key in item.features
                if key.startswith("context.device.sku.sku_family=")
            }
        ),
    }
    return observations, report


def _usable_for_latency(item: Observation[object]) -> bool:
    """Failures carry legality evidence; unstable successes carry no latency target."""

    if not item.successful:
        return True
    return not _explicitly_nonstationary(item.outcome)


class NormalizedCostModel:
    """Predict latency as an analytical lower bound times a learned residual."""

    def __init__(self, model: GradientBoostedCostModel | None = None) -> None:
        self.model = model or GradientBoostedCostModel()

    @property
    def fitted(self) -> bool:
        return self.model.fitted

    @property
    def parameter_count(self) -> int:
        return self.model.parameter_count

    def fit(self, observations: Sequence[Observation[object]]) -> None:
        normalized: list[Observation[object]] = []
        for item in observations:
            if not item.successful or item.outcome.median_ms is None:
                continue
            features = portable_features(item.features)
            baseline = analytical_baseline_ms(features)
            normalized.append(
                replace(
                    item,
                    features=features,
                    outcome=replace(
                        item.outcome,
                        median_ms=float(item.outcome.median_ms) / baseline,
                        timings_ms=[value / baseline for value in item.outcome.timings_ms],
                    ),
                )
            )
        self.model.fit(normalized)

    def predict(self, features: Sequence[FeatureMap]) -> tuple[np.ndarray, np.ndarray]:
        portable = [portable_features(row) for row in features]
        ratio, ratio_std = self.model.predict(portable)
        baseline = np.asarray(
            [analytical_baseline_ms(row) for row in portable], dtype=np.float64
        )
        return ratio * baseline, ratio_std * baseline

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "type": "normalized_gradient_boosted_cost_model",
            "baseline": "max(memory_roofline,compute_roofline)",
            "model": self.model.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "NormalizedCostModel":
        return cls(
            GradientBoostedCostModel.from_state_dict(state["model"])  # type: ignore[arg-type]
        )

    def save(self, path: Path | str) -> None:
        _atomic_json(Path(path), self.state_dict())

    @classmethod
    def load(cls, path: Path | str) -> "NormalizedCostModel":
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))


class ContextRankingModel:
    """Learn configuration order after removing each context's latency scale."""

    def __init__(self, model: GradientBoostedCostModel | None = None) -> None:
        self.model = model or GradientBoostedCostModel()

    @property
    def fitted(self) -> bool:
        return self.model.fitted

    @property
    def parameter_count(self) -> int:
        return self.model.parameter_count

    def fit(self, observations: Sequence[Observation[object]]) -> None:
        grouped: dict[str, list[Observation[object]]] = defaultdict(list)
        for item in observations:
            if item.successful and item.score > 0:
                grouped[item.context_id].append(item)
        centered: list[Observation[object]] = []
        for rows in grouped.values():
            center = float(np.median([math.log(item.score) for item in rows]))
            for item in rows:
                relative = math.exp(math.log(item.score) - center)
                centered.append(
                    replace(
                        item,
                        features=portable_features(item.features),
                        outcome=replace(
                            item.outcome,
                            median_ms=relative,
                            timings_ms=[],
                        ),
                    )
                )
        self.model.fit(centered)

    def predict(self, features: Sequence[FeatureMap]) -> tuple[np.ndarray, np.ndarray]:
        return self.model.predict([portable_features(row) for row in features])

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "type": "context_centered_gradient_boosted_ranking_model",
            "target": "exp(log_latency-context_median_log_latency)",
            "model": self.model.state_dict(),
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "ContextRankingModel":
        return cls(
            GradientBoostedCostModel.from_state_dict(state["model"])  # type: ignore[arg-type]
        )

    def save(self, path: Path | str) -> None:
        _atomic_json(Path(path), self.state_dict())

    @classmethod
    def load(cls, path: Path | str) -> "ContextRankingModel":
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True, slots=True)
class ConditionalEffectRule:
    coordinate: str
    coordinate_value_json: str
    feature: str | None
    low: float | None
    high: float | None
    effect_log_latency: float
    ci_low: float
    ci_high: float
    support: int
    contexts: int
    devices: int

    def as_dict(self) -> dict[str, object]:
        return {
            "coordinate": self.coordinate,
            "coordinate_value": json.loads(self.coordinate_value_json),
            "condition": (
                None
                if self.feature is None
                else {"feature": self.feature, "low": self.low, "high": self.high}
            ),
            "effect_log_latency": self.effect_log_latency,
            "median_speedup": math.exp(-self.effect_log_latency),
            "ci_low": self.ci_low,
            "ci_high": self.ci_high,
            "support": self.support,
            "contexts": self.contexts,
            "devices": self.devices,
        }


class ConditionalRuleSet:
    """Confidence-qualified local-move effects used as a soft ranking prior."""

    def __init__(self, rules: Sequence[ConditionalEffectRule] = ()) -> None:
        self.rules = tuple(rules)

    def adjustment(
        self,
        features: FeatureMap,
        *,
        coordinate: str | None,
        coordinate_value: object,
    ) -> tuple[float, list[int]]:
        if coordinate is None:
            return 0.0, []
        encoded = canonical_json(coordinate_value)
        matches: list[tuple[int, ConditionalEffectRule]] = []
        for index, rule in enumerate(self.rules):
            if rule.coordinate != coordinate or rule.coordinate_value_json != encoded:
                continue
            if rule.feature is not None:
                value = features.get(rule.feature)
                if value is None:
                    continue
                if rule.low is not None and value < rule.low:
                    continue
                if rule.high is not None and value >= rule.high:
                    continue
            matches.append((index, rule))
        if not matches:
            return 0.0, []
        # Prefer conditional evidence; confidence/support shrink effects rather
        # than turning observational rules into hard legality constraints.
        conditional = [item for item in matches if item[1].feature is not None]
        selected = conditional or matches
        weights = np.asarray(
            [math.sqrt(rule.support) * min(1.0, rule.contexts / 4.0) for _, rule in selected],
            dtype=np.float64,
        )
        effects = np.asarray(
            [rule.effect_log_latency for _, rule in selected], dtype=np.float64
        )
        return float(np.average(effects, weights=weights)), [index for index, _ in selected]

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "type": "conditional_effect_rules",
            "rules": [rule.as_dict() for rule in self.rules],
        }

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "ConditionalRuleSet":
        rules = []
        for item in state.get("rules", []):  # type: ignore[union-attr]
            condition = item.get("condition")
            rules.append(
                ConditionalEffectRule(
                    coordinate=str(item["coordinate"]),
                    coordinate_value_json=canonical_json(item.get("coordinate_value")),
                    feature=None if condition is None else str(condition["feature"]),
                    low=None if condition is None or condition.get("low") is None else float(condition["low"]),
                    high=None if condition is None or condition.get("high") is None else float(condition["high"]),
                    effect_log_latency=float(item["effect_log_latency"]),
                    ci_low=float(item["ci_low"]),
                    ci_high=float(item["ci_high"]),
                    support=int(item["support"]),
                    contexts=int(item["contexts"]),
                    devices=int(item["devices"]),
                )
            )
        return cls(rules)

    def save(self, path: Path | str) -> None:
        _atomic_json(Path(path), self.state_dict())

    @classmethod
    def load(cls, path: Path | str) -> "ConditionalRuleSet":
        return cls.from_state_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def _device_key(item: Observation[object]) -> str:
    for key in item.features:
        if key.startswith("context.device.sku.sku_family="):
            return key.split("=", 1)[1]
    sms = item.features.get("context.device.multiprocessor_count", -1.0)
    bus = item.features.get("context.device.sku.memory_bus_width_bits", -1.0)
    return f"sms-{sms:g}-bus-{bus:g}"


def _shape_group_key(item: Observation[object]) -> tuple[object, ...]:
    """Validation group which cannot leak through regimes or replicates."""

    return (
        item.family,
        item.kernel_revision,
        int(item.features.get("context.workload.m", 0)),
        int(item.features.get("context.workload.n", 0)),
        int(item.features.get("context.workload.k", 0)),
    )


def _bootstrap_median_ci(
    values: Sequence[float], *, seed: int, resamples: int = 300
) -> tuple[float, float, float]:
    array = np.asarray(values, dtype=np.float64)
    median = float(np.median(array))
    if len(array) < 2:
        return median, median, median
    rng = np.random.default_rng(seed)
    samples = rng.choice(array, size=(resamples, len(array)), replace=True)
    medians = np.median(samples, axis=1)
    return median, float(np.quantile(medians, 0.025)), float(np.quantile(medians, 0.975))


def extract_conditional_rules(
    observations: Sequence[Observation[object]],
    *,
    min_support: int = 12,
    minimum_effect: float = 0.01,
    max_rules: int = 256,
    seed: int = 0,
) -> ConditionalRuleSet:
    """Extract paired parent→child effects and robust conditional reversals."""

    by_config = {
        (item.context_id, item.config_id): item
        for item in observations
        if item.successful
    }
    moves = []
    for child in observations:
        if (
            not child.successful
            or child.parent_config_id is None
            or child.coordinate is None
            or child.score <= 0
        ):
            continue
        parent = by_config.get((child.context_id, child.parent_config_id))
        if parent is None or parent.score <= 0:
            continue
        moves.append(
            (
                child.coordinate,
                canonical_json(child.coordinate_value),
                math.log(child.score / parent.score),
                child,
            )
        )
    if not moves:
        return ConditionalRuleSet()

    condition_suffixes = (
        "effective_cta_waves",
        "estimated_resident_ctas_per_sm",
        "smem_fraction_per_cta",
        "register_fraction_per_cta",
        "working_set_l2_ratio",
        "memory_roofline_ms",
        "arithmetic_intensity_flops_per_byte",
        "grid_ctas",
    )
    feature_names = sorted(
        {
            name
            for _coordinate, _value, _effect, item in moves
            for name in item.features
            if name.startswith("derived.") and name.endswith(condition_suffixes)
        }
    )
    grouped: dict[tuple[str, str], list[tuple[float, Observation[object]]]] = defaultdict(list)
    for coordinate, value, effect, item in moves:
        grouped[(coordinate, value)].append((effect, item))

    candidates: list[ConditionalEffectRule] = []

    def add_rule(
        coordinate: str,
        value: str,
        rows: Sequence[tuple[float, Observation[object]]],
        feature: str | None,
        low: float | None,
        high: float | None,
        salt: int,
    ) -> None:
        if len(rows) < min_support:
            return
        effects = [effect for effect, _item in rows]
        median, ci_low, ci_high = _bootstrap_median_ci(
            effects, seed=seed ^ salt
        )
        if abs(median) < minimum_effect or not (ci_high < 0 or ci_low > 0):
            return
        candidates.append(
            ConditionalEffectRule(
                coordinate,
                value,
                feature,
                low,
                high,
                median,
                ci_low,
                ci_high,
                len(rows),
                len({item.context_id for _effect, item in rows}),
                len({_device_key(item) for _effect, item in rows}),
            )
        )

    for group_index, ((coordinate, value), rows) in enumerate(sorted(grouped.items())):
        add_rule(coordinate, value, rows, None, None, None, group_index)
        for feature_index, feature in enumerate(feature_names):
            available = [(effect, item) for effect, item in rows if feature in item.features]
            if len(available) < min_support * 2:
                continue
            values = np.asarray([item.features[feature] for _effect, item in available])
            boundaries = np.unique(np.quantile(values, [0.0, 1 / 3, 2 / 3, 1.0]))
            if len(boundaries) < 3:
                continue
            for bin_index in range(len(boundaries) - 1):
                low = float(boundaries[bin_index])
                high = None if bin_index == len(boundaries) - 2 else float(boundaries[bin_index + 1])
                selected = [
                    (effect, item)
                    for effect, item in available
                    if item.features[feature] >= low
                    and (high is None or item.features[feature] < high)
                ]
                add_rule(
                    coordinate,
                    value,
                    selected,
                    feature,
                    low,
                    high,
                    group_index * 100_003 + feature_index * 101 + bin_index,
                )
    candidates.sort(
        key=lambda rule: (
            rule.devices >= 2,
            rule.contexts,
            math.sqrt(rule.support) * abs(rule.effect_log_latency),
        ),
        reverse=True,
    )
    return ConditionalRuleSet(candidates[:max_rules])


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=np.float64)
    ranks[order] = np.arange(len(values), dtype=np.float64)
    return ranks


def evaluate_latency_model(
    model, observations: Sequence[Observation[object]]
) -> dict[str, object]:
    usable = [item for item in observations if item.successful and item.score > 0]
    if not usable:
        return {"rows": 0}
    predicted, _uncertainty = model.predict([item.features for item in usable])
    actual = np.asarray([item.score for item in usable], dtype=np.float64)
    log_mae = float(np.mean(np.abs(np.log(predicted) - np.log(actual))))
    rank_a = _rank(actual)
    rank_p = _rank(predicted)
    spearman = float(np.corrcoef(rank_a, rank_p)[0, 1]) if len(actual) > 1 else 1.0
    by_context: dict[str, list[int]] = defaultdict(list)
    for index, item in enumerate(usable):
        by_context[item.context_id].append(index)
    regrets = []
    within_context_correlations = []
    budget_regrets: dict[int, list[float]] = {
        budget: [] for budget in (1, 4, 8, 16, 32)
    }
    random_budget_regrets: dict[int, list[float]] = {
        budget: [] for budget in budget_regrets
    }
    for context_id, indices in by_context.items():
        chosen = min(indices, key=lambda index: float(predicted[index]))
        best = min(float(actual[index]) for index in indices)
        regrets.append(float(actual[chosen]) / best - 1.0)
        predicted_order = sorted(indices, key=lambda index: float(predicted[index]))
        for budget, values in budget_regrets.items():
            chosen_rows = predicted_order[: min(budget, len(predicted_order))]
            found = min(float(actual[index]) for index in chosen_rows)
            values.append(found / best - 1.0)
            random_rng = np.random.default_rng(
                int(hashlib.sha256(f"{context_id}:{budget}".encode()).hexdigest()[:16], 16)
            )
            random_regrets = []
            for _ in range(200):
                chosen_random = random_rng.choice(
                    indices, size=min(budget, len(indices)), replace=False
                )
                random_found = min(float(actual[index]) for index in chosen_random)
                random_regrets.append(random_found / best - 1.0)
            random_budget_regrets[budget].append(float(np.median(random_regrets)))
        if len(indices) > 1:
            within_context_correlations.append(
                float(
                    np.corrcoef(
                        _rank(actual[indices]), _rank(predicted[indices])
                    )[0, 1]
                )
            )
    return {
        "rows": len(usable),
        "contexts": len(by_context),
        "log_mae": log_mae,
        "spearman": spearman,
        "within_context_median_spearman": float(
            np.median(within_context_correlations)
        ),
        "predicted_best_median_regret": float(np.median(regrets)),
        "predicted_best_p90_regret": float(np.quantile(regrets, 0.9)),
        "catalog_replay_regret": {
            str(budget): {
                "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.9)),
            }
            for budget, values in budget_regrets.items()
        },
        "random_catalog_replay_regret": {
            str(budget): {
                "median": float(np.median(values)),
                "p90": float(np.quantile(values, 0.9)),
            }
            for budget, values in random_budget_regrets.items()
        },
    }


def evaluate_feasibility_model(
    model: GradientBoostedFeasibilityModel,
    observations: Sequence[Observation[object]],
) -> dict[str, object]:
    positive = set(model.positive_statuses)
    negative = set(model.negative_statuses)
    usable = [item for item in observations if item.outcome.status in positive | negative]
    failures = [item for item in usable if item.outcome.status in negative]
    if not usable or not failures or len(failures) == len(usable) or not model.fitted:
        return {"rows": len(usable), "failures": len(failures), "evaluable": False}
    probability, _uncertainty = model.predict([item.features for item in usable])
    labels = np.asarray(
        [1.0 if item.outcome.status in positive else 0.0 for item in usable]
    )
    positive_scores = probability[labels == 1]
    negative_scores = probability[labels == 0]
    auc = float(
        np.mean(
            positive_scores[:, None] > negative_scores[None, :]
        )
        + 0.5
        * np.mean(positive_scores[:, None] == negative_scores[None, :])
    )
    prediction = probability >= 0.5
    true_positive_rate = float(np.mean(prediction[labels == 1]))
    true_negative_rate = float(np.mean(~prediction[labels == 0]))
    order = np.argsort(probability)
    capture = {}
    total_failures = int(np.count_nonzero(labels == 0))
    for fraction in (0.01, 0.05, 0.10):
        count = max(1, int(math.ceil(len(usable) * fraction)))
        captured = int(np.count_nonzero(labels[order[:count]] == 0))
        capture[str(fraction)] = captured / total_failures
    return {
        "rows": len(usable),
        "failures": total_failures,
        "evaluable": True,
        "auc": auc,
        "balanced_accuracy_at_0.5": 0.5 * (true_positive_rate + true_negative_rate),
        "failure_capture_in_lowest_probability_fraction": capture,
    }


def _fold_head_beats_random(
    folds: Sequence[Mapping[str, object]], head: str, budget: str = "4"
) -> tuple[bool, dict[str, object]]:
    comparisons: list[tuple[float, float, float, float]] = []
    for fold in folds:
        metrics = fold[head]
        model_metrics = metrics["catalog_replay_regret"][budget]
        random_metrics = metrics["random_catalog_replay_regret"][budget]
        comparisons.append(
            (
                float(model_metrics["median"]),
                float(model_metrics["p90"]),
                float(random_metrics["median"]),
                float(random_metrics["p90"]),
            )
        )
    if not comparisons:
        return False, {"folds": 0, "wins": 0}
    wins = sum(
        model_median <= random_median and model_p90 <= random_p90
        for model_median, model_p90, random_median, random_p90 in comparisons
    )
    required = max(1, math.ceil(0.75 * len(comparisons)))
    model_median = float(np.median([item[0] for item in comparisons]))
    model_p90 = float(np.median([item[1] for item in comparisons]))
    random_median = float(np.median([item[2] for item in comparisons]))
    random_p90 = float(np.median([item[3] for item in comparisons]))
    summary = {
        "folds": len(comparisons),
        "wins": wins,
        "required_wins": required,
        "model_median_regret": model_median,
        "model_p90_regret": model_p90,
        "random_median_regret": random_median,
        "random_p90_regret": random_p90,
    }
    return (
        wins >= required
        and model_median <= random_median
        and model_p90 <= random_p90,
        summary,
    )


def _balanced_training_rows(
    observations: Sequence[Observation[object]], *, seed: int, per_group: int = 512
) -> list[Observation[object]]:
    groups: dict[tuple[str, str], list[Observation[object]]] = defaultdict(list)
    for item in observations:
        groups[(item.context_id, item.strategy)].append(item)
    rng = random.Random(seed)
    result = []
    for key in sorted(groups):
        rows = groups[key]
        if len(rows) > per_group:
            rows = rng.sample(rows, per_group)
        result.extend(rows)
    return result


def _aggregate_config_replicates(
    observations: Sequence[Observation[object]],
) -> list[Observation[object]]:
    """Give each context/config one robust training target.

    Separate portfolio sessions can legitimately benchmark the same candidate.
    Those repeats estimate timing noise, but feeding every repeat to a cost
    model would weight frequently rediscovered configurations more heavily.
    Keep a deterministic representative and replace its latency with the
    median across successful repeats.  Raw observations remain untouched for
    strategy-efficiency and timing-convergence studies.
    """

    groups: dict[tuple[str, str], list[Observation[object]]] = defaultdict(list)
    for item in observations:
        groups[(item.context_id, item.config_id)].append(item)
    result: list[Observation[object]] = []
    for (context_id, config_id), rows in sorted(groups.items()):
        if len(rows) == 1:
            result.append(rows[0])
            continue
        successful = [
            item
            for item in rows
            if item.successful and item.outcome.median_ms is not None
        ]
        candidates = successful or rows
        if successful:
            target = float(
                np.median([float(item.outcome.median_ms) for item in successful])
            )
            representative = min(
                successful,
                key=lambda item: (
                    abs(float(item.outcome.median_ms) - target),
                    item.observation_id,
                ),
            )
            timings = [
                value for item in successful for value in item.outcome.timings_ms
            ]
            compile_values = [
                float(item.outcome.compile_ms)
                for item in successful
                if item.outcome.compile_ms is not None
            ]
            outcome = replace(
                representative.outcome,
                median_ms=target,
                timings_ms=timings,
                compile_ms=(
                    None
                    if not compile_values
                    else float(np.median(compile_values))
                ),
            )
        else:
            statuses = Counter(item.outcome.status for item in rows)
            selected_status = min(statuses, key=lambda status: (-statuses[status], status))
            representative = min(
                (item for item in candidates if item.outcome.status == selected_status),
                key=lambda item: item.observation_id,
            )
            outcome = representative.outcome
        constituent_ids = sorted(item.observation_id for item in rows)
        aggregate_id = "aggregate-" + hashlib.sha256(
            canonical_json(
                {
                    "context_id": context_id,
                    "config_id": config_id,
                    "observation_ids": constituent_ids,
                }
            ).encode()
        ).hexdigest()[:24]
        result.append(
            replace(
                representative,
                observation_id=aggregate_id,
                session_id="offline-aggregate",
                outcome=outcome,
                elapsed_s=float(np.median([item.elapsed_s for item in rows])),
                metadata={
                    **representative.metadata,
                    "offline_config_replicates": len(rows),
                    "offline_successful_replicates": len(successful),
                    "offline_observation_ids": constituent_ids,
                },
            )
        )
    return result


@dataclass(frozen=True, slots=True)
class PretrainedFamilyModels:
    family: str
    kernel_revision: int
    cost_model: NormalizedCostModel
    ranking_model: ContextRankingModel
    feasibility_model: GradientBoostedFeasibilityModel
    rules: ConditionalRuleSet
    deployment: Mapping[str, object]
    artifact_id: str
    model_scope: str = "cross_device"


def load_pretrained_family(
    root: Path | str,
    family: str,
    kernel_revision: int,
    *,
    device_family: str | None = None,
) -> PretrainedFamilyModels:
    root = Path(root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != PRETRAINED_SCHEMA_VERSION
        or manifest.get("type") != "rtx_pretrained_autotune_bundle"
    ):
        raise ValueError("unsupported pretrained autotune artifact")
    if manifest.get("trainer_revision") != PRETRAINED_TRAINER_REVISION:
        raise ValueError(
            "pretrained autotune artifact uses a stale trainer revision; "
            "retrain it before runtime deployment"
        )
    key = f"{family}@{kernel_revision}"
    try:
        entry = manifest["families"][key]
    except KeyError as exc:
        raise KeyError(f"pretrained artifact has no {key}") from exc
    for relative, expected in entry.get("files_sha256", {}).items():
        path = root / relative
        if not path.is_file() or _file_sha256(path) != expected:
            raise RuntimeError(f"pretrained artifact file failed integrity check: {path}")
    model_entry = entry
    model_scope = "cross_device"
    if device_family is not None:
        device_entry = entry.get("device_models", {}).get(device_family)
        if (
            device_entry is not None
            and device_entry.get("deployment", {}).get("selected_cost_head", "none")
            != "none"
        ):
            model_entry = device_entry
            model_scope = f"device:{device_family}"
    deployment = dict(entry.get("deployment", {}))
    deployment.update(model_entry.get("deployment", {}))
    return PretrainedFamilyModels(
        family,
        kernel_revision,
        NormalizedCostModel.load(root / model_entry["cost_model"]),
        ContextRankingModel.load(root / model_entry["ranking_model"]),
        GradientBoostedFeasibilityModel.load(root / entry["feasibility_model"]),
        ConditionalRuleSet.load(root / entry["rules"]),
        deployment,
        str(manifest.get("artifact_id", "unknown")),
        model_scope,
    )


def evaluate_pretrained_bundle(
    root: Path | str,
    paths: Sequence[Path | str],
    *,
    campaign: str | Sequence[str] | None = None,
    allow_source_overlap: bool = False,
) -> dict[str, object]:
    """Evaluate an immutable pretrained bundle on disjoint copied observations.

    This is deliberately separate from training and never changes deployment
    gates in the artifact.  The report records both input identities so a
    prospective study cannot silently become an in-sample score.
    """

    root = Path(root).expanduser()
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("schema_version") != PRETRAINED_SCHEMA_VERSION
        or manifest.get("type") != "rtx_pretrained_autotune_bundle"
    ):
        raise ValueError("unsupported pretrained autotune artifact")
    if manifest.get("trainer_revision") != PRETRAINED_TRAINER_REVISION:
        raise ValueError(
            "pretrained autotune artifact uses a stale trainer revision; "
            "retrain it before held-out evaluation"
        )
    observations, evaluation_input = load_offline_observations(
        paths, campaign=campaign
    )
    if not observations:
        raise ValueError("no held-out autotuning observations were found")
    training_input = manifest.get("input", {})
    training_sources = {
        str(value) for value in training_input.get("source_files", [])
    } if isinstance(training_input, Mapping) else set()
    evaluation_sources = {
        str(value) for value in evaluation_input.get("source_files", [])
    }
    overlapping_sources = sorted(training_sources & evaluation_sources)
    training_identities = {
        str(value)
        for value in training_input.get("observation_identity_hashes", [])
    } if isinstance(training_input, Mapping) else set()
    evaluation_identities = {
        str(value)
        for value in evaluation_input.get("observation_identity_hashes", [])
    }
    overlapping_observations = training_identities & evaluation_identities
    same_dataset = (
        isinstance(training_input, Mapping)
        and training_input.get("dataset_sha256")
        == evaluation_input.get("dataset_sha256")
    )
    if (
        overlapping_sources or overlapping_observations or same_dataset
    ) and not allow_source_overlap:
        detail = (
            "shared observations"
            if overlapping_observations
            else "identical dataset digest"
            if same_dataset
            else "shared source files"
        )
        raise ValueError(
            f"held-out evaluation overlaps training data ({detail}); "
            "pass allow_source_overlap=True only for an explicitly in-sample diagnostic"
        )

    grouped: dict[tuple[str, int], list[Observation[object]]] = defaultdict(list)
    for item in observations:
        grouped[(item.family, item.kernel_revision)].append(item)
    results: dict[str, object] = {}
    artifact_families = manifest.get("families", {})
    if not isinstance(artifact_families, Mapping):
        raise ValueError("pretrained artifact has no family catalog")
    for key, entry in sorted(artifact_families.items()):
        if not isinstance(entry, Mapping):
            continue
        family = str(entry.get("family", ""))
        revision = int(entry.get("kernel_revision", -1))
        rows = grouped.get((family, revision), [])
        if not rows:
            results[str(key)] = {"rows": 0, "contexts": 0, "devices": {}}
            continue
        cross_device = load_pretrained_family(root, family, revision)
        by_device: dict[str, list[Observation[object]]] = defaultdict(list)
        for item in rows:
            by_device[_device_key(item)].append(item)
        device_results: dict[str, object] = {}
        for device, device_rows in sorted(by_device.items()):
            exact = load_pretrained_family(
                root, family, revision, device_family=device
            )
            device_results[device] = {
                "rows": len(device_rows),
                "contexts": len({item.context_id for item in device_rows}),
                "cross_device": {
                    "latency": evaluate_latency_model(
                        cross_device.cost_model, device_rows
                    ),
                    "ranking": evaluate_latency_model(
                        cross_device.ranking_model, device_rows
                    ),
                    "deployment": dict(cross_device.deployment),
                },
                "selected_exact_device_scope": exact.model_scope,
                "exact_device": {
                    "latency": evaluate_latency_model(exact.cost_model, device_rows),
                    "ranking": evaluate_latency_model(exact.ranking_model, device_rows),
                    "deployment": dict(exact.deployment),
                },
                "feasibility": evaluate_feasibility_model(
                    cross_device.feasibility_model, device_rows
                ),
            }
        results[str(key)] = {
            "rows": len(rows),
            "contexts": len({item.context_id for item in rows}),
            "cross_device": {
                "latency": evaluate_latency_model(cross_device.cost_model, rows),
                "ranking": evaluate_latency_model(cross_device.ranking_model, rows),
                "feasibility": evaluate_feasibility_model(
                    cross_device.feasibility_model, rows
                ),
                "deployment": dict(cross_device.deployment),
            },
            "devices": device_results,
        }
    return {
        "schema_version": 1,
        "type": "rtx_pretrained_heldout_evaluation",
        "recorded_at": _utc_now(),
        "artifact": {
            "path": str(root.resolve()),
            "artifact_id": manifest.get("artifact_id"),
            "training_input": {
                key: value
                for key, value in training_input.items()
                if key != "observation_identity_hashes"
            } if isinstance(training_input, Mapping) else training_input,
        },
        "evaluation_input": {
            key: value
            for key, value in evaluation_input.items()
            if key != "observation_identity_hashes"
        },
        "separation": {
            "held_out": not overlapping_sources and not overlapping_observations and not same_dataset,
            "same_dataset_sha256": same_dataset,
            "overlapping_source_files": overlapping_sources,
            "overlapping_observations": len(overlapping_observations),
            "overlap_override": allow_source_overlap,
        },
        "families": results,
    }


def train_pretrained_bundle(
    paths: Sequence[Path | str],
    output: Path | str,
    *,
    seed: int = 0,
    n_estimators: int = 28,
    ensembles: int = 3,
    max_depth: int = 3,
    min_leaf: int = 4,
    max_features: int = 96,
    min_rule_support: int = 12,
    max_rules: int = 256,
    validate_devices: bool = True,
    campaign: str | Sequence[str] | None = None,
) -> dict[str, object]:
    observations, load_report = load_offline_observations(paths, campaign=campaign)
    if not observations:
        raise ValueError("no autotuning observations were found")
    destination = Path(output).expanduser()
    destination.mkdir(parents=True, exist_ok=True)
    grouped: dict[tuple[str, int], list[Observation[object]]] = defaultdict(list)
    for item in observations:
        grouped[(item.family, item.kernel_revision)].append(item)
    families: dict[str, object] = {}
    for family_index, ((family, revision), raw_rows) in enumerate(sorted(grouped.items())):
        latency_raw_rows = [item for item in raw_rows if _usable_for_latency(item)]
        rows = _aggregate_config_replicates(latency_raw_rows)
        family_seed = seed ^ (family_index + 1) * 104729
        balanced = _balanced_training_rows(rows, seed=family_seed)
        feasibility_balanced = _balanced_training_rows(
            raw_rows, seed=family_seed ^ 0xFEA5
        )
        cost = NormalizedCostModel(
            GradientBoostedCostModel(
                n_estimators=n_estimators,
                ensembles=ensembles,
                max_depth=max_depth,
                min_leaf=min_leaf,
                max_features=max_features,
                seed=family_seed,
            )
        )
        cost.fit(balanced)
        ranking = ContextRankingModel(
            GradientBoostedCostModel(
                n_estimators=n_estimators,
                ensembles=ensembles,
                max_depth=max_depth,
                min_leaf=min_leaf,
                max_features=max_features,
                seed=family_seed ^ 0xA11CE,
            )
        )
        ranking.fit(balanced)
        feasibility = GradientBoostedFeasibilityModel(
            n_estimators=max(16, n_estimators // 2),
            ensembles=ensembles,
            max_depth=max_depth,
            min_leaf=max(2, min_leaf // 2),
            max_features=max_features,
            positive_statuses=("ok",),
            negative_statuses=_ALL_FAILURES,
            negative_fraction=0.2,
            seed=family_seed ^ 0xFEA5,
        )
        feasibility.fit(feasibility_balanced)
        rules = extract_conditional_rules(
            rows,
            min_support=min_rule_support,
            max_rules=max_rules,
            seed=family_seed ^ 0xC0DE,
        )
        stem = f"{family}-r{revision}"
        cost_path = Path("models") / f"{stem}-cost.json"
        ranking_path = Path("models") / f"{stem}-ranking.json"
        feasibility_path = Path("models") / f"{stem}-feasibility.json"
        rules_path = Path("rules") / f"{stem}.json"
        cost.save(destination / cost_path)
        ranking.save(destination / ranking_path)
        feasibility.save(destination / feasibility_path)
        rules.save(destination / rules_path)
        devices = sorted({_device_key(item) for item in rows})
        model_paths = [cost_path, ranking_path, feasibility_path, rules_path]
        validation = {}
        if validate_devices and len(devices) >= 2:
            for device_index, device in enumerate(devices):
                train = [item for item in balanced if _device_key(item) != device]
                test = [item for item in rows if _device_key(item) == device]
                fold = NormalizedCostModel(
                    GradientBoostedCostModel(
                        n_estimators=max(12, n_estimators // 2),
                        ensembles=max(2, ensembles - 1),
                        max_depth=max_depth,
                        min_leaf=min_leaf,
                        max_features=max_features,
                        seed=family_seed ^ (device_index + 1) * 65537,
                    )
                )
                fold.fit(train)
                fold_ranking = ContextRankingModel(
                    GradientBoostedCostModel(
                        n_estimators=max(12, n_estimators // 2),
                        ensembles=max(2, ensembles - 1),
                        max_depth=max_depth,
                        min_leaf=min_leaf,
                        max_features=max_features,
                        seed=family_seed ^ (device_index + 1) * 99991,
                    )
                )
                fold_ranking.fit(train)
                fold_feasibility = GradientBoostedFeasibilityModel(
                    n_estimators=max(12, n_estimators // 2),
                    ensembles=max(2, ensembles - 1),
                    max_depth=max_depth,
                    min_leaf=max(2, min_leaf // 2),
                    max_features=max_features,
                    positive_statuses=("ok",),
                    negative_statuses=_ALL_FAILURES,
                    negative_fraction=0.2,
                    seed=family_seed ^ (device_index + 1) * 131071,
                )
                fold_feasibility.fit(train)
                validation[device] = {
                    "latency": evaluate_latency_model(fold, test),
                    "ranking": evaluate_latency_model(fold_ranking, test),
                    "feasibility": evaluate_feasibility_model(
                        fold_feasibility, test
                    ),
                }

        # A single portable head can fail when SKU behavior differs sharply.
        # Exact-SKU heads are validated on unseen contexts from that SKU; they
        # are never used as a fallback for another device family.
        device_models = {}
        if validate_devices:
            for device_index, device in enumerate(devices):
                device_rows = [item for item in rows if _device_key(item) == device]
                context_ids = sorted({item.context_id for item in device_rows})
                shape_groups = sorted({_shape_group_key(item) for item in device_rows})
                if len(shape_groups) < 4:
                    continue
                fold_count = min(4, len(shape_groups))
                ordered_shapes = sorted(
                    shape_groups,
                    key=lambda shape: hashlib.sha256(
                        f"{family_seed}:{device}:{shape}".encode()
                    ).hexdigest(),
                )
                shape_fold = {
                    shape: index % fold_count
                    for index, shape in enumerate(ordered_shapes)
                }
                fold_metrics = []
                for fold_index in range(fold_count):
                    fold_train = [
                        item
                        for item in device_rows
                        if shape_fold[_shape_group_key(item)] != fold_index
                    ]
                    fold_test = [
                        item
                        for item in device_rows
                        if shape_fold[_shape_group_key(item)] == fold_index
                    ]
                    fold_train = _balanced_training_rows(
                        fold_train,
                        seed=family_seed ^ (device_index + 1) * 524287 ^ fold_index,
                    )
                    fold_cost = NormalizedCostModel(
                        GradientBoostedCostModel(
                            n_estimators=max(12, n_estimators // 2),
                            ensembles=max(2, ensembles - 1),
                            max_depth=max_depth,
                            min_leaf=min_leaf,
                            max_features=max_features,
                            seed=family_seed ^ (device_index + 1) * 8191 ^ fold_index,
                        )
                    )
                    fold_ranking = ContextRankingModel(
                        GradientBoostedCostModel(
                            n_estimators=max(12, n_estimators // 2),
                            ensembles=max(2, ensembles - 1),
                            max_depth=max_depth,
                            min_leaf=min_leaf,
                            max_features=max_features,
                            seed=family_seed ^ (device_index + 1) * 12289 ^ fold_index,
                        )
                    )
                    fold_cost.fit(fold_train)
                    fold_ranking.fit(fold_train)
                    fold_metrics.append(
                        {
                            "fold": fold_index,
                            "train_contexts": len(
                                {item.context_id for item in fold_train}
                            ),
                            "test_contexts": len(
                                {item.context_id for item in fold_test}
                            ),
                            "train_shapes": len(
                                {_shape_group_key(item) for item in fold_train}
                            ),
                            "test_shapes": len(
                                {_shape_group_key(item) for item in fold_test}
                            ),
                            "latency": evaluate_latency_model(fold_cost, fold_test),
                            "ranking": evaluate_latency_model(
                                fold_ranking, fold_test
                            ),
                        }
                    )
                ranking_ok, ranking_summary = _fold_head_beats_random(
                    fold_metrics, "ranking"
                )
                latency_ok, latency_summary = _fold_head_beats_random(
                    fold_metrics, "latency"
                )
                selected_device_head = (
                    "ranking" if ranking_ok else "latency" if latency_ok else "none"
                )
                device_balanced = _balanced_training_rows(
                    device_rows, seed=family_seed ^ (device_index + 1) * 1048573
                )
                device_cost = NormalizedCostModel(
                    GradientBoostedCostModel(
                        n_estimators=n_estimators,
                        ensembles=ensembles,
                        max_depth=max_depth,
                        min_leaf=min_leaf,
                        max_features=max_features,
                        seed=family_seed ^ (device_index + 1) * 4099,
                    )
                )
                device_ranking = ContextRankingModel(
                    GradientBoostedCostModel(
                        n_estimators=n_estimators,
                        ensembles=ensembles,
                        max_depth=max_depth,
                        min_leaf=min_leaf,
                        max_features=max_features,
                        seed=family_seed ^ (device_index + 1) * 6151,
                    )
                )
                device_cost.fit(device_balanced)
                device_ranking.fit(device_balanced)
                device_stem = f"{stem}-{device}"
                device_cost_path = Path("models") / f"{device_stem}-cost.json"
                device_ranking_path = Path("models") / f"{device_stem}-ranking.json"
                device_cost.save(destination / device_cost_path)
                device_ranking.save(destination / device_ranking_path)
                model_paths.extend((device_cost_path, device_ranking_path))
                device_models[device] = {
                    "rows": len(device_rows),
                    "contexts": len(context_ids),
                    "shapes": len(shape_groups),
                    "cost_model": str(device_cost_path),
                    "ranking_model": str(device_ranking_path),
                    "shape_group_folds": fold_metrics,
                    "validation_summary": {
                        "latency": latency_summary,
                        "ranking": ranking_summary,
                    },
                    "deployment": {
                        "validation_gated": True,
                        "validation_scope": "same_device_unseen_shape",
                        "catalog_replay_budget": 4,
                        "selected_cost_head": selected_device_head,
                        "pretrained_random_warmup_trials": (
                            4 if selected_device_head != "none" else None
                        ),
                    },
                }
        def head_beats_random(head: str, budget: str = "4") -> bool:
            if not validation:
                return False
            for metrics in validation.values():
                model_metrics = metrics[head]["catalog_replay_regret"][budget]
                random_metrics = metrics[head]["random_catalog_replay_regret"][budget]
                if (
                    float(model_metrics["median"])
                    > float(random_metrics["median"])
                    or float(model_metrics["p90"])
                    > float(random_metrics["p90"])
                ):
                    return False
            return True

        selected_head = (
            "ranking"
            if head_beats_random("ranking")
            else "latency"
            if head_beats_random("latency")
            else "none"
        )
        feasibility_enabled = bool(validation) and all(
            bool(metrics["feasibility"].get("evaluable"))
            and float(metrics["feasibility"].get("auc", 0.0)) >= 0.8
            for metrics in validation.values()
        )
        rules_enabled = any(rule.devices >= 2 for rule in rules.rules)
        deployment = {
            "validation_gated": True,
            "catalog_replay_budget": 4,
            "selected_cost_head": selected_head,
            "feasibility_enabled": feasibility_enabled,
            "conditional_rules_enabled": rules_enabled,
            "pretrained_random_warmup_trials": 4 if selected_head != "none" else None,
        }
        files_sha256 = {
            str(path): _file_sha256(destination / path)
            for path in model_paths
        }
        key = f"{family}@{revision}"
        families[key] = {
            "family": family,
            "kernel_revision": revision,
            "rows": len(rows),
            "raw_rows": len(raw_rows),
            "latency_raw_rows": len(latency_raw_rows),
            "latency_excluded_nonstationary": len(raw_rows) - len(latency_raw_rows),
            "feasibility_rows": len(feasibility_balanced),
            "aggregated_replicates": len(latency_raw_rows) - len(rows),
            "successful_rows": sum(item.successful for item in rows),
            "statuses": dict(Counter(item.outcome.status for item in rows)),
            "contexts": len({item.context_id for item in rows}),
            "devices": devices,
            "model_parameters": cost.parameter_count,
            "ranking_parameters": ranking.parameter_count,
            "feasibility_parameters": feasibility.parameter_count,
            "conditional_rules": len(rules.rules),
            "cost_model": str(cost_path),
            "ranking_model": str(ranking_path),
            "feasibility_model": str(feasibility_path),
            "rules": str(rules_path),
            "leave_one_device_out": validation,
            "device_models": device_models,
            "deployment": deployment,
            "files_sha256": files_sha256,
        }
    manifest = {
        "schema_version": PRETRAINED_SCHEMA_VERSION,
        "trainer_revision": PRETRAINED_TRAINER_REVISION,
        "type": "rtx_pretrained_autotune_bundle",
        "recorded_at": _utc_now(),
        "seed": seed,
        "training": {
            "n_estimators": n_estimators,
            "ensembles": ensembles,
            "max_depth": max_depth,
            "min_leaf": min_leaf,
            "max_features": max_features,
            "min_rule_support": min_rule_support,
            "max_rules": max_rules,
            "portable_feature_filter": 1,
            "latency_target": "log(latency/analytical_baseline)",
            "ranking_target": "log_latency-context_median_log_latency",
            "feasibility_positive": ["ok"],
            "feasibility_negative": list(_ALL_FAILURES),
            "feasibility_negative_fraction": 0.2,
            "latency_stationarity_policy": (
                "exclude_explicitly_nonstationary_successes"
            ),
            "feasibility_stationarity_policy": "retain_all",
        },
        "input": load_report,
        "families": families,
    }
    manifest["artifact_id"] = hashlib.sha256(
        canonical_json(
            {
                "schema_version": PRETRAINED_SCHEMA_VERSION,
                "trainer_revision": PRETRAINED_TRAINER_REVISION,
                "seed": seed,
                "dataset_sha256": load_report["dataset_sha256"],
                "training": manifest["training"],
                "files": {
                    key: value["files_sha256"]
                    for key, value in sorted(families.items())
                },
                "deployment": {
                    key: {
                        "global": value["deployment"],
                        "devices": {
                            device: device_value["deployment"]
                            for device, device_value in sorted(
                                value["device_models"].items()
                            )
                        },
                    }
                    for key, value in sorted(families.items())
                },
            }
        ).encode()
    ).hexdigest()[:24]
    _atomic_json(destination / "manifest.json", manifest)
    return manifest


__all__ = [
    "ConditionalEffectRule",
    "ConditionalRuleSet",
    "ContextRankingModel",
    "NormalizedCostModel",
    "PRETRAINED_SCHEMA_VERSION",
    "PRETRAINED_TRAINER_REVISION",
    "PretrainedFamilyModels",
    "analytical_baseline_ms",
    "evaluate_latency_model",
    "evaluate_feasibility_model",
    "evaluate_pretrained_bundle",
    "extract_conditional_rules",
    "load_offline_observations",
    "load_pretrained_family",
    "portable_features",
    "train_pretrained_bundle",
]
