"""Small dependency-free gradient-boosted latency model for tabular schedules."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
from typing import Iterable, Mapping, Protocol, Sequence

import numpy as np

from .core import FeatureMap, Observation


class CostModel(Protocol):
    def fit(self, observations: Sequence[Observation[object]]) -> None: ...

    def predict(self, features: Sequence[FeatureMap]) -> tuple[np.ndarray, np.ndarray]: ...


class SparseFeatureVectorizer:
    """Stable union-schema vectorizer; feature names are persisted with the model."""

    def __init__(self) -> None:
        self.names: tuple[str, ...] = ()
        self.index: dict[str, int] = {}

    def fit(self, rows: Iterable[FeatureMap]) -> None:
        self.names = tuple(sorted({key for row in rows for key in row}))
        self.index = {name: index for index, name in enumerate(self.names)}

    def transform(self, rows: Sequence[FeatureMap]) -> np.ndarray:
        result = np.zeros((len(rows), len(self.names)), dtype=np.float32)
        for row_index, row in enumerate(rows):
            for name, value in row.items():
                column = self.index.get(name)
                if column is not None:
                    result[row_index, column] = float(value)
        return result

    def state_dict(self) -> dict[str, object]:
        return {"names": list(self.names)}

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.names = tuple(str(name) for name in state["names"])  # type: ignore[union-attr]
        self.index = {name: index for index, name in enumerate(self.names)}


@dataclass(slots=True)
class _TreeNode:
    value: float
    feature: int = -1
    threshold: float = 0.0
    left: int = -1
    right: int = -1

    def as_dict(self) -> dict[str, object]:
        return {
            "value": self.value,
            "feature": self.feature,
            "threshold": self.threshold,
            "left": self.left,
            "right": self.right,
        }


class _RegressionTree:
    def __init__(
        self,
        *,
        max_depth: int,
        min_leaf: int,
        max_thresholds: int,
        max_features: int,
        seed: int,
    ) -> None:
        self.max_depth = max_depth
        self.min_leaf = min_leaf
        self.max_thresholds = max_thresholds
        self.max_features = max_features
        self.rng = random.Random(seed)
        self.nodes: list[_TreeNode] = []

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.nodes = []
        self._build(x, y, np.arange(x.shape[0]), 0)

    def _build(
        self,
        x: np.ndarray,
        y: np.ndarray,
        rows: np.ndarray,
        depth: int,
    ) -> int:
        node_index = len(self.nodes)
        self.nodes.append(_TreeNode(float(np.mean(y[rows]))))
        if depth >= self.max_depth or rows.size < self.min_leaf * 2:
            return node_index

        feature_count = x.shape[1]
        if feature_count == 0:
            return node_index
        candidates = list(range(feature_count))
        if len(candidates) > self.max_features:
            candidates = self.rng.sample(candidates, self.max_features)

        best: tuple[float, int, float, np.ndarray] | None = None
        for feature in candidates:
            values = x[rows, feature]
            unique = np.unique(values)
            if unique.size <= 1:
                continue
            if unique.size > self.max_thresholds + 1:
                quantiles = np.linspace(0.0, 1.0, self.max_thresholds + 2)[1:-1]
                thresholds = np.unique(np.quantile(values, quantiles))
            else:
                thresholds = (unique[:-1] + unique[1:]) * 0.5
            for threshold in thresholds:
                left_mask = values <= threshold
                left_count = int(np.count_nonzero(left_mask))
                right_count = rows.size - left_count
                if left_count < self.min_leaf or right_count < self.min_leaf:
                    continue
                left_y = y[rows[left_mask]]
                right_y = y[rows[~left_mask]]
                loss = float(np.var(left_y) * left_count + np.var(right_y) * right_count)
                if best is None or loss < best[0]:
                    best = (loss, feature, float(threshold), left_mask)

        if best is None:
            return node_index
        _loss, feature, threshold, left_mask = best
        left = self._build(x, y, rows[left_mask], depth + 1)
        right = self._build(x, y, rows[~left_mask], depth + 1)
        self.nodes[node_index].feature = feature
        self.nodes[node_index].threshold = threshold
        self.nodes[node_index].left = left
        self.nodes[node_index].right = right
        return node_index

    def predict(self, x: np.ndarray) -> np.ndarray:
        output = np.empty(x.shape[0], dtype=np.float64)
        for row_index in range(x.shape[0]):
            node_index = 0
            while self.nodes[node_index].feature >= 0:
                node = self.nodes[node_index]
                node_index = (
                    node.left
                    if x[row_index, node.feature] <= node.threshold
                    else node.right
                )
            output[row_index] = self.nodes[node_index].value
        return output

    def state_dict(self) -> dict[str, object]:
        return {"nodes": [node.as_dict() for node in self.nodes]}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, object]) -> "_RegressionTree":
        tree = cls(max_depth=1, min_leaf=1, max_thresholds=1, max_features=1, seed=0)
        tree.nodes = [_TreeNode(**node) for node in state["nodes"]]  # type: ignore[arg-type,union-attr]
        return tree


@dataclass(slots=True)
class _BoostedEnsemble:
    base: float
    trees: list[_RegressionTree]


class GradientBoostedCostModel:
    """Bagged gradient-boosted CART regressors predicting log latency.

    Defaults are a few thousand tree nodes, far below one million parameters.
    Bagging supplies an empirical uncertainty used by exploration-aware ranking.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 40,
        max_depth: int = 3,
        learning_rate: float = 0.08,
        min_leaf: int = 4,
        ensembles: int = 4,
        row_subsample: float = 0.8,
        max_training_rows: int = 50_000,
        max_thresholds: int = 12,
        max_features: int = 96,
        seed: int = 0,
    ) -> None:
        if n_estimators <= 0 or ensembles <= 0 or not 0 < learning_rate <= 1:
            raise ValueError("invalid gradient-boosting parameters")
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_leaf = min_leaf
        self.ensembles = ensembles
        self.row_subsample = row_subsample
        self.max_training_rows = max_training_rows
        self.max_thresholds = max_thresholds
        self.max_features = max_features
        self.seed = seed
        self.vectorizer = SparseFeatureVectorizer()
        self.models: list[_BoostedEnsemble] = []
        self.residual_std = 1.0

    @property
    def fitted(self) -> bool:
        return bool(self.models)

    @property
    def parameter_count(self) -> int:
        return sum(len(tree.nodes) * 5 + 1 for model in self.models for tree in model.trees)

    def fit(self, observations: Sequence[Observation[object]]) -> None:
        usable = [item for item in observations if item.successful and item.score > 0]
        if len(usable) < max(4, self.min_leaf):
            self.models = []
            return
        rng = np.random.default_rng(self.seed)
        if len(usable) > self.max_training_rows:
            indices = rng.choice(len(usable), self.max_training_rows, replace=False)
            usable = [usable[int(index)] for index in indices]
        feature_rows = [item.features for item in usable]
        self.vectorizer.fit(feature_rows)
        x = self.vectorizer.transform(feature_rows)
        y = np.log(np.asarray([item.score for item in usable], dtype=np.float64))
        self.models = []
        fitted_training: list[np.ndarray] = []
        for ensemble_index in range(self.ensembles):
            ensemble_rng = np.random.default_rng(self.seed + ensemble_index * 104729)
            sample_count = max(self.min_leaf * 2, int(len(y) * self.row_subsample))
            sample_rows = ensemble_rng.choice(len(y), sample_count, replace=True)
            train_x = x[sample_rows]
            train_y = y[sample_rows]
            base = float(np.mean(train_y))
            prediction = np.full(train_y.shape, base, dtype=np.float64)
            trees: list[_RegressionTree] = []
            for tree_index in range(self.n_estimators):
                residual = train_y - prediction
                tree = _RegressionTree(
                    max_depth=self.max_depth,
                    min_leaf=self.min_leaf,
                    max_thresholds=self.max_thresholds,
                    max_features=min(self.max_features, max(1, x.shape[1])),
                    seed=self.seed + ensemble_index * 1009 + tree_index,
                )
                tree.fit(train_x, residual)
                prediction += self.learning_rate * tree.predict(train_x)
                trees.append(tree)
            model = _BoostedEnsemble(base, trees)
            self.models.append(model)
            fitted_training.append(self._predict_log_one(model, x))
        mean_training = np.mean(np.stack(fitted_training), axis=0)
        self.residual_std = max(1e-6, float(np.std(y - mean_training)))

    def _predict_log_one(self, model: _BoostedEnsemble, x: np.ndarray) -> np.ndarray:
        result = np.full(x.shape[0], model.base, dtype=np.float64)
        for tree in model.trees:
            result += self.learning_rate * tree.predict(x)
        return result

    def predict(self, features: Sequence[FeatureMap]) -> tuple[np.ndarray, np.ndarray]:
        if not self.models:
            raise RuntimeError("cost model has not been fitted")
        x = self.vectorizer.transform(features)
        predictions = np.stack([self._predict_log_one(model, x) for model in self.models])
        mean_log = np.mean(predictions, axis=0)
        std_log = np.sqrt(np.var(predictions, axis=0) + self.residual_std**2)
        mean = np.exp(mean_log)
        # Early cross-device datasets can have very noisy log residuals. Keep
        # acquisition finite while retaining a broad uncertainty signal.
        log_variance = np.minimum(std_log**2, 25.0)
        std = mean * np.sqrt(np.maximum(0.0, np.exp(log_variance) - 1.0))
        return mean, std

    def state_dict(self) -> dict[str, object]:
        return {
            "schema_version": 1,
            "type": "gradient_boosted_cost_model",
            "params": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "learning_rate": self.learning_rate,
                "min_leaf": self.min_leaf,
                "ensembles": self.ensembles,
                "row_subsample": self.row_subsample,
                "max_training_rows": self.max_training_rows,
                "max_thresholds": self.max_thresholds,
                "max_features": self.max_features,
                "seed": self.seed,
            },
            "vectorizer": self.vectorizer.state_dict(),
            "residual_std": self.residual_std,
            "models": [
                {"base": model.base, "trees": [tree.state_dict() for tree in model.trees]}
                for model in self.models
            ],
        }

    def save(self, path: Path | str) -> None:
        Path(path).write_text(json.dumps(self.state_dict(), sort_keys=True), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "GradientBoostedCostModel":
        state = json.loads(Path(path).read_text(encoding="utf-8"))
        model = cls(**state["params"])
        model.vectorizer.load_state_dict(state["vectorizer"])
        model.residual_std = float(state["residual_std"])
        model.models = [
            _BoostedEnsemble(
                float(item["base"]),
                [_RegressionTree.from_state_dict(tree) for tree in item["trees"]],
            )
            for item in state["models"]
        ]
        return model


class GradientBoostedFeasibilityModel:
    """Small bagged classifier for whether generated device IR will compile.

    Only explicit ``ok`` and ``compile_error`` outcomes are labels. Correctness
    and runtime failures are intentionally excluded because they do not provide
    a reliable compile-boundary label across harness implementations.
    """

    def __init__(
        self,
        *,
        n_estimators: int = 16,
        max_depth: int = 3,
        learning_rate: float = 0.1,
        min_leaf: int = 3,
        ensembles: int = 3,
        row_subsample: float = 0.85,
        max_training_rows: int = 50_000,
        max_thresholds: int = 10,
        max_features: int = 64,
        seed: int = 0,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.min_leaf = min_leaf
        self.ensembles = ensembles
        self.row_subsample = row_subsample
        self.max_training_rows = max_training_rows
        self.max_thresholds = max_thresholds
        self.max_features = max_features
        self.seed = seed
        self.vectorizer = SparseFeatureVectorizer()
        self.models: list[_BoostedEnsemble] = []

    @property
    def fitted(self) -> bool:
        return bool(self.models)

    @property
    def parameter_count(self) -> int:
        return sum(
            len(tree.nodes) * 5 + 1 for model in self.models for tree in model.trees
        )

    @staticmethod
    def labeled_count(observations: Sequence[Observation[object]]) -> int:
        return sum(
            item.outcome.status in ("ok", "compile_error") for item in observations
        )

    def fit(self, observations: Sequence[Observation[object]]) -> None:
        usable = [
            item
            for item in observations
            if item.outcome.status in ("ok", "compile_error")
        ]
        labels = np.asarray(
            [1.0 if item.outcome.status == "ok" else 0.0 for item in usable],
            dtype=np.float64,
        )
        if (
            len(usable) < max(8, self.min_leaf * 2)
            or labels.size == 0
            or float(np.min(labels)) == float(np.max(labels))
        ):
            self.models = []
            return
        rng = np.random.default_rng(self.seed)
        if len(usable) > self.max_training_rows:
            indices = rng.choice(len(usable), self.max_training_rows, replace=False)
            usable = [usable[int(index)] for index in indices]
            labels = labels[indices]
        self.vectorizer.fit(item.features for item in usable)
        x = self.vectorizer.transform([item.features for item in usable])
        self.models = []
        for ensemble_index in range(self.ensembles):
            ensemble_rng = np.random.default_rng(self.seed + ensemble_index * 130363)
            sample_count = max(self.min_leaf * 2, int(len(labels) * self.row_subsample))
            sample_rows = ensemble_rng.choice(len(labels), sample_count, replace=True)
            train_x = x[sample_rows]
            train_y = labels[sample_rows]
            base = float(np.mean(train_y))
            prediction = np.full(train_y.shape, base, dtype=np.float64)
            trees: list[_RegressionTree] = []
            for tree_index in range(self.n_estimators):
                residual = train_y - prediction
                tree = _RegressionTree(
                    max_depth=self.max_depth,
                    min_leaf=self.min_leaf,
                    max_thresholds=self.max_thresholds,
                    max_features=min(self.max_features, max(1, x.shape[1])),
                    seed=self.seed + ensemble_index * 2017 + tree_index,
                )
                tree.fit(train_x, residual)
                prediction += self.learning_rate * tree.predict(train_x)
                trees.append(tree)
            self.models.append(_BoostedEnsemble(base, trees))

    def _predict_one(self, model: _BoostedEnsemble, x: np.ndarray) -> np.ndarray:
        result = np.full(x.shape[0], model.base, dtype=np.float64)
        for tree in model.trees:
            result += self.learning_rate * tree.predict(x)
        return np.clip(result, 0.0, 1.0)

    def predict(self, features: Sequence[FeatureMap]) -> tuple[np.ndarray, np.ndarray]:
        if not self.models:
            raise RuntimeError("feasibility model has not been fitted")
        x = self.vectorizer.transform(features)
        predictions = np.stack([self._predict_one(model, x) for model in self.models])
        probability = np.clip(np.mean(predictions, axis=0), 0.0, 1.0)
        uncertainty = np.sqrt(
            np.var(predictions, axis=0)
            + probability * (1.0 - probability) / max(1, len(self.models))
        )
        return probability, uncertainty


__all__ = [
    "CostModel",
    "GradientBoostedCostModel",
    "GradientBoostedFeasibilityModel",
    "SparseFeatureVectorizer",
]
