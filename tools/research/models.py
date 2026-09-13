"""Extensible, CPU-only candidate registry. Every candidate predicts lower=better."""
from collections import defaultdict
from dataclasses import dataclass
import time

import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data import feature_view


@dataclass(frozen=True)
class Candidate:
    name: str
    family_heads: bool = False
    variants: tuple = (0, 1)


REGISTRY = {name: Candidate(name, family) for name, family in (
    ('ridge', False), ('additive_boost', False), ('extra_trees', True),
    ('hist_global', False), ('hist_family', True), ('xgboost', True),
    ('catboost', True), ('pairwise_xgb', True), ('small_mlp', True),
    ('roofline_residual', True),
)}


def estimator(name, variant, threads):
    if name == 'ridge':
        return make_pipeline(StandardScaler(), Ridge(alpha=(10., 100.)[variant]))
    if name == 'small_mlp':
        return make_pipeline(StandardScaler(), MLPRegressor(
            hidden_layer_sizes=((32,), (64, 32))[variant], max_iter=40,
            batch_size=256, alpha=.1, random_state=1729,
            early_stopping=False, learning_rate_init=.002))
    if name == 'extra_trees':
        return ExtraTreesRegressor(n_estimators=48, max_depth=(12, 20)[variant],
                                   min_samples_leaf=4, max_features=.25, n_jobs=threads, random_state=1729)
    if name in ('xgboost', 'pairwise_xgb'):
        from xgboost import XGBRegressor, XGBRanker
        cls = XGBRanker if name == 'pairwise_xgb' else XGBRegressor
        return cls(n_estimators=80, max_depth=(4, 7)[variant], learning_rate=.08,
                   tree_method='hist', n_jobs=threads, random_state=1729,
                   objective='rank:pairwise' if name == 'pairwise_xgb' else 'reg:squarederror')
    if name == 'catboost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=100, depth=(4, 6)[variant], learning_rate=.08,
                                 thread_count=threads, random_seed=1729, verbose=False,
                                 allow_writing_files=False)
    return HistGradientBoostingRegressor(max_iter=(60, 100)[variant],
        max_depth=1 if name == 'additive_boost' else (4, 6)[variant],
        min_samples_leaf=20, l2_regularization=1., random_state=1729,
        early_stopping=False)  # No row-random validation leakage within fit.


def labels(rows):
    grouped = defaultdict(list)
    for r in rows:
        grouped[r['context']].append(np.log(r['ms']))
    centers = {k: np.median(v) for k, v in grouped.items()}
    return np.array([np.log(r['ms']) - centers[r['context']] for r in rows])


def roofline(rows):
    # An intentionally limited memory-only analytical baseline, not a claimed
    # full compute roofline. No measured latency is used to form this feature.
    result = []
    for row in rows:
        values = [v for k, v in row['features'].items()
                  if k.startswith('derived.') and k.endswith('memory_roofline_ms') and v > 0]
        result.append(np.log(max(sum(values), 1e-6)))
    return np.array(result)


class FittedCandidate:
    def __init__(self, spec, view, variant, threads):
        self.spec, self.view, self.variant, self.threads = spec, view, variant, threads

    def fit(self, rows, *, balance_contexts=False):
        # Opt-in: old tournament/export fits remain reproducible. Equal family
        # and context mass prevents dense collection pockets dominating loss.
        from .coverage_audit import context_weights
        weights=np.asarray(context_weights(rows)) if balance_contexts else None
        self.vectorizer = DictVectorizer(sparse=False, dtype=np.float32)
        X = self.vectorizer.fit_transform([feature_view(r, self.view) for r in rows])
        y = labels(rows)
        self.models = {}
        keys = sorted({r['family'] for r in rows}) if self.spec.family_heads else ['*']
        if self.spec.name == 'roofline_residual':
            base = roofline(rows)
            grouped = defaultdict(list)
            for i, r in enumerate(rows):
                grouped[r['context']].append(base[i])
            center = {k: np.median(v) for k, v in grouped.items()}
            y -= base - np.array([center[r['context']] for r in rows])
        for key in keys:
            ids = np.array([i for i, r in enumerate(rows) if key == '*' or r['family'] == key])
            model = estimator(self.spec.name, self.variant, self.threads)
            if self.spec.name == 'pairwise_xgb':
                ids = np.array(sorted(ids, key=lambda i: rows[i]['context']))
                contexts = [rows[i]['context'] for i in ids]
                group = np.unique(contexts, return_counts=True)[1]
                # Rank pair labels are within-context ranks, higher is faster.
                target = np.empty(len(ids))
                start = 0
                for count in group:
                    part = y[ids[start:start+count]]
                    target[start:start+count] = np.argsort(np.argsort(-part))
                    start += count
                if weights is not None:
                    raise ValueError('context balancing for ranker requires query-group weights; use regression heads for this experiment')
                model.fit(X[ids], target, group=group)
            else:
                options={}
                if weights is not None:
                    import inspect
                    estimators=model.steps if hasattr(model,'steps') else [('',model)]
                    for step,fit_step in estimators:
                        if 'sample_weight' not in inspect.signature(fit_step.fit).parameters:
                            raise ValueError(f'{type(fit_step).__name__} does not support sample_weight')
                        options[(step+'__' if step else '')+'sample_weight']=weights[ids]
                model.fit(X[ids], y[ids], **options)
            self.models[key] = model
        self.feature_count = X.shape[1]
        return self

    def predict(self, rows):
        X = self.vectorizer.transform([feature_view(r, self.view) for r in rows])
        result = np.zeros(len(rows))
        for key, model in self.models.items():
            ids = np.array([i for i, r in enumerate(rows) if key == '*' or r['family'] == key], dtype=int)
            if len(ids):
                result[ids] = model.predict(X[ids])
        if self.spec.name == 'pairwise_xgb':
            result *= -1
        if self.spec.name == 'roofline_residual':
            result += roofline(rows)
        return result


class Retrieval:
    """Nearest prior configuration/workload observations; unseen-config fallback.

    Matching config IDs use all matching rows. Novel configs consult a fixed
    bounded family bank; no label-dependent candidate-pool construction.
    """
    keys = ('context.workload.m', 'context.workload.n', 'context.workload.k',
            'context.workload.h', 'context.device.properties.multiprocessor_count',
            'context.device.properties.l2_cache_size',
            'context.device.sku.theoretical_memory_bandwidth_gbps')

    @classmethod
    def vector(cls, r):
        return np.log1p([max(0, r['features'].get(k, 0)) for k in cls.keys])

    def fit(self, rows):
        self.X, self.y = np.array([self.vector(r) for r in rows]), labels(rows)
        self.bank, self.exact = defaultdict(list), defaultdict(list)
        for i, r in enumerate(rows):
            contract = (r['family'], r['revision'])
            self.bank[contract].append(i)
            self.exact[(*contract, r['config_id'])].append(i)
        rng = np.random.default_rng(1729)
        self.bank = {k: rng.choice(v, min(len(v), 256), replace=False) for k, v in self.bank.items()}
        return self

    def predict(self, rows):
        result = []
        for r in rows:
            contract = (r['family'], r['revision'])
            ids = np.array(self.exact.get((*contract, r['config_id']), self.bank[contract]))
            distances = ((self.X[ids] - self.vector(r)) ** 2).mean(1)
            near = np.argsort(distances)[:3]
            result.append(np.average(self.y[ids[near]], weights=1/(distances[near]+1e-4)))
        return np.array(result)
