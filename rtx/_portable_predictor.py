"""Data-only research predictors. NumPy only: no sklearn, pickle, Torch or CUDA.

Scores order candidates, never authorize kernels or supply unmeasured winners.
"""
import gzip
import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

SCHEMA = 1
FEATURE_REVISION = 'tournament_interactions_v1'
SEMANTIC_TAGS = ('precision', 'phase', 'scaling', 'scale_policy', 'operand_state', 'activation')
POLICY_TAGS = (*SEMANTIC_TAGS, 'pass', 'silu_math', 'outer_scale_scope')


def contract(family, revision, features):
    policy = {k: v for k, v in features.items()
              if k.startswith('context.regime=')
              or (k.startswith('context.workload.') and '=' in k)
              or any(k.startswith('context.tags.'+tag+'=') for tag in POLICY_TAGS)}
    return dict(family=family, revision=revision, policy=policy)


def feature_view(row):
    result = {}
    for k, v in row['features'].items():
        keep = k.startswith(('config.', 'context.workload.', 'context.regime=', 'derived.',
                             'context.device.properties.', 'context.device.architecture.', 'context.device.sku.'))
        keep |= any(k.startswith('context.tags.'+tag+'=') for tag in SEMANTIC_TAGS)
        if not keep or not math.isfinite(v) or any(t in k.lower() for t in (
            'compiled', 'spill', 'telemetry', 'measured', 'source', 'name=', 'sku_family=', 'uuid',
        )):
            continue
        result[k] = float(v) if '=' in k else math.copysign(math.log1p(abs(v)), v)
    result['family='+row['family']] = 1.
    result['revision='+str(row['revision'])] = 1.
    return result


def memory_baseline(rows):
    return np.array([math.log(max(sum(v for k, v in r['features'].items()
        if k.startswith('derived.') and k.endswith('memory_roofline_ms') and v > 0), 1e-6)) for r in rows])


def _array(value, dtype=np.float64):
    result = np.asarray(value, dtype=dtype)
    if not np.isfinite(result).all():
        raise ValueError('nonfinite predictor parameters')
    result.flags.writeable = False
    return result


class PortablePredictor:
    def __init__(self, document, *, research=False):
        if document.get('schema') != SCHEMA or document.get('feature_revision') != FEATURE_REVISION:
            raise ValueError('unsupported predictor schema/features')
        if not research:
            raise ValueError('predictor requires explicit research opt-in')
        self.digest = hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()
        self.document = document
        self.feature_names = document['feature_names']
        if (not self.feature_names or len(self.feature_names) > 65536
                or any(not isinstance(n, str) for n in self.feature_names)
                or len(set(self.feature_names)) != len(self.feature_names)):
            raise ValueError('invalid predictor feature names')
        self.positions = {name: i for i, name in enumerate(self.feature_names)}
        self.models = {}
        n = len(self.feature_names)
        for head, families in document['models'].items():
            if head not in ('mlp', 'residual', 'trees'):
                raise ValueError('unknown predictor head')
            self.models[head] = {}
            for family, model in families.items():
                if head == 'mlp':
                    mean, scale = _array(model['mean']), _array(model['scale'])
                    if mean.shape != (n,) or scale.shape != (n,) or (scale <= 0).any():
                        raise ValueError('invalid scaler')
                    layers, width = [], n
                    for layer in model['layers']:
                        w, b = _array(layer['weight'], np.float32), _array(layer['bias'], np.float32)
                        if w.ndim != 2 or w.shape[0] != width or b.shape != (w.shape[1],):
                            raise ValueError('invalid MLP layer')
                        layers.append((w, b))
                        width = w.shape[1]
                    if not layers or width != 1:
                        raise ValueError('MLP must output one score')
                    self.models[head][family] = (mean, scale, layers)
                else:
                    base = float(model['baseline'])
                    if not math.isfinite(base):
                        raise ValueError('invalid tree baseline')
                    trees = []
                    for tree in model['trees']:
                        a = _array(tree)
                        # value, feature, threshold, left, right, leaf, missing_left.
                        if a.ndim != 2 or a.shape[1] != 7 or not len(a):
                            raise ValueError('invalid tree')
                        for i, node in enumerate(a):
                            if node[5] not in (0, 1) or node[6] not in (0, 1):
                                raise ValueError('invalid tree flags')
                            if not node[5]:
                                if node[1] != int(node[1]) or not 0 <= node[1] < n:
                                    raise ValueError('invalid tree feature')
                                if any(child != int(child) or not i < child < len(a) for child in node[3:5]):
                                    raise ValueError('cyclic or invalid tree children')
                        trees.append(a)
                    self.models[head][family] = (base, trees)

    def predict(self, rows, head):
        X = np.zeros((len(rows), len(self.feature_names)), dtype=np.float32)
        for i, row in enumerate(rows):
            for key, value in feature_view(row).items():
                index = self.positions.get(key)
                if index is not None:
                    X[i, index] = value
        result = np.empty(len(rows), dtype=np.float64)
        for family in {r['family'] for r in rows}:
            ids = np.array([i for i, r in enumerate(rows) if r['family'] == family])
            model = self.models[head].get(family)
            if model is None:
                raise ValueError('unsupported predictor family')
            if head == 'mlp':
                mean, scale, layers = model
                a = X[ids].copy()
                a -= mean
                a /= scale
                for j, (w, b) in enumerate(layers):
                    a = a @ w + b
                    if j != len(layers)-1:
                        np.maximum(a, 0, out=a)
                result[ids] = a.ravel()
            else:
                baseline, trees = model
                data = X[ids].astype(np.float64)
                score = np.full(len(ids), baseline)
                for tree in trees:
                    nodes = np.zeros(len(ids), dtype=np.int64)
                    active = np.arange(len(ids))
                    while len(active):
                        current = tree[nodes[active]]
                        leaf = current[:, 5].astype(bool)
                        score[active[leaf]] += current[leaf, 0]
                        active, current = active[~leaf], current[~leaf]
                        if len(active):
                            values = data[active, current[:, 1].astype(np.int64)]
                            left = np.where(np.isnan(values), current[:, 6].astype(bool), values <= current[:, 2])
                            nodes[active] = np.where(left, current[:, 3], current[:, 4]).astype(np.int64)
                result[ids] = score
        if head == 'residual':
            result += memory_baseline(rows)
        if not np.isfinite(result).all():
            raise ValueError('nonfinite predictor output')
        return result

    def bundle(self, adapter):
        q = contract(adapter.context.family, adapter.context.kernel_revision, adapter.context.features())
        matches = []
        for bundle in self.document['bundles']:
            c = bundle['contract']
            if (c['family'], c['revision']) != (q['family'], q['revision']):
                continue
            if any(c['policy'].get(k) != v for k, v in q['policy'].items()):
                continue
            # Never infer missing categorical workload or cache-regime inputs.
            if any(k not in q['policy'] for k in c['policy'] if not k.startswith('context.tags.')):
                continue
            matches.append(bundle)
        # Missing campaign tags can be completed only when unambiguous.
        return matches[0] if len(matches) == 1 else None

    def retrieval(self, adapter, limit):
        bundle = self.bundle(adapter)
        if bundle is None:
            return []
        target = adapter.context.features()
        def distance(row):
            return sum(abs(math.log2(max(1, row['features'].get('context.workload.'+k, 1)) /
                                     max(1, target.get('context.workload.'+k, 1)))) for k in ('m', 'n', 'k', 'h', 'o'))
        result, seen = [], set()
        for row in sorted(bundle['winners'], key=distance):
            try:
                config = adapter.deserialize(row['config'])
                cid = adapter.config_id(config)
                if cid in seen or adapter.rejection(config) is not None:
                    continue
            except (KeyError, ValueError, TypeError):
                continue
            seen.add(cid)
            result.append(config)
            if len(result) >= limit:
                break
        return result

    def order(self, adapter, pool, head):
        bundle = self.bundle(adapter)
        if bundle is None:
            return list(pool)
        rows = [dict(family=adapter.context.family, revision=adapter.context.kernel_revision,
                     features={**bundle['contract']['policy'], **adapter.features(c)}) for c in pool]
        score = self.predict(rows, head)
        return [pool[i] for i in np.argsort(score, kind='stable')]


@lru_cache(maxsize=4)
def _load(path, stamp, research):
    opener = gzip.open if path.endswith('.gz') else open
    with opener(path, 'rt') as stream:
        encoded = stream.read(128*1024*1024+1)
    if len(encoded) > 128*1024*1024:
        raise ValueError('predictor exceeds the 128 MiB decoded-text limit')
    return PortablePredictor(json.loads(encoded), research=research)


def load_predictor(path, *, research=False):
    path = Path(path).expanduser().resolve()
    stat = path.stat()
    return _load(str(path), (stat.st_mtime_ns, stat.st_size, stat.st_ino), research)
