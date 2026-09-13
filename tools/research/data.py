"""Audited data and pre-execution feature contracts for model comparison."""
import json
import math
from collections import Counter, defaultdict

import numpy as np

from tools.research_effort import identity, rows

VIEWS = ('shape', 'hardware', 'interactions', 'operating')
SEMANTIC_TAGS = ('precision', 'phase', 'scaling', 'scale_policy', 'operand_state', 'activation')
OPERATING = ('power.limit', 'power.draw', 'clocks.sm', 'clocks.mem', 'temperature.gpu')


def finite(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def shape_group(features):
    # Same workload across devices, regimes and numerical policies stays in
    # one fold. Do not hash the family, config, device or campaign identity.
    return identity({k: v for k, v in features.items()
                     if k.startswith('context.workload.') and '=' not in k})


def quarantine(record):
    c, family = record['config'], record['family']
    if family == 'nvfp4_weight_prequant_fwd' and c.get('quant_x', {}).get('scale_reciprocal', 'direct') != 'direct':
        return True
    gemms = [c.get('gemm', {})]
    if family == 'mxfp8_recompute_mlp':
        gemms += [c.get('gate_prequant', {}).get('gemm', {}), c.get('down_gemm', {})]
    return any(g.get('tile_m') == 64 and g.get('stages') == 1
               and g.get('scale_role', 'consumers') == 'consumers'
               and g.get('scale_layout', 'row_major') == 'row_major' for g in gemms)


def feature_view(row, view):
    if view not in VIEWS:
        raise ValueError(view)
    result = {}
    for k, v in row['features'].items():
        keep = k.startswith(('config.', 'context.workload.', 'context.regime='))
        keep |= any(k.startswith('context.tags.' + tag + '=') for tag in SEMANTIC_TAGS)
        if view != 'shape':
            keep |= k.startswith(('context.device.properties.', 'context.device.architecture.', 'context.device.sku.'))
        if view in ('interactions', 'operating'):
            keep |= k.startswith('derived.')
        if not keep or not finite(v) or any(token in k.lower() for token in (
            'compiled', 'spill', 'telemetry', 'measured', 'source', 'name=', 'sku_family=', 'uuid',
        )):
            continue
        result[k] = float(v) if '=' in k else math.copysign(math.log1p(abs(v)), v)
    if view == 'operating':
        # Only top-level telemetry_before (before candidate evaluation), never
        # telemetry_after or selected-attempt measurements from inside timing.
        for key in OPERATING:
            value = row.get('operating', {}).get(key)
            result['operating.' + key + '.missing'] = float(value is None)
            if value is not None:
                result['operating.' + key] = math.log1p(max(0, value))
    result['family=' + row['family']] = 1.
    result['revision=' + str(row['revision'])] = 1.
    return result


def load_dataset(paths):
    counts = Counter()
    seen, groups, attempts, inventory = set(), defaultdict(list), [], defaultdict(lambda: defaultdict(list))
    for record in rows(paths):
        oid = record.get('observation_id') or identity(record)
        if oid in seen:
            counts['duplicates'] += 1
            continue
        seen.add(oid)
        outcome, f = record['outcome'], record['features']
        counts['raw'] += 1
        counts[outcome['status']] += 1
        device = next((k.split('=', 1)[1] for k in f if k.startswith('context.device.name=')), 'unknown')
        semantic = {k: v for k, v in f.items() if k.startswith(('context.workload.', 'context.regime='))
                    or any(k.startswith('context.tags.' + tag + '=') for tag in SEMANTIC_TAGS)}
        context = identity([record['family'], record['kernel_revision'], device, semantic])
        metadata = outcome.get('metadata', {})
        before = metadata.get('telemetry_before', {})
        operating = {k: float(before[k]) for k in OPERATING if finite(before.get(k))}
        row = dict(context=context, config_id=record['config_id'], family=record['family'],
                   revision=record['kernel_revision'], device=device, shape=shape_group(f),
                   features=f, operating=operating, observation_id=oid)
        for k, v in f.items():
            if k.startswith(('context.device.properties.', 'context.device.sku.')) and '=' not in k and finite(v):
                inventory[device][k].append(float(v))
        for k, v in operating.items():
            inventory[device]['pre_evaluation.' + k].append(v)
        bad = quarantine(record)
        if bad:
            counts['quarantined'] += 1
            continue
        # Static rejects are not attempted configurations and not compiler failures.
        if outcome['status'] in ('ok', 'compile_error', 'runtime_error', 'correctness_error'):
            elapsed = metadata.get('elapsed_s')
            attempts.append({**row, 'failed': int(outcome['status'] != 'ok'),
                             'status': outcome['status'],
                             'evaluation_s': float(elapsed) if finite(elapsed) and float(elapsed) > 0 else None})
        if outcome['status'] != 'ok':
            continue
        if metadata.get('sampling', {}).get('collection', {}).get('stationary') is False:
            counts['unstable'] += 1
            continue
        ms = outcome.get('median_ms')
        if not finite(ms) or float(ms) <= 0:
            counts['invalid_latency'] += 1
            continue
        groups[(context, row['config_id'])].append({**row, 'ms': float(ms)})
    latency = []
    for values in groups.values():
        row = dict(values[0])
        row['ms'] = float(np.median([v['ms'] for v in values]))
        row['replicates'] = len(values)
        row['operating'] = {k: float(np.median([v['operating'][k] for v in values if k in v['operating']]))
                            for k in OPERATING if any(k in v['operating'] for v in values)}
        latency.append(row)
    counts['latency_rows'], counts['attempt_rows'] = len(latency), len(attempts)
    hardware = {device: {key: dict(count=len(values), minimum=min(values), maximum=max(values),
                                  median=float(np.median(values))) for key, values in fields.items()}
                for device, fields in inventory.items()}
    return latency, attempts, dict(counts), hardware


def split_rows(train, test, protocol):
    if protocol == 'chronological':
        tr, te = train, test
    elif protocol == 'unseen_shapes':
        held = {r['shape'] for r in test if int(r['shape'][:8], 16) % 5 == 0}
        tr = [r for r in train if r['shape'] not in held]
        te = [r for r in test if r['shape'] in held]
    elif protocol.startswith('heldout:'):
        device = protocol.split(':', 1)[1]
        tr = [r for r in train if r['device'] != device]
        te = [r for r in test if r['device'] == device]
    else:
        raise ValueError(protocol)
    supported = {(r['family'], r['revision']) for r in tr}
    return tr, [r for r in te if (r['family'], r['revision']) in supported]


def inner_split(train):
    validation = {r['shape'] for r in train if int(r['shape'][8:16], 16) % 5 == 0}
    return ([r for r in train if r['shape'] not in validation],
            [r for r in train if r['shape'] in validation])
