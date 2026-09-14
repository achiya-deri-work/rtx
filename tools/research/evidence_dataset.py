"""Immutable, provenance-backed historical dataset views; never runtime promotion."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import statistics
import zipfile

import pyarrow as pa
import pyarrow.parquet as pq

from tools.predictor_study import atomic
from .data import SEMANTIC_TAGS, shape_group, quarantine
from tools.research_effort import identity


def journals(source):
    source = Path(source)
    if source.suffix == '.zip':
        with zipfile.ZipFile(source) as archive:
            for name in sorted(archive.namelist()):
                if name.endswith('/observations.jsonl'):
                    with archive.open(name) as stream:
                        yield name, stream
    else:
        for path in sorted(source.rglob('observations.jsonl')):
            with path.open('rb') as stream:
                yield str(path.relative_to(source)), stream


def timing_quality(outcome):
    if outcome.get('status') != 'ok':
        return outcome.get('status', 'missing_status')
    ms = outcome.get('median_ms')
    if not isinstance(ms, (int, float)) or not math.isfinite(ms) or ms <= 0:
        return 'invalid_latency'
    stationary = outcome.get('metadata', {}).get('sampling', {}).get('collection', {}).get('stationary')
    if stationary is False:
        return 'unstable'
    timings = outcome.get('timings_ms', [])
    if len(timings) < 3:
        return 'timing_unverified'
    if any(not isinstance(t, (int, float)) or not math.isfinite(t) or t <= 0 for t in timings):
        return 'invalid_samples'
    median = statistics.median(timings)
    if statistics.pstdev(timings) / median > .05 or (max(timings)-min(timings))/median > .15:
        return 'unstable'
    return 'stable'


def decode(record):
    f = record['features']
    device = next((k.split('=', 1)[1] for k, v in f.items()
                   if k.startswith('context.device.name=') and v), 'unknown')
    semantic = {k: v for k, v in f.items() if k.startswith(('context.workload.', 'context.regime='))
                or any(k.startswith('context.tags.' + tag + '=') for tag in SEMANTIC_TAGS)}
    context = identity([record['family'], record['kernel_revision'], device, semantic])
    workload = {k.removeprefix('context.workload.'): v for k, v in f.items()
                if k.startswith('context.workload.') and '=' not in k}
    m, k = workload.get('m', 0), workload.get('k', 0)
    n = workload.get('h', workload.get('n', 0))
    shape_class = ('tiny_m' if m <= 128 else 'contraction' if n < k / 2
                   else 'expansion' if n > 2 * k else 'ordinary')
    return dict(context=context, device=device, shape=shape_group(f), shape_class=shape_class,
                config_id=record['config_id'], family=record['family'], revision=record['kernel_revision'],
                features=f, config=record['config'], operating={})


def build(sources, output):
    output = Path(output)
    if (output / 'manifest.json').exists():
        raise ValueError('choose a fresh output; completed dataset views are immutable')
    output.mkdir(parents=True, exist_ok=False)
    groups = defaultdict(list)
    seen = {}
    counts = Counter()
    source_hashes = []
    contracts = Counter()
    known_bad = {'8bcd7e142b11078517930357', '78fe1b37ce2118ae79d99365'}
    for split, paths in sources.items():
        for source in paths:
            file_hashes = []
            for name, stream in journals(source):
                digest = hashlib.sha256()
                audit = []
                for line_number, raw in enumerate(stream, 1):
                    digest.update(raw)
                    if not raw.strip():
                        continue
                    location = dict(source=str(source), member=name, line=line_number)
                    try:
                        record = json.loads(raw)
                        row = decode(record)
                        outcome = record['outcome']
                        oid = record.get('observation_id') or identity(record)
                    except (ValueError, KeyError, TypeError) as exc:
                        counts['malformed'] += 1
                        audit.append(dict(**location, classification='malformed', error=str(exc)))
                        continue
                    content_hash = identity(record)
                    if oid in seen:
                        label = 'duplicate' if seen[oid] == content_hash else 'conflicting_observation_id'
                        counts[label] += 1
                        audit.append(dict(**location, observation_id=oid, classification=label))
                        continue
                    seen[oid] = content_hash
                    label = timing_quality(outcome)
                    if outcome['status'] == 'ok' and (quarantine(record) or record['config_id'] in known_bad):
                        label = 'numerical_quarantine'
                    counts[label] += 1
                    contracts[(row['family'], row['revision'])] += 1
                    audit.append(dict(**location, observation_id=oid, context=row['context'],
                        config_id=row['config_id'], classification=label, status=outcome['status'],
                        strategy=record.get('strategy'), elapsed_s=record.get('elapsed_s'),
                        compile_ms=outcome.get('compile_ms')))
                    if label in ('stable', 'correctness_error', 'runtime_error'):
                        groups[(row['context'], row['config_id'])].append(dict(
                            **row, split=split, ms=outcome.get('median_ms'), classification=label,
                            observation_id=oid, source=str(source)))
                file_hash = digest.hexdigest()
                file_hashes.append((name, file_hash))
                # One durable residual per source journal; no shared append file.
                atomic(output/'audit'/f'{identity([str(source), name])}.json',
                       dict(source=str(source), member=name, sha256=file_hash, rows=audit))
            source_hashes.append(dict(source=str(source), split=split, journals=len(file_hashes),
                                      journal_inventory_sha256=identity(file_hashes)))
            print('AUDIT', split, source, dict(counts), flush=True)
    retained = []
    excluded = []
    for key, values in groups.items():
        reasons = {v['classification'] for v in values} - {'stable'}
        if reasons:
            reason = 'same_context_config_failure'
        elif max(v['ms'] for v in values) / min(v['ms'] for v in values) > 1.2:
            reason = 'cross_observation_instability'
        elif len({v['split'] for v in values}) > 1:
            reason = 'cross_split_overlap'
        else:
            reason = None
        if reason:
            counts[reason] += 1
            excluded.append(dict(context=key[0], config_id=key[1], reason=reason,
                                 observations=[v['observation_id'] for v in values]))
            continue
        row = dict(values[0])
        row.update(ms=statistics.median(v['ms'] for v in values), replicates=len(values),
                   observation_ids=[v['observation_id'] for v in values])
        retained.append(row)
    atomic(output/'excluded_groups.json', excluded)
    for offset in range(0, len(retained), 4096):
        shard = []
        for row in retained[offset:offset+4096]:
            shard.append({**row, 'features':json.dumps(row['features'], sort_keys=True),
                          'config':json.dumps(row['config'], sort_keys=True), 'operating':'{}'})
        pq.write_table(pa.Table.from_pylist(shard), output/f'latency-{offset//4096:04d}.parquet', compression='zstd')
    manifest = dict(schema=1, counts=dict(counts), rows=len(retained), sources=source_hashes,
                    contracts=[dict(family=f, revision=r, observations=n) for (f,r),n in sorted(contracts.items())],
                    policy=dict(within_cv_max=.05, within_range_max=.15, cross_observation_ratio_max=1.2,
                                known_bad_configs=sorted(known_bad), minimum_timing_samples=3),
                    runtime_promotion=False, historical_only=True,
                    policy_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    atomic(output/'manifest.json', manifest)
    return manifest


def load(output):
    rows = []
    for path in sorted(Path(output).glob('latency-*.parquet')):
        for row in pq.read_table(path).to_pylist():
            for field in ('features', 'config', 'operating'):
                row[field] = json.loads(row[field])
            rows.append(row)
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--sources', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(build(json.loads(Path(args.sources).read_text()), args.output), indent=2))
