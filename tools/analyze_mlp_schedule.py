"""Read-only analysis of recorded RTX forward/backward ownership metadata.

Host intervals describe operator ordering, not GPU kernel duration. Pending-input
bytes are a subset of application live memory, never an additive peak estimate.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def signature(row):
    return tuple(row['x']['shape']), tuple(row['gu']['shape']), tuple(row['down']['shape'])


def analyze(records, peak_us):
    pending = defaultdict(list)
    pairs, unmatched, shapes = [], [], Counter()
    # Match in host completion order. Tokens can reuse CPU addresses; never
    # match a backward to a future forward or reuse a consumed forward.
    for row in sorted(records, key=lambda r: r['end_us']):
        key = (row['token']['address'], row['x']['address'],
               row['gu']['address'], row['down']['address'])
        if row['phase'] == 'forward':
            shapes[(signature(row), bool(row['training']))] += 1
            if row['training']:
                pending[key].append(row)
        elif row['phase'] == 'backward':
            eligible = [i for i, f in enumerate(pending[key]) if f['end_us'] <= row['start_us']]
            if not eligible:
                unmatched.append(row)
                continue
            f = pending[key].pop(eligible[-1])
            if signature(f) != signature(row):
                raise ValueError('matched token/storage has inconsistent shapes')
            pairs.append((f, row))
        else:
            raise ValueError(f"unknown phase {row['phase']}")
    events = []
    at_peak = []
    for index, (f, b) in enumerate(pairs):
        events.extend(((f['end_us'], 1, index), (b['start_us'], -1, index)))
        if f['end_us'] <= peak_us < b['start_us']:
            at_peak.append(index)
    active = set()
    max_bytes = 0
    max_state = []
    max_at = None
    def storages(indices):
        result = {}
        for i in indices:
            x = pairs[i][0]['x']
            address, size = x['address'], x['storage_bytes']
            if address in result and result[address] != size:
                raise ValueError('simultaneous address has inconsistent storage size')
            result[address] = size
        return result
    # All changes at a timestamp applied together, avoiding invented peaks from
    # host timestamp ties. Zero-length intervals never enter the pending set.
    grouped = defaultdict(list)
    for timestamp, delta, index in events:
        grouped[timestamp].append((delta, index))
    for timestamp in sorted(grouped):
        for delta, index in sorted(grouped[timestamp], reverse=True):
            if delta == 1:
                active.add(index)
            else:
                active.discard(index)
        size = sum(storages(active).values())
        if size > max_bytes:
            max_bytes, max_state, max_at = size, sorted(active), timestamp
    def describe(indices):
        return [dict(shape=signature(pairs[i][0]), x_storage=pairs[i][0]['x']['address'],
                     bytes=pairs[i][0]['x']['storage_bytes'],
                     forward_end_us=pairs[i][0]['end_us'],
                     backward_start_us=pairs[i][1]['start_us']) for i in indices]
    # Short window around the allocator peak, including no-grad/recomputed calls.
    ordered = sorted(records, key=lambda r: r['start_us'])
    pivot = min(range(len(ordered)), key=lambda i: abs(ordered[i]['start_us']-peak_us)) if ordered else 0
    window = [dict(phase=r['phase'], training=r.get('training'), shape=signature(r),
                   start_offset_us=r['start_us']-peak_us, end_offset_us=r['end_us']-peak_us)
              for r in ordered[max(0, pivot-16):pivot+17]]
    return dict(matched_backward_calls=len(pairs), unmatched_backward_calls=len(unmatched),
        unconsumed_training_forwards=sum(map(len, pending.values())),
        forward_contexts=[dict(shape=s, training=t, calls=n) for (s,t),n in sorted(shapes.items())],
        peak_time_us=peak_us, pending_at_allocator_peak=describe(at_peak),
        pending_at_allocator_peak_bytes=sum(storages(at_peak).values()),
        maximum_pending_input_bytes=max_bytes, maximum_pending_time_us=max_at,
        maximum_pending_inputs=describe(max_state), operator_window_at_allocator_peak=window,
        scope='matched completed forwards awaiting backward start; unique input storage only; host timestamps, not GPU durations; not additive to allocator peak')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('capture', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    ownership = args.capture/'saved-storage-ownership.json'
    ledger = args.capture/'allocation-analysis.json'
    result = analyze(json.loads(ownership.read_text()), json.loads(ledger.read_text())['peak_time_us'])
    result['sources'] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in (ownership,ledger)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({k: result[k] for k in ('matched_backward_calls','unmatched_backward_calls',
        'unconsumed_training_forwards','pending_at_allocator_peak_bytes','maximum_pending_input_bytes')}))


if __name__ == '__main__':
    main()
