"""Attribute explicit live RTX allocations in a trusted local Torch snapshot.

Pickle can execute code: use only snapshots you trust. This reports snapshot
residency, never historical peak ownership or driver-wide allocation.
"""
import argparse
import hashlib
import json
import pickle
from collections import Counter
from pathlib import Path


def summarize(snapshot):
    pools = {}
    sites = Counter()
    for segment in snapshot['segments']:
        key = str(segment.get('segment_pool_id', 'unknown'))
        pool = pools.setdefault(key, Counter())
        pool['reserved_bytes'] += segment['total_size']
        for block in segment['blocks']:
            pool[block['state'] + '_bytes'] += block['size']
            if block['state'] != 'active_allocated':
                continue
            # First RTX frame is the nearest visible allocation call site,
            # not proof of sole ownership or the originating application module.
            frame = next((f for f in block.get('frames', [])
                          if '/rtx/rtx/' in f.get('filename', '')), None)
            label = (frame['filename'].split('/rtx/rtx/', 1)[1] + ':' + str(frame['line'])
                     if frame else 'non_rtx_or_unattributed')
            sites[label] += block['size']
    traces = [dict(events=len(trace), actions=dict(Counter(e['action'] for e in trace)))
              for trace in snapshot.get('device_traces', [])]
    return dict(pools=pools, active_allocation_sites=dict(sites), trace_summary=traces,
                scope='snapshot-time rounded live allocation; not peak attribution',
                limitations=['history may be truncated', 'pool IDs are process-local',
                             'stack attribution is not unique semantic ownership',
                             'external device allocations are not represented'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('snapshot', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    raw = args.snapshot.read_bytes()
    report = summarize(pickle.loads(raw))
    report.update(source=str(args.snapshot.resolve()), source_sha256=hashlib.sha256(raw).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report, stream, indent=2)


if __name__ == '__main__':
    main()
