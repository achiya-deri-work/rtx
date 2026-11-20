"""Paired, ordinary CUDA-event timings for the two routed movement legs.

These are synthetic balanced routes and warmed same-buffer CUDA graphs; this
is not the full MoE, and isolated leg deltas do not predict its speedup.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

import torch

from rtx.kernels import mxfp8_moe, mxfp8_moe_token_reduce


KINDS = ('combine', 'dx')
VARIANTS = ('scalar', 'v4_l32', 'v8_l16', 'v8_l32', 'v4_l16')
# N, D, E, K, sequence lengths, allocated logical C (when larger than active C).
CASES = {
    'tiny': (128, 384, 32, 4, (128,), 0),
    'small_k3': (3072, 384, 96, 3, (3072,), 0),
    'narrow_k8': (32768, 128, 64, 8, (32768,), 0),
    'long': (65536, 384, 32, 4, (65536,), 0),
    'wide_long': (32768, 768, 32, 4, (32768,), 0),
    'new_k6': (24576, 640, 48, 6, (24576,), 0),
    'new_k6_packed': (24576, 640, 48, 6, (95, 817, 3321, 6847, 13496), 0),
    'packed32': (8192, 384, 32, 4, (256,) * 32, 2016),
}


def _fixture(n, d, e, k, lengths, alloc_c, seed):
    assert sum(lengths) == n
    rng = torch.Generator().manual_seed(seed)
    starts = []
    capacity = 0
    for length in lengths:
        rows = math.ceil(length * k / e)
        starts.append(capacity)
        capacity += math.ceil(rows / 32) * 32 if len(lengths) > 1 else rows
    c = max(capacity, alloc_c)
    cs = c
    inverse = torch.empty((n, k), dtype=torch.int64)
    ids = torch.full((e, c), -1, dtype=torch.int64)
    rank = torch.arange(k)
    offset = 0
    for length, start in zip(lengths, starts, strict=True):
        # Balanced memberships with expert rows in input-token order, and
        # randomized token/rank memberships to avoid a periodic warp pattern.
        memberships = (torch.randperm(length, generator=rng)[:, None] * k + rank) % e
        perm = torch.rand((length, k), generator=rng).argsort(1)
        memberships = memberships.gather(1, perm).flatten()
        order = memberships.argsort(stable=True)
        sizes = memberships.bincount(minlength=e)
        group_start = torch.repeat_interleave(sizes.cumsum(0) - sizes, sizes)
        local = torch.arange(length * k) - group_start + start
        inverse.flatten()[offset * k + order] = memberships[order] * c + local
        ids[memberships[order], local] = offset + order // k
        offset += length
    weights = torch.rand((e, c), generator=rng, dtype=torch.float32)
    weights[ids < 0] = 0.
    hash_routes = hashlib.sha256(inverse.numpy().tobytes()).hexdigest()
    gpu_rng = torch.Generator(device='cuda').manual_seed(seed)
    source = {kind: torch.randn((e * cs, d), device='cuda', dtype=torch.bfloat16,
                                generator=gpu_rng) for kind in KINDS}
    return source, inverse.cuda(), ids.cuda(), weights.cuda(), c, cs, hash_routes


def _graph(fn, nodes):
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(nodes):
            fn()
    return graph


def _time(graph, repetitions):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repetitions):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / repetitions


def _measure(graphs, nodes, rounds, samples):
    names = list(graphs)
    batch = {name: max(1, min(64, math.ceil(5000 / max(_time(graph, 4), 1))))
             for name, graph in graphs.items()}
    result = {name: [] for name in names}
    for round_index in range(rounds):
        rotated = names[round_index % len(names):] + names[:round_index % len(names)]
        for name in rotated:
            result[name].append([_time(graphs[name], batch[name]) / nodes for _ in range(samples)])
    return {name: {'median_us': statistics.median(x for visit in visits for x in visit),
                   'minimum_us': min(x for visit in visits for x in visit),
                   'visits_us': visits, 'graph_repetitions_per_sample': batch[name]}
            for name, visits in result.items()}


def _run_case(name, problem, variants, threads, nodes, rounds, samples, seed):
    n, d, e, k, lengths, alloc_c = problem
    source, inverse, ids, w, c, cs, route_hash = _fixture(*problem, seed)
    launch = {}
    for kind in KINDS:
        src = source[kind]
        dtype, itype = ((torch.float32, torch.int32) if kind == 'combine'
                        else (torch.bfloat16, torch.int16))
        out = torch.empty((n, d), device='cuda', dtype=dtype)
        ref = torch.empty_like(out)
        legacy = mxfp8_moe.compile_movement(n, d, e, c, k, threads, kind, 'flat', cs)
        legacy(src, ids, inverse, w, src, ref, w)
        if not torch.isfinite(ref).all():
            raise AssertionError(f'{name}/{kind}: reference is nonfinite')
        for variant in variants:
            if variant == 'scalar':
                kernel = legacy
            else:
                v, lanes = (int(x[1:]) for x in variant.split('_'))
                kernel = mxfp8_moe_token_reduce.compile_token_reduce(n, d, e, c, k, threads,
                                                                    kind, cs, v, lanes)

            def run(kernel=kernel, src=src, out=out):
                kernel(src, ids, inverse, w, src, out, w)

            run()
            if not torch.equal(out.view(itype), ref.view(itype)):
                raise AssertionError(f'{name}/{kind}/{variant}: not bitwise identical')
            launch[kind, variant] = run
    del ref
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    pre_graph_live = torch.cuda.memory_allocated()
    legs = {}
    for kind in KINDS:
        graphs = {variant: _graph(launch[kind, variant], nodes) for variant in variants}
        legs[kind] = _measure(graphs, nodes, rounds, samples)
    # Compare both legs in one graph (not the rest of the layer), including
    # each singly modified composition. Select geometry on isolated timings.
    choice = {kind: min((v for v in variants if v != 'scalar'),
                        key=lambda v: legs[kind][v]['median_us']) for kind in KINDS}
    pairs = {'scalar': ('scalar', 'scalar'),
             'combine_only': (choice['combine'], 'scalar'),
             'dx_only': ('scalar', choice['dx']),
             'both': (choice['combine'], choice['dx'])}
    graph_pairs = {}
    for cell, (combine_kind, dx_kind) in pairs.items():
        def run_pair(combine=launch['combine', combine_kind], dx=launch['dx', dx_kind]):
            combine()
            dx()
        graph_pairs[cell] = _graph(run_pair, nodes)
    # Graph creation can allocate control state (e.g. RNG state tensors).
    # Only synchronized post-capture replay samples diagnose live growth.
    for graph in graph_pairs.values():
        graph.replay()
    torch.cuda.synchronize()
    live_before = torch.cuda.memory_allocated()
    paired = _measure(graph_pairs, nodes, rounds, samples)
    live_after = torch.cuda.memory_allocated()
    if live_before != live_after:
        raise AssertionError(f'{name}: live movement bytes changed from {live_before} to {live_after}')
    return {'case': name, 'shape': [n, d, e, c, k], 'lengths': lengths, 'physical_capacity': cs,
            'inverse_sha256': route_hash, 'threads': threads, 'bitwise': True,
            'pre_graph_live_bytes': pre_graph_live, 'replay_live_bytes': [live_before, live_after],
            'peak_bytes': torch.cuda.max_memory_allocated(),
            'leg_us': legs, 'pair_geometries': pairs, 'movement_pair_us': paired}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', action='append', choices=CASES, help='Repeat for several cases')
    parser.add_argument('--variant', action='append', choices=VARIANTS[1:])
    parser.add_argument('--threads', type=int, default=128, choices=(64, 128, 256, 512))
    parser.add_argument('--samples', type=int, default=7)
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--graph-nodes', type=int, default=16)
    parser.add_argument('--seed', type=int, default=861019)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    props = torch.cuda.get_device_properties(0)
    variants = ('scalar', *args.variant) if args.variant else VARIANTS
    report = {'device': props.name, 'uuid': str(props.uuid), 'torch': torch.__version__,
              'cuda': torch.version.cuda, 'sm_count': props.multi_processor_count,
              'argv': sys.argv, 'seed': args.seed, 'variants': variants,
              'rounds': args.rounds, 'samples': args.samples, 'graph_nodes': args.graph_nodes,
              'sources': {mod.__name__: hashlib.sha256(Path(mod.__file__).read_bytes()).hexdigest()
                          for mod in (mxfp8_moe, mxfp8_moe_token_reduce)},
              'results': []}
    for name in args.case or CASES:
        row = _run_case(name, CASES[name], variants, args.threads, args.graph_nodes,
                        args.rounds, args.samples, args.seed)
        report['results'].append(row)
        print(name, {kind: {v: round(row['leg_us'][kind][v]['median_us'], 2) for v in variants}
                     for kind in KINDS}, flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as output:
        json.dump(report, output, indent=2)
        output.write('\n')


if __name__ == '__main__':
    main()
