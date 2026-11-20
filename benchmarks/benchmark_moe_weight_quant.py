"""Exact-fixture comparison of separate and one-read backward weight prep.

The timed portfolio always includes original-token X. Optional planner timing
includes gather and routed-dY/XT prep and the complete backward launch plan.
Neither phase is a claim about router/shared/optimizer or whole-model speedup.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import median
import sys


FIELDS = ('weight_tile_rows', 'weight_tile_cols', 'weight_load_values',
          'weight_store_bits', 'weight_warps', 'weight_raster')
DEFAULTS = ((32, 64, 4, 32, 4, 'features'), (32, 32, 8, 32, 4, 'features'),
            (32, 128, 8, 32, 4, 'features'), (64, 64, 8, 32, 8, 'features'),
            (64, 32, 8, 32, 4, 'rows'), (32, 64, 8, 32, 4, 'rows'),
            (32, 64, 8, 16, 4, 'features'))


def _same(torch, left, right):
    return all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(left, right, strict=True))


def _fixture(torch, file, archive, digest):
    data = torch.load(file, map_location='cpu', weights_only=True)
    observed = {name: digest(data[name]) for name in ('x', 'gu', 'down', 'ids', 'weights', 'inverse', 'dy')}
    identity = archive['identities']
    assert observed['x'] == identity['input']
    assert digest(data['dy'].bfloat16()) == identity['upstream']
    assert observed['ids'] == identity['routing']['expert_ids']
    assert observed['inverse'] == identity['routing']['inverse']
    for e, (gu, down) in enumerate(zip(data['gu'], data['down'])):
        gate, up = gu.chunk(2)
        for parameter, tensor in ((f'gate_up.projections.e{e}_gate.weight', gate),
                                  (f'gate_up.projections.e{e}_up.weight', up),
                                  (f'down.projections.e{e}.weight', down)):
            assert digest(tensor) == identity['parameters'][parameter]
    return {k: v.cuda() for k, v in data.items() if isinstance(v, torch.Tensor)}, observed


def _measure(torch, calls, rounds, samples, iterations, *, graph):
    from rtx.autotune.preparation_runtime import _warm_measurement_graph

    graph_list = {}
    stream = torch.cuda.Stream()
    for name, call in calls.items():
        call()
        torch.cuda.synchronize()
        if graph:
            captured = torch.cuda.CUDAGraph()
            with torch.cuda.graph(captured, stream=stream):
                call()
            graph_list[name] = captured
    torch.cuda.synchronize()
    live = [torch.cuda.memory_allocated()]
    order = tuple(calls)
    records = {name: [] for name in order}
    for r in range(rounds):
        visits = order[r % len(order):] + order[:r % len(order)]
        for name in visits:
            if graph:
                _warm_measurement_graph(graph_list[name])
                call = graph_list[name].replay
            else:
                call = calls[name]
                for _ in range(8):
                    call()
            values = []
            for _ in range(samples):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iterations):
                    call()
                end.record()
                end.synchronize()
                values.append(start.elapsed_time(end) * 1000 / iterations)
            records[name].append(values)
        live.append(torch.cuda.memory_allocated())
    assert len(set(live)) == 1, live
    return {'median_us': {name: median(map(median, v)) for name, v in records.items()},
            'visits_us': records, 'live_bytes': live, 'peak_bytes': torch.cuda.max_memory_allocated()}


def _planner(torch, data, configs, shape, args):
    from rtx._moe_execution import allocate, compiled_plan, layout, plan
    from rtx.moe import _config_key
    from rtx.runtime import load_kernel_symbol

    x, gu, down, ids, weights, inverse, dy = (data[k] for k in ('x', 'gu', 'down', 'ids', 'weights', 'inverse', 'dy'))
    inputs = [x, gu.flatten(0, 1), down.flatten(0, 1), ids, weights, inverse, dy]
    prefix, full, results, reports = {}, {}, {}, {}
    for name, config in configs.items():
        key, dtypes, sizes, views, outputs = layout(shape, _config_key(config), 'backward')
        buffers = allocate(x, dtypes, sizes, views)
        builder, *_ = plan(key)
        compiled = compiled_plan(key, x.device.index)
        full[name] = lambda op=compiled, buffers=buffers: op(inputs, buffers)
        results[name] = tuple(buffers[i] for i in outputs)

        def resolve(ref):
            if isinstance(ref, tuple):
                return tuple(resolve(r) for r in ref)
            return inputs[-ref - 1] if ref < 0 else buffers[ref]

        ops = []
        for module, symbol, parameters, references in builder.steps:
            if symbol == 'compile_expert_packed_gemm':
                break
            ops.append((load_kernel_symbol(module, symbol)(*parameters), resolve(references)))

        def prepare(ops=ops):
            for fn, fn_args in ops:
                fn(*fn_args)

        prefix[name] = prepare
        reports[name] = dict(arena_bytes=sum(n * t.itemsize for n, t in zip(sizes, dtypes)),
                             prep_steps=[s[1] for s in builder.steps[:len(ops)]],
                             complete_steps=[s[1] for s in builder.steps])

    # Exact comparisons depend on only the weight-copy schedule changing.
    # Audit the replay after in-place weights and dY change, not merely a
    # warm eager reference against an unchanged graph.
    graphs = {}
    stream = torch.cuda.Stream()
    for name, call in full.items():
        call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
        graphs[name] = graph
    for stage in ('original', 'changed', 'restored'):
        if stage == 'changed':
            gu.neg_()
            down.mul_(2)
            dy.neg_()
        elif stage == 'restored':
            gu.neg_()
            down.mul_(0.5)
            dy.neg_()
        for graph in graphs.values():
            graph.replay()
        for name, values in results.items():
            assert all(a.isfinite().all() for a in values), (stage, name)
            assert _same(torch, values, results['separate']), (stage, name)
            reports[name][stage] = ('dx', 'dgu', 'ddown', 'route_dw')
    del graphs
    timings = {}
    for name, calls in (('prep_prefix', prefix), ('complete_backward', full)):
        timings[name] = _measure(torch, calls, args.rounds, args.samples, max(2, args.iterations // 4), graph=True)
    return dict(validation=reports, **timings)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-result', required=True, type=Path)
    parser.add_argument('--fixture', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--variant', action='append', nargs=6,
                        metavar=('ROWS', 'COLS', 'LOAD', 'STORE', 'WARPS', 'RASTER'))
    parser.add_argument('--rounds', type=int, default=4)
    parser.add_argument('--samples', type=int, default=5)
    parser.add_argument('--iterations', type=int, default=16)
    parser.add_argument('--planner', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.rounds, args.samples, args.iterations) <= 0:
        parser.error('Every repetition count must be positive')

    import torch
    from benchmarks.benchmark_moe_execution import _python_source_digest, _tensor_digest
    from rtx.configs.moe_backward import backward_weight_quant_rejection
    from rtx.kernels.mxfp8_moe_buffers import compile_expert_quant
    from rtx.kernels.mxfp8_moe_weight_quant import compile_backward_weight_quant
    from rtx.moe import MXFP8MoEConfig

    archive = json.loads(args.source_result.read_text())
    config = MXFP8MoEConfig.from_dict(archive['expert_config'])
    data, hashes = _fixture(torch, args.fixture, archive, _tensor_digest)
    x, gu, down = (data[k] for k in ('x', 'gu', 'down'))
    e, c = data['ids'].shape
    n, d = x.shape
    h, k = down.shape[-1], data['inverse'].shape[-1]
    p = config.prepared
    specs = ((n, d, 'row', False, 1), (2*h, d, 'row', True), (d, h, 'row', False),
             (h, d, 'transpose', False), (d, 2*h, 'transpose', False))
    gg, dd = gu.flatten(0, 1), down.flatten(0, 1)
    baseline = compile_expert_quant(e, specs, p.quant, p.transpose_mode, p.quant_grid,
                                    p.transpose_warp_tokens, p.transpose_warp_raster)
    def pair(rows, cols):
        blocks = (cols + 31) // 32
        return (torch.empty((rows, blocks * 32), device=x.device, dtype=torch.float8_e4m3fn),
                torch.empty((rows, blocks), device=x.device, dtype=torch.uint8))

    outputs = tuple(t for rows, cols in ((n, d), (e * 2*h, d), (e*d, h), (e*h, d), (e*d, 2*h))
                    for t in pair(rows, cols))

    def separate():
        baseline(x, (x, gg, dd, dd, gg), outputs[::2], outputs[1::2])

    calls, schedules = {'separate': separate}, {}
    variants = [(*map(int, v[:5]), v[5]) for v in args.variant] if args.variant else DEFAULTS
    separate()
    expected = tuple(t.clone() for t in outputs)
    for variant in variants:
        if reason := backward_weight_quant_rejection(n, d, e, h, p.quant, p.quant_grid, *variant):
            raise ValueError((variant, reason))
        name = 'x'.join(map(str, variant))
        fn = compile_backward_weight_quant(n, d, e, h, p.quant, p.quant_grid, *variant)
        calls[name] = lambda fn=fn: fn(x, gg, dd, *outputs)
        schedules[name] = dict(zip(FIELDS, variant))
        calls[name]()
        assert _same(torch, outputs, expected), name
    del expected
    graph = _measure(torch, calls, args.rounds, args.samples, args.iterations, graph=True)
    ordinary = _measure(torch, calls, args.rounds, args.samples, args.iterations, graph=False)
    best = min(schedules, key=graph['median_us'].__getitem__)
    shape = n, d, e, c, k, h
    weight_bytes = 2 * e * (2*h*d + d*h)
    result = dict(command=sys.argv, fixture=str(args.fixture.resolve()), identities=hashes,
                  source_digest=_python_source_digest(Path.cwd()), exact_archive_match=True,
                  device=str(torch.cuda.get_device_properties(0)), torch=torch.__version__, cuda=torch.version.cuda,
                  shape=shape, quant=asdict(p.quant), baseline_transpose=p.transpose_mode, grid=p.quant_grid,
                  schedules=schedules, byte_exact_all=True, logical_bf16_weight_reads={'separate':2*weight_bytes, 'dual':weight_bytes},
                  includes_original_token_x=True, launches_per_portfolio=1,
                  ordinary_event=ordinary, graph_event=graph, fastest_dual=best)
    print(json.dumps(graph['median_us']), flush=True)
    if args.planner:
        del calls, outputs
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        result['planner'] = _planner(torch, data, {'separate': config, best: replace(
            config, prepared=replace(p, weight_preparation='dual_orientation', **schedules[best]))}, shape, args)
        for name in ('prep_prefix', 'complete_backward'):
            print(name, json.dumps(result['planner'][name]['median_us']), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
