"""Time indexed-X preparation plus gate dW against the on-chip consumer.

Use a saved expert fixture from benchmark_moe_varlen.py. Freeze routing,
weights, X, upstream dY, qP transpose, and all preceding preparations. The
ordinary CUDA graph measurements include both baseline launches; the fused
path does not write or read an X FP8 operand in global memory.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics


def get_operands(fixture, config):
    from rtx._moe_execution import allocate, compiled_plan, layout, plan
    from rtx.moe import _config_key

    x, gu, down, ids, weights, inverse, dy = (
        fixture[name].cuda() for name in ('x', 'gu', 'down', 'ids', 'weights', 'inverse', 'dy'))
    shape = (x.shape[0], x.shape[1], ids.shape[0], ids.shape[1], inverse.shape[1], down.shape[-1])
    key, dtypes, sizes, views, _ = layout(shape, _config_key(config), 'backward')
    builder, *_ = plan(key)
    buffers = allocate(x, dtypes, sizes, views)
    inputs = [x, gu.flatten(0, 1), down.flatten(0, 1), ids, weights, inverse, dy.contiguous()]

    def resolve(ref):
        if isinstance(ref, int):
            return inputs[-ref-1] if ref < 0 else buffers[ref]
        return tuple(map(resolve, ref))

    functions = compiled_plan(key, x.device.index).__rtx_resources__
    transpose = None
    target = builder.outputs[1]
    for step, fn in zip(builder.steps, functions):
        _, symbol, _, args = step
        if symbol == 'compile_indexed_expert_transpose':
            mapping = resolve(args[1])
            transpose = fn
        if symbol == 'compile_expert_packed_gemm' and args[4] == target:
            if transpose is None:
                raise ValueError('Expected independent indexed transpose and packed unsplit gate dW')
            qpt, _, spt, _, _, _ = map(resolve, args)
            return shape, x, mapping, qpt, spt, fn, transpose
        fn(*map(resolve, args))
    raise ValueError('Expected a packed gate dW step')


def synthetic_fixture(shape):
    """Seeded fixed-K probe; does not stand in for the learned-route fixture."""
    import torch
    from tests.test_mxfp8_moe import make_route

    torch.manual_seed(40607)
    n, d, e, k, h = shape
    ids, weights, inverse = make_route(n, e, k, 'cpu')
    return dict(x=torch.randn(n, d).bfloat16()*.3, gu=torch.randn(e, 2*h, d).bfloat16()*.1,
                down=torch.randn(e, d, h).bfloat16()*.1, dy=torch.randn(n, d),
                ids=ids, weights=weights, inverse=inverse)


def run(args):
    import torch
    from rtx.configs.moe_backward import indexed_gate_dw_rejection
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.kernels.mxfp8_moe_indexed_dw import compile_indexed_gate_dw
    from rtx.moe import MXFP8MoEConfig
    from benchmarks.benchmark_moe_joint_recompute import _capture
    from benchmarks.benchmark_moe_varlen import statistics_ms
    from tests.test_moe_indexed_dw import error_summary, oracle

    fixture = (torch.load(args.fixture, map_location='cpu', weights_only=True)
               if args.fixture else synthetic_fixture(args.synthetic_shape))
    raw = json.loads(args.expert_config.read_text())
    config = MXFP8MoEConfig.from_dict(raw.get('config', raw))
    if args.synthetic_shape and args.synthetic_shape[-1] % config.prepared.down_dx.tile_n:
        # The original target's TMA store is legal only for full H tiles.
        p = config.prepared
        config = replace(config, prepared=replace(p, down_dx=replace(p.down_dx,
                        epilogue='direct', store_vec=1, epilogue_stages=1)))
    shape, x, mapping, qpt, spt, packed, transpose = get_operands(fixture, config)
    n, d, e, c, top_k, h = shape
    cs = config.prepared.physical_capacity(c)
    problem = MXFP8Problem(2*h, d, cs)
    p = config.prepared
    reference = torch.empty((e*2*h, d), device=x.device, dtype=x.dtype)
    qxt = torch.empty((e*d, cs), device=x.device, dtype=torch.float8_e4m3fn)
    sxt = torch.empty((e*d, cs//32), device=x.device, dtype=torch.uint8)
    out = torch.empty_like(reference)

    def materialized():
        transpose(x, mapping, qxt, sxt)
        packed(qpt, qxt, spt, sxt, reference, reference)

    graph_stream = torch.cuda.Stream()
    materialized()
    samples_expert, samples_hidden, samples_features = (0, e-1), min(8, 2*h), min(32, d)
    expert_map = mapping.view(e, cs)[list(samples_expert)].cpu()
    sampled = torch.stack([qpt.view(e, 2*h, cs)[i, :samples_hidden].cpu().double() for i in samples_expert])
    scales = spt.view(e, 2*h, cs//32)[list(samples_expert), :samples_hidden].cpu().to(torch.int32) - 127
    sampled *= torch.ldexp(torch.ones_like(scales, dtype=torch.float64), scales).repeat_interleave(32, dim=-1)
    oracle_out = oracle(x[:, :samples_features].cpu(), sampled, expert_map, p.quant)

    def sampled_error(result):
        return error_summary(result.view(e, 2*h, d)[list(samples_expert), :samples_hidden, :samples_features].cpu(), oracle_out)

    baseline_error = sampled_error(reference)
    if baseline_error['nonfinite'] or baseline_error['nrms'] >= .001:
        raise ValueError(f'Baseline sample failed independent oracle: {baseline_error}')
    graphs = {'packed': _capture(materialized, graph_stream)[0]}
    results = {'packed': dict(oracle=baseline_error, visits=[])}
    # Rows, staged slots, token lanes, BF16 pair vector width, quant producer
    # warps. Keep the same N geometry and all precision choices as control.
    candidates = args.candidate or [(64,128,8,1,0), (64,128,8,2,4), (64,128,4,2,8),
                                   (64,64,4,2,8), (128,128,8,2,4), (128,128,4,2,8),
                                   (128,128,2,2,8)]
    for rows, slots, lanes, pairs, warps in candidates:
        name = f'm{rows}-k{slots}-lanes{lanes}-pairs{pairs}-producer{warps}'
        g = replace(p.gate_dw, tile_m=rows, tile_k=slots, atom_layout_m=2 if rows == 64 else 4)
        reason = indexed_gate_dw_rejection(problem, g, p.quant, lanes, pairs, warps)
        if reason:
            results[name] = dict(rejected=reason)
            continue
        launch = compile_indexed_gate_dw(n, e, problem, g, p.quant, lanes, pairs, warps)

        def indexed():
            launch(qpt, spt, x, mapping, out)

        indexed()
        error, independent = error_summary(out, reference), sampled_error(out)
        print('VALID', name, error, independent, flush=True)
        results[name] = dict(vs_packed=error, oracle=independent, visits=[])
        if not error['nonfinite'] and error['nrms'] < .001 and independent['nrms'] < .001:
            graphs[name] = _capture(indexed, graph_stream)[0]
    torch.cuda.synchronize()
    live = [torch.cuda.memory_allocated()]
    phases = list(graphs)
    for round_id in range(args.rounds):
        visit = phases if round_id % 2 == 0 else phases[::-1]
        if round_id >= 2:
            rotate = 1 + (round_id-2) % len(phases)
            visit = visit[rotate:] + visit[:rotate]
        for name in visit:
            graph = graphs[name]
            graph.replay(); graph.replay()
            events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                      for _ in range(args.samples)]
            for start, end in events:
                start.record()
                for _ in range(args.replays):
                    graph.replay()
                end.record()
            events[-1][1].synchronize()
            measures = statistics_ms([start.elapsed_time(end)/args.replays for start, end in events])
            results[name]['visits'].append(measures)
            print('TIME', round_id, name, measures['median_ms'], flush=True)
        live.append(torch.cuda.memory_allocated())
    for result in results.values():
        if result.get('visits'):
            result['median_visit_ms'] = statistics.median(v['median_ms'] for v in result['visits'])
    properties = torch.cuda.get_device_properties(0)
    record = dict(shape=shape, fixture=str(args.fixture) if args.fixture else None,
                  synthetic_shape=args.synthetic_shape, expert_config=str(args.expert_config),
                  source_sha256=hashlib.sha256(Path('rtx/kernels/mxfp8_moe_indexed_dw.py').read_bytes()).hexdigest(),
                  map_sha256=hashlib.sha256(mapping.cpu().numpy().tobytes()).hexdigest(),
                  workspace_removed_bytes=qxt.numel()+sxt.numel(), torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=str(properties), rounds=args.rounds, samples=args.samples, replays=args.replays,
                  live_allocated_bytes=live, peak_allocated_bytes=torch.cuda.max_memory_allocated(), results=results,
                  oracle_samples=dict(experts=samples_expert, hidden=samples_hidden, features=samples_features))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    fixture = parser.add_mutually_exclusive_group(required=True)
    fixture.add_argument('--fixture', type=Path)
    fixture.add_argument('--synthetic-shape', nargs=5, type=int, metavar=('N','D','E','K','H'))
    parser.add_argument('--expert-config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--candidate', nargs=5, type=int, action='append', metavar=('M','K','LANES','PAIRS','PRODUCERS'))
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--samples', type=int, default=7)
    parser.add_argument('--replays', type=int, default=4)
    args = parser.parse_args()
    if min(args.rounds, args.samples, args.replays) < 1:
        parser.error('rounds, samples and replays must be positive')
    run(args)


if __name__ == '__main__':
    main()
