"""Weighted-dY preparation only: isolated, byte-exact, paired CUDA graph timing."""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--case-id', default='d384_e32_k4_h128')
    p.add_argument('--lengths', type=int, nargs='+', default=(512, 2000, 768, 4096))
    p.add_argument('--expert-config', type=Path)
    p.add_argument('--router-config', type=Path)
    p.add_argument('--joint-warp-baseline', action='store_true',
                   help='Retarget a saved different-family quant schedule to indexed joint row-major warp prep')
    p.add_argument('--fused-tile', type=int, nargs=2, metavar=('FEATURES', 'SLOTS'))
    p.add_argument('--fused-padding', type=int)
    p.add_argument('--include-x', action='store_true',
                   help='Compare fused dY plus separate warp-XT with one combined preparation launch')
    p.add_argument('--fixture', type=Path, help='Optional saved ids/weights/inverse/dy fixture')
    p.add_argument('--samples', type=int, default=9)
    p.add_argument('--sample-iters', type=int, default=32)
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--capture', choices=('baseline', 'fused'))
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()

    import torch
    from rtx.moe import DEFAULT_MOE_CONFIG, MXFP8MoEConfig
    from rtx.kernels.mxfp8_moe import compile_movement
    from rtx.kernels.mxfp8_moe_dy import compile_routed_dy_quant
    from rtx.kernels.mxfp8_moe_prepared import compile_expert_quant
    from rtx.routing import BalancedRouter, RoutingKernelConfig

    source = Path(__file__).resolve().parents[1]
    cases = json.loads((source / 'benchmarks/balanced_moe_cases.json').read_text())['cases']
    case = next(c for c in cases if c['id'] == args.case_id)
    d, e, k = case['d_model'], case['num_experts'], case['top_k']
    n = sum(args.lengths)
    config = DEFAULT_MOE_CONFIG
    if args.expert_config:
        saved = json.loads(args.expert_config.read_text())
        config = MXFP8MoEConfig.from_dict(saved.get('config', saved))
    if args.joint_warp_baseline:
        config = replace(config, prepared=replace(config.prepared, transpose_mode='warp',
                         backward_preparation='joint', input_preparation='indexed_transpose',
                         quant=replace(config.prepared.quant, scale_layout='row_major', native_scale_store='scalar')))
    prepared = config.prepared
    router_config = RoutingKernelConfig(qb_backend='native_column', metadata_backend='cute', quota_threshold=0)
    if args.router_config:
        saved = json.loads(args.router_config.read_text())
        router_config = RoutingKernelConfig.from_dict(saved.get('config', saved))

    torch.manual_seed(17391)
    if args.fixture:
        fixture = torch.load(args.fixture, map_location='cuda', weights_only=True)
        ids, weights, inverse, dy = (fixture[key] for key in ('ids', 'weights', 'inverse', 'dy'))
        x = fixture['x'] if args.include_x else None
    else:
        r = BalancedRouter(d, e, top_k=k, device='cuda', backend='cute', autotune='off', config=router_config)
        with torch.no_grad():
            r.projection_weight.normal_(std=0.05)
            x = torch.randn(n, d, device='cuda', dtype=torch.bfloat16) * 0.3
            cu = torch.tensor([0, *torch.tensor(args.lengths).cumsum(0).tolist()], device='cuda', dtype=torch.int32)
            ids, weights, inverse = r(x, cu_seqlens=cu)
            dy = torch.randn_like(x).float()
        del r, cu
        if not args.include_x:
            del x
    assert dy.shape == (n, d) and dy.dtype == torch.float32
    assert ids.shape[0] == e and inverse.shape == (n, k)
    c = ids.shape[1]
    cs = (prepared.physical_capacity(c) + 31) // 32 * 32
    slots = torch.arange(e * c, device='cuda').view(e, c)
    valid = (ids >= 0) & (ids < n) & (inverse[ids.clamp(0, n - 1)] == slots[..., None]).any(-1)
    # The exact same map as the shared input gather; this cost is paid by
    # neither measured phase because indexed X reuse already needs it.
    mapping = torch.full((e, cs), -1, device='cuda', dtype=torch.int32)
    mapping[:, :c] = torch.where(valid, ids, -1)
    mapping = mapping.view(e * cs, 1)
    width = (d + 31) // 32 * 32

    def pairs():
        return tuple(torch.empty(shape, device='cuda', dtype=dtype) for shape, dtype in (
            ((e * cs, width), torch.float8_e4m3fn), ((e * cs, width // 32), torch.uint8),
            ((e * d, cs), torch.float8_e4m3fn), ((e * d, cs // 32), torch.uint8),
        ))

    base, candidate = pairs(), pairs()
    staged = None if args.include_x else torch.empty((e * cs, d), device='cuda', dtype=torch.bfloat16)
    fused_config = prepared.quant
    if args.fused_tile:
        fused_config = replace(fused_config, transposed_tile_rows=args.fused_tile[0], transposed_tile_k=args.fused_tile[1])
    if args.fused_padding is not None:
        fused_config = replace(fused_config, transposed_smem_padding=args.fused_padding)
    labels = ('qdy', 'sdy', 'qdyt', 'sdyt')
    common = (prepared.transpose_warp_tokens, prepared.transpose_warp_raster, prepared.quant_grid)
    if args.include_x:
        from rtx.kernels.mxfp8_moe_backward_quant import compile_routed_backward_quant
        from rtx.kernels.mxfp8_moe_warp_quant import compile_indexed_expert_transpose

        base += tuple(torch.empty_like(value) for value in base[2:])
        candidate += tuple(torch.empty_like(value) for value in candidate[2:])
        labels += ('qxt', 'sxt')
        fused = compile_routed_dy_quant(n, d, e, c, cs, prepared.quant, *common)
        xt = compile_indexed_expert_transpose(n, d, e, cs, prepared.quant, *common)
        both = compile_routed_backward_quant(n, d, e, c, cs, fused_config, *common)

        def baseline():
            fused(dy, mapping, weights, *base[:4])
            xt(x, mapping, *base[4:])

        def candidate_call():
            both(dy, mapping, weights, *candidate[:4], x, *candidate[4:])
    else:
        movement = compile_movement(n, d, e, c, k, config.movement_threads, 'slot_dy', 'flat', cs)
        quant = compile_expert_quant(e, ((cs, d, 'row', False), (d, cs, 'transpose', False)),
                                    prepared.quant, prepared.transpose_mode, prepared.quant_grid,
                                    *common[:2], _byte_scales=True)
        fused = compile_routed_dy_quant(n, d, e, c, cs, fused_config, *common)

        def baseline():
            movement(dy, ids, inverse, weights, staged, staged, weights)
            quant(staged, (staged, staged), (base[0], base[2]), (base[1], base[3]))

        def candidate_call():
            fused(dy, mapping, weights, *candidate)

    baseline()
    candidate_call()
    torch.cuda.synchronize()
    for label, a, b in zip(labels, base, candidate):
        assert torch.equal(a.view(torch.uint8), b.view(torch.uint8)), label
    calls = dict(baseline=baseline, fused=candidate_call)
    record = dict(command=sys.argv, source=str(source), device=str(torch.cuda.get_device_properties(0)),
                  torch=torch.__version__, cuda=torch.version.cuda, case=case, lengths=args.lengths,
                  shape=dict(n=n, d=d, experts=e, capacity=c, physical_capacity=cs, top_k=k),
                  expert_config=config.to_dict(), fused_quant_config=asdict(fused_config),
                  router_config=router_config.to_dict(), byte_exact=True, includes_indexed_x=args.include_x,
                  staging_removal_bytes=0 if args.include_x else staged.numel() * staged.element_size(),
                  global_intermediate_traffic_removed_bytes=0 if args.include_x else 3 * staged.numel() * staged.element_size())
    if args.capture:
        start = torch.cuda.cudart().cudaProfilerStart()
        assert start.value == 0, start
        try:
            with torch.cuda.nvtx.range(f'routed_dy/{args.capture}'):
                calls[args.capture]()
                torch.cuda.synchronize()
        finally:
            torch.cuda.cudart().cudaProfilerStop()
        record['capture'] = args.capture
    else:
        from rtx.autotune.preparation_runtime import _warm_measurement_graph

        stream = torch.cuda.Stream()
        graphs = {}
        for label, call in calls.items():
            g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g, stream=stream):
                call()
            graphs[label] = g
        torch.cuda.current_stream().wait_stream(stream)
        live_before = torch.cuda.memory_allocated()
        record['timings'] = {label: [] for label in graphs}
        for round_id in range(args.rounds):
            order = ('baseline', 'fused') if round_id % 2 == 0 else ('fused', 'baseline')
            for label in order:
                g = graphs[label]
                _warm_measurement_graph(g)
                samples = []
                for _ in range(args.samples):
                    start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    start.record()
                    for _ in range(args.sample_iters):
                        g.replay()
                    stop.record()
                    stop.synchronize()
                    samples.append(start.elapsed_time(stop) * 1000 / args.sample_iters)
                record['timings'][label].append(dict(round=round_id, median_us=statistics.median(samples),
                                                      min_us=min(samples), samples_us=samples, wall_time=time.time()))
        record['live_growth_bytes'] = torch.cuda.memory_allocated() - live_before
        record['median_us'] = {label: statistics.median(r['median_us'] for r in rounds)
                               for label, rounds in record['timings'].items()}
        record['speedup'] = record['median_us']['baseline'] / record['median_us']['fused']
        print(json.dumps(dict(shape=record['shape'], median_us=record['median_us'], speedup=record['speedup'])), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
