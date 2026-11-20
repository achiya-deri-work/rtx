"""Bounded original-layout MXFP8 producers: exact bytes and changed replay."""

from dataclasses import replace

import pytest
import torch

from tests.test_moe_routed_dy_quant import empty_pairs, quantize_oracle, weighted_oracle


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="SM120 CUDA required")


def _round_up(value, tile, storage):
    return min((value + tile - 1) // tile * tile, storage)


def _config():
    from tests.test_moe_prefix_composition import _composition_config

    cfg = _composition_config()
    return replace(cfg, prepared=replace(cfg.prepared, dy_preparation="fused", weight_preparation="dual_orientation"))


def test_bounded_plan_keeps_arena_layout_and_initializes_maps_for_full_readers():
    from rtx._moe_execution import Builder
    from tests.test_moe_active_prefix import _buffers
    from tests.test_moe_prefix_composition import SHAPE

    base = _config()
    for bound_dy, bound_xt in ((False, False), (True, False), (False, True), (True, True)):
        p = replace(base.prepared, bounded_gather=True, bounded_dy=bound_dy, bounded_xt=bound_xt)
        cfg = replace(base, prepared=p)
        planner = Builder(SHAPE, cfg, "backward").build()
        _, params, operands = next(step[1:] for step in planner.steps if step[1] == "compile_packed_input_gather")
        assert params[-1] is not (bound_dy and bound_xt)
        assert operands[-1] == planner.active_capacity
        for phase in ("forward", "backward"):
            # Bounds are compile-time launch coordinates; no extra arena
            # allocations or different public metadata are needed.
            assert _buffers(SHAPE, phase, cfg)[1:] == _buffers(SHAPE, phase, base)[1:]

    illegal = replace(base.prepared, bounded_gather=True, input_preparation="quantize_once")
    assert "requires indexed input" in illegal.bounded_preparation_rejection("train")


def test_bounded_producer_factories_reject_unqualified_layouts():
    from rtx.configs.moe import DEFAULT_PREPARED_MOE_CONFIG
    from rtx.kernels.mxfp8_moe_buffers import compile_packed_input_gather
    from rtx.kernels.mxfp8_moe_dy import compile_routed_dy_quant
    from rtx.kernels.mxfp8_moe_warp_quant import compile_indexed_expert_transpose

    args = (7, 128, 2, 129, 2, 256)
    with pytest.raises(ValueError, match="row-major"):
        compile_packed_input_gather(*args, scale_layout="mma128", active_m_alignment=64, row_tile_m=64)
    with pytest.raises(ValueError, match="indexed destination"):
        compile_packed_input_gather(*args, active_m_alignment=64, row_tile_m=64, map_tile_ks=(128,))
    with pytest.raises(ValueError, match="row-major"):
        compile_packed_input_gather(*args, copy_bf16=True, active_m_alignment=64, row_tile_m=64)

    native = replace(DEFAULT_PREPARED_MOE_CONFIG.quant, scale_layout="mma128")
    with pytest.raises(ValueError, match="row-major"):
        compile_routed_dy_quant(7, 128, 2, 129, 256, native,
                                active_m_alignment=64, transpose_tile_k=128, row_tile_m=64)
    with pytest.raises(ValueError, match="row-major"):
        compile_indexed_expert_transpose(7, 128, 2, 256, native, active_tile_k=128)


@requires_cuda
@pytest.mark.parametrize("raster,quant_k,lanes,full_map", [
    ("features", 128, 8, False), ("tokens", 64, 4, False), ("features", 128, 8, True),
])
def test_bounded_producers_exact_read_envelopes_and_untouched_poison(raster, quant_k, lanes, full_map, record_property):
    from rtx.configs.moe import DEFAULT_PREPARED_MOE_CONFIG
    from rtx.kernels.mxfp8_moe_active import compile_active_prefix
    from rtx.kernels.mxfp8_moe_buffers import compile_expert_quant, compile_packed_input_gather
    from rtx.kernels.mxfp8_moe_dy import compile_routed_dy_quant
    from rtx.kernels.mxfp8_moe_warp_quant import compile_indexed_expert_transpose

    n, d, e, c, k, cs = 19, 67, 3, 397, 2, 416
    width = _round_up(d, 32, 128)
    sample = torch.Generator(device="cuda").manual_seed(921803)
    # CuTe contracts are contiguous, not necessarily vector-aligned. The
    # odd-width, shifted raw source tensors exercise their scalar fallback.
    x = torch.randn(n*d+1, generator=sample, device="cuda", dtype=torch.bfloat16)[1:].view(n, d)
    dy = torch.randn(n*d+1, generator=sample, device="cuda")[1:].view(n, d)
    ids = torch.zeros(e, c, device="cuda", dtype=torch.int64)
    inverse = torch.full((n, k), -1, device="cuda", dtype=torch.int64)
    weights = torch.randn(e, c, generator=sample, device="cuda")
    prefix = torch.empty((1,), device="cuda", dtype=torch.int32)
    cfg = replace(DEFAULT_PREPARED_MOE_CONFIG.quant, transposed_tile_rows=64, transposed_tile_k=quant_k)
    global_q = torch.empty((n, width), device="cuda", dtype=torch.float8_e4m3fn)
    global_sf = torch.empty((n, width // 32), device="cuda", dtype=torch.uint8)
    quant = compile_expert_quant(e, ((n, d, "row", False, 1),), cfg)
    get_prefix = compile_active_prefix(n, e, c, k)

    fwd = empty_pairs(e, cs, d)[:2]
    bwd = tuple(torch.empty_like(t) for t in fwd)
    mapping = torch.empty((e*cs, 1), device="cuda", dtype=torch.int32)
    weighted = empty_pairs(e, cs, d)
    xt = tuple(torch.empty_like(t) for t in weighted[2:])
    gather_forward = compile_packed_input_gather(n, d, e, c, k, cs, active_m_alignment=96, row_tile_m=64)
    gather_backward = compile_packed_input_gather(n, d, e, c, k, cs, emit_index=True,
        active_m_alignment=64, row_tile_m=64, map_tile_ks=(128, 256), full_map=full_map)
    route_dy = compile_routed_dy_quant(n, d, e, c, cs, cfg, lanes, raster, "expert", 64, 128, 96)
    transpose_x = compile_indexed_expert_transpose(n, d, e, cs, cfg, lanes, raster, "expert", 256)

    def step():
        get_prefix(inverse, prefix)
        quant(x, (x,), (global_q,), (global_sf,))
        gather_forward(global_q, global_sf, x, ids, inverse, *fwd, x, prefix)
        gather_backward(global_q, global_sf, x, ids, inverse, *bwd, mapping, prefix)
        route_dy(dy, mapping, weights, *weighted, prefix)
        transpose_x(x, mapping, *xt, prefix)

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()

    def test_pair(pair, expected, extent, transposed=False):
        for j, (value, reference) in enumerate(zip(pair, expected, strict=True)):
            rows = d if transposed else cs
            view = value.view(torch.uint8).view(e, rows, -1)
            ref = reference.view(torch.uint8).view_as(view)
            if transposed:
                extent_j = extent // (32 if j else 1)
                result, target, suffix = view[..., :extent_j], ref[..., :extent_j], view[..., extent_j:]
            else:
                result, target, suffix = view[:, :extent], ref[:, :extent], view[:, extent:]
            assert torch.equal(result, target), (raster, extent, j, transposed)
            assert bool((suffix == 0xfe).all()), (raster, extent, j, transposed, "wrote suffix")

    cases = {}
    try:
        for count in (161, 0, 1, 397, 33, 97):
            for value in (*fwd, *bwd, *weighted, *xt):
                value.view(torch.uint8).fill_(0xfe)
            mapping.fill_(255)
            ids.zero_()
            inverse.fill_(-1)
            x.mul_(-.875)
            dy.mul_(-.9375)
            weights.mul_(-.8125)
            expected_map = torch.full((e, cs), -1, device="cuda", dtype=torch.int32)
            if count:
                ids[0, 0] = 1
                inverse[1, 0] = 0
                expected_map[0, 0] = 1
                weights[0, 0] = 0
                # The highest real row has zero weight on alternate visits.
                # A stale map cannot substitute zero-weight routes for dummies.
                ids[2, count-1] = 2
                inverse[2, 0] = 2*c+count-1
                expected_map[2, count-1] = 2
                weights[2, count-1] = 0 if count % 3 == 1 else 1
            # ids in all other rows deliberately name an in-range token
            # without a corresponding inverse route, including empty expert 1.
            graph.replay()

            a = _round_up(count, 32, c)
            fwd_rows = _round_up(_round_up(a, 96, cs), 64, cs)
            bwd_rows = _round_up(a, 64, cs)
            dy_rows = _round_up(bwd_rows, 96, cs)
            dyt_cols, xt_cols = _round_up(a, 128, cs), _round_up(a, 256, cs)
            map_rows = cs if full_map else max(bwd_rows, dyt_cols, xt_cols)
            assert int(prefix[0]) == a
            map2d = mapping.view(e, cs)
            assert torch.equal(map2d[:, :map_rows], expected_map[:, :map_rows])
            assert bool((map2d[:, map_rows:] == 255).all())

            index = expected_map.long()
            is_real = index[..., None] >= 0
            gather_ref = tuple(torch.where(is_real, t.view(torch.uint8)[index.clamp_min(0)], 0)
                               for t in quantize_oracle(x.unsqueeze(0), cfg))
            test_pair(fwd, gather_ref, fwd_rows)
            test_pair(bwd, gather_ref, bwd_rows)
            materialized = weighted_oracle(dy, expected_map.reshape(-1, 1), weights, cs)
            test_pair(weighted[:2], quantize_oracle(materialized, cfg), dy_rows)
            test_pair(weighted[2:], quantize_oracle(materialized.transpose(1, 2), cfg), dyt_cols, True)
            gathered_bf16 = torch.where(is_real, x[index.clamp_min(0)], 0)
            test_pair(xt, quantize_oracle(gathered_bf16.transpose(1, 2), cfg), xt_cols, True)
            cases[count] = dict(active=a, fwd=fwd_rows, bwd=bwd_rows, dy=dy_rows, dyt=dyt_cols,
                                xt=xt_cols, map=map_rows)
        torch.cuda.synchronize()
        live = []
        for _ in range(8):
            graph.replay()
            torch.cuda.synchronize()
            live.append(torch.cuda.memory_allocated())
        assert len(set(live)) == 1
        record_property("defined_rows_and_columns", cases)
        record_property("live_bytes", live)
    finally:
        graph.reset()


@requires_cuda
def test_bounded_experts_poisoned_scratch_and_public_fullgraph_changed_input(record_property):
    from rtx._moe_execution import allocate, execute
    from rtx._moe_reference import operational_reference
    from tests.test_moe_active_prefix import _buffers, _public, _route
    from tests.test_moe_prefix_composition import SHAPE, _compare_all, _operands

    n, d, e, c, k, _ = SHAPE
    unbounded = _config()
    bounded = replace(unbounded, prepared=replace(unbounded.prepared, bounded_gather=True,
                                                  bounded_dy=True, bounded_xt=True))
    x, gu, down = _operands()
    candidates = [_route(lengths, c, e=e, k=k)[0]
                  for lengths in ((0, 1, 31, 64, 384), (480, 0, 0, 0, 0), (63, 1, 0, 32, 384))]
    ids, weights, inverse = (value.clone() for value in candidates[0])
    dy = torch.randn(d, n, device="cuda", generator=torch.Generator(device="cuda").manual_seed(192810)).T
    weights.flatten()[inverse[-1]] = 0
    args = (x, gu, down, ids, weights, inverse)
    expected = operational_reference(*args, dy)
    poisoned = []
    for phase in ("forward", "backward"):
        key, dtypes, sizes, views, outputs = _buffers(SHAPE, phase, bounded)
        buffers = allocate(x, dtypes, sizes, views)
        for buffer in buffers:
            if buffer.dtype == torch.float8_e4m3fn:
                buffer.view(torch.uint8).fill_(0x7f)
            elif buffer.is_floating_point():
                buffer.fill_(torch.nan)
            else:
                buffer.fill_(255)
        # Plan launches consume contiguous dY; the public fullgraph probe
        # below separately exercises normalization of a strided upstream.
        launch_args = [x, gu.flatten(0, 1), down.flatten(0, 1), ids, weights, inverse]
        execute([*launch_args, dy.contiguous()] if phase == "backward" else launch_args, buffers, key)
        poisoned.extend(buffers[i].view_as(reference) for i, reference in
                        zip(outputs, expected[1:] if phase == "backward" else expected[:1], strict=True))
        del buffers
    _compare_all(poisoned, expected, record_property, "bounded_poisoned_quantized_fp64")
    del poisoned, expected

    leaves = (x, gu, down, weights)
    for value in leaves:
        value.requires_grad_()
    native = torch.compile(_public(bounded), fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
    control = _public(unbounded)

    def step(fn):
        out = fn(*args)
        return out.detach(), *torch.autograd.grad(out, leaves, dy)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step(native)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = step(native)
    try:
        for i, (new_ids, new_weights, new_inverse) in enumerate(candidates):
            with torch.no_grad():
                ids.copy_(new_ids)
                weights.copy_(new_weights).mul_(.9375)
                inverse.copy_(new_inverse)
                weights.flatten()[inverse[-1]] = 0
                x.mul_(-.75)
                gu.mul_(-.875)
                down.mul_(.875)
                dy.mul_(-.8125)
                reference = operational_reference(*args, dy)
            graph.replay()
            _compare_all(captured, reference, record_property, f"bounded_replay_quantized_fp64_{i}")
            current = step(control)
            assert all(torch.equal(a, b) for a, b in zip(captured, current, strict=True))
            invalid = torch.ones_like(weights, dtype=torch.bool)
            invalid.flatten()[inverse] = False
            assert torch.count_nonzero(captured[-1][invalid]) == 0
            assert torch.count_nonzero(captured[-1].flatten()[inverse[-1]]) > 0
        del current, reference
        live = []
        for _ in range(8):
            graph.replay()
            torch.cuda.synchronize()
            live.append(torch.cuda.memory_allocated())
        assert len(set(live)) == 1
        record_property("bounded_public_live_bytes", live)
    finally:
        graph.reset()
