"""The packed prefix is a runtime bound, not a different quantization recipe."""

from dataclasses import replace
import math

import pytest
import torch

from tests.test_moe_fragment_glu import (
    _buffers as _fragment_buffers,
    _decode_fp64,
    _inputs as _fragment_inputs,
    _oracle as _fragment_oracle,
    _schedule,
)
from tests.test_moe_persistent_dw import CONFIG as DW_CONFIG, _packed


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="SM120 CUDA required")


def _norms(a, b):
    assert a.shape == b.shape
    assert bool(a.isfinite().all())
    assert bool(b.isfinite().all())
    a, b = a.double(), b.double()
    delta = a - b
    return {
        "max_abs": float(delta.abs().max()),
        "rms": float(delta.square().mean().sqrt()),
        "reference_rms": float(b.square().mean().sqrt()),
        "normalized_max": float(delta.abs().max() / b.abs().max().clamp_min(1e-12)),
        "nrms": float(delta.norm() / b.norm().clamp_min(1e-12)),
    }


def test_prefix_factories_reject_unsupported_abi_before_compile():
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.kernels.mxfp8_moe_buffers import compile_expert_packed_gemm
    from rtx.kernels.mxfp8_moe_persistent_dw import compile_persistent_dw
    from rtx.kernels.mxfp8_moe_prepared import compile_expert_packed_gemm as typed

    p = MXFP8Problem(288, 128, 160)
    g = _schedule(64, 128, 2, "consumers")
    for illegal in ({"active_k": True}, {"split": 2}, {"transport": "cooperative"}):
        with pytest.raises(ValueError, match="unsplit TMA forward with static K"):
            compile_expert_packed_gemm(2, p, g, "glu", glu_engine="fragment_direct", **illegal)
    with pytest.raises(ValueError, match="byte-scale"):
        typed(2, p, g, "glu", active_m_alignment=64, glu_engine="fragment_packed")
    with pytest.raises(ValueError, match="single tile per CTA"):
        compile_expert_packed_gemm(2, p, replace(g, tiles_per_cta=2), "glu",
                                   active_m_alignment=64, glu_engine="fragment_packed")
    with pytest.raises(ValueError, match="nonnegative integer"):
        compile_expert_packed_gemm(2, p, g, "glu", active_m_alignment=-32)
    with pytest.raises(ValueError, match="bool"):
        compile_persistent_dw(2, p, DW_CONFIG, active_k=1)
    with pytest.raises(ValueError, match="32-value"):
        compile_persistent_dw(2, replace(p, k=159), DW_CONFIG, active_k=True)


@requires_cuda
@pytest.mark.parametrize("engine,tile_m,tile_n,alignment,role", [
    ("fragment_direct", 64, 64, 128, "producer"),
    ("fragment_packed", 64, 128, 128, "consumers"),
    ("fragment_direct", 128, 128, 64, "consumers"),
    ("fragment_packed", 128, 64, 64, "producer"),
])
def test_fragment_prefix_graph_bytes_and_guarded_rows(engine, tile_m, tile_n, alignment, role, record_property):
    from rtx.configs.moe import DEFAULT_PREPARED_MOE_CONFIG
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.kernels.mxfp8_moe_buffers import compile_expert_packed_gemm, compile_expert_quant

    e, m, h, k = 2, 288, 65, 160
    x, w = _fragment_inputs(e, m, h, k)
    inputs, actual = _fragment_buffers(e, m, h, k)
    qx, qw, sx, sw = inputs
    qh, sh = (value.view(e, m, -1).view(torch.uint8) for value in actual)
    _, expected = _fragment_buffers(e, m, h, k)
    eq, es = (value.view(e, m, -1).view(torch.uint8) for value in expected)
    quant = compile_expert_quant(e, ((m, k, "row", False), (2*h, k, "row", True)),
                                 DEFAULT_PREPARED_MOE_CONFIG.quant)
    quant(x, (x, w), (qx, qw), (sx, sw))
    p, g = MXFP8Problem(m, 2*h, k), _schedule(tile_m, tile_n, 1 if role == "producer" else 2, role)
    static = compile_expert_packed_gemm(e, p, g, "glu", glu_engine=engine)
    active = compile_expert_packed_gemm(e, p, g, "glu", active_m_alignment=alignment, glu_engine=engine)
    prefix = torch.tensor([128], dtype=torch.int32, device="cuda")
    active(*inputs, *actual, prefix)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        active(*inputs, *actual, prefix)
    results = {}
    for size in (192, 0, 32, 288, 96, 128):
        # Rebuild operands at the same addresses, with changes to both the
        # bound and weights. Poison suffix rows that no active CTA owns.
        w.mul_(-.875)
        quant(x, (x, w), (qx, qw), (sx, sw))
        static(*inputs, *expected)
        oq, os = _fragment_oracle(x, w, e, m, h, k)
        rows = min(m, math.ceil(math.ceil(size / alignment) * alignment / tile_m) * tile_m)
        qx.view(e, m, k).view(torch.uint8)[:, rows:].fill_(0x7f)
        sx.view(e, m, -1)[:, rows:].fill_(0xff)
        qh.fill_(0xfe)
        sh.fill_(0xfe)
        prefix.fill_(size)
        graph.replay()
        assert torch.equal(qh[:, :rows], eq[:, :rows])
        assert torch.equal(sh[:, :rows], es[:, :rows])
        assert bool((qh[:, rows:] == 0xfe).all()) and bool((sh[:, rows:] == 0xfe).all())
        if rows:
            oq = oq.view(e, m, -1)[:, :rows]
            os = os.view(e, m, -1)[:, :rows]
            metrics = _norms(_decode_fp64(qh[:, :rows].view(torch.float8_e4m3fn), sh[:, :rows]),
                             _decode_fp64(oq.view(torch.float8_e4m3fn), os))
            assert metrics["nrms"] < .003, metrics
            results[size] = metrics
    # The oracle tensors are retained in the loop locals at nonempty sizes;
    # drain validation allocations with a replay before measuring live bytes.
    graph.replay()
    torch.cuda.synchronize()
    live = []
    for _ in range(8):
        graph.replay()
        torch.cuda.synchronize()
        live.append(torch.cuda.memory_allocated())
    record_property("fragment_quantized_fp64", results)
    record_property("live_bytes", live)
    assert len(set(live)) == 1
    graph.reset()


@requires_cuda
@pytest.mark.parametrize("axis,tiles,m,n,stages,scale_schedule", [
    ("m", 2, 131, 79, 3, "after_wait"),
    ("m", 3, 207, 71, 2, "before_wait"),
    ("n", 2, 73, 149, 2, "before_wait"),
    ("n", 3, 78, 229, 2, "after_wait"),
])
def test_persistent_prefix_graph_matches_quantized_oracle(axis, tiles, m, n, stages, scale_schedule,
                                                          record_property):
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.kernels.mxfp8_moe_persistent_dw import compile_persistent_dw

    e, k = 2, 288
    rng = torch.Generator(device="cuda").manual_seed(912729)
    qa, sa = _packed(e, m, k, rng)
    qb, sb = _packed(e, n, k, rng)
    hold = torch.full((e*m*n + 32,), 73, device="cuda", dtype=torch.bfloat16)
    out = hold[16:-16].view(e*m, n)
    p = MXFP8Problem(m, n, k)
    g = replace(DW_CONFIG, stages=stages, scale_schedule=scale_schedule)
    launch = compile_persistent_dw(e, p, g, axis, tiles, True)
    prefix = torch.tensor([k], device="cuda", dtype=torch.int32)
    launch(qa, qb, sa, sb, out, prefix)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(qa, qb, sa, sb, out, prefix)

    def decode(q, s, rows, active):
        blocks = q.view(e, rows, k)[:, :, :active].double().unflatten(-1, (active // 32, 32))
        return (blocks * torch.exp2(s.view(e, rows, k//32)[:, :, :active//32].double() - 127)
                .unsqueeze(-1)).flatten(-2)

    results = {}
    for size in (160, 0, 128, 32, 288, 256):
        qa_new, sa_new = _packed(e, m, k, rng)
        qb_new, sb_new = _packed(e, n, k, rng)
        if size:
            ref = (decode(qa_new, sa_new, m, size) @ decode(qb_new, sb_new, n, size).mT).bfloat16().flatten(0, 1)
        else:
            ref = torch.zeros_like(out)
        tail = min(k, math.ceil(size / 128) * 128)
        qa.copy_(qa_new)
        qb.copy_(qb_new)
        sa.copy_(sa_new)
        sb.copy_(sb_new)
        # Only A is scrubbed in this tile; B is finite but intentionally
        # nonzero. The skipped physical suffix would corrupt either operand.
        qa.view(torch.uint8)[:, size:tail].zero_()
        sa[:, size//32:tail//32].zero_()
        for q, s in ((qa, sa), (qb, sb)):
            q.view(torch.uint8)[:, tail:].fill_(0x7f)
            s[:, tail//32:].fill_(0xff)
        prefix.fill_(size)
        out.fill_(torch.nan)
        graph.replay()
        metrics = _norms(out, ref)
        assert metrics["nrms"] < .002, metrics
        assert metrics["max_abs"] <= .004 * float(ref.float().abs().max()), metrics
        assert bool((hold[:16] == 73).all()) and bool((hold[-16:] == 73).all())
        results[size] = metrics
    graph.replay()
    torch.cuda.synchronize()
    live = []
    for _ in range(8):
        graph.replay()
        torch.cuda.synchronize()
        live.append(torch.cuda.memory_allocated())
    record_property("persistent_quantized_fp64", results)
    record_property("live_bytes", live)
    assert len(set(live)) == 1
    graph.reset()


def _composition_config(*, active=True, mode="fragment_packed", input_preparation="indexed_transpose"):
    from tests.test_moe_active_prefix import _config

    cfg = _config(active=active, engine="dual_operand")
    p = cfg.prepared
    if input_preparation == "indexed_transpose":
        p = replace(p, transpose_mode="warp", dy_preparation="fused_all")
    return replace(cfg, prepared=replace(
        p, input_preparation=input_preparation, gate_up=_schedule(128, 64, 1, "producer"),
        forward_glu_engine=mode, gate_dw_engine="persistent", down_dw_engine="persistent",
        gate_dw=replace(DW_CONFIG, scale_schedule="before_wait"), gate_dw_reuse_axis="n", gate_dw_reuse_tiles=3,
        down_dw=replace(DW_CONFIG, scale_schedule="after_wait"), down_dw_reuse_axis="m", down_dw_reuse_tiles=2,
    ))


SHAPE = (480, 192, 8, 288, 2, 64)


def test_composed_prefix_plan_reuses_arenas_and_passes_only_internal_bound():
    from rtx._moe_execution import Builder

    train = _composition_config()
    static = _composition_config(active=False)
    for phase in ("forward", "backward"):
        plan = Builder(SHAPE, train, phase).build()
        baseline = Builder(SHAPE, static, phase).build()
        assert [(plan.buffers[i].shape, plan.buffers[i].dtype) for i in plan.outputs] == [
            (baseline.buffers[i].shape, baseline.buffers[i].dtype) for i in baseline.outputs]
        assert sorted((buf.shape, str(buf.dtype)) for i, buf in enumerate(plan.buffers)
                      if i != plan.active_capacity) == sorted((buf.shape, str(buf.dtype)) for buf in baseline.buffers)
        bound = plan.buffers[plan.active_capacity]
        assert bound.shape == (1,) and bound.dtype == torch.int32 and not bound.output
        if phase == "forward":
            glu = next(s for s in plan.steps if s[1] == "compile_expert_packed_gemm" and s[2][3] == "glu")
            assert glu[2][-3:] == (train.prepared.down.tile_m, False, "fragment_packed")
            assert glu[3][-1] == plan.active_capacity
        else:
            scrub = next(s for s in plan.steps if s[1] == "compile_initialize_reduction_tail")
            assert scrub[2] == (8, 288, 64, 64, 128, 128)
            assert scrub[3][0] == plan.active_capacity
            reductions = [(i, s) for i, s in enumerate(plan.steps) if s[1] == "compile_persistent_dw"]
            assert len(reductions) == 2 and all(i > plan.steps.index(scrub) for i, _ in reductions)
            assert all(s[2][-1] is True and s[3][-1] == plan.active_capacity for _, s in reductions)


def _operands():
    n, d, e, _, _, h = SHAPE
    rng = torch.Generator().manual_seed(354378)

    def sample(shape, scale):
        return (torch.randn(shape, generator=rng) * scale).bfloat16().cuda()

    return sample((n, d), .3), sample((e, 2*h, d), .04), sample((e, d, h), .04)


def _compare_all(actual, expected, record_property, case):
    from tests.test_moe_active_prefix import FIELDS

    metrics = {}
    for field, out, ref in zip(FIELDS, actual, expected, strict=True):
        assert out.dtype == ref.dtype
        m = _norms(out, ref)
        metrics[field] = m
        assert m["nrms"] < (.003 if field == "output" else .006) or m["max_abs"] < 2e-7, (field, m)
    record_property(case, metrics)


@requires_cuda
@pytest.mark.parametrize("input_preparation,mode,lengths", [
    ("quantize_once", "fragment_direct", (0, 1, 31, 64, 384)),
    ("indexed_transpose", "fragment_packed", (0, 1, 31, 64, 384)),
    ("indexed_transpose", "fragment_packed", (480, 0, 0, 0, 0)),
])
@torch.no_grad()
def test_composed_prefix_poisoned_arenas_all_grads(input_preparation, mode, lengths, record_property):
    from rtx._moe_execution import allocate, execute, plan
    from rtx._moe_reference import operational_reference
    from tests.test_moe_active_prefix import _buffers, _route

    n, d, e, c, k, _ = SHAPE
    x, gu, down = _operands()
    (ids, weights, inverse), prefix = _route(lengths, c, e=e, k=k)
    weights.flatten()[inverse[-1]] = 0
    rng = torch.Generator().manual_seed(467129)
    dy = torch.randn((d, n), generator=rng).cuda().T.contiguous()
    expected = operational_reference(x, gu, down, ids, weights, inverse, dy)
    cfg = _composition_config(mode=mode, input_preparation=input_preparation)
    outputs = []
    for phase in ("forward", "backward"):
        key, dtypes, sizes, views, indices = _buffers(SHAPE, phase, cfg)
        buffers = allocate(x, dtypes, sizes, views)
        for buffer in buffers:
            if buffer.dtype == torch.float8_e4m3fn:
                buffer.view(torch.uint8).fill_(0x7f)
            elif buffer.is_floating_point():
                buffer.fill_(torch.nan)
            else:
                buffer.fill_(255)
        args = [x, gu.flatten(0, 1), down.flatten(0, 1), ids, weights, inverse]
        if phase == "backward":
            args.append(dy)
        execute(args, buffers, key)
        builder, *_ = plan(key)
        assert int(buffers[builder.active_capacity][0]) == prefix
        outputs.extend(buffers[i] for i in indices)
    outputs[2] = outputs[2].view_as(gu)
    outputs[3] = outputs[3].view_as(down)
    _compare_all(outputs, expected, record_property, "combined_poisoned_quantized_fp64")


@requires_cuda
def test_compiled_composed_prefix_same_graph_changed_routes_weights_and_upstream(record_property):
    from rtx._moe_reference import operational_reference
    from tests.test_moe_active_prefix import _public, _route

    n, d, e, c, k, _ = SHAPE
    x, gu, down = (x.requires_grad_() for x in _operands())
    metadata = [_route(lengths, c, e=e, k=k) for lengths in ((0, 1, 31, 64, 384), (480, 0, 0, 0, 0))]
    ids, weights, inverse = [arg.clone() for arg in metadata[0][0]]
    weights.requires_grad_()
    dy = torch.randn(d, n, device="cuda", generator=torch.Generator(device="cuda").manual_seed(81977)).T
    args = (x, gu, down, ids, weights, inverse)
    leaves = (x, gu, down, weights)
    active = torch.compile(_public(_composition_config()), fullgraph=True, dynamic=False,
                           options={"triton.cudagraphs": False})
    static = _public(_composition_config(active=False))

    def step(fn):
        out = fn(*args)
        return out, *torch.autograd.grad(out, leaves, dy)

    for _ in range(3):
        warmed = step(active)
        del warmed
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = step(active)
    try:
        for iteration in range(4):
            (current_ids, current_weights, current_inverse), _ = metadata[iteration % 2]
            with torch.no_grad():
                ids.copy_(current_ids)
                weights.copy_(current_weights).mul_(.9375)
                inverse.copy_(current_inverse)
                weights.flatten()[inverse[-1]] = 0
                gu.mul_(-.875)
                down.mul_(.875)
                dy.mul_(-.8125)
            graph.replay()
            expected = operational_reference(*args, dy)
            _compare_all(captured, expected, record_property, f"combined_replay_quantized_fp64_{iteration}")
            baseline = step(static)
            for out, ref in zip(captured, baseline, strict=True):
                assert torch.equal(out, ref), _norms(out, ref)
            del expected, baseline, out, ref
        graph.replay()
        torch.cuda.synchronize()
        live = []
        for _ in range(8):
            graph.replay()
            torch.cuda.synchronize()
            live.append(torch.cuda.memory_allocated())
        record_property("live_bytes", live)
        assert len(set(live)) == 1
    finally:
        graph.reset()
        torch.compiler.reset()
