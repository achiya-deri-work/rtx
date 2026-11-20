"""Packed active-prefix geometry, poison safety, and mutable graph replay."""
from dataclasses import replace
import math

import pytest
import torch


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="SM120/121 CUDA required")
FIELDS = ("output", "dx", "gate_dw", "down_dw", "route_dw")


def _config(*, active=True, epilogue="fused", engine="legacy"):
    from rtx.moe import DEFAULT_MOE_CONFIG

    original = DEFAULT_MOE_CONFIG
    p = original.prepared
    recompute = replace(p.recompute, glu_grad_store="exchange")
    if engine != "legacy":
        recompute = replace(recompute, tile_n=64, stages=1, scale_role="producer", scale_smem_store="scalar",
                            glu_grad_load="interleaved", glu_quant_vec=4, glu_scratch_swizzle="xor_16b")
    return replace(original, prepared=replace(
        p, active_prefix=active, backward_preparation="joint", input_preparation="quantize_once",
        route_gradient_epilogue=epilogue, recompute=recompute, joint_recompute_engine=engine,
    ))


def _buffers(shape, phase, config):
    from rtx._moe_execution import layout
    from rtx.moe import _config_key

    return layout(shape, _config_key(config), phase)


def test_active_prefix_plan_retains_static_shapes_and_zero_boundaries():
    from rtx._moe_execution import plan

    shape = (2000, 384, 32, 352, 4, 128)
    for phase in ("forward", "backward"):
        key, _, _, views, outputs = _buffers(shape, phase, _config())
        old_key, _, _, old_views, old_outputs = _buffers(shape, phase, _config(active=False))
        builder, *_ = plan(key)
        old_builder, *_ = plan(old_key)
        assert [views[i][:2] for i in outputs] == [old_views[i][:2] for i in old_outputs]
        assert sorted((v.shape, str(v.dtype)) for v in builder.buffers) == sorted([
            *((v.shape, str(v.dtype)) for v in old_builder.buffers), ((1,), "torch.int32"),
        ])
        p = builder.active_capacity
        assert builder.steps[0][1:] == ("compile_active_prefix", (2000, 32, 352, 4), (-6, p))
        tails = [s for s in builder.steps if s[1] == "compile_initialize_reduction_tail"]
        if phase == "forward":
            assert not tails
        else:
            assert len(tails) == 1 and tails[0][2] == (32, 352, 128, 64, 128, 128)
            # A 288-row live prefix computes through row 319, but both dW
            # descriptors can read up to 351 in their last 128-K tile.
            assert min(352, math.ceil(288 / 128) * 128) - math.ceil(288 / 64) * 64 == 32
            reductions = [s for s in builder.steps if s[1] == "compile_expert_packed_gemm"
                          and s[2][-1:] == (True,)]
            assert len(reductions) == 2 and all(s[3][-1] == p for s in reductions)
            assert all(builder.steps.index(tails[0]) < builder.steps.index(s) for s in reductions)


def test_active_prefix_disabled_by_default_and_illegal_plan_fails():
    from rtx._moe_execution import Builder, execute
    from rtx.moe import DEFAULT_MOE_CONFIG

    assert not DEFAULT_MOE_CONFIG.prepared.active_prefix
    schema = execute._opoverload._schema
    assert not schema.returns
    assert [arg.name for arg in schema.arguments if arg.alias_info and arg.alias_info.is_write] == ["buffers"]
    config = _config()
    broken = replace(config, prepared=replace(config.prepared, backward_preparation="separate"))
    with pytest.raises(ValueError, match="joint recompute"):
        Builder((2000, 384, 32, 352, 4, 128), broken, "backward").build()


def test_active_prefix_composed_dual_plan_preserves_transposed_tail_init():
    from rtx._moe_execution import Builder
    from rtx.autotune.moe import rejection

    shape = (2000, 384, 32, 352, 4, 128)
    cfg = _config(engine="dual_operand")
    assert rejection(cfg, shape) is None
    builder = Builder(shape, cfg, "backward").build()
    recompute = next(s for s in builder.steps if s[1] == "compile_expert_joint_recompute")
    scrub = next(s for s in builder.steps if s[1] == "compile_initialize_reduction_tail")
    assert recompute[2][-3:-1] == ("dual_operand", 64)
    assert recompute[3][-1][-1] == scrub[3][0] == builder.active_capacity
    assert builder.steps.index(recompute) < builder.steps.index(scrub)


def test_packed_search_offers_both_legal_prefix_policies_early():
    from rtx.autotune import AutotunePolicy
    from rtx.autotune.moe import joint_recompute_seeds, make_moe_adapter, make_moe_tuner
    from rtx.autotune.store import InMemoryTuningStore

    shape = (2000, 384, 32, 352, 4, 128)
    seed = _config(active=False)
    axis = "prepared.active_prefix"
    assert axis not in make_moe_adapter(shape, None, initial=seed).axes
    adapter = make_moe_adapter(shape, None, initial=seed, tags={
        "batch_contract": {"layout": "independent_varlen", "sequences": 4, "segment_alignment": 32},
    })
    assert adapter.axes[axis] == (False, True)
    alternative = adapter.update_fn(adapter.initial_config, axis, True)
    assert adapter.rejection(alternative) is None
    assert alternative == replace(adapter.initial_config, prepared=replace(
        adapter.initial_config.prepared, active_prefix=True,
    ))
    assert adapter.config_id(alternative) != adapter.config_id(adapter.initial_config)
    engine_axis = "prepared.joint_recompute_engine"
    dual = adapter.update_fn(_config(active=False, engine="dual_operand"), engine_axis, "dual_operand")
    combined = adapter.update_fn(dual, axis, True)
    assert combined == adapter.update_fn(adapter.update_fn(dual, axis, True), engine_axis, "dual_operand")
    assert combined.prepared.active_prefix and combined.prepared.joint_recompute_engine == "dual_operand"
    assert adapter.rejection(combined) is None
    assert adapter.update_fn(combined, axis, False) == dual
    legacy = adapter.update_fn(combined, engine_axis, "legacy")
    assert legacy.prepared.active_prefix and adapter.rejection(legacy) is None
    assert adapter.update_fn(legacy, engine_axis, "dual_operand") == combined
    assert any(c.prepared.active_prefix and c.prepared.joint_recompute_engine == "dual_operand"
               for c in joint_recompute_seeds(shape, alternative))
    tuner = make_moe_tuner(adapter, InMemoryTuningStore(), AutotunePolicy(max_trials=6, candidate_mix="mixed"))
    assert any(c.prepared.active_prefix for c in tuner.strategies["structural"].pool[:6])


def _public(config):
    from rtx.moe import mxfp8_routed_experts
    from rtx.routing import BalancedRouting

    def call(x, gu, down, ids, weights, inverse):
        return mxfp8_routed_experts(x, gu, down, BalancedRouting(ids, weights, inverse), config=config)

    return call


def test_active_prefix_fullgraph_fake_cuda():
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    def fake(mode, shape, dtype):
        return FakeTensor(mode, torch.empty(shape, dtype=dtype, device="meta"), torch.device("cuda:0"))

    graphs = []

    def backend(graph, _):
        graphs.append(graph)
        return graph.forward

    with FakeTensorMode() as mode:
        specs = [((480,128),torch.bfloat16),((8,128,128),torch.bfloat16),
                 ((8,128,64),torch.bfloat16),((8,288),torch.int64),
                 ((8,288),torch.float32),((480,2),torch.int64)]
        args = [fake(mode, *spec) for spec in specs]
        result = torch.compile(_public(_config()), backend=backend, fullgraph=True, dynamic=False)(*args)
        assert result.shape == args[0].shape and result.dtype == torch.float32
    assert len(graphs) == 1 and "moe_execute" in graphs[0].code
    assert "_local_scalar_dense" not in graphs[0].code and ".item(" not in graphs[0].code


def _route(lengths, capacity, e=8, k=2):
    """Balanced cyclic route, built independently of the router and device scan."""
    ids = torch.zeros((e, capacity), dtype=torch.int64)
    weights = torch.zeros_like(ids, dtype=torch.float32)
    inverse = torch.empty((sum(lengths), k), dtype=torch.int64)
    row = physical = 0
    for length in lengths:
        for local in range(length):
            for rank in range(k):
                pos = local * k + rank
                expert, slot = pos % e, physical + pos // e
                ids[expert, slot] = row + local
                weights[expert, slot] = (rank + 1) / (k * (k + 1) // 2)
                inverse[row + local, rank] = expert * capacity + slot
        physical += math.ceil(length * k / (e * 32)) * 32
        row += length
    assert physical <= capacity
    return tuple(v.cuda() for v in (ids, weights, inverse)), physical


def _inputs(n):
    generator = torch.Generator().manual_seed(1937)

    def sample(shape, scale):
        return (torch.randn(shape, generator=generator) * scale).bfloat16().cuda()

    return sample((n,128),.3), sample((8,128,128),.04), sample((8,128,64),.04)


def _compare(actual, reference, record_property=None):
    for name, value, expected in zip(FIELDS, actual, reference, strict=True):
        assert value.dtype == expected.dtype and value.shape == expected.shape, name
        assert bool(value.isfinite().all()), name
        delta = value.float() - expected.float()
        nrms = float(delta.norm() / expected.float().norm().clamp_min(1e-12))
        maximum = float(delta.abs().max())
        if record_property is not None:
            record_property(name, {"nrms": nrms, "max_abs": maximum})
        assert nrms < (.003 if name == "output" else .006) or maximum < 2e-7, (name,nrms,maximum)


@cuda
@pytest.mark.parametrize("epilogue,preparation,engine", [
    ("separate", "gather_bf16", "legacy"),
    ("fused", "quantize_once", "legacy"),
    ("fused", "indexed_transpose", "legacy"),
    ("fused", "indexed_transpose", "dual_operand"),
])
@pytest.mark.parametrize("lengths,expected_prefix", [((0,1,31,64,384),192), ((480,0,0,0,0),128)])
@torch.no_grad()
def test_poisoned_oversized_scratch_and_quantized_oracle(lengths, expected_prefix, epilogue, preparation, engine,
                                                       record_property):
    from rtx._moe_execution import allocate, execute, plan
    from rtx._moe_reference import operational_reference

    n = sum(lengths)
    torch.manual_seed(1938)
    c = 288  # Oversized storage with a partial static 128-K tile.
    x, gu, down = _inputs(n)
    (ids, weights, inverse), physical = _route(lengths, c)
    assert physical == expected_prefix
    # A bona fide real route can have zero weight and a nonzero route adjoint.
    # A prefix scan cannot infer liveness from nonzero weights.
    weights.flatten()[inverse[-1]] = 0
    dy = torch.randn(128, n, device="cuda", dtype=torch.float32).T.contiguous()
    expected = operational_reference(x, gu, down, ids, weights, inverse, dy)
    outputs = []
    cfg = _config(epilogue=epilogue, engine=engine)
    cfg = replace(cfg, prepared=replace(cfg.prepared, input_preparation=preparation,
        **({"transpose_mode": "warp", "dy_preparation": "fused_all"}
           if preparation == "indexed_transpose" else {})))
    for phase in ("forward", "backward"):
        key, dtypes, sizes, views, out = _buffers((n,128,8,c,2,64), phase, cfg)
        buffers = allocate(x, dtypes, sizes, views)
        for buffer in buffers:
            if buffer.dtype == torch.float8_e4m3fn:
                buffer.view(torch.uint8).fill_(0x7f)
            elif buffer.is_floating_point():
                buffer.fill_(float("nan"))
            else:
                buffer.fill_(255)
        args = [x, gu.flatten(0,1), down.flatten(0,1), ids, weights, inverse]
        if phase == "backward":
            args.append(dy)
        check_abi = phase == "forward" and preparation == "quantize_once" and expected_prefix == 192
        saved = [arg.clone() for arg in args] if check_abi else []
        execute(args, buffers, key)
        if check_abi:
            # PyTorch 2.13's schema checker calls allclose on every argument,
            # which is not implemented for CUDA E4M3 scratch tensors. Check
            # the declared writes and actual input immutability separately.
            assert all(torch.equal(x, y) for x, y in zip(args, saved, strict=True))
            report = torch.library.opcheck(execute, (args, buffers, key),
                                           test_utils=("test_faketensor",))
            assert set(report.values()) == {"SUCCESS"}
        builder, *_ = plan(key)
        assert int(buffers[builder.active_capacity][0]) == expected_prefix
        outputs.extend(buffers[i] for i in out)
    outputs[2] = outputs[2].view_as(gu)
    outputs[3] = outputs[3].view_as(down)
    _compare(outputs, expected, record_property)


@cuda
@pytest.mark.parametrize("engine", ["legacy", "dual_operand"])
def test_active_prefix_compiled_graph_replays_changed_packed_boundaries(engine, record_property):
    from copy import deepcopy
    from rtx.moe import BalancedMoE, _selection_spec
    from rtx.routing import RoutingKernelConfig

    torch.compiler.reset()
    torch.manual_seed(7532)
    cfg = _config(engine=engine)
    native = BalancedMoE(128, 8, 64, top_k=2, shared_hidden_features=64, num_shared_experts=0, router_dim=16,
        config=cfg, device="cuda", dtype=torch.bfloat16, autotune="off",
        router_options={"config": RoutingKernelConfig(qb_backend="native_column", metadata_backend="cute"),
                        "autotune": "off"})
    with torch.no_grad():
        native.router.projection_weight.normal_(std=.03)
    old = _config(active=False, engine=engine)
    legacy = deepcopy(native)
    legacy.config = old
    legacy._expert_spec = _selection_spec(old, "off", None, None)
    x, _, _ = _inputs(480)
    x.requires_grad_()
    # The bound is unchanged, but the initial profile's prefix is 192 while
    # the one-nonempty-pool replay's prefix is 128. Keep the same graph/address.
    cu_values = [torch.tensor(v, dtype=torch.int32, device="cuda")
                 for v in ([0,0,1,32,96,480], [0,480,480,480,480,480])]
    cu = cu_values[0].clone()
    dy = torch.randn(128, 480, device="cuda", dtype=torch.bfloat16).T
    leaves = (x, *native.parameters())
    compile_options = dict(fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
    compiled = torch.compile(native, **compile_options)
    legacy_compiled = torch.compile(legacy, **compile_options)

    def step():
        y = compiled(x, cu_seqlens=cu)
        return y, *torch.autograd.grad(y, leaves, dy)

    def expected(values):
        # Match compiler lowering, routing, and every parameter. Change only
        # the expert kernel schedule in the reference model.
        out = legacy_compiled(x, cu_seqlens=values)
        return out.detach(), *torch.autograd.grad(out, (x, *legacy.parameters()), dy)

    references = [expected(value) for value in cu_values]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            result = step()
            del result
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=stream):
            captured = step()
        allocations = []
        for iteration in range(8):
            index = iteration % 2
            cu.copy_(cu_values[index])
            graph.replay()
            torch.cuda.synchronize()
            for value, ref in zip(captured, references[index], strict=True):
                assert bool(value.isfinite().all())
                delta = value.detach().float() - ref.float()
                assert float(delta.norm() / ref.float().norm().clamp_min(1e-12)) < .01 or float(delta.abs().max()) < 2e-7
            del value, ref, delta
            torch.cuda.synchronize()
            allocations.append(torch.cuda.memory_allocated())
        record_property("post_replay_live_allocations", allocations)
        assert len(set(allocations)) == 1
    finally:
        graph.reset()
        torch.compiler.reset()
