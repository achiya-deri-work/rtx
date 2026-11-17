"""Focused contracts for compiler-owned MoE buffers and launch-only kernels.

These tests qualify the fixed-config prepared path. First-hit selection has
its own tests and must not silently masquerade as an already-bound plan.
"""
from dataclasses import replace

import pytest
import torch


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="SM120 CUDA required")


def _example(shape=(32, 128, 4, 2, 64), *, seed=847):
    from tests.test_mxfp8_moe import make_route
    from rtx.routing import BalancedRouting

    n, d, e, k, h = shape
    torch.manual_seed(seed)
    x = (torch.randn(n, d, device="cuda", dtype=torch.bfloat16) * .3).requires_grad_()
    gu = (torch.randn(e, 2*h, d, device="cuda", dtype=torch.bfloat16) * .04).requires_grad_()
    down = (torch.randn(e, d, h, device="cuda", dtype=torch.bfloat16) * .04).requires_grad_()
    route = make_route(n, e, k, "cuda")
    route = BalancedRouting(route.expert_ids, route.route_weights.detach().requires_grad_(), route.inverse)
    # Ordinary noncontiguous upstream gradient; no physical transpose is needed.
    dy = torch.randn(d, n, device="cuda", dtype=torch.float32).T
    return (x, gu, down, *route), dy


def _public(config):
    from rtx.moe import mxfp8_routed_experts
    from rtx.routing import BalancedRouting

    def call(x, gu, down, ids, weights, inverse):
        return mxfp8_routed_experts(x, gu, down, BalancedRouting(ids, weights, inverse), config=config)
    return call


def _result(call, operands, dy):
    x, gu, down, _ids, weights, _inverse = operands
    y = call(*operands)
    return y, *torch.autograd.grad(y, (x, gu, down, weights), dy)


@torch.no_grad()
def _check(actual, expected):
    for name, value, reference in zip(("output", "dx", "gate_dw", "down_dw", "route_dw"), actual, expected):
        assert bool(value.isfinite().all()), name
        error = value.float() - reference.float()
        nrms = float(error.norm() / reference.float().norm().clamp_min(1e-12))
        maximum = float(error.abs().max())
        assert nrms < (.003 if name == "output" else .006) or maximum < 2e-7, (name, nrms, maximum)


def test_prepared_launch_schema_has_explicit_mutation_and_no_cpu_output():
    # Importing registration is CPU safe; actual CUDA compilation is lazy.
    import rtx._moe_execution  # noqa: F401

    op = torch.ops.rtx.moe_execute.default
    schema = op._schema
    assert not schema.returns
    buffers = next(a for a in schema.arguments if a.name == "buffers")
    assert buffers.alias_info is not None and buffers.alias_info.is_write
    assert torch.Tag.cudagraph_unsafe not in op.tags


def test_static_moe_policy_metadata_is_constant_under_fullgraph():
    # Torch 2.13 unwraps lru_cache. Mark the underlying immutable decoder, not
    # only the cache wrapper, or JSON's Scanner becomes an unsupported FX op.
    from rtx.moe import DEFAULT_MOE_CONFIG, _selection_spec, _parse_spec
    spec = _selection_spec(DEFAULT_MOE_CONFIG, "off", None, None)
    def metadata_only(x):
        return x + int(_parse_spec(spec)["mode"] == "off")
    compiled = torch.compile(metadata_only, backend="eager", fullgraph=True, dynamic=False)
    torch.testing.assert_close(compiled(torch.ones(1)), torch.full((1,), 2.))


def _fake_operands(mode):
    from torch._subclasses.fake_tensor import FakeTensor
    def fake(shape, dtype):
        return FakeTensor(mode, torch.empty(shape, device="meta", dtype=dtype), torch.device("cuda:0"))
    return (fake((32, 128), torch.bfloat16), fake((4, 128, 128), torch.bfloat16),
            fake((4, 128, 64), torch.bfloat16), fake((4, 16), torch.int64),
            fake((4, 16), torch.float32), fake((32, 2), torch.int64))


def _fake_prepared_graph():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    from rtx.moe import DEFAULT_MOE_CONFIG
    with FakeTensorMode() as mode:
        operands = _fake_operands(mode)
        return make_fx(_public(DEFAULT_MOE_CONFIG), tracing_mode="fake")(*operands)


def test_public_fixed_policy_fullgraph_traces_fake_cuda_without_driver():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from rtx.moe import DEFAULT_MOE_CONFIG
    with FakeTensorMode() as mode:
        operands = _fake_operands(mode)
        compiled = torch.compile(_public(DEFAULT_MOE_CONFIG), backend="eager", fullgraph=True, dynamic=False)
        result = compiled(*operands)
        assert result.shape == operands[0].shape and result.dtype == torch.float32
        assert result.device.type == "cuda"


def test_fixed_prepared_fx_exposes_output_and_scratch_allocations():
    graph = _fake_prepared_graph()
    calls = [n for n in graph.graph.nodes if n.op == "call_function"]
    allocations = [n for n in calls if any(s in str(n.target) for s in ("aten.empty.", "aten.empty_strided.", "aten.empty_like."))]
    launches = [n for n in calls if str(n.target).startswith("rtx.moe_execute")]
    assert len(allocations) >= 2, graph.code
    assert launches, graph.code
    assert not any("mxfp8_routed_experts" in str(n.target) for n in calls), graph.code
    # No materialized concatenation/copy is needed to build the expert inputs.
    assert not any(str(n.target).startswith(("aten.cat.", "aten.clone.", "aten._to_copy.")) for n in calls), graph.code
    for n in allocations:
        value = n.meta.get("val")
        assert value is None or value.device.type == "cuda", graph.code


def _storage_dtype_collisions(values):
    by_storage = {}
    def visit(value):
        if isinstance(value, torch.Tensor):
            key = (str(value.device), value.untyped_storage()._cdata)
            by_storage.setdefault(key, set()).add(value.dtype)
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item)
    visit(values)
    return {key: dtypes for key, dtypes in by_storage.items() if len(dtypes) > 1}


def test_prepared_plan_uses_separately_typed_storage_not_dtype_aliases():
    graph = _fake_prepared_graph()
    values = [node.meta.get("val") for node in graph.graph.nodes]
    assert not _storage_dtype_collisions(values), graph.code
    # E8M0 bits cross this private mutable ABI in canonical uint8 storage.
    # CuTe reinterprets the pointer; Torch must not see an unsupported E8M0
    # functionalization or a view(dtype) alias.
    assert all(value.dtype != torch.float8_e8m0fnu for value in values if isinstance(value, torch.Tensor)), graph.code
    assert not any(str(node.target) == "aten.view.dtype" for node in graph.graph.nodes
                   if node.op == "call_function"), graph.code


def test_dtype_alias_audit_catches_e8m0_uint8_reinterpretation():
    # Torch 2.13 functionalization must not reinterpret mutated scale buffers.
    scales = torch.empty((8,), device="meta", dtype=torch.float8_e8m0fnu)
    assert _storage_dtype_collisions([scales, scales.view(torch.uint8)])
    assert not _storage_dtype_collisions([scales, torch.empty((8,), device="meta", dtype=torch.uint8)])


def test_compact_structural_seeds_preserve_aligned_weight_gradient_tma_outputs():
    from rtx.moe import DEFAULT_MOE_CONFIG
    from rtx.autotune.moe import structural_seeds, rejection

    p = DEFAULT_MOE_CONFIG.prepared
    stages = ("down", "down_dx", "down_dw", "gate_dx", "gate_dw")
    p = replace(p, capacity_multiple=128,
                **{name:replace(getattr(p, name), epilogue="tma") for name in stages})
    initial = replace(DEFAULT_MOE_CONFIG, prepared=p)
    shape = (128, 768, 48, 16, 6, 256)
    assert rejection(initial, shape) is None
    candidates = list(structural_seeds(shape, initial))
    compact = [c for c in candidates[:4] if c.family == "grouped_prepared"
               and c.prepared.capacity_multiple == 1
               and c.prepared.down_dw.epilogue == c.prepared.gate_dw.epilogue == "tma"]
    assert compact, "Compact seed lost coalesced weight-gradient stores"
    for candidate in compact:
        assert rejection(candidate, shape) is None
        assert all(getattr(candidate.prepared, stage).epilogue == "direct"
                   for stage in ("down", "down_dx", "gate_dx"))


@cuda
@pytest.mark.parametrize("shape", [(32, 128, 4, 2, 64), (13, 35, 5, 2, 49)])
@pytest.mark.parametrize("compiled", [False, True])
def test_prepared_execution_all_gradients_and_noncontiguous_upstream(shape, compiled):
    from rtx.moe import DEFAULT_MOE_CONFIG
    from rtx._moe_reference import operational_reference

    operands, dy = _example(shape)
    call = _public(DEFAULT_MOE_CONFIG)
    if compiled:
        call = torch.compile(call, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
    _check(_result(call, operands, dy), operational_reference(*operands, dy))


@cuda
def test_prepared_saves_only_caller_operands_and_outstanding_configs_are_independent():
    from rtx.moe import DEFAULT_MOE_CONFIG
    from rtx._moe_reference import operational_reference
    from rtx.autotune.moe import _choices

    operands, dy = _example()
    first = DEFAULT_MOE_CONFIG
    p = first.prepared
    second = replace(first, prepared=replace(p, backward_preparation="joint",
                                            recompute=replace(p.recompute, glu_grad_store="exchange")))
    saved = []
    with torch.autograd.graph.saved_tensors_hooks(lambda t: (saved.append(t), t)[1], lambda t: t):
        a = _public(first)(*operands)
        b = _public(second)(*operands)
    pointers = {t.data_ptr() for t in operands}
    assert saved and all(t.device.type == "cuda" and t.data_ptr() in pointers for t in saved)
    _choices.clear()
    targets = (operands[0], operands[1], operands[2], operands[4])
    # Reverse-order adjoints must not pick up another forward's plan.
    actual_b = b, *torch.autograd.grad(b, targets, dy)
    actual_a = a, *torch.autograd.grad(a, targets, dy)
    expected = operational_reference(*operands, dy)
    _check(actual_b, expected)
    _check(actual_a, expected)


@cuda
def test_prepared_launch_body_does_not_allocate_eager_tensors(monkeypatch):
    from rtx.moe import DEFAULT_MOE_CONFIG
    from torch._library.custom_ops import OPDEFS

    operands, dy = _example()
    call = _public(DEFAULT_MOE_CONFIG)
    _result(call, operands, dy)
    torch.cuda.synchronize()
    op = OPDEFS["rtx::moe_execute"]
    original = op._backend_fns["cuda"]
    executions = []
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Allocation is hidden inside the opaque MoE launch op")
    def protected(*args, **kwargs):
        with monkeypatch.context() as scoped:
            for name in ("empty", "empty_like", "empty_strided", "zeros", "zeros_like", "cat", "stack"):
                scoped.setattr(torch, name, forbidden)
            executions.append(True)
            return original(*args, **kwargs)
    monkeypatch.setitem(op._backend_fns, "cuda", protected)
    actual = _result(call, operands, dy)
    assert executions and all(bool(t.isfinite().all()) for t in actual)


@cuda
def test_prepared_private_execute_schema_and_fake_contracts():
    from rtx.moe import DEFAULT_MOE_CONFIG, _config_key
    from rtx._moe_execution import execute, layout, allocate

    operands, _dy = _example()
    x, gu, down, ids, weights, inverse = (t.detach() for t in operands)
    shape = (x.shape[0], x.shape[1], gu.shape[0], ids.shape[1], inverse.shape[1], down.shape[2])
    key, dtypes, sizes, views, _outputs = layout(shape, _config_key(DEFAULT_MOE_CONFIG), "forward")
    buffers = allocate(x, dtypes, sizes, views)
    inputs = [x, gu.flatten(0, 1), down.flatten(0, 1), ids, weights, inverse]
    execute(inputs, buffers, key)
    # This mutation-only private launch op is not the public autograd boundary.
    result = torch.library.opcheck(execute, (inputs, buffers, key),
                                   test_utils=("test_schema", "test_faketensor"))
    assert set(result.values()) == {"SUCCESS"}


@cuda
def test_prepared_public_external_graph_replays_mutable_inputs_and_weights():
    from rtx.moe import DEFAULT_MOE_CONFIG
    from rtx._moe_reference import operational_reference

    operands, upstream = _example()
    dy = upstream.contiguous()
    call = torch.compile(_public(DEFAULT_MOE_CONFIG), fullgraph=True, dynamic=False,
                         options={"triton.cudagraphs": False})
    _result(call, operands, dy)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            _result(call, operands, dy)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        captured = _result(call, operands, dy)
    torch.cuda.current_stream().wait_stream(stream)
    try:
        for _ in range(2):
            with torch.no_grad():
                operands[0].add_(.01)
                operands[1].mul_(.97)
                operands[2].mul_(1.01)
            graph.replay()
            torch.cuda.synchronize()
            _check(captured, operational_reference(*operands, dy))
    finally:
        graph.reset()


@cuda
def test_prepared_compiled_execution_stream_isolation_and_live_memory():
    from rtx.moe import DEFAULT_MOE_CONFIG
    from rtx._moe_reference import operational_reference

    operands_a, dy_a = _example(seed=850)
    operands_b, dy_b = _example(seed=851)
    call = torch.compile(_public(DEFAULT_MOE_CONFIG), fullgraph=True, dynamic=False,
                         options={"triton.cudagraphs": False})
    _result(call, operands_a, dy_a)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(streams[0]):
        result_a = _result(call, operands_a, dy_a)
    with torch.cuda.stream(streams[1]):
        result_b = _result(call, operands_b, dy_b)
    for stream in streams:
        stream.synchronize()
    _check(result_a, operational_reference(*operands_a, dy_a))
    _check(result_b, operational_reference(*operands_b, dy_b))
    del result_a, result_b
    allocations = []
    for _ in range(8):
        result = _result(call, operands_a, dy_a)
        del result
        torch.cuda.synchronize()
        allocations.append(torch.cuda.memory_allocated())
    assert max(allocations) - min(allocations) <= 65536, allocations


@cuda
def test_prepared_public_allows_user_side_inductor_cudagraph_replay():
    from rtx.moe import DEFAULT_MOE_CONFIG
    from torch._dynamo.utils import counters
    from torch._inductor.cudagraph_trees import get_manager

    operands, upstream = _example()
    dy = upstream.contiguous()
    before = counters["inductor"]["cudagraph_skips"]
    call = torch.compile(_public(DEFAULT_MOE_CONFIG), fullgraph=True, dynamic=False,
                         options={"triton.cudagraphs": True})
    for _ in range(3):
        torch.compiler.cudagraph_mark_step_begin()
        result = _result(call, operands, dy)
        assert all(bool(t.isfinite().all()) for t in result)
        del result
    torch.cuda.synchronize()
    manager = get_manager(torch.cuda.current_device(), create_if_none_exists=False)
    assert manager is not None and manager.roots, "No user-side Inductor CUDA graph was captured"
    assert counters["inductor"]["cudagraph_skips"] == before


@cuda
def test_first_hit_binds_hot_execution_without_reentering_selector(tmp_path, monkeypatch):
    from rtx.moe import mxfp8_routed_experts
    from rtx.routing import BalancedRouting
    from rtx.autotune.effort import AutotunePolicy
    import rtx.autotune.moe as tuning
    from rtx._moe_reference import operational_reference
    from torch._library.custom_ops import OPDEFS

    operands, dy = _example((16, 128, 4, 2, 64))
    policy = AutotunePolicy(max_trials=2, max_seconds=30, samples=3, calls_per_sample=1,
        warmup=1, confirmation_repeats=0, search_schedule="progressive", candidate_mix="mixed",
        pool_size=16, structural_proposals=16)
    selected = OPDEFS["rtx::mxfp8_routed_experts"]
    original = selected._backend_fns["cuda"]
    cold_calls = []
    def count_selected(*args, **kwargs):
        cold_calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setitem(selected._backend_fns, "cuda", count_selected)
    def fn(x, gu, down, ids, weights, inverse):
        return mxfp8_routed_experts(x, gu, down, BalancedRouting(ids, weights, inverse),
            autotune="balanced", tuning_policy=policy, autotune_cache_dir=str(tmp_path))
    call = torch.compile(fn, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
    first = call(*operands)
    assert len(cold_calls) == 1
    tuning._choices.clear()
    def forbidden(*args, **kwargs):
        raise AssertionError("A bound hot call or outstanding adjoint unexpectedly tuned")
    monkeypatch.setattr(tuning, "tune_moe", forbidden)
    second = call(*operands)
    targets = (operands[0], operands[1], operands[2], operands[4])
    actual_second = second, *torch.autograd.grad(second, targets, dy)
    actual_first = first, *torch.autograd.grad(first, targets, dy)
    assert len(cold_calls) == 1, "Hot execution still enters the opaque selector"
    expected = operational_reference(*operands, dy)
    _check(actual_first, expected)
    _check(actual_second, expected)
