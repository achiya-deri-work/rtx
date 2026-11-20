"""Compiler-visible router storage and immutable saved-state execution.

CPU checks establish framework contracts only. CUDA checks compare every
transport derivative against independent double-precision finite Sinkhorn.
"""
from dataclasses import replace
import gc

import pytest
import torch

from rtx.routing import RoutingKernelConfig, SinkhornKernelConfig


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="SM120 CUDA required")


def _config():
    return RoutingKernelConfig(
        forward=SinkhornKernelConfig(64, 1, 1),
        backward=SinkhornKernelConfig(256, 4, 8),
    )


def _selection_spec(cache_dir=None):
    from rtx.routing import BalancedRouter
    # Use the real frontend's workload identity, not an abbreviated test spec
    # that only works under fake execution and bypasses cache-contract fields.
    return BalancedRouter(16, 16, top_k=4, router_dim=8, sinkhorn_iterations=3,
                          config=_config(), autotune="cache", autotune_cache_dir=cache_dir,
                          dtype=torch.float32)._selection_spec


def _independent_transport(scores, k, iterations):
    # Form normalized transport directly. Do not call a production planner,
    # CuTe kernel or its saved-history/backward decomposition.
    import math
    value = scores
    for _ in range(iterations):
        value = value - torch.logsumexp(value, 1, keepdim=True) + math.log(k)
        value = value - torch.logsumexp(value, 0, keepdim=True) + math.log(scores.shape[0]*k/scores.shape[1])
    return value


def _fake_cuda(mode, shape, *, requires_grad=False):
    from torch._subclasses.fake_tensor import FakeTensor
    return FakeTensor(mode, torch.empty(shape, device="meta", dtype=torch.float32,
                                       requires_grad=requires_grad), torch.device("cuda:0"))


def _call(scores, *, k=4, iterations=3, config=None):
    from rtx._routing_execution import sinkhorn
    selected = _config() if config is None else config
    return sinkhorn(scores, k, iterations, tuple(selected.to_token()))


def test_sinkhorn_execution_schema_is_launch_only_and_fake_has_no_cpu_output():
    import rtx._routing_execution  # noqa: F401
    schema = torch.ops.rtx_routing.sinkhorn_execute.default._schema
    assert not schema.returns
    buffers = next(arg for arg in schema.arguments if arg.name == "buffers")
    assert buffers.alias_info is not None and buffers.alias_info.is_write
    assert torch.Tag.cudagraph_unsafe not in torch.ops.rtx_routing.sinkhorn_execute.default.tags


def test_fixed_sinkhorn_fake_cuda_fullgraph_without_driver():
    from torch._subclasses.fake_tensor import FakeTensorMode
    with FakeTensorMode() as mode:
        scores = _fake_cuda(mode, (68, 16), requires_grad=True)
        call = torch.compile(_call, backend="eager", fullgraph=True, dynamic=False)
        out, history = call(scores)
        assert out.shape == scores.shape and out.dtype == torch.float32
        assert out.device.type == history.device.type == "cuda"
        assert history.numel() == 2*3*scores.numel()
        assert out.requires_grad and not history.requires_grad


def test_complete_fixed_router_fullgraph_fake_cuda_has_static_policy_and_visible_outputs():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from rtx.routing import BalancedRouter
    router = BalancedRouter(16, 16, top_k=4, router_dim=8, config=_config(),
                            dtype=torch.bfloat16)
    with FakeTensorMode() as mode:
        for name, parameter in list(router.named_parameters()):
            fake = _fake_cuda(mode, tuple(parameter.shape), requires_grad=True)
            # Router parameters in this check use FP32 deliberately; its
            # ordinary compute recipe promotes the sensitive affinity path.
            setattr(router, name, torch.nn.Parameter(fake))
        scores = _fake_cuda(mode, (65, 16), requires_grad=True)
        captured = []
        def backend(graph, _inputs):
            captured.append(graph)
            return graph.forward
        compiled = torch.compile(router, backend=backend, fullgraph=True, dynamic=False)
        route = compiled(scores)
        assert route.expert_ids.shape == route.route_weights.shape == (16, 17)
        assert route.inverse.shape == (65, 4)
        assert route.route_weights.requires_grad
        assert route.expert_ids.dtype == route.inverse.dtype == torch.int64
    assert len(captured) == 1
    # Dynamo represents custom autograd as nested forward/backward FX bodies;
    # inspect those too instead of demanding the launch live in the root.
    code = "\n".join(module.code for module in captured[0].modules()
                     if isinstance(module, torch.fx.GraphModule))
    assert "sinkhorn_execute" in code and "hard_schedule_into" in code
    assert "selected_sinkhorn" not in code and "selected_hard" not in code


def test_sinkhorn_fx_exposes_allocations_without_cpu_token_or_numeric_torch_body():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    with FakeTensorMode() as mode:
        graph = make_fx(lambda scores: _call(scores), tracing_mode="fake")(_fake_cuda(mode, (68, 16)))
    calls = [node for node in graph.graph.nodes if node.op == "call_function"]
    launches = [node for node in calls if str(node.target) == "rtx_routing.sinkhorn_execute.default"]
    allocations = [node for node in calls if str(node.target).startswith(
        ("aten.empty.", "aten.empty_like.", "aten.empty_strided."))]
    assert len(launches) == 1, graph.code
    assert len(allocations) >= 2, graph.code
    forbidden = ("aten.cat.", "aten.permute.", "aten.exp.", "aten.logsumexp.",
                 "aten.clone.", "aten._to_copy.", "aten._local_scalar_dense.")
    assert not any(str(node.target).startswith(forbidden) for node in calls), graph.code
    for node in allocations:
        value = node.meta.get("val")
        assert value is not None and value.device.type == "cuda", graph.code


def test_fake_cuda_cold_binding_transitions_only_its_shape_to_static_execution():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch._dynamo.testing import CompileCounterWithBackend
    from rtx import _routing_autotune as startup
    from rtx import _moe_selection as selection
    spec = _selection_spec()
    counter = CompileCounterWithBackend("eager")
    call = torch.compile(lambda scores: startup.dispatch_sinkhorn(scores, 4, 3, spec, False, 65)[0],
                         backend=counter, fullgraph=True, dynamic=False)
    selection.clear_selection_cells()
    try:
        with FakeTensorMode() as mode:
            scores = _fake_cuda(mode, (68, 16))
            call(scores)
            assert counter.frame_count == 1
            startup.bind_selected_config(scores, 4, 3, spec, False, 65, _config())
            call(scores)
            assert counter.frame_count == 2
            startup.bind_selected_config(_fake_cuda(mode, (72, 16)), 4, 3, spec, False, 70, _config())
            call(scores)
            assert counter.frame_count == 2
        assert "selected_sinkhorn" in counter.graphs[0].code
        assert "selected_sinkhorn" not in counter.graphs[1].code
        assert "sinkhorn_execute" in counter.graphs[1].code
    finally:
        selection.clear_selection_cells()


def test_hard_dispatch_exposes_output_and_private_mutation_schema():
    from torch._subclasses.fake_tensor import FakeTensorMode
    from torch.fx.experimental.proxy_tensor import make_fx
    from rtx._routing_hard import dispatch_hard
    op = torch.ops.rtx_routing.hard_schedule_into.default
    assert not op._schema.returns
    out = next(arg for arg in op._schema.arguments if arg.name == "out")
    assert out.alias_info is not None and out.alias_info.is_write
    with FakeTensorMode() as mode:
        graph = make_fx(lambda scores: dispatch_hard(scores, 4, 5, 64), tracing_mode="fake")(
            _fake_cuda(mode, (68, 16)))
    calls = [node for node in graph.graph.nodes if node.op == "call_function"]
    assert sum(str(node.target) == "rtx_routing.hard_schedule_into.default" for node in calls) == 1
    assert any(str(node.target).startswith("aten.empty.") for node in calls), graph.code
    assert not any(str(node.target).startswith(("aten.exp.", "aten.topk.", "aten.clone.")) for node in calls)


def test_sinkhorn_layout_is_bounded_metadata_and_rejects_index_overflow():
    from rtx import _routing_execution as execution
    config = tuple(_config().to_token())
    key, sizes = execution.layout((68, 16), 4, 3, config, "forward")
    assert isinstance(key, str) and sizes == (68*16, 2*3*68*16, 68, 16)
    _, backward_sizes = execution.layout((68, 16), 4, 3, config, "backward")
    assert backward_sizes == (68*16, 68*16, 1)
    assert execution._COMPILED.max_entries == 128
    for shape, k, iterations in (((13, 16), 4, 3), ((68, 16), 17, 3),
                                  ((68, 16), 4, 0), ((1_000_000, 128), 8, 16)):
        with pytest.raises(ValueError):execution.layout(shape, k, iterations, config, "forward")


def test_routing_refinement_and_cleanup_keep_other_selection_namespaces(monkeypatch):
    import json
    from collections import OrderedDict
    from rtx import _routing_autotune as startup
    from rtx import _moe_selection as selection
    from rtx.runtime import BoundedCache
    monkeypatch.setattr(selection, "_CELLS", {})
    monkeypatch.setattr(selection, "_RECENCY", OrderedDict())
    monkeypatch.setattr(startup, "_selections", BoundedCache(256))
    scores = torch.empty(68, 16)
    spec = _selection_spec()
    unrelated = ((16, 128), "cpu", "moe-policy", True)
    selection.bind_selected(*unrelated, "unchanged")
    context = startup.binding_identity(scores, 4, 3, spec, True, 65)
    startup.bind_selected_config(scores, 4, 3, spec, True, 65, _config())
    original = selection.bound_config(*context)
    refined = replace(_config(), backward=SinkhornKernelConfig(128, 2, 4))
    startup.publish_selected_config(scores, 4, 3, spec, True, 65, refined)
    assert tuple(json.loads(original)) == tuple(_config().to_token())
    assert tuple(json.loads(selection.bound_config(*context))) == tuple(refined.to_token())
    assert startup._selections[(str(scores.device), tuple(scores.shape), spec, True, 65)] == refined
    selection.clear_selection_cells(namespace="routing:")
    # Simulate only the device context guard on CPU. This path must return
    # cached host metadata before reaching capture checks or any CUDA work.
    from contextlib import nullcontext
    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "device", lambda *_args, **_kwargs: nullcontext())
        assert startup._resolve(scores, spec, True, 65) == refined
    startup.bind_selected_config(scores, 4, 3, spec, True, 65, refined)
    assert startup._clear_runtime_caches(preserve_compiled=True) == {}
    assert selection.bound_config(*unrelated) == "unchanged"
    startup._clear_runtime_caches()
    assert selection.bound_config(*unrelated) == "unchanged"
    assert selection.bound_config(*context) is None


def _check(actual, expected, *, rtol, atol):
    assert bool(actual.isfinite().all())
    torch.testing.assert_close(actual.double(), expected, rtol=rtol, atol=atol)


@cuda
@pytest.mark.parametrize("shape,k", [((68, 16), 4), ((14, 7), 3)])
@pytest.mark.parametrize("compiled", [False, True])
def test_sinkhorn_visible_storage_independent_forward_and_backward(shape, k, compiled, record_property):
    torch.manual_seed(1771)
    scores = (torch.randn(shape, device="cuda")*.7-2).requires_grad_()
    reference = scores.detach().double().requires_grad_()
    upstream = torch.randn(shape[::-1], device="cuda").T
    call = lambda value: _call(value, k=k)
    if compiled:
        call = torch.compile(call, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
    actual, history = call(scores)
    expected = _independent_transport(reference, k, 3)
    got_grad, = torch.autograd.grad(actual, (scores,), upstream)
    want_grad, = torch.autograd.grad(expected, (reference,), upstream.double())
    assert history.numel() == 2*3*scores.numel() and not history.requires_grad
    _check(actual, expected, rtol=5e-5, atol=5e-6)
    _check(got_grad, want_grad, rtol=5e-4, atol=8e-6)
    for name, value, oracle in (("output", actual, expected), ("gradient", got_grad, want_grad)):
        error = value.double()-oracle
        rms = error.square().mean().sqrt()
        record_property(name+"_max_abs", error.abs().max().item())
        record_property(name+"_normalized_rms", (rms/oracle.square().mean().sqrt().clamp_min(1e-12)).item())
    record_property("seed", 1771)


@cuda
def test_sinkhorn_outstanding_forwards_keep_independent_static_backward_policy(monkeypatch):
    from rtx import _routing_autotune as startup
    torch.manual_seed(1772)
    other = replace(_config(), backward=SinkhornKernelConfig(128, 2, 4))
    operands = [(torch.randn(68, 16, device="cuda")*.5).requires_grad_() for _ in range(2)]
    upstream = torch.randn(68, 16, device="cuda")
    outputs = [_call(value, config=config)[0] for value, config in zip(operands, (_config(), other))]
    startup._clear_runtime_caches()
    monkeypatch.setattr(startup, "_resolve", lambda *_a, **_k: pytest.fail("Backward performed mutable policy selection"))
    for scores, output in zip(operands, outputs):
        reference = scores.detach().double().requires_grad_()
        expected = _independent_transport(reference, 4, 3)
        actual, = torch.autograd.grad(output, (scores,), upstream)
        want, = torch.autograd.grad(expected, (reference,), upstream.double())
        _check(actual, want, rtol=5e-4, atol=8e-6)


@cuda
def test_private_sinkhorn_opcheck_and_hot_body_has_no_eager_allocations(monkeypatch):
    from rtx import _routing_execution as execution
    torch.manual_seed(1775)
    scores = torch.randn(68, 16, device="cuda")
    key, sizes = execution.layout(tuple(scores.shape), 4, 3, tuple(_config().to_token()), "forward")
    buffers = execution.allocate(scores, sizes)
    inputs = [scores.flatten()]
    execution.sinkhorn_execute(inputs, buffers, key)
    checked = torch.library.opcheck(execution.sinkhorn_execute, (inputs, buffers, key),
                                   test_utils=("test_schema", "test_faketensor"))
    assert set(checked.values()) == {"SUCCESS"}
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Hot Sinkhorn custom op allocated an eager tensor")
    with monkeypatch.context() as patch:
        for name in ("empty", "empty_like", "zeros", "zeros_like", "ones", "tensor"):
            patch.setattr(torch, name, forbidden)
        execution.sinkhorn_execute(inputs, buffers, key)
    torch.cuda.synchronize()
    expected = _independent_transport(scores.double(), 4, 3)
    _check(buffers[0].view_as(scores), expected, rtol=5e-5, atol=5e-6)


@cuda
def test_real_cold_then_bound_dispatch_selects_once_and_pending_backwards_survive(tmp_path, monkeypatch):
    from rtx import _routing_autotune as startup
    from rtx import _moe_selection as selection
    torch.manual_seed(1776)
    spec = _selection_spec(tmp_path)
    selection.clear_selection_cells()
    original = startup._resolve
    resolved = []
    def counted(scores, *args, **kwargs):
        resolved.append(tuple(scores.shape))
        return original(scores, *args, **kwargs)
    monkeypatch.setattr(startup, "_resolve", counted)
    call = torch.compile(lambda scores: startup.dispatch_sinkhorn(scores, 4, 3, spec, True, 65)[0],
                         fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
    values = [torch.randn(68, 16, device="cuda", requires_grad=True) for _ in range(2)]
    outputs = [call(scores) for scores in values]
    assert resolved == [(68, 16)], "Hot dispatch still enters the selection operator"
    selection.clear_selection_cells()
    monkeypatch.setattr(startup, "_resolve", lambda *_a, **_k: pytest.fail("Pending backward selected again"))
    upstream = torch.randn(68, 16, device="cuda")
    for scores, output in zip(values, outputs):
        reference = scores.detach().double().requires_grad_()
        expected = _independent_transport(reference, 4, 3)
        actual, = torch.autograd.grad(output, (scores,), upstream)
        want, = torch.autograd.grad(expected, (reference,), upstream.double())
        _check(output, expected, rtol=5e-5, atol=5e-6)
        _check(actual, want, rtol=5e-4, atol=8e-6)


@cuda
def test_private_hard_into_opcheck_and_no_hot_output_allocation(monkeypatch):
    from rtx._routing_hard import hard_schedule_into
    from rtx._routing_reference import assignment_loads
    torch.manual_seed(1777)
    scores = torch.rand(68, 16, device="cuda")
    out = torch.empty((68, 4), device="cuda", dtype=torch.int64)
    args = (scores, out, 4, 5, 64, True, 0, 1024, "torch", 128)
    hard_schedule_into(*args)
    checked = torch.library.opcheck(hard_schedule_into, args,
                                   test_utils=("test_schema", "test_faketensor"))
    assert set(checked.values()) == {"SUCCESS"}
    def forbidden(*_args, **_kwargs):
        raise AssertionError("Hot hard-routing custom op allocated an eager tensor")
    with monkeypatch.context() as patch:
        for name in ("empty", "empty_like", "zeros", "zeros_like", "ones", "tensor"):
            patch.setattr(torch, name, forbidden)
        hard_schedule_into(*args)
    torch.cuda.synchronize()
    assert bool((assignment_loads(out, 16) == 17).all())
    assert not bool((out.sort(1).values.diff(1) == 0).any())


@cuda
def test_sinkhorn_external_graph_mutable_inputs_and_no_live_growth():
    torch.manual_seed(1773)
    scores = torch.randn(68, 16, device="cuda", requires_grad=True)
    upstream = torch.randn(68, 16, device="cuda")
    call = torch.compile(lambda value: _call(value)[0], fullgraph=True, dynamic=False,
                         options={"triton.cudagraphs": False})
    def step():
        output = call(scores)
        gradient, = torch.autograd.grad(output, (scores,), upstream)
        return output, gradient
    for _ in range(3):step()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):output, gradient = step()
    for _ in range(2):
        with torch.no_grad():scores.add_(.013)
        graph.replay();torch.cuda.synchronize()
        reference = scores.detach().double().requires_grad_()
        expected = _independent_transport(reference, 4, 3)
        want, = torch.autograd.grad(expected, (reference,), upstream.double())
        _check(output, expected, rtol=5e-5, atol=5e-6)
        _check(gradient, want, rtol=5e-4, atol=8e-6)
    graph.reset()
    del output, gradient, reference, expected, want
    gc.collect();torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    for _ in range(8):step()
    gc.collect();torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() <= baseline+65536


@cuda
def test_sinkhorn_two_streams_have_independent_scratch_and_owned_outputs():
    torch.manual_seed(1774)
    values = [torch.randn(68, 16, device="cuda", requires_grad=True) for _ in range(2)]
    streams = [torch.cuda.Stream() for _ in range(2)]
    for stream in streams:stream.wait_stream(torch.cuda.current_stream())
    results = []
    for stream, value in zip(streams, values):
        with torch.cuda.stream(stream):
            output, history = _call(value)
            gradient, = torch.autograd.grad(output.square().sum(), (value,))
            results.append((output, history, gradient))
    for stream in streams:torch.cuda.current_stream().wait_stream(stream)
    assert len({t.untyped_storage()._cdata for row in results for t in row}) == 6
    for value, (output, _, gradient) in zip(values, results):
        reference = value.detach().double().requires_grad_()
        expected = _independent_transport(reference, 4, 3)
        want, = torch.autograd.grad(expected.square().sum(), (reference,))
        _check(output, expected, rtol=5e-5, atol=5e-6)
        _check(gradient, want, rtol=5e-4, atol=8e-6)
