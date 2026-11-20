"""Composed router execution choices retain policy, gradients and first-hit ABI."""
from dataclasses import replace

import pytest
import torch

from rtx.routing import BalancedRouter, RoutingKernelConfig


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="RTX CUDA required")


def candidate():
    return RoutingKernelConfig(qb_backend="native_warp", metadata_backend="cute",
        metadata_threads=256, quota_threshold=0, quota_cache_items=8,
        quota_threads=64, quota_early_exit=True)


def test_execution_token_migration_and_inactive_axes():
    from rtx._routing_autotune import execution_config
    from rtx.autotune.routing import make_routing_adapter, rejection
    from rtx.autotune.outcomes import TrialOutcome
    default = RoutingKernelConfig()
    for length in (10, 14, 15):
        token = torch.tensor(default.to_token()[:length], dtype=torch.int64)
        assert execution_config(token) == default
    chosen = candidate()
    assert RoutingKernelConfig.from_token(chosen.to_token()) == chosen
    assert RoutingKernelConfig.from_dict(chosen.to_dict()) == chosen
    assert RoutingKernelConfig.from_token(chosen.to_token()[:14]) == replace(chosen, quota_early_exit=False)
    with pytest.raises(ValueError):
        RoutingKernelConfig.from_token(chosen.to_token()[:12])
    with pytest.raises(ValueError):
        replace(chosen, metadata_backend="unknown")
    with pytest.raises(ValueError):
        replace(chosen, quota_cache_items=32)
    adapter = make_routing_adapter(68, 16, 4, lambda _: TrialOutcome("ok", median_ms=1.))
    assert {"metadata_backend", "metadata_threads", "quota_cache_items", "quota_threads", "quota_early_exit"} <= set(adapter.axes)
    a = replace(default, segmented_quotas=False, metadata_threads=512)
    b = replace(a, quota_cache_items=16, quota_early_exit=True, metadata_threads=256)
    assert adapter.config_id(a) == adapter.config_id(b)
    assert rejection(chosen, 128, 256, 8)
    assert rejection(replace(chosen, qb_backend="native", metadata_backend="torch"), 128, 256, 8) is None


def test_torch_metadata_adjoint_has_independent_float64_oracle_with_dummies():
    from rtx._routing_reference import routing_metadata
    from rtx.autotune.routing import _metadata_gradient_reference
    n, e, k, real = 14, 7, 2, 13
    generator = torch.Generator().manual_seed(3849)
    assignments = (torch.arange(n*k) % e).reshape(n, k)
    scores = torch.randn(n, e, generator=generator, dtype=torch.float64, requires_grad=True)
    _, weights, inverse = routing_metadata(scores, assignments, num_real_tokens=real)
    upstream = torch.randn(e, n*k//e, generator=generator, dtype=torch.float64)
    expected = torch.autograd.grad(weights, scores, upstream)[0]
    actual = _metadata_gradient_reference(upstream.float(), weights.detach().float(), assignments,
                                           inverse, n, e)
    torch.testing.assert_close(actual.double(), expected, rtol=3e-6, atol=2e-7)
    assert not bool(actual[real:].count_nonzero())


def test_composed_router_fake_fullgraph_uses_visible_metadata_and_static_launches():
    from torch._subclasses.fake_tensor import FakeTensorMode, FakeTensor
    def fake(mode, shape):
        return FakeTensor(mode, torch.empty(shape, device="meta", dtype=torch.float32,
                                           requires_grad=True), torch.device("cuda:0"))
    router = BalancedRouter(16, 16, top_k=4, router_dim=8, config=candidate())
    graphs = []
    def backend(graph, _inputs):
        graphs.append(graph)
        return graph.forward
    with FakeTensorMode() as mode:
        for name, parameter in list(router.named_parameters()):
            setattr(router, name, torch.nn.Parameter(fake(mode, tuple(parameter.shape))))
        route = torch.compile(router, backend=backend, fullgraph=True, dynamic=False)(fake(mode, (65, 16)))
    assert route.route_weights.shape == (16, 17) and route.inverse.shape == (65, 4)
    code = "\n".join(module.code for module in graphs[0].modules() if isinstance(module, torch.fx.GraphModule))
    assert "metadata_execute" in code and "hard_schedule_into" in code and "sinkhorn_execute" in code
    assert "selected_sinkhorn" not in code and "selected_hard" not in code


@cuda
def test_composed_router_gradients_and_compiled_cold_to_hot(tmp_path):
    from rtx import _routing_autotune as startup
    from rtx import _moe_selection as selection
    torch.manual_seed(7439)
    base = BalancedRouter(16, 16, top_k=4, router_dim=8, device="cuda",
        config=replace(candidate(), metadata_backend="torch", quota_cache_items=0,
                       quota_early_exit=False), autotune="off")
    with torch.no_grad():
        base.projection_weight.normal_(std=.05)
    selected = BalancedRouter(16, 16, top_k=4, router_dim=8, device="cuda",
        config=candidate(), autotune="cache", autotune_cache_dir=tmp_path)
    selected.load_state_dict(base.state_dict())
    x = torch.randn(65, 16, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_other = x.detach().clone().requires_grad_()
    upstream = torch.randn(16, 17, device="cuda")
    fn = torch.compile(selected, fullgraph=True, dynamic=False, options={"triton.cudagraphs":False})
    selection.clear_selection_cells()
    try:
        # Cache miss uses the explicit seed, then the guarded hot invocation
        # consumes the bound CuTe metadata configuration without a CPU token.
        for _ in range(2):
            expected = base(x)
            actual = fn(x_other)
            torch.testing.assert_close(actual.expert_ids, expected.expert_ids, rtol=0, atol=0)
            torch.testing.assert_close(actual.inverse, expected.inverse, rtol=0, atol=0)
            torch.testing.assert_close(actual.route_weights, expected.route_weights, rtol=5e-5, atol=5e-7)
            a = torch.autograd.grad(actual.route_weights, (x_other,*selected.parameters()), upstream)
            b = torch.autograd.grad(expected.route_weights, (x,*base.parameters()), upstream)
            for value, reference in zip(a,b):
                assert bool(value.isfinite().all())
                # BF16 affinity GEMM derivatives can differ in final rounding;
                # exact hard assignments above are gated separately.
                nrms = (value.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-12)
                assert nrms < .015
    finally:
        startup._clear_runtime_caches()


@cuda
def test_shared_evaluator_qualifies_composed_forward_and_adjoint(tmp_path):
    from rtx._routing_autotune import parse_spec
    from rtx.autotune import AutotunePolicy
    from rtx.autotune.routing import RoutingEvaluator
    router = BalancedRouter(16, 16, top_k=4, router_dim=8, config=candidate(),
                            autotune_cache_dir=tmp_path)
    spec = parse_spec(router._selection_spec)
    spec = {**spec, "workload": {**spec["workload"], "real_tokens":65}}
    policy = AutotunePolicy(max_trials=1, samples=3, calls_per_sample=1,
                            warmup=1, confirmation_repeats=0)
    scores = torch.nn.functional.logsigmoid(torch.randn(68,16,device="cuda"))
    evaluator = RoutingEvaluator(scores, spec, "train", policy)
    try:
        outcome = evaluator.evaluate(candidate())
        assert outcome.status == "ok", outcome
        assert outcome.median_ms > 0 and len(outcome.metadata["validation"]) == 2
    finally:
        evaluator.close()
