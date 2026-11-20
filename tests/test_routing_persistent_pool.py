"""Bounded persistent transport retains segmented FP32 saved state."""
from dataclasses import replace
from itertools import accumulate
import math

import pytest
import torch

from rtx.routing import BalancedRouter, RoutingKernelConfig, SinkhornKernelConfig


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="assigned RTX CUDA required")


def _cu(lengths):
    return torch.tensor([0, *accumulate(lengths)], device="cuda", dtype=torch.int32)


def _oracle(source, lengths, k, steps, upstream):
    """Direct FP64 logsumexp, without production offset decomposition."""
    e = source.shape[1]
    multiple = e // math.gcd(e, k)
    oracle = source.detach().double().requires_grad_()
    pieces, states, expected = 0, [[] for _ in range(2 * steps)], []
    offset = virtual = 0
    for length in lengths:
        if not length:
            continue
        n = (length + multiple - 1) // multiple * multiple
        value = torch.nn.functional.pad(
            oracle[offset:offset + length], (0, 0, 0, n - length), value=-math.log(2)
        )
        for stage, (axis, mass) in enumerate(((1, k), (0, n * k // e)) * steps):
            value = value - torch.logsumexp(value, axis, keepdim=True)
            states[stage].append(value.exp())
            value = value + math.log(mass)
        expected.append(value)
        pieces = pieces + (value * upstream[virtual:virtual + n].double()).sum()
        offset += length
        virtual += n
    return torch.cat(expected), torch.stack([torch.cat(stage) for stage in states]), torch.autograd.grad(pieces, oracle)[0]


@pytest.mark.parametrize("mix", ["legacy", "mixed"])
def test_small_pool_modes_present_in_opening_and_large_preserves_legacy(tmp_path, mix):
    from rtx._routing_varlen import _decode_config, _flat, layout, persistent_hybrid_rows, persistent_pool_rows
    from rtx.autotune.effort import AutotunePolicy, make_effort_autotuner
    from rtx.autotune.outcomes import TrialOutcome
    from rtx.autotune.store import ResidualTuningStore
    from rtx.autotune.varlen_routing import make_adapter, packed_structural_seeds

    initial = RoutingKernelConfig(qb_backend="native_column", metadata_backend="cute")
    seen = []
    def measure(config):
        seen.append(config)
        return TrialOutcome("ok", median_ms=1., timings_ms=[1.])
    adapter = make_adapter(128, 1, 32, 4, 5, initial, measure)
    seeds = list(packed_structural_seeds(adapter.initial_config, persistent_eligible=True))
    policy = AutotunePolicy(max_trials=8, max_seconds=10, candidate_mix=mix, confirmation_repeats=0)
    make_effort_autotuner(adapter, ResidualTuningStore(tmp_path), policy, candidates=seeds).tune()
    assert [c.packed_sinkhorn_backend for c in seen[:3]] == ["staged", "persistent", "persistent_backward"]
    assert len({adapter.config_id(c) for c in seen}) == len(seen) == 8
    assert adapter.features(seen[0])["derived.transport_launches"] == 23
    assert adapter.features(seen[1])["derived.transport_launches"] == 2
    assert adapter.features(seen[2])["derived.transport_launches"] == 13
    assert persistent_pool_rows(506, 32, 4) == 512 and persistent_pool_rows(513, 32, 4) == 0
    assert persistent_pool_rows(512, 128, 4) == 0
    assert persistent_pool_rows(1, 2, 1, pools=2**21) == 0
    assert RoutingKernelConfig.from_token(initial.to_token()[:20]) == initial
    assert RoutingKernelConfig.from_token(seen[1].to_token()) == seen[1]
    small = layout(128, 1, 32, 4, 5, tuple(seen[1].to_token()))
    assert _decode_config(tuple(_flat(small))) == small and small[17] == 3
    assert _decode_config(tuple(_flat(small))[:22]) == small
    assert small[7][0] == small[8][0] == 1024
    assert adapter.features(seen[1])["derived.backward_persistent_cta_threads"] == 1024
    large = make_adapter(2000, 4, 32, 4, 5, initial, measure)
    assert "packed_sinkhorn_backend" not in large.axes
    assert large.config_id(seen[1]) == large.config_id(replace(seen[1], packed_sinkhorn_backend="staged"))
    assert persistent_hybrid_rows(2000, 32, 4, pools=4) == 128
    assert layout(2000, 4, 32, 4, 5, tuple(seen[1].to_token()))[17:20] == (3, 128, 1024)
    assert layout(2000, 1, 32, 4, 5, tuple(seen[1].to_token()))[17:20] == (0, 0, 0)
    infer = make_adapter(128, 1, 32, 4, 5, initial, measure, training=False)
    assert infer.axes["packed_sinkhorn_backend"] == ("staged", "persistent_forward")
    assert infer.config_id(seen[1]) == infer.config_id(replace(seen[1], packed_sinkhorn_backend="persistent_forward"))


@pytest.mark.parametrize("lengths,hybrid_enabled", [((128,) * 6, True), ((640, 128, 0), True), ((384, 384), False)])
def test_cold_hybrid_search_uses_actual_pool_profile(lengths, hybrid_enabled):
    from rtx._routing_varlen import layout
    from rtx.autotune.varlen_routing import make_adapter, packed_structural_seeds
    from rtx.autotune.outcomes import TrialOutcome

    prior = RoutingKernelConfig(packed_row_backend="warp", packed_column_shards_forward=16,
                                packed_column_shards_backward=8)
    adapter = make_adapter(sum(lengths), len(lengths), 32, 4, 5, prior,
                           lambda _: TrialOutcome("ok", median_ms=1.), lengths=lengths)
    assert ("packed_sinkhorn_backend" in adapter.axes) == hybrid_enabled
    if hybrid_enabled:
        mixed = max(lengths) > 128
        opening, second, *_ = packed_structural_seeds(adapter.initial_config, persistent_eligible=True, mixed=mixed)
        assert opening.packed_sinkhorn_backend == "persistent"
        assert second.packed_sinkhorn_backend == ("staged" if mixed else "persistent_backward")
        decoded = adapter.deserialize(adapter.serialize(opening))
        key = layout(sum(lengths), len(lengths), 32, 4, 5, tuple(decoded.to_token()))
        assert key[14:16] == (16, 8)  # long pools still need the unchanged geometry
        assert key[17:20] == (3, 128, 1024)
        assert adapter.features(decoded)["derived.transport_launches"] == (37 if mixed else 4)


@cuda
@pytest.mark.parametrize("e,k,lengths,steps,threads,backend", [
    (32, 4, (128,), 5, 256, "persistent"),
    (7, 2, (5, 0, 14, 1), 3, 64, "persistent"),
    (64, 8, (0, 65, 0, 129), 5, 128, "persistent_forward"),
    (32, 4, (512,), 5, 256, "persistent_backward"),
    (32, 4, (255, 0, 129), 5, 1024, "persistent"),
    (128, 32, (1, 0, 4), 1, 64, "persistent"),
    (32, 4, (128, 0, 129, 361), 5, 1024, "persistent"),
    (7, 2, (126, 0, 130, 392), 3, 64, "persistent_backward"),
])
def test_saved_transport_fp64(e, k, lengths, steps, threads, backend):
    from rtx._routing_varlen import layout, transport_forward, transport_backward

    generator = torch.Generator(device="cuda").manual_seed(19345 + e)
    source = torch.nn.functional.logsigmoid(
        torch.randn((sum(lengths), e), device="cuda", generator=generator) * 1.5
    )
    geometry = SinkhornKernelConfig(threads=min(256, threads))
    config = RoutingKernelConfig(forward=geometry, backward=geometry, packed_sinkhorn_backend=backend,
                                 packed_persistent_threads=threads if threads > 256 else 0,
                                 packed_column_shards_forward=4 if sum(lengths) > 512 else 0,
                                 packed_column_shards_backward=4 if sum(lengths) > 512 else 0)
    key = layout(source.shape[0], len(lengths), e, k, steps, tuple(config.to_token()))
    assert key[17] == config.to_token()[20] != 0
    value, history, scheduled, _, sequence, _, rowmap, probabilities = transport_forward(source, _cu(lengths), key)
    upstream = torch.randn((e, value.shape[0]), device="cuda", generator=generator).T
    gradient = transport_backward(upstream, history, scheduled, sequence, rowmap, key)
    expected, state, ref_grad = _oracle(source, lengths, k, steps, upstream)
    n = expected.shape[0]
    saved = history.view(2 * steps, value.shape[0], e)
    for actual in (value, history, probabilities, gradient):
        assert bool(actual.isfinite().all())
    torch.testing.assert_close(value[:n].double(), expected, rtol=5e-5, atol=2e-6)
    torch.testing.assert_close(probabilities[:n].double(), expected.exp(), rtol=5e-5, atol=2e-6)
    torch.testing.assert_close(saved[:, :n].double(), state, rtol=5e-5, atol=2e-6)
    torch.testing.assert_close(gradient.double(), ref_grad, rtol=5e-4, atol=8e-6)
    assert bool((value[n:] == 0).all()) and bool((probabilities[n:] == 1).all())
    assert bool((saved[:, n:] == 0).all())


@cuda
@pytest.mark.parametrize("hybrid", [False, True])
def test_compiled_replay_exact_routes_runtime_boundaries_and_live_allocations(monkeypatch, hybrid):
    import gc
    from torch._dynamo.testing import CompileCounterWithBackend
    from rtx._routing_varlen import workspace_bounds
    monkeypatch.setattr(torch._inductor.config.triton, "cudagraphs", False)

    torch.manual_seed(5121)
    geometry = SinkhornKernelConfig(threads=128, column_groups=4)
    staged = RoutingKernelConfig(
        forward=geometry, backward=geometry, packed_row_backend="warp",
        qb_backend="native_warp", metadata_backend="cute", quota_threshold=0,
    )
    candidate = replace(staged, packed_sinkhorn_backend="persistent", packed_persistent_threads=1024)
    model = BalancedRouter(32, 32, top_k=4, router_dim=16, device="cuda", dtype=torch.float32, config=candidate)
    reference = BalancedRouter(32, 32, top_k=4, router_dim=16, device="cuda", dtype=torch.float32, config=staged)
    with torch.no_grad():
        model.projection_weight.normal_(std=.05)
    reference.load_state_dict(model.state_dict())
    beginning, mutations = ((124,) * 5, ((520, 0, 35, 0, 65), (620, 0, 0, 0, 0), (124,) * 5)) if hybrid else (
        (31, 0, 33), ((0, 64, 0), (1, 1, 62), (31, 0, 33)))
    t, b = sum(beginning), len(beginning)
    x = torch.randn(t, 32, device="cuda", requires_grad=True)
    cu = _cu(beginning)
    upstream = torch.randn(32, workspace_bounds(t, b, 32, 4)[1], device="cuda")
    counter = CompileCounterWithBackend("inductor")
    run = torch.compile(model, backend=counter, fullgraph=True, dynamic=False)
    # Keep affinity evaluation compiled on both sides. Eager/Inductor FP32
    # rounding can change a near-tie QB decision independently of transport.
    reference_run = torch.compile(reference, fullgraph=True, dynamic=False)

    def step(call, layer):
        route = call(x, cu_seqlens=cu)
        gradients = torch.autograd.grad(route.route_weights, (x, *layer.parameters()), upstream)
        return route, gradients

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            step(run, model)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            result, gradients = step(run, model)
    torch.cuda.current_stream().wait_stream(stream)
    try:
        for lengths in mutations:
            with torch.no_grad():
                cu.copy_(_cu(lengths))
                x.add_(.01)
            graph.replay()
            expected, ref_gradients = step(reference_run, reference)
            torch.testing.assert_close(result.expert_ids, expected.expert_ids, rtol=0, atol=0)
            torch.testing.assert_close(result.inverse, expected.inverse, rtol=0, atol=0)
            torch.testing.assert_close(result.route_weights, expected.route_weights, rtol=5e-5, atol=2e-6)
            for actual, target in zip(gradients, ref_gradients):
                assert bool(actual.isfinite().all())
                torch.testing.assert_close(actual, target, rtol=5e-4, atol=8e-6)
        assert counter.frame_count == 1
        del expected, ref_gradients, result, gradients
    finally:
        graph.reset()
    for _ in range(2):
        step(run, model)
    allocations = []
    for _ in range(5):
        output = step(run, model)
        del output
        gc.collect()
        torch.cuda.synchronize()
        allocations.append(torch.cuda.memory_allocated())
    assert len(set(allocations)) == 1, allocations


@cuda
def test_hybrid_graph_skips_staged_launches_with_all_short_pools():
    from rtx._routing_varlen import _flat, layout, prepare_execute, sinkhorn_execute

    e,k,steps,t,b=32,4,3,620,5
    c=RoutingKernelConfig(packed_persistent_threads=1024,packed_sinkhorn_backend="persistent")
    key=layout(t,b,e,k,steps,tuple(c.to_token()))
    n=key[4];flat=_flat(key)
    assert key[18]==128
    empty=lambda size,dtype=torch.float32:torch.empty(size,device="cuda",dtype=dtype)
    source=torch.randn(t*e,device="cuda")
    cu=_cu((124,)*5)
    scheduled=empty(b+1,torch.int32);physical=torch.empty_like(scheduled)
    sequence=empty(n,torch.int64);local=torch.empty_like(sequence)
    rowmap=empty(n,torch.int32);packed=empty(n*e);status=empty(1,torch.int32)
    out=empty(n*e);history=empty(2*steps*n*e);probability=empty(n*e)
    row=empty(max(n,b*e));column=torch.empty_like(row)
    def step():
        row.fill_(17.);column.fill_(17.)
        prepare_execute(source,cu,[scheduled,physical,sequence,local,rowmap,packed,status],flat)
        sinkhorn_execute([packed,scheduled,sequence,rowmap],[out,history,row,column,probability],flat,False)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        step()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):step()
    torch.cuda.current_stream().wait_stream(stream)
    try:
        for lengths,staged in (((124,)*5,False),((520,0,35,0,65),True),((124,)*5,False)):
            cu.copy_(_cu(lengths));graph.replay()
            assert bool((column!=17.).any())==staged
            assert bool(out.isfinite().all()) and bool(history.isfinite().all())
    finally:graph.reset()


@cuda
def test_runtime_persistent_selected_schema_fake_and_registered_adjoint(tmp_path):
    from rtx._routing_varlen_autotune import publish, selected_route

    config = RoutingKernelConfig(qb_backend="native_column", metadata_backend="cute", quota_threshold=0,
                                 packed_sinkhorn_backend="persistent")
    model = BalancedRouter(16, 8, top_k=2, router_dim=8, device="cuda", dtype=torch.float32,
                           config=config, autotune="cache", autotune_cache_dir=tmp_path)
    scores = torch.nn.functional.logsigmoid(torch.randn(19, 8, device="cuda")).requires_grad_()
    cu = _cu((4, 0, 15))
    spec = model._selection_spec
    publish(scores, cu, 2, 5, spec, True, False, config)
    output = torch.library.opcheck(selected_route, (scores, cu, 2, 5, 5, 64, spec, True, False, 15322))
    assert set(output.values()) == {"SUCCESS"}
