"""Stable private M compaction, byte copies, and original-rank combine."""

from dataclasses import replace
from math import gcd

import pytest
import torch


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='SM120 CUDA required')


@pytest.fixture
def cuda_capture_stream():
    with torch.cuda.stream(torch.cuda.Stream()) as stream:
        yield stream


def _compositions(total, parts):
    if parts == 1:
        yield (total,)
    else:
        for leading in range(total + 1):
            for tail in _compositions(total - leading, parts - 1):
                yield (leading, *tail)


@pytest.mark.parametrize('e,k', [(8, 2), (6, 4), (5, 3), (4, 4)])
def test_shape_only_bound_covers_every_tiny_packed_boundary(e, k):
    from rtx._moe_compact_layout import compact_forward_capacity

    g = gcd(e, k)
    period, quota = e // g, k // g
    for tokens in range(1, 12):
        for pools in range(1, 6):
            bound = compact_forward_capacity(tokens, pools, e, k, 4096, 1)
            lengths = _compositions(tokens, pools)
            maximum = max(sum((length + period - 1) // period * quota for length in ls) for ls in lengths)
            assert bound == maximum


def test_capacity_preserves_opaque_routes_and_no_storage_win():
    from rtx._moe_compact_layout import compact_forward_capacity as capacity

    assert capacity(8192, 32, 32, 4, 2016, 1) == 1052
    assert capacity(8192, 32, 32, 4, 2016, 128) == 1152
    assert capacity(8192, 1, 32, 4, 1024, 128) == 0
    assert capacity(8192, 1, 32, 4, 2048, 128) == 1024
    assert capacity(8192, None, 32, 4, 2016, 128) == 0
    assert capacity(8192, 0, 32, 4, 2016, 128) == 0


def _policies():
    from tests.test_moe_active_prefix import _config
    from tests.test_moe_fragment_glu import _schedule

    baseline = _config(engine='dual_operand', epilogue='separate')
    p = replace(baseline.prepared, forward_glu_engine='fragment_direct',
                gate_up=_schedule(64, 64, 1, 'producer'))
    baseline = replace(baseline, prepared=p, combine_reduce_engine='token', dx_reduce_engine='token')
    candidate = replace(baseline, prepared=replace(p, forward_row_layout='compact'))
    return baseline, candidate


def test_compact_plan_has_smaller_forward_storage_and_same_backward_launches():
    from rtx._moe_execution import layout, plan
    from rtx.moe import _config_key

    shape = (512, 128, 8, 256, 2, 64)
    baseline, candidate = _policies()
    with pytest.raises(ValueError, match='pool bound'):
        layout(shape, _config_key(candidate), 'forward')
    candidate = replace(candidate, prepared=replace(candidate.prepared, compact_pool_count=4))
    before = plan(layout(shape, _config_key(baseline), 'forward')[0])
    after = plan(layout(shape, _config_key(candidate), 'forward')[0])
    assert [x[1] for x in after[0].steps if x[0] == 'mxfp8_moe_compact'] == [
        'compile_compact_route', 'compile_compact_gather', 'compile_compact_combine',
    ]
    assert any(v.shape == (8 * 192, 128) and v.dtype == torch.bfloat16 for v in after[0].buffers)
    assert not any(v.shape == (8 * 256, 128) and v.dtype == torch.bfloat16 for v in after[0].buffers)
    def arena_bytes(cell):
        return sum(size * torch.empty((), dtype=dtype).element_size() for dtype, size in zip(cell[1], cell[2]))
    assert arena_bytes(after) < arena_bytes(before)
    old, new = (plan(layout(shape, _config_key(config), 'backward')[0]) for config in (baseline, candidate))
    assert old[0].steps == new[0].steps
    assert old[1:] == new[1:]


def test_regular_fully_used_stride_rejects_compaction():
    from rtx.autotune.moe import make_moe_adapter

    shape = (512, 128, 8, 128, 2, 64)  # One pool exactly fills both 64-row tiles.
    original, _ = _policies()
    adapter = make_moe_adapter(shape, None, initial=original, tags={
        'batch_contract': {'layout': 'regular', 'sequences': 1, 'segment_alignment': 32},
    })
    assert adapter.rejection(adapter.initial_config) is None
    compact = adapter.update_fn(adapter.initial_config, 'prepared.forward_row_layout', 'compact')
    assert compact.prepared.compact_pool_count == 1
    assert 'no shape-bound storage reduction' in adapter.rejection(compact)[1]


def _route(n, e, c, k, seed, *, empty=False):
    """Independent routes with holes in each metadata tile and a real zero weight."""
    prng = torch.Generator().manual_seed(seed)
    ids = torch.randint(n, (e, c), generator=prng, dtype=torch.int64)
    ids[:, 2::7] = -1
    ids[:, 3::7] = 2**32  # Cannot be narrowed to token zero before validation.
    inverse = torch.full((n, k), -1, dtype=torch.int64)
    weights = torch.full((e, c), float('nan'), dtype=torch.float32)
    if not empty:
        count = [0] * e
        order = [torch.randperm(c, generator=prng).tolist() for _ in range(e)]
        for token in range(n):
            for rank in range(k):
                expert = ((n - token + seed) * k + (k - rank - 1)) % e
                row = order[expert][count[expert]]
                count[expert] += 1
                ids[expert, row] = token
                inverse[token, rank] = expert * c + row
                weights[expert, row] = ((rank + token) % 7 - 3) / 8
    return ids, inverse, weights


def _compact_oracle(ids, inverse, capacity):
    """Enumerate by token rank first, then sort the actual slots per expert."""
    e, c = ids.shape
    ranked = [[] for _ in range(e)]
    for token, route in enumerate(inverse.tolist()):
        for slot in route:
            if 0 <= slot < e * c:
                expert, row = divmod(slot, c)
                assert ids[expert, row].item() == token
                ranked[expert].append((row, token))
    forward = torch.full((e, c), -1, dtype=torch.int32)
    reverse = torch.full((e, capacity), -1, dtype=torch.int32)
    tokens = torch.full((e, capacity), -1, dtype=torch.int32)
    for expert, entries in enumerate(ranked):
        for dense, (row, token) in enumerate(sorted(entries)):
            forward[expert, row], reverse[expert, dense], tokens[expert, dense] = dense, row, token
    return forward, reverse, tokens, torch.tensor([len(x) for x in ranked], dtype=torch.int32)


@cuda
@pytest.mark.parametrize('d,engine,shifted', [(35, 'scalar', False), (128, 'token', False), (96, 'token', True)])
def test_compact_launches_replay_changed_routes_and_scrub_poison(d, engine, shifted):
    from rtx.runtime import load_kernel_symbol, retain_runtime_resources

    n, e, c, k, compact_c = 37, 5, 1281, 3, 32
    width = (d + 31) // 32 * 32
    factory = lambda symbol: load_kernel_symbol('mxfp8_moe_compact', symbol)
    mapping = factory('compile_compact_route')(n, e, c, k, compact_c)
    gather = factory('compile_compact_gather')(n, d, e, c, compact_c, 128)
    combine = factory('compile_compact_combine')(n, d, e, c, k, compact_c, 128, engine, 4, 16)
    original = load_kernel_symbol('mxfp8_moe', 'compile_movement')(n, d, e, c, k, 128, 'combine')

    ids = torch.empty((e, c), dtype=torch.int64, device='cuda')
    inverse = torch.empty((n, k), dtype=torch.int64, device='cuda')
    weights = torch.empty((e, c), dtype=torch.float32, device='cuda')
    original_to_compact = torch.empty((e, c), dtype=torch.int32, device='cuda')
    compact_to_original = torch.empty((e, compact_c), dtype=torch.int32, device='cuda')
    counts = torch.empty(e, dtype=torch.int32, device='cuda')
    global_q = torch.empty((n, width), dtype=torch.uint8, device='cuda')
    global_sf = torch.empty((n, width // 32), dtype=torch.uint8, device='cuda')
    qx = torch.empty((e * compact_c, width), dtype=torch.uint8, device='cuda')
    sx = torch.empty((e * compact_c, width // 32), dtype=torch.uint8, device='cuda')
    prefix = torch.empty(1, dtype=torch.int32, device='cuda')
    source = torch.empty(e * compact_c * d + shifted, dtype=torch.bfloat16, device='cuda')[shifted:].view(e * compact_c, d)
    expanded = torch.empty((e * c, d), dtype=torch.bfloat16, device='cuda')
    out = torch.empty(n * d + shifted, dtype=torch.float32, device='cuda')[shifted:].view(n, d)
    direct = torch.empty_like(out)
    global_fp8, qx_fp8 = global_q.view(torch.float8_e4m3fn), qx.view(torch.float8_e4m3fn)

    def phase():
        original_to_compact.fill_(9999)
        compact_to_original.fill_(9999)
        counts.fill_(9999)
        qx.fill_(255)
        sx.fill_(255)
        prefix.fill_(9999)
        out.fill_(float('nan'))
        mapping(ids, inverse, original_to_compact, compact_to_original, counts)
        gather(global_fp8, global_sf, ids, compact_to_original, counts, qx_fp8, sx, prefix)
        combine(source, inverse, weights, original_to_compact, out)

    def fixture(seed, empty):
        host_ids, host_inverse, host_weights = _route(n, e, c, k, seed, empty=empty)
        maps = _compact_oracle(host_ids, host_inverse, compact_c)
        forward, reverse, tokens, expected_counts = maps
        for dst, src in zip((ids, inverse, weights), (host_ids, host_inverse, host_weights)):
            dst.copy_(src)
        generator = torch.Generator(device='cuda').manual_seed(seed)
        global_q.random_(generator=generator)
        global_sf.random_(generator=generator)
        source.normal_(std=.25, generator=generator)
        source.view(e, compact_c, d).masked_fill_((tokens < 0).cuda()[..., None], float('nan'))
        expanded.fill_(float('nan'))
        experts, rows = (forward >= 0).nonzero(as_tuple=True)
        original_flat = (experts * c + rows).cuda()
        compact_flat = (experts * compact_c + forward[experts, rows]).cuda()
        expanded[original_flat] = source[compact_flat]
        expected = torch.zeros((n, d), dtype=torch.float64, device='cuda')
        for rank in range(k):
            slots = inverse[:, rank]
            present = slots >= 0
            selected = slots[present]
            expected[present] += expanded[selected].double() * weights.flatten()[selected, None].double()
        return forward, reverse, tokens, expected_counts, expected

    fixture(12, False)
    phase()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with retain_runtime_resources(), torch.cuda.graph(graph):
        phase()

    for seed, empty in [(13, False), (14, True), (15, False)]:
        forward, reverse, tokens, expected_counts, expected = fixture(seed, empty)
        graph.replay()
        original(expanded, ids, inverse, weights, expanded, direct, weights)
        assert torch.equal(out, direct)
        torch.testing.assert_close(out.double(), expected, rtol=1e-6, atol=1e-7)
        assert torch.equal(original_to_compact.cpu(), forward)
        assert torch.equal(compact_to_original.cpu(), reverse)
        assert torch.equal(counts.cpu(), expected_counts)
        assert prefix.item() == expected_counts.max().item()
        tok = tokens.flatten().cuda()
        assert torch.equal(qx, global_q[tok.clamp_min(0)].masked_fill((tok < 0)[:, None], 0))
        assert torch.equal(sx, global_sf[tok.clamp_min(0)].masked_fill((tok < 0)[:, None], 0))


@cuda
def test_private_compact_map_overflow_cannot_write_past_stride():
    """No public route may violate its bound; guarded raw launches stay in-bounds."""
    from rtx.runtime import load_kernel_symbol

    n, e, c, k, stride = 11, 2, 11, 1, 4
    mapping = load_kernel_symbol('mxfp8_moe_compact', 'compile_compact_route')(n, e, c, k, stride)
    ids = torch.arange(n, device='cuda', dtype=torch.int64).repeat(e, 1)
    inverse = torch.arange(n, device='cuda', dtype=torch.int64).view(n, k)
    # All n assignments hit expert 0, impossible for the proven balanced route.
    # Expert 1 and the backing guard must not be overwritten by overflow.
    original = torch.empty((e, c), dtype=torch.int32, device='cuda')
    storage = torch.full((e + 1, stride), 177, dtype=torch.int32, device='cuda')
    counts = torch.empty(e, dtype=torch.int32, device='cuda')
    mapping(ids, inverse, original, storage[:e], counts)
    assert counts.tolist() == [stride, 0]
    assert storage.tolist() == [list(range(stride)), [-1] * stride, [177] * stride]
    assert original.tolist() == [list(range(stride)) + [-1] * (c - stride), [-1] * c]


@cuda
def test_compact_dispatch_fullgraph_replays_changed_packing_and_all_gradients(record_property, cuda_capture_stream):
    from rtx._moe_reference import operational_reference
    from rtx.moe import _dispatch_experts, _packed_expert_spec, _selection_spec
    from rtx.runtime import retain_runtime_resources
    from tests.test_moe_active_prefix import _inputs, _route as packed_route

    baseline, candidate = _policies()
    n, e, c, k, d = 512, 8, 256, 2, 128
    x, gu, down = _inputs(n)
    ids, weights, inverse = packed_route((32, 32, 224, 224), c)[0]
    leaves = tuple(v.requires_grad_() for v in (x, gu, down, weights))
    saved_inputs = (x, gu, down, ids, weights, inverse)
    generator = torch.Generator(device='cuda').manual_seed(8302)
    dy = torch.randn((d, n), device='cuda', generator=generator).T  # strided upstream

    def model(config):
        spec = _packed_expert_spec(_selection_spec(config, 'off', None, None), 4)

        def call(x, gu, down, ids, weights, inverse):
            return _dispatch_experts(x, gu, down, ids, weights, inverse, spec, True)

        return call

    def step(call):
        z = call(*saved_inputs)
        return z, *torch.autograd.grad(z, leaves, dy)

    saved = []
    def pack(tensor):
        saved.append(tensor.data_ptr())
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        eager = step(model(candidate))
    assert sorted(saved) == sorted(v.data_ptr() for v in saved_inputs)
    del eager
    compiled = torch.compile(model(candidate), fullgraph=True, dynamic=False,
                             options={'triton.cudagraphs': False})
    step(compiled)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with retain_runtime_resources(), torch.cuda.graph(graph, stream=cuda_capture_stream):
        outputs = step(compiled)

    previous = None
    for case, lengths in enumerate(((32, 32, 224, 224), (0, 1, 128, 383), (1, 1, 1, 509))):
        changed, _ = packed_route(lengths, c, e, k)
        with torch.no_grad():
            for dst, src in zip((ids, weights, inverse), changed):
                dst.copy_(src)
            x.mul_(.8125)
            dy.mul_(-.875)
            # Zero-weight real slots must keep their public route adjoints.
            weights.flatten()[inverse[0, 0]] = 0
        graph.replay()
        reference = step(model(baseline))
        assert all(torch.equal(a, b) for a, b in zip(outputs, reference))
        if previous is not None:
            assert not torch.equal(previous, outputs[0])
        previous = outputs[0].clone()
        oracle = operational_reference(x, gu, down, ids, weights, inverse, dy)
        for name, actual, expected in zip(('out', 'dx', 'dgu', 'ddown', 'dweights'), outputs, oracle):
            delta = actual.detach().float() - expected.float()
            nrms = float(delta.norm() / expected.float().norm().clamp_min(1e-12))
            max_abs = float(delta.abs().max())
            record_property(f'{case}.{name}.nrms', nrms)
            record_property(f'{case}.{name}.max_abs', max_abs)
            assert bool(actual.isfinite().all()) and (nrms < .006 or max_abs < 2e-7), (name, nrms, max_abs)
        assert outputs[4].flatten()[inverse[0, 0]] != 0
        del reference, oracle

    live = []
    for _ in range(5):
        graph.replay()
        torch.cuda.synchronize()
        live.append(torch.cuda.memory_allocated())
    record_property('live_allocated_bytes', str(live))
    assert len(set(live)) == 1


@cuda
def test_compact_execute_mutation_and_fake_abi(monkeypatch):
    from rtx._moe_execution import allocate, execute, layout
    from rtx.moe import _config_key
    from tests.test_moe_active_prefix import _inputs, _route as packed_route

    n, d, e, c, k, h = 512, 128, 8, 256, 2, 64
    x, gu, down = _inputs(n)
    ids, weights, inverse = packed_route((0, 1, 128, 383), c, e, k)[0]
    _, candidate = _policies()
    candidate = replace(candidate, prepared=replace(candidate.prepared, compact_pool_count=4))
    key, dtypes, sizes, views, outputs = layout((n, d, e, c, k, h), _config_key(candidate), 'forward')
    buffers = allocate(x, dtypes, sizes, views)
    args = ([x, gu.flatten(0, 1), down.flatten(0, 1), ids, weights, inverse], buffers, key)
    execute(*args)
    assert buffers[outputs[0]].isfinite().all()
    original_allclose = torch.allclose
    def schema_equal(lhs, rhs, *values, **kwargs):
        # Torch 2.13 SchemaCheckMode uses allclose to detect mutation, but
        # CUDA allclose has no E4M3 arithmetic. Raw-byte equality is stricter
        # than that numerical comparator and leaves alias checks untouched.
        if lhs.dtype == rhs.dtype == torch.float8_e4m3fn:
            return torch.equal(lhs.view(torch.uint8), rhs.view(torch.uint8))
        return original_allclose(lhs, rhs, *values, **kwargs)
    with monkeypatch.context() as scope:
        scope.setattr(torch, 'allclose', schema_equal)
        observed = torch.library.opcheck(execute, args, test_utils=('test_schema', 'test_faketensor'))
    assert set(observed.values()) == {'SUCCESS'}
