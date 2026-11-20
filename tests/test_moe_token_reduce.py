"""Independent slot semantics, scalar bit parity and locked-GPU replay tests."""
from dataclasses import replace

import pytest
import torch

from rtx.kernels.mxfp8_moe import compile_movement
from rtx.kernels.mxfp8_moe_token_reduce import compile_token_reduce


def _oracle(source, inverse, weights, storage_c, kind):
    """Independent logical-view oracle, with one FP32 rounding per rank."""
    e, c = weights.shape
    d = source.shape[-1]
    # Remove physical padding before interpreting logical inverse slots.
    rows = source.reshape(e, storage_c, d)[:, :c].reshape(e * c, d).double()
    w = weights.flatten().double()
    accum = torch.zeros((inverse.shape[0], d), dtype=torch.float32)
    for slots in inverse.unbind(1):
        valid = (slots >= 0) & (slots < e * c)
        indexes = slots[valid]
        contrib = rows[indexes]
        if kind == 'combine':
            contrib = contrib * w[indexes, None]
        # For the moderate mantissa/exponent range in these fixtures the
        # float64 operation implements a single correctly rounded FP32 FMA.
        accum[valid] = (accum[valid].double() + contrib).float()
    return accum if kind == 'combine' else accum.bfloat16()


@pytest.mark.parametrize('kind', ['combine', 'dx'])
def test_oracle_hand_case(kind):
    source = torch.tensor([[1., -2.], [3., 4.], [float('nan')] * 2,
                           [5., -6.], [-7., 8.], [float('nan')] * 2]).bfloat16()
    inverse = torch.tensor([[0, 3, 0], [3, 2, -1], [-4, 4, 7]])
    weights = torch.tensor([[.25, .5], [-1., 0.]])
    expected = (torch.tensor([[.5, -1.], [-5., 6.], [0., 0.]]) if kind == 'combine'
                else torch.tensor([[-5., 4.], [-2., 2.], [0., 0.]]).bfloat16())
    torch.testing.assert_close(_oracle(source, inverse, weights, 3, kind), expected,
                               atol=0, rtol=0)


def _config(combine, dx):
    from rtx.moe import DEFAULT_MOE_CONFIG

    return replace(DEFAULT_MOE_CONFIG, combine_reduce_engine=combine, dx_reduce_engine=dx,
                   combine_vector_width=8, combine_lanes_per_token=16,
                   dx_vector_width=4, dx_lanes_per_token=8)


def test_old_config_loads_scalar_and_new_config_roundtrips():
    from rtx.moe import DEFAULT_MOE_CONFIG, MXFP8MoEConfig

    old = {key: value for key, value in DEFAULT_MOE_CONFIG.to_dict().items()
           if not key.startswith(('combine_', 'dx_'))}
    assert MXFP8MoEConfig.from_dict(old) == DEFAULT_MOE_CONFIG
    c = _config('token', 'scalar')
    assert MXFP8MoEConfig.from_dict(c.to_dict()) == c


@pytest.mark.parametrize('combine,dx', [('token', 'scalar'), ('scalar', 'token'), ('token', 'token')])
def test_token_plan_replaces_only_selected_leg_without_new_buffers(combine, dx):
    from rtx._moe_execution import layout, plan
    from rtx.moe import DEFAULT_MOE_CONFIG, _config_key

    shape = (128, 96, 4, 101, 3, 64)
    for phase in ('forward', 'backward'):
        leg = 'combine' if phase == 'forward' else 'dx'
        choices = {'combine': combine, 'dx': dx}
        base, *base_storage = layout(shape, _config_key(DEFAULT_MOE_CONFIG), phase)
        candidate, *new_storage = layout(shape, _config_key(_config(combine, dx)), phase)
        assert base_storage == new_storage
        scalar_builder, *_ = plan(base)
        candidate_builder, *_ = plan(candidate)
        steps = list(zip(scalar_builder.steps, candidate_builder.steps, strict=True))
        changes = [(a, b) for a, b in steps if a != b]
        if choices[leg] == 'scalar':
            assert not changes
            continue
        [(a, b)] = changes
        assert a[:2] == ('mxfp8_moe', 'compile_movement')
        assert b[:2] == ('mxfp8_moe_token_reduce', 'compile_token_reduce')
        assert a[2][:7] == b[2][:7] and b[2][6] == leg
        assert a[2][8] == b[2][7]  # physical C in either ABI
        assert b[2][8:] == ((8, 16) if leg == 'combine' else (4, 8))
        assert a[3] == b[3] and len(b[3]) == 7


requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def _problem(n, d, e, c, k, cs, seed=887):
    generator = torch.Generator().manual_seed(seed)
    source = torch.randn((e, cs, d), generator=generator).bfloat16()
    source[:, c:] = float('nan')  # padding must never be consumed
    weights = torch.randn((e, c), generator=generator)
    weights[0, 0] = 0.  # valid zero weight is not a missing assignment
    inverse = torch.randint(-3, e * c + 3, (n, k), generator=generator)
    inverse[0] = 0  # duplicate ranks must each contribute
    inverse[1] = -1
    inverse[2] = e * c
    inverse[3] = torch.arange(k) % (e * c)
    return source.flatten(0, 1), inverse, weights


def _bits(t):
    return t.view(torch.int32 if t.dtype == torch.float32 else torch.int16)


def _guarded_view(shape, dtype, shift):
    rows, cols = shape
    start = 32 + shift
    buffer = torch.full((start + rows * cols + 32,), -123, dtype=dtype, device='cuda')
    return buffer[start:start + rows * cols].view(shape), buffer, start


@requires_cuda
@pytest.mark.parametrize('kind', ['combine', 'dx'])
@pytest.mark.parametrize('d,k,vector,lanes,threads,source_shift,out_shift', [
    (128, 1, 4, 32, 64, 0, 0),
    (384, 2, 4, 32, 128, 0, 0),
    (576, 3, 4, 16, 256, 0, 0),
    (640, 6, 8, 32, 128, 0, 0),
    (768, 4, 8, 16, 512, 0, 0),
    (1024, 8, 2, 8, 128, 0, 0),
    (70, 3, 2, 8, 64, 0, 0),  # partial feature stripe with vector-safe D
    (137, 4, 4, 32, 128, 0, 0),  # scalar fallback for unaligned row stride
    (384, 3, 8, 16, 128, 1, 0),  # shifted source only
    (384, 3, 8, 16, 128, 0, 1),  # shifted output only
])
def test_vector_and_scalar_bitwise(kind, d, k, vector, lanes, threads, source_shift, out_shift):
    n, e, c, cs = 37, 13, 13, 19
    source, inverse, weights = _problem(n, d, e, c, k, cs)
    expect = _oracle(source, inverse, weights, cs, kind)
    x, x_buffer, x_start = _guarded_view(source.shape, source.dtype, source_shift)
    x.copy_(source)
    gpu_inv, gpu_w = inverse.cuda(), weights.cuda()
    ids = torch.full((e, c), -1, device='cuda', dtype=torch.int64)  # unused by both modes
    out, out_buffer, out_start = _guarded_view(expect.shape, expect.dtype, out_shift)
    reference = torch.empty_like(out)
    scalar = compile_movement(n, d, e, c, k, threads, kind, 'flat', cs)
    legacy_x = source.cuda() if source_shift else x
    scalar(legacy_x, ids, gpu_inv, gpu_w, legacy_x, reference, gpu_w)
    launch = compile_token_reduce(n, d, e, c, k, threads, kind, cs, vector, lanes)
    launch(x, ids, gpu_inv, gpu_w, x, out, gpu_w)
    assert torch.isfinite(out).all()
    assert torch.equal(_bits(out), _bits(reference))
    assert torch.equal(_bits(out.cpu()), _bits(expect))
    assert torch.all(out_buffer[:out_start] == -123)
    assert torch.all(out_buffer[out_start + out.numel():] == -123)
    assert torch.all(x_buffer[:x_start] == -123)
    assert torch.all(x_buffer[x_start + x.numel():] == -123)
    assert torch.equal(_bits(x.cpu()), _bits(source))
    assert torch.equal(gpu_inv.cpu(), inverse)
    assert torch.equal(_bits(gpu_w.cpu()), _bits(weights))


@requires_cuda
@pytest.mark.parametrize('kind', ['combine', 'dx'])
def test_nonfinite_is_not_masked_at_valid_slots(kind):
    n, d, e, c, k, cs = 5, 32, 4, 4, 3, 5
    source, inverse, weights = _problem(n, d, e, c, k, cs)
    source[0, 0] = float('inf')
    source[1, 1] = float('-inf')
    source[2, 2] = float('nan')
    weights[0, 1] = 1.
    source, inverse, weights = source.cuda(), inverse.cuda(), weights.cuda()
    ids = torch.zeros_like(weights, dtype=torch.int64)
    out = torch.empty((n, d), device='cuda', dtype=torch.float32 if kind == 'combine' else torch.bfloat16)
    ref = torch.empty_like(out)
    scalar = compile_movement(n, d, e, c, k, 128, kind, 'flat', cs)
    vector = compile_token_reduce(n, d, e, c, k, 128, kind, cs, 8, 8)
    scalar(source, ids, inverse, weights, source, ref, weights)
    vector(source, ids, inverse, weights, source, out, weights)
    assert (~torch.isfinite(ref)).any()
    for predicate in (torch.isnan, torch.isposinf, torch.isneginf):
        assert torch.equal(predicate(ref), predicate(out))
    finite = torch.isfinite(ref)
    assert torch.equal(_bits(ref)[finite], _bits(out)[finite])
    assert torch.all(out[1:3] == 0.)


@requires_cuda
def test_both_legs_replay_updates_at_same_addresses():
    n, d, e, c, k, cs = 91, 640, 16, 19, 6, 32
    source, inverse, weights = _problem(n, d, e, c, k, cs)
    x, inv, w = source.cuda(), inverse.cuda(), weights.cuda()
    ids = torch.zeros_like(w, dtype=torch.int64)
    launch = {
        kind: compile_token_reduce(n, d, e, c, k, 128, kind, cs, 8, 16)
        for kind in ('combine', 'dx')
    }
    out = {
        'combine': torch.empty((n, d), device='cuda', dtype=torch.float32),
        'dx': torch.empty((n, d), device='cuda', dtype=torch.bfloat16),
    }

    def run():
        for kind, kernel in launch.items():
            kernel(x, ids, inv, w, x, out[kind], w)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    graph.replay()
    torch.cuda.synchronize()
    before = {kind: value.cpu() for kind, value in out.items()}
    source, inverse, weights = _problem(n, d, e, c, k, cs, seed=41)
    x.copy_(source)
    inv.copy_(inverse)
    w.copy_(weights)
    torch.cuda.synchronize()
    live = torch.cuda.memory_allocated()
    for _ in range(10):
        graph.replay()
        torch.cuda.synchronize()
        assert torch.cuda.memory_allocated() == live
    for kind, result in out.items():
        assert not torch.equal(_bits(result.cpu()), _bits(before[kind]))
        assert torch.equal(_bits(result.cpu()), _bits(_oracle(source, inverse, weights, cs, kind)))


@requires_cuda
def test_public_fullgraph_replay_changed_inputs_and_all_gradients(record_property):
    from rtx._moe_execution import allocate, execute, layout
    from rtx._moe_reference import operational_reference
    from rtx._routing_reference import padded_token_count, routing_metadata
    from rtx.moe import DEFAULT_MOE_CONFIG, _config_key, mxfp8_routed_experts
    from rtx.routing import BalancedRouting

    n, d, e, k, h = 45, 96, 4, 3, 64
    torch.manual_seed(408177)
    x = (torch.randn(n, d, device='cuda', dtype=torch.bfloat16) * .3).requires_grad_()
    gu = (torch.randn(e, 2 * h, d, device='cuda', dtype=torch.bfloat16) * .05).requires_grad_()
    down = (torch.randn(e, d, h, device='cuda', dtype=torch.bfloat16) * .05).requires_grad_()
    ns = padded_token_count(n, e, k)
    balanced = torch.arange(ns * k, device='cuda').view(ns, k) % e
    routes = [BalancedRouting(*routing_metadata(torch.randn(ns, e, device='cuda'),
                                               torch.randperm(e, device='cuda')[balanced],
                                               num_real_tokens=n)) for _ in range(2)]
    ids, weights, inverse = (t.clone() for t in routes[0])
    weights.requires_grad_()
    leaves = x, gu, down, weights
    dy = torch.randn(d, n, device='cuda').T
    assert not dy.is_contiguous()

    def public(config):
        def apply(x, gu, down, ids, weights, inverse):
            return mxfp8_routed_experts(x, gu, down, BalancedRouting(ids, weights, inverse),
                                       config=config, autotune='off')
        return apply

    def step(fn):
        out = fn(x, gu, down, ids, weights, inverse)
        return out, *torch.autograd.grad(out, leaves, dy)

    c = _config('token', 'token')
    key, dtypes, sizes, views, _ = layout((n, d, e, ids.shape[1], k, h), _config_key(c), 'forward')
    args = [x.detach(), gu.detach().flatten(0, 1), down.detach().flatten(0, 1), ids,
            weights.detach(), inverse]
    checks = torch.library.opcheck(execute, (args, allocate(x, dtypes, sizes, views), key),
                                   test_utils=('test_schema', 'test_faketensor'))
    assert set(checks.values()) == {'SUCCESS'}

    baseline = public(DEFAULT_MOE_CONFIG)
    compiled = torch.compile(public(c), fullgraph=True, dynamic=False,
                             options={'triton.cudagraphs': False})
    main_stream = torch.cuda.current_stream()
    side_stream = torch.cuda.Stream()
    side_stream.wait_stream(main_stream)
    with torch.cuda.stream(side_stream):
        warm = step(compiled)
        initial = warm[0].detach().clone()
        del warm
    side_stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side_stream):
        captured = step(compiled)
    main_stream.wait_stream(side_stream)
    try:
        for index in (1, 0):
            with torch.no_grad():
                ids.copy_(routes[index].expert_ids)
                weights.copy_(routes[index].route_weights).mul_(.97)
                inverse.copy_(routes[index].inverse)
                weights.flatten()[inverse[-1]] = 0.
                x.add_(.0625)
                gu.mul_(.9375)
                down.mul_(1.125)
                dy.mul_(-.875)
            graph.replay()
            side_stream.wait_stream(main_stream)
            with torch.cuda.stream(side_stream):
                same_arithmetic = step(baseline)
            main_stream.wait_stream(side_stream)
            independent = operational_reference(x, gu, down, ids, weights, inverse, dy)
            assert not torch.equal(_bits(captured[0]), _bits(initial))
            for name, actual, scalar, oracle in zip(
                    ('output', 'dx', 'dgu', 'ddown', 'droute'), captured, same_arithmetic,
                    independent, strict=True):
                assert torch.isfinite(actual).all(), name
                assert torch.equal(_bits(actual), _bits(scalar)), name
                delta = actual.detach().double() - oracle.double()
                maximum = delta.abs().max().item()
                nrms = (delta.norm() / oracle.double().norm().clamp_min(1e-12)).item()
                record_property(f'{name}_replay{index}_max_abs_nrms', [maximum, nrms])
                assert nrms < .006 or maximum < 2e-7, (name, maximum, nrms)
            del same_arithmetic, independent, delta
        torch.cuda.synchronize()
        live = torch.cuda.memory_allocated()
        for _ in range(5):
            graph.replay()
            torch.cuda.synchronize()
            assert torch.cuda.memory_allocated() == live
        record_property('replay_live_bytes', live)
    finally:
        graph.reset()
        torch.compiler.reset()
