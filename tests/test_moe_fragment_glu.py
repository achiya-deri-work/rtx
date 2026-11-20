"""Forward-only fragment exchange: raw bytes, independent FP64, and replay."""
from dataclasses import replace

import pytest
import torch


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='SM120 CUDA required')
ENGINES = ('fragment_direct', 'fragment_packed')


def _schedule(tile_m, tile_n, stages, scale_role, vec=4):
    from rtx.configs.moe import DEFAULT_PREPARED_MOE_CONFIG
    return replace(DEFAULT_PREPARED_MOE_CONFIG.gate_up, tile_m=tile_m, tile_n=tile_n,
                   atom_layout_m=2 if tile_m == 64 else 8,
                   stages=stages, scale_role=scale_role, glu_quant_vec=vec,
                   consumer_registers=192 if tile_m == 64 else 112)


def _quantize_fp64(source):
    """Independent mathematical E8M0 recipe and nearest-even E4M3 cast."""
    d = source.double()
    d = torch.nn.functional.pad(d, (0, (-d.shape[-1]) % 32))
    blocks = d.unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(-1)
    exponent = (amax / 448).log2().ceil().clamp(-127, 127)
    # E8M0 code 0 decodes to 2^-127; the production FP32 reciprocal
    # intentionally clamps its denominator to the smallest normal FP32.
    divisor = torch.exp2(exponent.clamp_min(-126))
    q = (blocks / divisor[..., None]).to(torch.float8_e4m3fn).flatten(-2)
    return q, (exponent + 127).to(torch.uint8)


def _decode_fp64(q, code):
    return (q.double().unflatten(-1, (-1, 32)) * torch.exp2(code.double() - 127)[..., None]).flatten(-2)


def _oracle(x, weights, e, m, h, k):
    qx, sx = _quantize_fp64(x.view(e, m, k))
    qw, sw = _quantize_fp64(weights.view(e, 2 * h, k))
    # No dependency on the implementation's interleaving, fragments,
    # warp reductions, E8 bit manipulation, or native GEMM summation order.
    a = torch.bmm(_decode_fp64(qx, sx), _decode_fp64(qw, sw).transpose(1, 2))
    gate, up = a.chunk(2, dim=-1)
    qh, sh = _quantize_fp64(gate / (1 + (-gate).exp()) * up)
    return qh.flatten(0, 1).view(torch.uint8), sh.flatten(0, 1)


def _inputs(e, m, h, k):
    g = torch.Generator(device='cuda').manual_seed(129994)
    x = torch.randint(-3, 4, (e, m, k), generator=g, device='cuda').bfloat16()
    w = torch.randint(-3, 4, (e, 2 * h, k), generator=g, device='cuda').bfloat16()
    # Vary scales along row, hidden block, axis, and K partition rather than
    # comparing a product with only unit or one constant scale code.
    x *= torch.exp2(-4 + (torch.arange(m, device='cuda') % 5)[:, None] - torch.arange(k, device='cuda') // 32)
    w *= torch.exp2(-6 + (torch.arange(2*h, device='cuda') % h // 32)[:, None])
    x[:, -3:] = 0
    if e > 1:
        x[1, 0] = 0
    return x.flatten(0, 1), w.flatten(0, 1)


def _buffers(e, m, h, k):
    q = lambda r, d: torch.empty((e*r, d), device='cuda', dtype=torch.float8_e4m3fn)
    s = lambda r, d: torch.empty((e*r, d//32), device='cuda', dtype=torch.uint8)
    p = (h+31)//32*32
    return (q(m, k), q(2*h, k), s(m, k), s(2*h, k)), (q(m, p), s(m, p))


def _hidden(buffers):
    qh, sh = buffers
    return qh.view(torch.uint8), sh


def _equal(left, right):
    for a, b in zip(_hidden(left), _hidden(right)):
        assert torch.equal(a, b), int(torch.count_nonzero(a != b))


def test_fragment_glu_cpu_contract():
    from rtx.configs.moe_backward import fragment_glu_rejection, fragment_glu_scratch_bytes
    from rtx.kernels.mxfp8 import MXFP8Problem

    problem = MXFP8Problem(79, 130, 160)
    config = _schedule(64, 128, 2, 'consumers')
    assert [fragment_glu_scratch_bytes(config, engine) for engine in ENGINES] == [1024, 5120]
    for engine in ENGINES:
        assert fragment_glu_rejection(problem, config, engine) is None
        assert 'nonsingleton' in fragment_glu_rejection(replace(problem, m=1), config, engine)
        assert 'paired' in fragment_glu_rejection(replace(problem, n=131), config, engine)
        assert 'layout' in fragment_glu_rejection(problem, replace(config, tile_n=256, atom_layout_n=4), engine)
        assert 'SiLU' in fragment_glu_rejection(problem, config, engine, 'relu')
        assert 'SiLU' in fragment_glu_rejection(problem, replace(config, glu_silu_math='shared_reciprocal'), engine)
        assert 'static' in fragment_glu_rejection(problem, replace(config, tiles_per_cta=2), engine)


@cuda
@pytest.mark.parametrize('e,m,h,k,tile_m,tile_n,stages,role', [
    (2, 8, 32, 32, 64, 64, 1, 'producer'),
    (3, 79, 65, 160, 64, 128, 2, 'consumers'),
    (2, 97, 96, 160, 128, 64, 1, 'producer'),
    (2, 137, 65, 128, 128, 128, 2, 'consumers'),
])
def test_fragment_glu_bytes_oracle_and_tails(e, m, h, k, tile_m, tile_n, stages, role):
    from rtx.configs.moe import DEFAULT_PREPARED_MOE_CONFIG
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.kernels.mxfp8_moe_buffers import compile_expert_packed_gemm, compile_expert_quant

    x, w = _inputs(e, m, h, k)
    inputs, expected = _buffers(e, m, h, k)
    _, actual = _buffers(e, m, h, k)
    qx, qw, sx, sw = inputs
    prepare = compile_expert_quant(e, ((m, k, 'row', False), (2*h, k, 'row', True)),
                                   DEFAULT_PREPARED_MOE_CONFIG.quant)
    prepare(x, (x, w), (qx, qw), (sx, sw))
    problem = MXFP8Problem(m, 2*h, k)
    config = _schedule(tile_m, tile_n, stages, role)
    old = compile_expert_packed_gemm(e, problem, config, 'glu')
    old(*inputs, *expected)
    oq, os = _oracle(x, w, e, m, h, k)
    assert os.unique().numel() >= 5
    assert torch.equal(expected[1], os)
    # The integer-power-of-two fixture aims for exact quantization. If
    # SiLU evaluation differs near FP8 midpoints, bound the decoded error as
    # well as the number of changed codes without changing the schedule gate.
    qref, qactual = oq.view(torch.float8_e4m3fn), expected[0]
    r, a = _decode_fp64(qref, os), _decode_fp64(qactual, os)
    delta = (a-r).abs()
    count = int((oq != qactual.view(torch.uint8)).sum())
    nrms = float(delta.norm()/r.norm().clamp_min(1e-12))
    assert torch.isfinite(a).all() and count <= .0005 * oq.numel() and nrms < .0015, (count, nrms, float(delta.max()))
    for engine in ENGINES:
        new = compile_expert_packed_gemm(e, problem, config, 'glu', glu_engine=engine)
        for poison in (255, 191):
            for out in _hidden(actual):
                out.fill_(poison)
            new(*inputs, *actual)
            _equal(actual, expected)
            assert (actual[0].view(torch.uint8)[:, h:] == 0).all()
            assert (actual[1].view(e, m, -1)[:, -3:] == 0).all()


@cuda
@pytest.mark.parametrize('engine', ENGINES)
def test_fragment_glu_changed_input_graph_replay_and_live_bytes(engine):
    from rtx.configs.moe import DEFAULT_PREPARED_MOE_CONFIG
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.kernels.mxfp8_moe_buffers import compile_expert_packed_gemm, compile_expert_quant

    e, m, h, k = 2, 83, 65, 160
    x, w = _inputs(e, m, h, k)
    inputs, expected = _buffers(e, m, h, k)
    _, actual = _buffers(e, m, h, k)
    qx, qw, sx, sw = inputs
    prepare = compile_expert_quant(e, ((m, k, 'row', False), (2*h, k, 'row', True)),
                                   DEFAULT_PREPARED_MOE_CONFIG.quant)
    problem = MXFP8Problem(m, 2*h, k)
    config = _schedule(64, 128, 2, 'consumers', vec=1)
    old = compile_expert_packed_gemm(e, problem, config, 'glu')
    new = compile_expert_packed_gemm(e, problem, config, 'glu', glu_engine=engine)
    def quantize():
        prepare(x, (x, w), (qx, qw), (sx, sw))
    def operation():
        quantize()
        new(*inputs, *actual)

    quantize()
    old(*inputs, *expected)
    operation()
    _equal(actual, expected)
    original_q = actual[0].view(torch.uint8).clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    torch.cuda.synchronize()
    live = [torch.cuda.memory_allocated()]
    for repeat in range(4):
        # Change both BF16 input and gate/up weights in-place at the captured
        # addresses, rerunning quantization, with no new launcher allocations.
        x.mul_(-.75)
        w.mul_(.875)
        graph.replay()
        old(*inputs, *expected)
        torch.cuda.synchronize()
        _equal(actual, expected)
        assert not torch.equal(original_q, actual[0].view(torch.uint8)), repeat
        live.append(torch.cuda.memory_allocated())
    assert len(set(live)) == 1, live


@cuda
def test_fragment_glu_public_fullgraph_keeps_backward_and_saved_tensor_contract():
    from rtx._moe_reference import operational_reference
    from rtx.moe import DEFAULT_MOE_CONFIG, mxfp8_routed_experts
    from rtx.routing import BalancedRouting
    from tests.test_mxfp8_moe import make_route

    n, d, e, k, h = 137, 97, 4, 2, 64
    torch.manual_seed(221019)
    x = (torch.randn(n, d, device='cuda', dtype=torch.bfloat16)*.3).requires_grad_()
    gu = (torch.randn(e, 2*h, d, device='cuda', dtype=torch.bfloat16)*.05).requires_grad_()
    down = (torch.randn(e, d, h, device='cuda', dtype=torch.bfloat16)*.05).requires_grad_()
    ids, rw, inverse = make_route(n, e, k, 'cuda')
    weights = rw.detach().requires_grad_()
    leaves = x, gu, down, weights
    dy = torch.randn(d, n, device='cuda').T
    cfg = replace(DEFAULT_MOE_CONFIG, prepared=replace(DEFAULT_MOE_CONFIG.prepared,
                      gate_up=_schedule(64, 64, 1, 'producer')))
    packed = replace(cfg, prepared=replace(cfg.prepared, forward_glu_engine='fragment_packed'))

    def model(plan):
        def forward(x, gu, down, weights):
            return mxfp8_routed_experts(x, gu, down, BalancedRouting(ids, weights, inverse), config=plan)
        return forward

    def step(f):
        y = f(*leaves)
        return y, *torch.autograd.grad(y, leaves, dy)

    saved = []
    def pack(value):
        saved.append(value.data_ptr())
        return value
    with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
        eager = step(model(packed))
    assert set(saved) == {value.data_ptr() for value in (x, gu, down, ids, weights, inverse)}
    assert len(saved) == 6

    compiled = torch.compile(model(packed), fullgraph=True, dynamic=False,
                             options={'triton.cudagraphs': False})
    actual, old = step(compiled), step(model(cfg))
    ref = operational_reference(x, gu, down, ids, weights, inverse, dy)
    for i, (a, b, c, r) in enumerate(zip(actual, old, eager, ref)):
        assert torch.equal(a, b) and torch.equal(a, c), i
        assert torch.isfinite(a).all()
        delta = (a.detach().double()-r.double()).abs()
        nrms = float(delta.norm()/r.double().norm().clamp_min(1e-12))
        assert nrms < (.003 if i == 0 else .006) or float(delta.max()) < 2e-7, (i, nrms)
