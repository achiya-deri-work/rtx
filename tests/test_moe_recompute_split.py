"""Independent byte-level qualification of split-team joint recompute."""

from dataclasses import replace

import pytest
import torch


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='SM120 CUDA required')


def _encode(values, math='fp32'):
    """Torch block quantization, with mathematical E8M0 rounding and normal floor."""
    blocks = values.float().unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(-1, keepdim=True)
    fraction, shift = torch.frexp(amax)
    exponent = torch.where(amax == 0, -127, shift - 9 + (fraction > .875)).clamp(-127, 127).float()
    scale = torch.exp2(exponent)
    scaled = blocks / scale.clamp_min(2.**-126)
    if math == 'bf16x2':
        # Nonzero blocks in this fixture have an exactly representable BF16
        # reciprocal. Round the multiply as standalone BF16x2 would do.
        scaled = scaled.bfloat16().float()
    return scaled.clamp(-448, 448).flatten(-2).to(torch.float8_e4m3fn), (exponent[..., 0] + 127).byte()


def _decode(data, scale):
    return data.double() * torch.exp2(scale.double() - 127).repeat_interleave(32, dim=-1)


def _oracle(qx, qw, sx, sw, dh, math):
    # The fixture uses small integer FP8 entries and dyadic scales, making
    # the projection itself exact also in FP32. No production kernels or tile
    # geometry enter this reference, including quantization and transposes.
    projection = (_decode(qx, sx) @ _decode(qw, sw).mT).float()
    gate, up = projection[..., 0::2], projection[..., 1::2]
    denom = 1 + torch.exp(-gate)
    sigmoid = denom.reciprocal()
    act = gate / denom
    derivative = sigmoid * (1 + gate * (1 - sigmoid))
    dg = (dh.float() * up * derivative).bfloat16()
    du = (dh.float() * act).bfloat16()
    qh, sh = _encode(act * up)
    qht, sht = _encode(_decode(qh, sh).bfloat16().mT, math)
    qp, sp = _encode(torch.stack((dg, du), dim=2).flatten(1, 2), math)
    qpt, spt = _encode(torch.cat((dg, du), dim=-1).mT, math)
    return qh, sh, qht, sht, qp, sp, qpt, spt


def _assert_bytes(actual, expected):
    names = ('qh', 'sh', 'qht', 'sht', 'qp', 'sp', 'qpt', 'spt')
    mismatches = {}
    for name, a, b in zip(names, actual, expected, strict=True):
        a, b = a.view(torch.uint8).reshape(-1), b.view(torch.uint8).reshape(-1)
        if not torch.equal(a, b):
            bad = (a != b).nonzero().flatten()
            sample = bad[:6]
            mismatches[name] = (bad.numel(), sample.tolist(), a[sample].tolist(), b[sample].tolist())
    assert not mismatches, mismatches


def test_split_recompute_scratch_and_warp_ownership():
    from rtx.configs.moe import DEFAULT_PREPARED_MOE_CONFIG, joint_recompute_rejection
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.kernels.mxfp8_moe_recompute_split import split_recompute_scratch_elems
    cfg = replace(DEFAULT_PREPARED_MOE_CONFIG.recompute, tile_m=64, tile_n=64, tile_k=128,
                  stages=1, scale_role='producer', glu_grad_store='exchange')
    problem = MXFP8Problem(96, 192, 160)
    assert split_recompute_scratch_elems(cfg) * 4 == 4160
    for warps in range(1, cfg.num_threads // 32):
        assert joint_recompute_rejection(problem, cfg, 'split_warp', gradient_warps=warps) is None
    for warps in (0, cfg.num_threads // 32, -1, True):
        assert joint_recompute_rejection(problem, cfg, 'split_warp', gradient_warps=warps)
    assert joint_recompute_rejection(problem, replace(cfg, stages=1, tile_n=128), 'split_warp')
    assert joint_recompute_rejection(problem, replace(cfg, persistent_waves=1), 'split_warp')


def test_independent_infinity_scale_boundary():
    maximum = torch.tensor([448 * 2.**e for e in (-14, -9, -3, 0, 3)])
    values = torch.stack((maximum, torch.nextafter(maximum, torch.full_like(maximum, torch.inf))))
    _, scale = _encode(values[..., None].expand(-1, -1, 32))
    expected = torch.tensor([113, 118, 124, 127, 130], dtype=torch.uint8)
    assert torch.equal(scale[0, :, 0], expected)
    assert torch.equal(scale[1, :, 0], expected+1)


@cuda
@pytest.mark.parametrize('gradient_warps,n_tile,stages,vec,swizzle,math,amax', [
    (2, 64, 1, 1, 'none', 'fp32', 'bf16_bits'),
    (3, 64, 1, 4, 'xor_16b', 'fp32', 'bf16_bits'),
    (1, 128, 2, 2, 'xor_16b', 'bf16x2', 'fp32'),
    (4, 128, 2, 8, 'none', 'fp32', 'bf16_bits'),
])
def test_split_recompute_exact_replay(gradient_warps, n_tile, stages, vec, swizzle, math, amax):
    from rtx.configs.moe import DEFAULT_PREPARED_MOE_CONFIG
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.kernels.mxfp8_moe_buffers import compile_expert_joint_recompute

    base = DEFAULT_PREPARED_MOE_CONFIG
    cfg = replace(base.recompute, tile_m=64, tile_n=n_tile, tile_k=128, stages=stages,
                  scale_role='producer', glu_grad_store='exchange', glu_grad_load='interleaved',
                  glu_quant_vec=vec, glu_scratch_swizzle=swizzle)
    quant = replace(base.quant, quant_math=math, quant_amax=amax)
    # Expert tails: M clipped by 32, N clipped by 64 for 128-column tiles,
    # K has one 32-wide physical block in the final stage. Include distinct
    # padding holes per expert and an entire padded token-axis block.
    e, m, h, k = 3, 96, 96, 160
    device = 'cuda'
    f8 = torch.float8_e4m3fn
    byte = torch.uint8
    generator = torch.Generator(device=device).manual_seed(20913)

    def ints(low, high, *shape):
        return torch.randint(low, high, shape, device=device, generator=generator)

    qx = ints(-8, 9, e, m, k).to(f8)
    qw = ints(-8, 9, e, 2*h, k).to(f8)
    sx = ints(120, 123, e, m, k//32).byte()
    sw = ints(120, 123, e, 2*h, k//32).byte()
    dh = (ints(-16, 17, e, m, h).float() / 8).bfloat16()
    padded = torch.arange(m, device=device)[None, :] >= torch.tensor([87, 64, 93], device=device)[:, None]
    qx.view(byte)[padded] = 0
    dh[padded] = 0
    problem = MXFP8Problem(m, 2*h, k)

    def alloc():
        return tuple(torch.empty((e*r, c), device=device, dtype=dtype) for r, c, dtype in (
            (m,h,f8), (m,h//32,byte), (h,m,f8), (h,m//32,byte),
            (2*m,h,f8), (2*m,h//32,byte), (2*h,m,f8), (2*h,m//32,byte)))

    expected, actual = alloc(), alloc()
    baseline = compile_expert_joint_recompute(e, problem, cfg, quant, 'dual_operand')
    candidate = compile_expert_joint_recompute(e, problem, cfg, quant, 'split_warp',
                                               gradient_warps=gradient_warps)

    x, w, xs, ws, d = [v.flatten(0, 1) for v in (qx, qw, sx, sw, dh)]

    def run(outputs, fn):
        fn(x, w, xs, ws, d, (d, *outputs))

    def check():
        run(expected, baseline)
        _assert_bytes(actual, expected)
        oracle = _oracle(qx, qw, sx, sw, dh, math)
        # NVIDIA MMA may canonicalize the sign of exact projection zeros;
        # keep signed zero byte equality in the same-schedule comparison.
        # Oracle compares every nonzero code and scale, allowing signed zero.
        sanitized = tuple(torch.where((v.view(byte) & 127) == 0, 0, v.view(byte))
                          if v.dtype == f8 else v for v in actual)
        refs = tuple(torch.where((v.view(byte) & 127) == 0, 0, v.view(byte))
                     if v.dtype == f8 else v for v in oracle)
        _assert_bytes(sanitized, refs)

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for v in actual:
            v.view(byte).fill_(255)
        run(actual, candidate)
    torch.cuda.current_stream().wait_stream(stream)
    check()
    graph = torch.cuda.CUDAGraph()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.graph(graph, stream=stream):
        run(actual, candidate)

    # Replay changed inputs/upstream into the same storage, not just a
    # repeated exact graph; poison every byte between launches.
    qx.view(byte).copy_(ints(-8, 9, e, m, k).to(f8).view(byte))
    qx.view(byte)[padded] = 0
    dh.mul_(-.5)
    for v in actual:
        v.view(byte).fill_(119)
    graph.replay()
    check()
    torch.cuda.synchronize()
    live = [torch.cuda.memory_allocated()]
    for _ in range(5):
        run(actual, candidate)
        torch.cuda.synchronize()
        live.append(torch.cuda.memory_allocated())
    assert live == [live[0]] * len(live)
