"""Byte/scale contracts for dynamically quantizing both expert weight axes."""
from dataclasses import replace

import pytest
import torch

from rtx.configs.mxfp8 import MXFP8QuantConfig
from rtx.configs.moe_backward import backward_weight_quant_rejection


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='SM120 CUDA required')


def _oracle(source, config):
    """Compute both independent policies using numeric, not bitfield, scales."""
    padding = (-source.shape[-1]) % 32
    blocks = torch.nn.functional.pad(source.double(), (0, padding)).unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(-1, keepdim=True)
    # Use an exact power-of-two comparison, including the 448 threshold;
    # applying ceil to a floating-point log can misround an exact boundary.
    fraction, exponent = torch.frexp(amax)
    exponent = exponent - 9
    if config.scale_rounding == 'infinity':
        exponent = exponent + (fraction > 448.0 / 512.0)
    exponent = torch.where(amax == 0, -127, exponent).clamp(-127, 127)
    inverse = torch.ldexp(torch.ones_like(amax), -exponent.clamp_min(-126)).float()
    if config.quant_math == 'bf16x2':
        scaled = (blocks.float() * inverse.bfloat16().float()).bfloat16().float()
    else:
        scaled = blocks.float() * inverse
    output = scaled.clamp(-448, 448).to(torch.float8_e4m3fn).flatten(-2)
    scale = (exponent.squeeze(-1) + 127).to(torch.uint8)
    return output.reshape(-1, output.shape[-1]), scale.reshape(-1, scale.shape[-1])


def _pair(rows, cols, device='cuda', canaries=None):
    padded = (cols + 31) // 32 * 32
    result = []
    for width, dtype in ((padded, torch.float8_e4m3fn), (padded // 32, torch.uint8)):
        if canaries is None:
            result.append(torch.empty((rows, width), device=device, dtype=dtype))
        else:
            raw = torch.full((rows * width + 32,), 193, device=device, dtype=torch.uint8)
            canaries.append(raw[rows * width:])
            result.append(raw[:rows * width].view(dtype).view(rows, width))
    return tuple(result)


def pairs(n, d, e, h, canaries=None):
    return tuple(x for rows, cols in ((n, d), (e * 2 * h, d), (e * d, h), (e * h, d), (e * d, 2 * h))
                 for x in _pair(rows, cols, canaries=canaries))


def _test_config(vec=8, arithmetic='fp32', amax='bf16_bits', rounding='infinity', reduction='redux'):
    return replace(MXFP8QuantConfig(), num_warps=4, quant_vec=vec, load_bits=16 * vec,
                   quant_store_bits=min(32, 8 * vec), quant_math=arithmetic,
                   quant_amax=amax, scale_rounding=rounding, reduction=reduction)


def _expected(x, gu, down, config):
    # Row output is gate[0],up[0],gate[1],up[1],...; transpose is NOT.
    e, g, d = gu.shape
    interleaved = gu.view(e, 2, g // 2, d).transpose(1, 2).reshape(e, g, d)
    return tuple(y for source in (x, interleaved, down, down.mT, gu.mT) for y in _oracle(source, config))


def _inputs(n, d, e, h, shifted=False):
    torch.manual_seed(30431)

    def sample(shape):
        numel = 1
        for size in shape:
            numel *= size
        offset = int(shifted)
        return torch.randn(numel + offset, device='cuda', dtype=torch.bfloat16)[offset:].view(shape)

    x, gu, down = sample((n, d)), sample((e, 2 * h, d)), sample((e, d, h))
    # Force strongly asymmetric row/transpose scales, signed zero, empty
    # blocks, exact scale thresholds and BF16 values around those thresholds.
    gu[0, 0].zero_()
    gu[0, 0, 0] = -0.0
    gu[0, h:] *= 32
    down[0, :, 0] *= 0.03125
    if d >= 4:
        gu[0, h, :4] = torch.tensor([1.734375, 1.75, 1.7578125, -1.75], device='cuda', dtype=torch.bfloat16)
    return x, gu, down


def _assert_bytes(actual, expected):
    names = ('X', 'GU row', 'down row', 'down transpose', 'GU transpose')
    for i, (value, ref) in enumerate(zip(actual, expected, strict=True)):
        assert torch.equal(value.view(torch.uint8), ref.view(torch.uint8)), (names[i // 2], i % 2)


def test_backward_weight_quant_cpu_legality():
    q = _test_config()
    assert backward_weight_quant_rejection(19, 35, 3, 17, q) is None
    for override in ({'tile_rows': 16}, {'tile_rows': 128, 'tile_cols': 128}, {'load_values': 16},
                     {'store_bits': 64}, {'weight_warps': 1}, {'raster': 'tokens'}, {'grid_mode': 'global'}):
        assert backward_weight_quant_rejection(19, 35, 3, 17, q, **override)
    assert backward_weight_quant_rejection(19, 35, 3, 17, _test_config(1))
    assert backward_weight_quant_rejection(128, 128, 2, 128, replace(q, scale_layout='mma128'))
    assert backward_weight_quant_rejection(2**24, 128, 3, 17, q)


def test_backward_weight_quant_replaces_one_plan_step_without_changing_storage():
    from rtx._moe_execution import layout, plan
    from rtx.moe import DEFAULT_MOE_CONFIG, _config_key

    shape = (211, 128, 3, 160, 2, 96)
    p = DEFAULT_MOE_CONFIG.prepared
    base = replace(DEFAULT_MOE_CONFIG, prepared=replace(p, backward_preparation='joint',
                   input_preparation='indexed_transpose', transpose_mode='warp',
                   recompute=replace(p.recompute, glu_grad_store='exchange')))
    candidate = replace(base, prepared=replace(base.prepared, weight_preparation='dual_orientation'))
    for phase in ('forward', 'backward'):
        keys = [layout(shape, _config_key(cfg), phase) for cfg in (base, candidate)]
        assert keys[0][1:] == keys[1][1:], phase
        left, right = [plan(key[0])[0].steps for key in keys]
        if phase == 'forward':
            assert left == right
        else:
            assert len(left) == len(right)
            changed = [pair for pair in zip(left, right, strict=True) if pair[0] != pair[1]]
            assert len(changed) == 1
            separate, dual = changed[0]
            assert (separate[1], dual[1]) == ('compile_expert_quant', 'compile_backward_weight_quant')
            assert dual[3][:3] == (-1, -2, -3)  # X, GU, down
            assert dual[3][3:] == tuple(t for pair in zip(*separate[3][2:]) for t in pair)


@cuda
@pytest.mark.parametrize('shape,config,schedule,shifted', [
    ((103, 128, 3, 64), (8, 'fp32', 'bf16_bits', 'infinity', 'redux'),
     ('operand', 32, 64, 8, 32, 4, 'features'), False),
    ((1, 1, 1, 1), (1, 'fp32', 'fp32', 'floor', 'shuffle'),
     ('operand', 32, 32, 1, 8, 2, 'rows'), True),
    ((19, 35, 3, 17), (2, 'bf16x2', 'bf16_bits', 'infinity', 'shuffle'),
     ('expert', 64, 64, 4, 16, 4, 'features'), True),
    ((97, 71, 5, 65), (4, 'fp32', 'fp32', 'infinity', 'redux'),
     ('portfolio', 128, 32, 2, 32, 8, 'rows'), False),
    ((17, 64, 2, 33), (4, 'bf16x2', 'fp32', 'floor', 'shuffle'),
     ('operand', 32, 128, 8, 32, 8, 'features'), True),
])
def test_backward_weight_quant_independent_byte_oracle(shape, config, schedule, shifted):
    from rtx.kernels.mxfp8_moe_weight_quant import compile_backward_weight_quant

    x, gu, down = _inputs(*shape, shifted)
    cfg = _test_config(*config)
    guards = []
    outputs = pairs(*shape, canaries=guards)
    call = compile_backward_weight_quant(*shape, cfg, *schedule)
    call(x, gu.flatten(0, 1), down.flatten(0, 1), *outputs)
    _assert_bytes(outputs, _expected(x, gu, down, cfg))
    assert all(torch.all(guard == 193).item() for guard in guards)


@cuda
def test_backward_weight_quant_graph_fresh_dynamic_weights_and_no_live_growth():
    from rtx.kernels.mxfp8_moe_buffers import compile_expert_quant
    from rtx.kernels.mxfp8_moe_weight_quant import compile_backward_weight_quant

    n, d, e, h = 211, 128, 3, 96
    x, gu, down = _inputs(n, d, e, h)
    cfg = _test_config()
    outputs = pairs(n, d, e, h)
    control = pairs(n, d, e, h)
    call = compile_backward_weight_quant(n, d, e, h, cfg, 'portfolio')
    old = compile_expert_quant(e, ((n, d, 'row', False, 1), (2*h, d, 'row', True), (d, h, 'row', False),
                                   (h, d, 'transpose', False), (d, 2*h, 'transpose', False)),
                               cfg, 'warp', 'portfolio')
    gg, dd = gu.flatten(0, 1), down.flatten(0, 1)

    def launch():
        call(x, gg, dd, *outputs)

    launch()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch()
    live = []
    for exponent in (0, 3, -4, 2, -1):
        gu.mul_(-(2.0 ** exponent))
        down.mul_(2.0 ** (-exponent))
        # Original X still refreshes and uses the same singleton quantizer.
        x.neg_()
        graph.replay()
        old(x, (x, gg, dd, dd, gg), control[::2], control[1::2])
        _assert_bytes(outputs, control)
        _assert_bytes(outputs, _expected(x, gu, down, cfg))
        torch.cuda.synchronize()
        live.append(torch.cuda.memory_allocated())
    assert len(set(live)) == 1, live
