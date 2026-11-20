"""Independent quantized oracle for on-chip indexed gate dW."""
from dataclasses import replace

import pytest
import torch

from rtx.configs.moe import MXFP8PreparedMoEConfig
from rtx.kernels.mxfp8 import MXFP8Problem


def quantized_reference(x, config):
    """CPU/reference reduction; each last-axis block owns an E8M0 scale."""
    values = x.float().unflatten(-1, (-1, 32))
    amax = values.abs().amax(-1)
    mantissa, exponent = torch.frexp(amax)
    exponent = exponent - 9
    if config.scale_rounding == 'infinity':
        exponent = exponent + (mantissa > .875)
    exponent = torch.where(amax == 0, -127, exponent).clamp(-127, 127)
    scale = torch.ldexp(torch.ones_like(amax), exponent)
    normalized = values.double() / scale.double().unsqueeze(-1)
    if config.quant_math == 'bf16x2':
        # Match the BF16 encoding path's pre-E4M3 rounding.
        normalized = normalized.to(torch.bfloat16).float()
    encoded = normalized.clamp(-448, 448).to(torch.float8_e4m3fn)
    decoded = encoded.double() * scale.double().unsqueeze(-1)
    return encoded.flatten(-2), (exponent + 127).to(torch.uint8), decoded.flatten(-2)


def raw_fixture(tokens, experts, problem, seed, device, offset_x=False):
    m, d, slots = problem.m, problem.n, problem.k
    generator = torch.Generator().manual_seed(seed)
    x = torch.randn(tokens, d, generator=generator) * torch.exp2(torch.arange(d) % 11 - 5)
    x = x.bfloat16().to(device)
    if offset_x:
        offset = torch.empty(tokens*d+1, dtype=x.dtype, device=device)[1:].view_as(x)
        offset.copy_(x)
        x = offset
    p = torch.randn(experts, m, slots, generator=generator) * (torch.arange(slots)%20+1) / 100
    p = p.bfloat16().to(device)
    mapping = torch.randint(0, tokens, (experts, slots), generator=generator, dtype=torch.int32)
    mapping[0, 3::7] = -1
    mapping[0, slots//2:slots//2+13] = -1
    if experts >= 2:
        mapping[1, :] = -1
    if experts >= 3:
        mapping[2, -4:] = tokens + 123
    return x, p, mapping.to(device)


def oracle(x, p_decoded, mapping, quant):
    valid = (mapping >= 0) & (mapping < x.shape[0])
    gathered = x[mapping.long().clamp(0, x.shape[0]-1)].masked_fill(~valid[:, :, None], 0)
    _, _, x_decoded = quantized_reference(gathered.mT.contiguous(), quant)
    return (p_decoded @ x_decoded.mT).bfloat16()


def error_summary(out, reference):
    actual = out.float()
    ref = reference.float()
    delta = actual - ref
    rms = float(delta.square().mean().sqrt())
    ref_rms = float(ref.square().mean().sqrt())
    absolute = float(delta.abs().max())
    return dict(nonfinite=int((~actual.isfinite()).sum()), max_abs=absolute, rms=rms, ref_rms=ref_rms,
                nrms=rms / max(ref_rms, 1e-12), nmax=absolute / max(float(ref.abs().max()), 1e-12))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='assigned RTX CUDA required')
@pytest.mark.parametrize('tile_k,slots,lanes,pairs,producers,quant_math,offset', [
    (128, 160, 8, 1, 0, 'fp32', False),
    (128, 224, 8, 1, 4, 'fp32', False),
    (128, 224, 4, 2, 0, 'fp32', True),
    (128, 224, 4, 2, 4, 'fp32', True),
    (64, 96, 2, 1, 0, 'bf16x2', False),
    (32, 96, 8, 2, 2, 'fp32', False),
])
def test_indexed_gate_dw_matches_fp64_quantized_oracle(tile_k, slots, lanes, pairs, producers, quant_math, offset):
    from rtx.kernels.mxfp8_moe_indexed_dw import compile_indexed_gate_dw

    tokens, experts = 79, 3
    problem = MXFP8Problem(96, 96, slots)
    prepared = MXFP8PreparedMoEConfig()
    gemm = replace(prepared.gate_dw, tile_k=tile_k)
    quant = replace(prepared.quant, quant_math=quant_math)
    x, p, mapping = raw_fixture(tokens, experts, problem, 46140 + slots, 'cuda', offset_x=offset)
    qpt, spt, p_decoded = quantized_reference(p.cpu(), quant)
    expected = oracle(x.cpu(), p_decoded, mapping.cpu(), quant)
    qp = qpt.flatten(0, 1).cuda()
    sp = spt.flatten(0, 1).cuda()
    map_tensor = mapping.reshape(-1, 1)
    actual = torch.full((experts*problem.m, problem.n), float('nan'), dtype=torch.bfloat16, device='cuda')
    launch = compile_indexed_gate_dw(tokens, experts, problem, gemm, quant, lanes, pairs, producers)
    launch(qp, sp, x, map_tensor, actual)
    result = actual.cpu().reshape(experts, problem.m, problem.n)
    error = error_summary(result, expected)
    assert error['nonfinite'] == 0 and error['nrms'] < .001 and error['nmax'] < .005, error
    assert result[1].count_nonzero().item() == 0
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launch(qp, sp, x, map_tensor, actual)
    actual.fill_(float('nan'))
    graph.replay()
    assert torch.equal(result, actual.cpu().view_as(result))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='assigned RTX CUDA required')
def test_indexed_gate_dw_static_fullgraph_packed_pools_changed_dy():
    from rtx._moe_reference import operational_reference
    from rtx.moe import DEFAULT_MOE_CONFIG, mxfp8_routed_experts
    from rtx.routing import BalancedRouting
    from tests.test_mxfp8_moe import make_route

    lengths, e, k, d, h = (113, 157), 8, 2, 128, 64
    n = sum(lengths)
    torch.manual_seed(5401)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        routes = [make_route(size, e, k, 'cuda') for size in lengths]
        capacities = [((route[0].shape[1] + 31)//32)*32 for route in routes]
        capacity = sum(capacities)
        ids_parts, weight_parts, inverse_parts = [], [], []
        token_offset = slot_offset = 0
        for size, cap, (ids, weights, inverse) in zip(lengths, capacities, routes):
            c = ids.shape[1]
            ids_parts.append(torch.nn.functional.pad(ids+token_offset, (0,cap-c), value=token_offset))
            weight_parts.append(torch.nn.functional.pad(weights, (0,cap-c)))
            inverse_parts.append(inverse//c*capacity + inverse%c + slot_offset)
            token_offset += size
            slot_offset += cap
        ids = torch.cat(ids_parts, dim=1)
        inverse = torch.cat(inverse_parts)
        weights = torch.cat(weight_parts, dim=1).requires_grad_()
        x = (torch.randn(n,d,device='cuda').bfloat16()*.3).requires_grad_()
        gu = (torch.randn(e,2*h,d,device='cuda').bfloat16()*.04).requires_grad_()
        down = (torch.randn(e,d,h,device='cuda').bfloat16()*.04).requires_grad_()
        dy = torch.randn(d,n,device='cuda').T
        leaves = x, gu, down, weights
        recompute = replace(DEFAULT_MOE_CONFIG.prepared.recompute,
                            glu_grad_store='exchange', glu_grad_load='interleaved')
        p = replace(DEFAULT_MOE_CONFIG.prepared, backward_preparation='joint',
                    input_preparation='indexed_transpose', transpose_mode='warp',
                    dy_preparation='fused', recompute=recompute)
        baseline = replace(DEFAULT_MOE_CONFIG, family='grouped_prepared', prepared=p)
        candidate = replace(baseline, prepared=replace(p, gate_dw_engine='indexed',
                            indexed_dw_token_lanes=4, indexed_dw_gather_pairs=2,
                            indexed_dw_producer_warps=4))

        def model(x, gu, down, weights):
            return mxfp8_routed_experts(x, gu, down, BalancedRouting(ids,weights,inverse), config=candidate)

        compiled = torch.compile(model, fullgraph=True, dynamic=False, options={'triton.cudagraphs':False})

        def step():
            output = compiled(*leaves)
            return (output, *torch.autograd.grad(output, leaves, dy))

        for _ in range(2):
            warm = step()
            del warm
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            replayed = tuple(value.detach() for value in step())
        for perturbation in (0, .125):
            dy.add_(perturbation)
            graph.replay()
            output = mxfp8_routed_experts(x, gu, down, BalancedRouting(ids,weights,inverse), config=baseline)
            expected = (output, *torch.autograd.grad(output, leaves, dy))
            independent = operational_reference(x,gu,down,ids,weights,inverse,dy)
            for value, reference, ideal in zip(replayed, expected, independent):
                torch.testing.assert_close(value, reference, rtol=0, atol=0)
                error = error_summary(value, ideal)
                assert error['nonfinite'] == 0 and (error['nrms'] < .006 or error['max_abs'] < 2e-7), error
        del output, expected, independent, ideal, value, reference
        stream.synchronize()
        initial = torch.cuda.memory_allocated()
        for _ in range(4):
            graph.replay()
        stream.synchronize()
        assert torch.cuda.memory_allocated() == initial
