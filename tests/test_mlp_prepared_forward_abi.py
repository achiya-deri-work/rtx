"""One compiler-visible MXFP8 forward arena, with unchanged recompute."""
from dataclasses import replace
import json

import pytest
import torch

from rtx.mlp import DEFAULT_MLP_CONFIG, _config_key, mxfp8_mlp


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='requires SM12x CUDA')


def _allocations(shape=(128,384,256,384)):
    from benchmarks.benchmark_mlp_small_forward import tensors
    values, dy = tensors(shape, 8119)
    return values, dy


def test_launcher_schema_fake_and_poisoned_scale_scratch():
    from rtx._mlp_forward import _allocate, _execute
    from rtx.kernels.mxfp8_quant import compile_mxfp8_dual_quant, compile_mxfp8_quant
    x, gu, down = _allocations()[0]
    config = replace(DEFAULT_MLP_CONFIG, gate_backend='prequant', down_backend='prequant')
    key = _config_key(config)
    hidden, scales, gate_scratch, down_scratch = _allocate(x, gu, down, config)
    out = torch.empty((x.shape[0], down.shape[0]), device=x.device, dtype=x.dtype)
    buffers = [out, hidden, scales, *gate_scratch, *down_scratch]
    args = x, gu, down, buffers, 'silu', key
    op = torch.library.opcheck(_execute, args)
    assert all(v=='SUCCESS' for v in op.values()), op

    def launch(poison):
        for t in buffers:
            t.view(torch.uint8).fill_(poison)
        _execute(*args)
        return [t.view(torch.uint8).clone() for t in buffers]
    for actual, expected in zip(launch(0x7f), launch(0xde), strict=True):
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    qx, qgu, sx, sgu = (*gate_scratch[:2], *(s.view(torch.float8_e8m0fnu) for s in gate_scratch[2:]))
    qd, sd = down_scratch[0], down_scratch[1].view(torch.float8_e8m0fnu)
    expected = [torch.empty_like(t) for t in (qx,qgu,sx,sgu,qd,sd)]
    compile_mxfp8_dual_quant(x.shape[0],gu.shape[0],x.shape[1],config.gate_prequant.quant,
        'row_major',interleave_weight=True)(x,gu,*expected[:4])
    compile_mxfp8_quant(*down.shape,config.down_quant)(down,*expected[4:])
    for actual, ref in zip((qx,qgu,sx,sgu,qd,sd),expected,strict=True):
        torch.testing.assert_close(actual.view(torch.uint8),ref.view(torch.uint8),atol=0,rtol=0)


@pytest.mark.parametrize('mask',[(True,False,False),(False,True,False),(False,False,True),(True,False,True)])
def test_prepared_forward_partial_gradients_through_bound_fullgraph(mask):
    from rtx._execution_plan_token import read_plan_token
    from rtx._moe_shared import _cold_shared, selected_shared
    from benchmarks.benchmark_mlp_small_forward import quantized_fp64

    values,dy = _allocations()
    leaves = tuple(x.detach().requires_grad_(need) for x, need in zip(values,mask,strict=True))
    active = tuple(x for x in leaves if x.requires_grad)
    base = DEFAULT_MLP_CONFIG
    config = replace(base, gate_backend='prequant', down_backend='prequant')
    spec = json.dumps(dict(mode='fixed', initial=config.to_dict()), sort_keys=True)
    recorded=[]
    def pack(t):
        recorded.append(t.data_ptr())
        return t
    with torch.autograd.graph.saved_tensors_hooks(pack,lambda t:t):
        y = mxfp8_mlp(*leaves,config=config)
    assert recorded==[t.data_ptr() for t in leaves]
    del y
    cold, token = _cold_shared(*leaves,'silu',spec,True)
    assert read_plan_token(token)==_config_key(config)
    baseline=mxfp8_mlp(*leaves,config=base)
    baseline_grads=torch.autograd.grad(baseline,active,dy)
    for actual,expected in zip(torch.autograd.grad(cold,active,dy),baseline_grads,strict=True):
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
    compiled=torch.compile(lambda *xs:selected_shared(*xs,'silu',spec,True),
                           fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    output=compiled(*leaves)
    del cold, token
    expected=quantized_fp64(*values,dy)
    actual=(output,*torch.autograd.grad(output,active,dy))
    for a,e in zip(actual,(expected[0],*(v for v,need in zip(expected[1:],mask,strict=True) if need)),strict=True):
        diff=a.detach().float()-e.detach().float()
        assert bool(a.isfinite().all())
        assert float(diff.norm()/e.detach().float().norm().clamp_min(1e-12))<.025
    torch.testing.assert_close(output,baseline,rtol=0,atol=0)
    for a,e in zip(actual[1:],baseline_grads,strict=True):
        torch.testing.assert_close(a,e,rtol=0,atol=0)
