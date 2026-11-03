from dataclasses import replace

import pytest
import torch

from rtx.models.decoder import DecoderConfig, SwiGLU, situ_glu


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_situ_values_and_analytic_gradients(dtype):
    g = torch.tensor([-1000., -25., -4., -.5, 0., .5, 4., 25., 1000.], dtype=dtype, requires_grad=True)
    u = torch.linspace(-100, 100, 9, dtype=dtype).requires_grad_()
    actual = situ_glu(g, u)
    gd, ud = g.detach().double(), u.detach().double()
    t, s, v = (gd / 4).tanh(), gd.sigmoid(), (ud / 25).tanh()
    reference = 100 * t * s * v
    torch.testing.assert_close(actual, reference.to(dtype), rtol=1e-6, atol=1e-6)
    actual.sum().backward()
    dg = ((1 - t.square()) * s + 4 * t * s * (1 - s)) * 25 * v
    du = 4 * t * s * (1 - v.square())
    torch.testing.assert_close(g.grad, dg.to(dtype), rtol=.01, atol=2e-6)
    torch.testing.assert_close(u.grad, du.to(dtype), rtol=.01, atol=2e-6)
    assert actual.abs().max() <= 100


def test_activation_does_not_change_initial_weights_and_dispatches():
    config = DecoderConfig(hidden_size=128, intermediate_size=256, num_attention_heads=4,
                           num_layers=1, mlp_activation='situ')
    torch.manual_seed(17)
    situ = SwiGLU(config, layer_depth=1)
    torch.manual_seed(17)
    silu = SwiGLU(replace(config, mlp_activation='silu'), layer_depth=1)
    for a, b in zip(situ.parameters(), silu.parameters()):
        assert torch.equal(a, b)
    x = torch.randn(2, 128, dtype=torch.bfloat16)
    gate, up = situ.gate_up_proj(x).chunk(2, -1)
    assert torch.equal(situ(x), situ.down_proj(situ_glu(gate, up)))


@pytest.mark.parametrize('cap', [0., -1., float('inf'), float('nan')])
def test_invalid_caps(cap):
    with pytest.raises(ValueError, match='caps'):
        DecoderConfig(situ_beta=cap)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_situ_compiled_cuda_gradients():
    torch.manual_seed(71)
    g = (torch.randn(128, 256, device='cuda') * 8).bfloat16().requires_grad_()
    u = (torch.randn_like(g) * 30).requires_grad_()
    ref = situ_glu(g, u)
    expected_grads = torch.autograd.grad(ref.sum(), (g, u))
    compiled = torch.compile(situ_glu, fullgraph=True, dynamic=False)
    actual = compiled(g, u)
    grads = torch.autograd.grad(actual.sum(), (g, u))
    torch.testing.assert_close(actual, ref, rtol=.008, atol=.004)
    for actual_grad, expected in zip(grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected, rtol=.02, atol=.002)
