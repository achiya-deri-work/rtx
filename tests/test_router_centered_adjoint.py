"""Independent FP64 checks of cancellation-sensitive selected-softmax gradients."""
import pytest
import torch

from rtx._routing_reference import selected_softmax


@pytest.mark.parametrize('k',[1,4,6,8,32])
@pytest.mark.parametrize('offset',[0.,1.,1000.])
def test_selected_softmax_centered_gradient(k,offset):
    generator=torch.Generator().manual_seed(913+k)
    values=torch.randn(47,k,generator=generator).requires_grad_()
    upstream=offset+torch.randn(47,k,generator=generator)*1e-4
    reference=values.detach().double().requires_grad_()
    expected=torch.autograd.grad(reference.softmax(-1),reference,upstream.double())[0]
    result=selected_softmax(values)
    actual=torch.autograd.grad(result,values,upstream)[0]
    torch.testing.assert_close(result,values.softmax(-1),rtol=0,atol=0)
    torch.testing.assert_close(actual.double(),expected,rtol=3e-5,atol=2e-10)
    constant=selected_softmax(values)
    assert not torch.autograd.grad(constant,values,torch.ones_like(constant))[0].count_nonzero()


def test_tuner_torch_adjoint_matches_actual_centered_backward():
    from rtx.autotune.routing import _metadata_gradient_reference
    from rtx._routing_reference import routing_metadata
    from tests.test_routing_metadata import assignments
    scores=torch.randn(48,12,requires_grad=True)
    ids=assignments(48,12,6)
    _,weights,inverse=routing_metadata(scores,ids,num_real_tokens=47)
    for offset in (0.,1.,1000.):
        upstream=offset+torch.randn_like(weights)*1e-4
        actual=torch.autograd.grad(weights,scores,upstream,retain_graph=True)[0]
        estimated=_metadata_gradient_reference(upstream,weights,ids,inverse,48,12)
        torch.testing.assert_close(estimated,actual,rtol=0,atol=0)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('k',[1,4,6,32])
def test_native_metadata_centered_gradient(k):
    from rtx._routing_metadata import routing_metadata
    from tests.test_routing_metadata import assignments
    n,e=128,32
    torch.manual_seed(913+k)
    values=torch.randn(n,e,device='cuda',requires_grad=True)
    ids=assignments(n,e,k,'cuda')
    fn=torch.compile(lambda v:routing_metadata(v,ids,num_real_tokens=n),fullgraph=True,dynamic=False)
    _,weights,inverse=fn(values)
    reference=values.detach().double().requires_grad_()
    selected=reference.gather(1,ids).softmax(-1)
    ref_weights=torch.empty(n*k,device='cuda',dtype=torch.float64).scatter(0,inverse.flatten(),selected.flatten())
    for offset in [1.,1000.]:
        upstream=offset+torch.randn_like(weights)*1e-4
        actual=torch.autograd.grad(weights,values,upstream,retain_graph=True)[0]
        expected=torch.autograd.grad(ref_weights,reference,upstream.double().flatten(),retain_graph=True)[0]
        torch.testing.assert_close(actual.double(),expected,rtol=5e-5,atol=2e-10)
    assert not torch.autograd.grad(weights,values,torch.ones_like(weights))[0].count_nonzero()
