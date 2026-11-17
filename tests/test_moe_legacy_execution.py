"""Compiled allocator-visible execution for retained fused/on-chip families."""
from dataclasses import replace

import pytest
import torch

cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='SM120 CUDA required')


@cuda
@pytest.mark.parametrize('family',['grouped_staged','grouped_onchip'])
@pytest.mark.parametrize('compiled',[False,True])
def test_retained_family_all_gradients_and_inactive_prepared_padding(family,compiled):
    from rtx.moe import DEFAULT_MOE_CONFIG,mxfp8_routed_experts
    from rtx.autotune.moe import structural_seeds,rejection
    from rtx.routing import BalancedRouting
    from rtx._moe_reference import operational_reference
    from tests.test_mxfp8_moe import make_route
    n,d,e,k,h=32,128,4,2,128
    shape=(n,d,e,16,k,h)
    candidates=[c for c in structural_seeds(shape) if c.family==family and rejection(c,shape) is None]
    assert candidates,f'No legal {family} structural seed'
    config=candidates[0]
    config=replace(config,prepared=replace(config.prepared,capacity_multiple=128))
    torch.manual_seed(944)
    x=(torch.randn(n,d,device='cuda',dtype=torch.bfloat16)*.3).requires_grad_()
    gu=(torch.randn(e,2*h,d,device='cuda',dtype=torch.bfloat16)*.05).requires_grad_()
    down=(torch.randn(e,d,h,device='cuda',dtype=torch.bfloat16)*.05).requires_grad_()
    ids,weights,inverse=make_route(n,e,k,'cuda');weights=weights.detach().requires_grad_()
    def call(x,gu,down,weights):
        return mxfp8_routed_experts(x,gu,down,BalancedRouting(ids,weights,inverse),config=config)
    if compiled:call=torch.compile(call,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    dy=torch.randn_like(x,dtype=torch.float32);result=call(x,gu,down,weights)
    actual=(result,*torch.autograd.grad(result,(x,gu,down,weights),dy))
    actuals=[actual]
    if not compiled:
        from rtx.moe import _run_forward,_run_backward
        # Offline tuning must exercise the same allocator/launch plan as the
        # public path, including legacy families and inactive prepared fields.
        actuals.append((_run_forward(x,gu,down,ids,weights,inverse,config),
            *_run_backward(x,gu,down,ids,weights,inverse,dy,config)))
    expected=operational_reference(x,gu,down,ids,weights,inverse,dy)
    for variant,values in enumerate(actuals):
        for name,value,reference in zip(('out','dx','dgu','ddown','dweights'),values,expected):
            delta=value.detach().float()-reference.float()
            nrms=float(delta.norm()/reference.float().norm().clamp_min(1e-12));maximum=float(delta.abs().max())
            assert bool(value.isfinite().all()) and (nrms<(.003 if name=='out' else .006) or maximum<2e-7),(variant,name,nrms,maximum)
