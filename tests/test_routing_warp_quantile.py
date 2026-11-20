"""Exact warp-local row order statistics and unchanged QB policy."""
from dataclasses import replace

import pytest
import torch


cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='SM120 CUDA required')


def _ordered_keys(values):
    bits=values.contiguous().view(torch.int32).long()&0xffffffff
    bits=torch.where(values==0,torch.zeros_like(bits),bits)
    keys=torch.where((bits&0x80000000)!=0,(~bits)&0xffffffff,bits^0x80000000)
    return torch.where(values.isnan(),torch.full_like(keys,0xffffffff),keys)


def _network(keys):
    """CPU comparator-network model, checked against independent integer sort."""
    n,e=keys.shape;width=max(32,1<<(e-1).bit_length())
    values=torch.zeros(n,width,dtype=torch.int64);values[:,:e]=keys
    indices=torch.arange(width)
    for level in range(width.bit_length()-1):
        span=1<<(level+1)
        for part in range(level+1):
            distance=1<<(level-part)
            peer=values[:,indices^distance]
            choose_min=((indices&span)==0)==((indices&distance)==0)
            values=torch.where(choose_min[None,:],torch.minimum(values,peer),torch.maximum(values,peer))
    return values


@pytest.mark.parametrize('e',[1,2,16,24,32,48,64,96,128])
def test_comparator_network_preserves_exact_ordered_float_contract(e):
    generator=torch.Generator().manual_seed(761+e)
    values=torch.randn(7,e,generator=generator)
    specials=torch.tensor([float('nan'),float('inf'),float('-inf'),-0.,0.,1.,-1.])
    for row in range(7):values[row,row%e]=specials[row]
    keys=_ordered_keys(values)
    actual=_network(keys)
    for rank in set((0,e//2,e-1)):
        torch.testing.assert_close(actual[:,-1-rank],torch.sort(keys,descending=True).values[:,rank],rtol=0,atol=0)


def test_warp_backend_preserves_old_token_indices_and_legality():
    from rtx.routing import RoutingKernelConfig,ROUTING_TOKEN_SIZE
    from rtx.autotune.routing import make_routing_adapter
    for index,backend in enumerate(('torch','native_column','native','native_warp')):
        config=RoutingKernelConfig(qb_backend=backend)
        token=config.to_token()
        assert len(token)==ROUTING_TOKEN_SIZE and token[8]==index
        assert RoutingKernelConfig.from_token(token)==config
        assert RoutingKernelConfig.from_token(token[:10])==config
        assert RoutingKernelConfig.from_dict(config.to_dict())==config
    adapter=make_routing_adapter(258,129,3,lambda c:None)
    assert adapter.rejection(replace(adapter.initial_config,qb_backend='native_warp')) is not None
    assert adapter.rejection(replace(adapter.initial_config,qb_backend='native')) is None


@cuda
@pytest.mark.parametrize('n,e,rank,threads',[(33,16,0,64),(65,24,12,128),(129,48,6,256),
                                          (17,64,63,64),(9,128,64,128)])
def test_warp_quantile_exact_finite_rows(n,e,rank,threads):
    from rtx._routing_hard import quantile,warp_quantile
    torch.manual_seed(841+e)
    values=torch.randn(n,e,device='cuda',dtype=torch.float32)
    values[0]=0
    values[1]=torch.arange(e,device='cuda').float()%3-1
    actual=warp_quantile(values,rank,threads)
    expected=values.sort(dim=1,descending=True).values[:,rank]
    expected=torch.where(expected==0,torch.zeros_like(expected),expected)
    torch.testing.assert_close(actual.view(torch.int32),expected.contiguous().view(torch.int32),rtol=0,atol=0)
    radix=quantile(values,rank,threads,False)
    torch.testing.assert_close(actual.view(torch.int32),radix.view(torch.int32),rtol=0,atol=0)


@cuda
@pytest.mark.parametrize('rank',[0,12,23])
def test_warp_quantile_nan_infinity_and_signed_zero_match_radix_bits(rank):
    from rtx._routing_hard import quantile,warp_quantile
    special=torch.tensor([float('nan'),float('inf'),float('-inf'),-0.,0.,-1.,1.,-2.],
                         device='cuda',dtype=torch.float32)
    values=special.repeat(5,3).contiguous()
    actual=warp_quantile(values,rank,128)
    radix=quantile(values,rank,128,False)
    torch.testing.assert_close(actual.view(torch.int32),radix.view(torch.int32),rtol=0,atol=0)


@cuda
def test_warp_quantile_has_compiler_visible_output_and_schema_fake_contract():
    from rtx._routing_hard import warp_quantile,warp_quantile_into
    values=torch.randn(17,48,device='cuda',dtype=torch.float32)
    result=torch.empty(17,device='cuda',dtype=torch.float32)
    checks=torch.library.opcheck(warp_quantile_into,(values.flatten(),result,17,48,7,128),
                                test_utils=('test_schema','test_faketensor'))
    assert set(checks.values())=={'SUCCESS'}
    compiled=torch.compile(lambda x:warp_quantile(x,7,128),fullgraph=True,dynamic=False)
    torch.testing.assert_close(compiled(values),warp_quantile(values,7,128),rtol=0,atol=0)


@cuda
def test_warp_qb_full_router_matches_radix_assignments_and_all_gradients():
    from rtx.routing import BalancedRouter,RoutingKernelConfig
    torch.manual_seed(1298)
    a=BalancedRouter(32,24,top_k=6,router_dim=16,device='cuda',
                     config=RoutingKernelConfig(qb_backend='native_warp'))
    b=BalancedRouter(32,24,top_k=6,router_dim=16,device='cuda',
                     config=RoutingKernelConfig(qb_backend='native'))
    with torch.no_grad():a.projection_weight.normal_(std=.05)
    b.load_state_dict(a.state_dict())
    x=torch.randn(65,32,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    other=x.detach().clone().requires_grad_()
    actual=a(x);expected=b(other)
    torch.testing.assert_close(actual.expert_ids,expected.expert_ids,rtol=0,atol=0)
    torch.testing.assert_close(actual.inverse,expected.inverse,rtol=0,atol=0)
    torch.testing.assert_close(actual.route_weights,expected.route_weights,rtol=0,atol=0)
    gradient=torch.randn_like(actual.route_weights)
    ga=torch.autograd.grad(actual.route_weights,(x,*a.parameters()),gradient)
    gb=torch.autograd.grad(expected.route_weights,(other,*b.parameters()),gradient)
    for left,right in zip(ga,gb):
        assert bool(left.isfinite().all())
        torch.testing.assert_close(left,right,rtol=0,atol=0)
