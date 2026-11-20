"""The stronger BF16 reference must not profit by dropping routed work."""
import itertools
import math

import pytest
import torch
from torch import nn

from benchmarks.benchmark_moe_checkpoint import (
    BF16FFNReference,bf16_compact_capacity,torch_compacted_routed_experts,
)
from rtx.moe import torch_routed_experts
from rtx.routing import BalancedRouting


def _fixture(capacity=16):
    torch.manual_seed(1169)
    n,d,e,h,k=7,8,4,8,2
    x=(torch.randn(n,d,dtype=torch.bfloat16)*.3).requires_grad_()
    gu=(torch.randn(e,2*h,d,dtype=torch.bfloat16)*.02).requires_grad_()
    down=(torch.randn(e,d,h,dtype=torch.bfloat16)*.02).requires_grad_()
    ids=torch.zeros(e,capacity,dtype=torch.int64)
    inverse=torch.empty(n,k,dtype=torch.int64);counts=[0]*e
    weights=torch.zeros(e,capacity)
    for token in range(n):
        for rank in range(k):
            expert=(token+rank)%e;slot=2*counts[expert]+1;counts[expert]+=1
            ids[expert,slot]=token;inverse[token,rank]=expert*capacity+slot
            weights[expert,slot]=.25 if rank==0 else .75
    weights.requires_grad_()
    return x,gu,down,BalancedRouting(ids,weights,inverse)


def _close(values,expected):
    for a,b in zip(values,expected,strict=True):
        assert a is not None and a.dtype==b.dtype and a.shape==b.shape
        assert bool(a.isfinite().all()) and bool(b.isfinite().all())
        delta=a.float()-b.float()
        assert float(delta.norm()/b.float().norm().clamp_min(1e-12))<.006 or float(delta.abs().max())<2e-7


def test_shape_bound_covers_empty_pools_and_rounding_without_host_values():
    assert bf16_compact_capacity(8192,32,32,4,alignment=1)==1052
    assert bf16_compact_capacity(8192,32,32,4)==1152
    for pools in (1,2,3,4):
        for lengths in itertools.product(range(4),repeat=pools):
            tokens=sum(lengths)
            if not tokens:continue
            for e,k in ((4,1),(4,2),(6,3),(8,3)):
                divisor=math.gcd(e,k);period=e//divisor;q=k//divisor
                actual=q*sum((n+period-1)//period for n in lengths)
                assert actual<=bf16_compact_capacity(tokens,pools,e,k,alignment=1)
    for values in ((0,2,4,2),(1,0,4,2),(1,2,0,2),(1,2,4,0),(1,2,4,5)):
        with pytest.raises(ValueError):bf16_compact_capacity(*values)


def test_compaction_preserves_holes_token_zero_and_all_gradients():
    x,gu,down,route=_fixture()
    leaves=(x,gu,down,route.route_weights);dy=torch.randn(7,8)
    expected=torch_routed_experts(x,gu,down,route)
    expected_values=(expected,*torch.autograd.grad(expected,leaves,dy))
    actual=torch_compacted_routed_experts(x,gu,down,route,8)
    _close((actual,*torch.autograd.grad(actual,leaves,dy)),expected_values)
    assert torch.count_nonzero(actual[0])>0,'Token zero must survive invalid zero IDs'


def test_no_benefit_layout_retains_native_exactly():
    x,gu,down,route=_fixture()
    assert torch.equal(torch_compacted_routed_experts(x,gu,down,route,128),
                       torch_routed_experts(x,gu,down,route))


def test_compact_reference_traces_fullgraph_with_tensor_autograd():
    x,gu,down,route=_fixture();dy=torch.randn(7,8)
    def call(a,b,c,w):
        return torch_compacted_routed_experts(a,b,c,BalancedRouting(route.expert_ids,w,route.inverse),8)
    compiled=torch.compile(call,fullgraph=True,backend='aot_eager')
    leaves=(x,gu,down,route.route_weights)
    expected=call(*leaves);a=(expected,*torch.autograd.grad(expected,leaves,dy))
    actual=compiled(*leaves);_close((actual,*torch.autograd.grad(actual,leaves,dy)),a)


@pytest.mark.parametrize('checkpoint_ffns',[False,True])
def test_module_checkpoint_compacts_without_recomputing_router(checkpoint_ffns):
    x,gu,down,route=_fixture(256)
    class Router(nn.Module):
        def __init__(self):
            super().__init__();self.weights=nn.Parameter(route.route_weights.detach())
            self.register_buffer('ids',route.expert_ids);self.register_buffer('inverse',route.inverse);self.calls=0
        def forward(self,value,*,cu_seqlens):
            self.calls+=1;return BalancedRouting(self.ids,self.weights,self.inverse)
    model=nn.Module();model.router=Router()
    model.gate_up=nn.Module();model.gate_up.weight=nn.Parameter(gu.detach())
    model.down=nn.Module();model.down.weight=nn.Parameter(down.detach())
    model.shared_gate_up=nn.ModuleList();model.shared_down=nn.ModuleList()
    leaves=(x,*model.parameters());dy=torch.randn_like(x);cu=torch.tensor([0,3,3,7],dtype=torch.int32)
    native=BF16FFNReference(model,checkpoint_ffns=checkpoint_ffns)(x,cu_seqlens=cu);native.backward(dy)
    expected=(native.detach(),*(v.grad.detach().clone() for v in leaves))
    for v in leaves:v.grad=None
    model.router.calls=0
    compact=BF16FFNReference(model,checkpoint_ffns=checkpoint_ffns,expert_layout='compact')(x,cu_seqlens=cu)
    compact.backward(dy)
    assert model.router.calls==1
    _close((compact,*(v.grad for v in leaves)),expected)
