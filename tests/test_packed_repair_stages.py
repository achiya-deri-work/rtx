import pytest
import torch
from dataclasses import replace

from rtx._routing_varlen import _repair_targets


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("width,k", [(16,4), (24,16), (32,4)])
def test_packed_topk_matches_aten_ties_masks_and_tail(width,k):
    from rtx._routing_varlen import _packed_row_topk
    n=4097
    gen=torch.Generator(device="cuda").manual_seed(201)
    x=torch.randint(-2,3,(n,width),device="cuda",generator=gen).float()
    x[::8].fill_(-torch.inf)
    x[1::8].zero_()
    x[1::8,1::2]=-0.
    values=torch.empty((n,k),device=x.device)
    ids=torch.empty((n,k),device=x.device,dtype=torch.int64)
    args=(x.flatten(),values.flatten(),ids.flatten(),n,width,k)
    torch.library.opcheck(_packed_row_topk,args,test_utils=("test_schema","test_faketensor"))
    _packed_row_topk(*args)
    expected,indices=x.topk(k,dim=1)
    assert torch.equal(indices,ids) and torch.equal(expected.view(torch.int32),values.view(torch.int32))


def test_packed_repair_abi_and_search():
    from rtx.routing import RoutingKernelConfig
    from rtx.autotune.varlen_routing import make_adapter,packed_structural_seeds
    from rtx.autotune.routing import make_routing_adapter
    config=RoutingKernelConfig()
    assert 'packed_repair_backend' not in config.to_dict()
    assert RoutingKernelConfig.from_token(config.to_token()[:25])==config
    for mode in ('mixed','split','split_warp'):
        c=replace(config,packed_repair_backend=mode)
        assert RoutingKernelConfig.from_dict(c.to_dict())==RoutingKernelConfig.from_token(c.to_token())==c
    with pytest.raises(ValueError):RoutingKernelConfig.from_token(config.to_token()[:-1]+[-1])
    packed=make_adapter(21504,17,24,4,5,config,None)
    small=make_adapter(256,3,8,4,5,config,None)
    flat=make_routing_adapter(21504,24,4,None)
    assert packed.axes['packed_repair_backend']==('mixed','split','split_warp')
    assert small.axes['packed_repair_backend']==('mixed','split')
    w=replace(config,packed_repair_backend='split_warp')
    s=replace(config,packed_repair_backend='split')
    assert len({packed.config_id(c) for c in (config,s,w)})==3
    assert flat.config_id(w)==flat.config_id(config) and small.config_id(w)==small.config_id(s)
    assert any(c.packed_repair_backend=='split_warp' for c in packed_structural_seeds(config,repair_challenger='split_warp'))


@pytest.mark.parametrize('phase',[0,15,16,63])
def test_exact_phase_specialization_with_ties_and_empty_targets(phase):
    from torch.fx.experimental.proxy_tensor import make_fx
    rng=torch.Generator().manual_seed(98)
    values=torch.randint(0,3,(5,24),generator=rng).float()
    values[0]=-torch.inf
    pv,pt=values.topk(16,dim=1)
    pos=torch.tensor([0,2,31,57,2047])
    p=torch.tensor(phase)
    k=4
    hashed=(pos[:,None]*1103515245+pt*12345+pos[:,None]*pt*1013904223+
            p*(pt*2654435761+2246822519))%104729
    idx=hashed.float().masked_fill(~pv.isfinite(),-torch.inf).topk(k,dim=1).indices
    expected=(pt.gather(1,idx),pv.gather(1,idx)) if phase>=16 else (pt[:,:k],pv[:,:k])
    stage='early' if phase<16 else 'late'
    for mode in (stage,'mixed'):
        ids,scores=_repair_targets(pv,pt,pos,p,k,mode)
        assert torch.equal(ids,expected[0]) and torch.equal(scores,expected[1])
    graph=make_fx(lambda v,t:_repair_targets(v,t,pos,p,k,stage))(pv,pt)
    assert ('aten.topk.default' in graph.code)==(phase>=16)
