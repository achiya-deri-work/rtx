"""Exact quota policy with register-cached or streaming threshold selection."""
import pytest
import torch
import struct


cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='SM120 CUDA required')


def test_cached_quota_templates_validate_capacity_and_geometry():
    from rtx.kernels.routing_quota_cached import CachedQuotaSelect
    for threads in (64,128,256):
        for items in (4,8,16):
            kernel=CachedQuotaSelect(600,48,6,threads,items,False)
            assert kernel.capacity==threads*items and not kernel.target
    for parameters in ((0,48,6),(601,48,6),(600,1025,6),(600,48,0)):
        with pytest.raises(ValueError):CachedQuotaSelect(*parameters)
    with pytest.raises(ValueError):CachedQuotaSelect(600,48,6,32,8)
    with pytest.raises(ValueError):CachedQuotaSelect(600,48,6,128,2)


def test_cached_quota_factory_changes_only_two_selection_stages_without_cuda(monkeypatch):
    import rtx.kernels.routing_quota_cached as module
    module.compile_quota_cached.cache_clear()
    original=tuple(object() for _ in range(12));compiled=[]
    monkeypatch.setattr(module,'compile_quota',lambda *a:original)
    monkeypatch.setattr(module.cute.runtime,'make_fake_tensor',lambda *a,**kw:('tensor',a,kw))
    monkeypatch.setattr(module.cute.runtime,'make_fake_stream',lambda **kw:('stream',kw))
    def compile(kernel,*a,**kw):
        assert len(a)==18 and kw['options']=='--enable-tvm-ffi --opt-level 3'
        compiled.append(kernel);return kernel
    monkeypatch.setattr(module.cute,'compile',compile)
    try:
        assert module.compile_quota_cached(600,48,6,128,0) is original
        stages=module.compile_quota_cached(600,48,6,128,8)
        assert len(stages)==12 and len(compiled)==2
        for index,value in enumerate(stages):
            if index not in (4,9):assert value is original[index]
        assert stages[4].target is False and stages[9].target is True
        assert stages[4].capacity==stages[9].capacity==1024
        streamed=module.compile_quota_cached(600,48,6,128,0,True)
        assert streamed[4].early_exit and streamed[9].early_exit
        assert streamed[4].capacity==streamed[9].capacity==0
    finally:module.compile_quota_cached.cache_clear()


def _ordered_key(value,index):
    bits=struct.unpack('<I',struct.pack('<f',value))[0]
    if value==0:bits=0
    ordered=((~bits)&0xffffffff) if bits&0x80000000 else bits^0x80000000
    if value!=value:ordered=0xffffffff
    return (ordered<<32)|(0xffffffff-index)


def _early_threshold(keys,take):
    prefix=0;rank=take-1
    for step in range(16):
        shift=60-step*4
        mask=(((1<<64)-1)<<(shift+4))&((1<<64)-1)
        surviving=[key for key in keys if key&mask==prefix]
        counts=[sum(((key>>shift)&15)==digit for key in surviving) for digit in range(16)]
        for digit in reversed(range(16)):
            if rank>=counts[digit]:rank-=counts[digit]
            else:break
        prefix|=digit<<shift
        remaining=[key for key in surviving if ((key>>shift)&15)==digit]
        if len(remaining)==1:return remaining[0],step+1
    return prefix,16


@pytest.mark.parametrize('count',[1,17,250,513])
@pytest.mark.parametrize('tied',[False,True])
def test_early_threshold_recovers_full_unique_key_without_changing_ties(count,tied):
    values=torch.randn(count,generator=torch.Generator().manual_seed(1765+count)).tolist()
    if tied:values=[0.]*count
    elif count>=17:values[:5]=[float('nan'),float('inf'),float('-inf'),-0.,0.]
    keys=[_ordered_key(value,index) for index,value in enumerate(values)]
    expected=sorted(keys,reverse=True)
    passes=[]
    for take in set((1,max(1,count//2),count)):
        actual,steps=_early_threshold(keys,take)
        assert actual==expected[take-1]
        passes.append(steps)
    if not tied:assert min(passes)<16


def _inputs(n,k,e,overflow=False):
    m=n*k;device='cuda';generator=torch.Generator(device=device).manual_seed(1753+n)
    assert m%e==0 and 2*k<=e
    current=torch.rand((n,e),generator=generator,device=device).topk(k,1).indices
    slots=torch.rand((n,k),generator=generator,device=device).argsort(1)
    eligible=torch.ones((n,e),device=device,dtype=torch.bool).scatter(1,current,False)
    targets=torch.rand((n,e),generator=generator,device=device).masked_fill(~eligible,-torch.inf).topk(k,1).indices.flatten()
    if overflow:current.zero_();targets.fill_(1)
    sources=current.gather(1,slots).flatten()
    gains=torch.randn(m,generator=generator,device=device,dtype=torch.float32)
    gains[:min(32,m)]=0
    if m>40:gains[32:37]=torch.tensor([float('nan'),float('inf'),float('-inf'),-0.,0.],device=device)
    loads=torch.zeros(e,device=device,dtype=torch.int64).scatter_add(0,sources,torch.ones_like(sources))
    capacity=m//e
    surplus=(loads-capacity).clamp_min(0);deficit=(capacity-loads).clamp_min(0)
    phase=torch.tensor(17,device=device,dtype=torch.int64)
    return sources,targets,gains,surplus,deficit,slots,loads,phase


def _execute(stages,inputs):
    sources,targets,gains,surplus,deficit,slots,loads,phase=inputs
    m=sources.numel();e=loads.numel();device=sources.device
    counts=torch.empty(e,device=device,dtype=torch.int32)
    offsets=torch.empty(e+1,device=device,dtype=torch.int32)
    keys=torch.empty(m,device=device,dtype=torch.uint64)
    thresholds=torch.empty(e,device=device,dtype=torch.uint64)
    source_mask=torch.empty(m,device=device,dtype=torch.bool)
    target_mask=torch.empty_like(source_mask)
    result=torch.empty_like(slots);new_loads=torch.empty_like(loads);new_phase=torch.empty(1,device=device,dtype=torch.int64)
    args=(sources,targets,gains,surplus,deficit,slots.flatten(),loads,phase.view(-1),
          counts,offsets,keys,thresholds,source_mask,target_mask,result.flatten(),new_loads,new_phase)
    for stage in stages:stage(*args)
    return result,new_loads,new_phase.view(()),source_mask,target_mask


def _oracle(inputs):
    from rtx._routing_reference import _quota_accept
    sources,targets,gains,surplus,deficit,slots,loads,phase=inputs
    source=_quota_accept(sources,gains,surplus)
    target=_quota_accept(targets,gains.masked_fill(~source,float('-inf')),deficit)
    accepted=source&target&gains.isfinite()
    values=torch.where(accepted,targets,sources).reshape_as(slots)
    result=torch.empty_like(slots).scatter(1,slots,values)
    moved=accepted.long()
    new_loads=loads.scatter_add(0,sources,-moved).scatter_add(0,targets,moved)
    return result,new_loads,phase+1,source,target&source


@cuda
@pytest.mark.parametrize('n,k,e,threads,items,overflow',[
    (35,3,7,64,4,False),(128,4,16,128,8,False),
    (2000,6,48,128,8,False),(513,1,9,64,4,True),
    (121,2,11,256,16,False),
])
def test_cached_quota_exact_original_and_independent_policy(n,k,e,threads,items,overflow):
    from rtx.kernels.routing_selection import compile_quota
    from rtx.kernels.routing_quota_cached import compile_quota_cached
    inputs=_inputs(n,k,e,overflow)
    original=_execute(compile_quota(n*k,e,k,threads,True),inputs)
    cached=_execute(compile_quota_cached(n*k,e,k,threads,items),inputs)
    expected=_oracle(inputs)
    for index,(actual,previous,want) in enumerate(zip(cached,original,expected)):
        torch.testing.assert_close(actual,previous,rtol=0,atol=0,msg=lambda m:f'{index}: {m}')
        torch.testing.assert_close(actual,want,rtol=0,atol=0,msg=lambda m:f'{index}: {m}')
    # All references to prior outputs must survive subsequent launches.
    saved=tuple(value.clone() for value in cached)
    _execute(compile_quota_cached(n*k,e,k,threads,items),inputs)
    for actual,want in zip(cached,saved):torch.testing.assert_close(actual,want,rtol=0,atol=0)


@cuda
@pytest.mark.parametrize('items,overflow',[(0,False),(4,True),(8,False)])
def test_early_exit_exact_original_including_streaming_and_priority_ties(items,overflow):
    from rtx.kernels.routing_selection import compile_quota
    from rtx.kernels.routing_quota_cached import compile_quota_cached
    n,k,e,threads=513,1,9,64
    inputs=_inputs(n,k,e,overflow)
    stages=compile_quota_cached(n*k,e,k,threads,items,True)
    for tied in (False,True):
        if tied:inputs[2].zero_()
        actual=_execute(stages,inputs)
        original=_execute(compile_quota(n*k,e,k,threads,True),inputs)
        expected=_oracle(inputs)
        for result,previous,want in zip(actual,original,expected):
            torch.testing.assert_close(result,previous,rtol=0,atol=0)
            torch.testing.assert_close(result,want,rtol=0,atol=0)


@cuda
def test_source_count_reset_covers_more_experts_than_proposals():
    """Primitive-contract boundary: source mark must reset all E counters."""
    from rtx.kernels.routing_selection import compile_quota
    from rtx.kernels.routing_quota_cached import compile_quota_cached
    device='cuda';m,e,k=3,130,1
    sources=torch.tensor([128,128,128],device=device,dtype=torch.int64)
    targets=torch.tensor([129,129,129],device=device,dtype=torch.int64)
    gains=torch.tensor([.2,.1,0.],device=device,dtype=torch.float32)
    surplus=torch.zeros(e,device=device,dtype=torch.int64);surplus[128]=3
    deficit=torch.zeros_like(surplus);deficit[129]=2
    slots=torch.zeros((m,k),device=device,dtype=torch.int64)
    loads=torch.zeros_like(surplus);loads[128]=3
    phase=torch.zeros((),device=device,dtype=torch.int64)
    inputs=(sources,targets,gains,surplus,deficit,slots,loads,phase)
    expected=_oracle(inputs)
    for factory in (lambda:compile_quota(m,e,k,64,True),
                    lambda:compile_quota_cached(m,e,k,64,4,True)):
        actual=_execute(factory(),inputs)
        for result,want in zip(actual,expected):torch.testing.assert_close(result,want,rtol=0,atol=0)
