"""Explicit ownership contracts, plus actual graph replay under eviction."""
import gc
from dataclasses import dataclass, replace
import weakref

import pytest
import torch

from rtx.runtime import BoundedCache, runtime_memory_report, tensor_storage_bytes
from rtx.execution import ExecutionResources


def test_nested_cache_inventory_counts_unique_aliases():
    @dataclass
    class Owner:
        data: object
    a,b=BoundedCache(8),BoundedCache(8)
    storage=torch.empty(100)
    a[(0,101,1)]=Owner(storage)
    b[(0,102,1)]=Owner(storage[:20])
    report=runtime_memory_report()
    ids={r['cache_id'] for r in report['caches']}
    assert {id(a),id(b)}<=ids
    assert tensor_storage_bytes([*a.values(),*b.values()])==400


def test_callable_explicit_resources_accounted():
    def run():pass
    run.__rtx_resources__=(torch.empty(100),)
    assert tensor_storage_bytes(run)==400


def test_execution_owner_keeps_working_set_independent_of_lru():
    cache=BoundedCache(1)
    owner=ExecutionResources()
    first,second=object(),object()
    with owner.activate():
        cache[(0,12,1)]=first
        cache[(0,12,2)]=second
        assert len(cache)==1
        assert cache[(0,12,1)] is first
        assert cache[(0,12,2)] is second
    assert len(cache)==1
    with pytest.raises(KeyError):cache[(0,12,1)]


def test_retirement_rejects_capture_before_synchronizing():
    from unittest.mock import patch
    owner=ExecutionResources()
    with patch('torch.cuda.is_initialized',return_value=True), \
         patch('torch.cuda.is_current_stream_capturing',return_value=True), \
         patch('torch.cuda.synchronize') as synchronize:
        with pytest.raises(RuntimeError,match='outside CUDA graph capture'):
            owner.retire(streams=[0],device='cuda:0')
        synchronize.assert_not_called()


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_capture_working_set_survives_one_entry_lru(monkeypatch):
    from rtx import fp8_bwd as b
    from rtx.kernels.mxfp8_bwd import DEFAULT_SEPARATE_DECOMPOSED_MXFP8_BWD_CONFIG as cfg
    monkeypatch.setattr(b,'_RUNNERS',BoundedCache(1,sizeof=tensor_storage_bytes))
    values=[(torch.randn(m,128,device='cuda',dtype=torch.bfloat16)*.1,
             torch.randn(m,128,device='cuda',dtype=torch.bfloat16)*.1,
             torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.1) for m in (128,256,384)]
    def step():return [b.mxfp8_linear_backward(*v,config=cfg,autotune=False) for v in values]
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    owner=ExecutionResources(split_workspace_bytes=1<<20)
    with owner.activate(),torch.cuda.stream(stream):
        expected=[tuple(t.clone() for t in row) for row in step()]
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):actual=step()
    torch.cuda.synchronize()
    assert len(b._RUNNERS)==1
    del owner
    b._RUNNERS.clear();gc.collect()
    for _ in range(3):
        graph.replay();torch.cuda.synchronize()
        for a,e in zip(actual,expected):
            for av,ev in zip(a,e):torch.testing.assert_close(av,ev,rtol=0,atol=0)
    graph.reset()


@pytest.mark.parametrize('value',[-1,True,1.5])
def test_arena_capacity_validation(value):
    with pytest.raises(ValueError):ExecutionResources(split_workspace_bytes=value)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('dual',[False,True])
def test_graph_pins_and_selective_retirement(dual):
    from rtx import fp8_bwd as b
    from rtx._bwd_search import backward_structural_seeds
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.kernels.mxfp8_bwd import DEFAULT_FUSED_MXFP8_BWD_CONFIG
    from rtx.runtime import retire_runtime_contexts
    torch.manual_seed(37)
    p=MXFP8Problem(4096,256,256)
    cfg=next(c for c in backward_structural_seeds(p,DEFAULT_FUSED_MXFP8_BWD_CONFIG) if c.dw.backend=='decomposed')
    cfg=replace(cfg,stream_schedule='dual_stream' if dual else 'single')
    args=(torch.randn(p.m,p.n,device='cuda',dtype=torch.bfloat16)*.1,
          torch.randn(p.m,p.k,device='cuda',dtype=torch.bfloat16)*.1,
          torch.randn(p.n,p.k,device='cuda',dtype=torch.bfloat16)*.1)
    def call():return b.mxfp8_linear_backward(*args,config=cfg,autotune=False)
    expected=tuple(t.clone() for t in call())
    preparation=torch.cuda.current_stream()
    stream=torch.cuda.Stream();stream.wait_stream(preparation)
    with torch.cuda.stream(stream):
        call()
        graph=torch.cuda.CUDAGraph()
        # Deliberately no retain_runtime_resources scope: public CUDA capture
        # must pin resources, including autograd-worker accesses, automatically.
        with torch.cuda.graph(graph,stream=stream):actual=call()
    torch.cuda.synchronize()
    assert graph._rtx_resource_owners
    report=retire_runtime_contexts(streams=[preparation],device=args[0].device)
    assert report['retired_entries']>=1
    retire_runtime_contexts(streams=[stream],device=args[0].device)
    for _ in range(3):
        pressure=torch.empty(32<<20,device='cuda',dtype=torch.uint8);pressure.fill_(17)
        graph.replay();torch.cuda.synchronize()
        for a,e in zip(actual,expected):torch.testing.assert_close(a,e,rtol=0,atol=0)
        del pressure
    assert runtime_memory_report()['graph_pinned_storage_bytes']>0
    references=[weakref.ref(v.dw.quantized_a) for v in graph._rtx_resource_owners.values()
                if hasattr(v,'dx')]
    graph.reset()
    assert not hasattr(graph,'_rtx_resource_owners')
    gc.collect()
    assert references and all(r() is None for r in references)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_split_arena_serial_alias_and_concurrent_lane_isolation():
    from rtx.execution import split_workspace
    from rtx._workspace import persistent_scratch
    @persistent_scratch
    def allocate(device,count,lane):return split_workspace(count,device,lane=lane)
    owner=ExecutionResources(split_workspace_bytes=1<<20)
    device=torch.device('cuda',torch.cuda.current_device())
    with owner.activate():
        a=allocate(device,100,'ordered');b=allocate(device,200,'ordered')
        dx=allocate(device,100,'dx');dw=allocate(device,100,'dw')
        large=allocate(device,(1<<20)//4+1,'ordered')
    assert a.data_ptr()==b.data_ptr()
    assert len({a.data_ptr(),dx.data_ptr(),dw.data_ptr(),large.data_ptr()})==4
    assert owner.stats()['storage_bytes']==3<<20
    ref=weakref.ref(a)
    del a,b,dx,dw,large
    gc.collect();assert ref() is None


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_orphan_stream_pool_is_not_retained_by_routing_table():
    from rtx._workspace import persistent_scratch,_pools
    @persistent_scratch
    def allocate(device):return torch.empty(4<<20,device=device,dtype=torch.uint8)
    stream=torch.cuda.Stream()
    with torch.cuda.stream(stream):
        tensor=allocate(torch.device('cuda',torch.cuda.current_device()))
        key=(torch.cuda.current_device(),int(stream.cuda_stream))
        pool_ref=weakref.ref(_pools[key])
    stream.synchronize()
    del tensor
    gc.collect()
    assert pool_ref() is None
    assert key not in _pools


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_concurrent_host_submissions_use_isolated_stream_arenas():
    from concurrent.futures import ThreadPoolExecutor
    from rtx import fp8_bwd as b
    from rtx._bwd_search import backward_structural_seeds
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.kernels.mxfp8_bwd import DEFAULT_FUSED_MXFP8_BWD_CONFIG
    p=MXFP8Problem(2048,128,128)
    cfg=next(c for c in backward_structural_seeds(p,DEFAULT_FUSED_MXFP8_BWD_CONFIG)
             if c.dx.backend==c.dw.backend=='decomposed')
    streams=[torch.cuda.Stream(),torch.cuda.Stream()]
    owners=[ExecutionResources(split_workspace_bytes=2<<20) for _ in streams]
    values=[(torch.randn(p.m,p.n,device='cuda',dtype=torch.bfloat16)*.1,
             torch.randn(p.m,p.k,device='cuda',dtype=torch.bfloat16)*.1,
             torch.randn(p.n,p.k,device='cuda',dtype=torch.bfloat16)*.1) for _ in streams]
    expected=[]
    for s,o,v in zip(streams,owners,values):
        s.wait_stream(torch.cuda.current_stream())
        with o.activate(),torch.cuda.stream(s):
            expected.append(tuple(t.clone() for t in b.mxfp8_linear_backward(*v,config=cfg,autotune=False)))
    torch.cuda.synchronize()
    def worker(index):
        with owners[index].activate(),torch.cuda.stream(streams[index]):
            for _ in range(5):
                result=b.mxfp8_linear_backward(*values[index],config=cfg,autotune=False)
            streams[index].synchronize()
            for a,e in zip(result,expected[index]):torch.testing.assert_close(a,e,rtol=0,atol=0)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(worker,range(2)))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('family',['mxfp8','nvfp4_block','nvfp4_current','nvfp4_jit_row_region','nvfp4_delayed'])
def test_two_live_graphs_survive_global_cleanup_and_graph_destruction(family):
    import rtx
    from rtx.runtime import retire_runtime_contexts
    if family=='mxfp8':
        layer=rtx.MXFP8Linear(128,128,device='cuda',autotune=False,backend='materialized')
    else:
        layer=rtx.NVFP4Linear(128,128,device='cuda',autotune=False,backend='materialized',
                              scaling=family.removeprefix('nvfp4_'))
    inputs=[torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.1 for _ in range(2)]
    if family=='nvfp4_delayed':inputs[1]=inputs[0].clone()
    streams=[torch.cuda.Stream(),torch.cuda.Stream()]
    graphs=[]; outputs=[]; expected=[]
    with torch.no_grad():
        for x,s in zip(inputs,streams):
            s.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(s):
                for _ in range(3):layer(x)
                expected.append(layer(x).clone())
                g=torch.cuda.CUDAGraph()
                with torch.cuda.graph(g,stream=s):outputs.append(layer(x))
                graphs.append(g)
        torch.cuda.synchronize()
        rtx.clear_runtime_caches(preserve_compiled=True)
        for i,g in enumerate(graphs):
            g.replay();torch.cuda.synchronize()
            torch.testing.assert_close(outputs[i],expected[i],rtol=0,atol=0)
        graphs[0].reset();graphs.pop(0);outputs.pop(0);expected.pop(0)
        for _ in range(3):
            graphs[0].replay();torch.cuda.synchronize()
            torch.testing.assert_close(outputs[0],expected[0],rtol=0,atol=0)
        graphs[0].reset()
