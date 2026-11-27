"""Production flat transport choices: identities, lazy binding and saved adjoints."""
from dataclasses import replace

import pytest
import torch

from rtx.routing import RoutingKernelConfig,SinkhornKernelConfig,ROUTING_TOKEN_SIZE


def parallel():
    return RoutingKernelConfig(forward=SinkhornKernelConfig(128,4,2),
        backward=SinkhornKernelConfig(256,8,4),flat_row_backend='warp',
        flat_column_shards_forward=16,flat_column_shards_backward=8)


def test_token_migration_and_parallel_roundtrip():
    from rtx._routing_autotune import execution_config
    c=parallel();default=RoutingKernelConfig()
    assert ROUTING_TOKEN_SIZE==26 and RoutingKernelConfig.from_token(c.to_token())==c
    assert RoutingKernelConfig.from_dict(c.to_dict())==c
    for size in (10,14,15,17,19,20,22,25):
        assert execution_config(torch.tensor(default.to_token()[:size]))==default
    for kwargs in (dict(flat_row_backend='unknown'),dict(flat_column_shards_forward=True),
                   dict(flat_column_shards_backward=3)):
        with pytest.raises((ValueError,TypeError)):replace(c,**kwargs)
    with pytest.raises(ValueError):RoutingKernelConfig.from_token(c.to_token()[:24])


def test_layout_partial_sizes_and_inactive_row_ownership():
    from rtx._routing_execution import layout
    from rtx.autotune.routing import make_routing_adapter
    c=parallel();token=tuple(c.to_token())
    key,sizes=layout((68,16),4,3,token,'forward')
    assert sizes==(68*16,2*3*68*16,68,16,16*16)
    assert layout((68,16),4,3,token,'backward')[1]==(68*16,68*16,1,8*16)
    other=replace(c,forward=replace(c.forward,row_groups=8),backward=replace(c.backward,row_groups=2))
    assert layout((68,16),4,3,tuple(other.to_token()),'forward')[0]==key
    a=make_routing_adapter(68,16,4,None)
    assert a.config_id(c)==a.config_id(other)
    assert a.config_id(c)!=a.config_id(replace(c,flat_column_shards_forward=32))
    f=make_routing_adapter(68,16,4,None,phase='forward')
    assert f.config_id(c)==f.config_id(replace(c,flat_column_shards_backward=64))
    with pytest.raises(ValueError,match='E<=128'):
        layout((256,256),4,3,token,'forward')


def test_packed_adapter_does_not_sample_inactive_flat_variants():
    from rtx.autotune.varlen_routing import make_adapter
    a=make_adapter(128,2,16,4,3,RoutingKernelConfig(),None,lengths=[64,64])
    assert not any(k.startswith('flat_') for k in a.axes)
    assert a.config_id(parallel())==a.config_id(replace(parallel(),flat_row_backend='cta',
        flat_column_shards_forward=0,flat_column_shards_backward=0))


cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='RTX CUDA required')


@cuda
@pytest.mark.parametrize('n,e,k,shards',[(1,1,1,128),(35,7,1,16),(99,33,1,4),(192,64,4,32)])
def test_parallel_transport_matches_independent_fp64(n,e,k,shards):
    from rtx._routing_execution import sinkhorn
    from tests.test_routing_execution import _independent_transport,_check
    torch.manual_seed(5312+n)
    c=replace(parallel(),flat_column_shards_forward=shards,flat_column_shards_backward=shards)
    x=(torch.randn(n,e,device='cuda')*.7).requires_grad_()
    upstream=torch.randn_like(x)
    ref=x.detach().double().requires_grad_()
    expected=_independent_transport(ref,k,3)
    y,history=sinkhorn(x,k,3,tuple(c.to_token()))
    dx,=torch.autograd.grad(y,x,upstream)
    want,=torch.autograd.grad(expected,ref,upstream.double())
    _check(y,expected,rtol=5e-5,atol=5e-6)
    _check(dx,want,rtol=5e-4,atol=8e-6)
    assert history.numel()==2*3*n*e and not history.requires_grad


@cuda
def test_cold_disk_hit_hot_refinement_and_capture_use_identical_full_plan(tmp_path,monkeypatch):
    from rtx import _routing_autotune as startup,_moe_selection as selection
    from rtx.routing import BalancedRouter
    from rtx.autotune.routing import cache_contract
    from rtx.autotune.winners import save_runtime_winner
    from rtx._routing_execution import sinkhorn
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        torch.manual_seed(5313)
        x=torch.randn(68,16,device='cuda',requires_grad=True)
        upstream=torch.randn_like(x)
        initial=RoutingKernelConfig();chosen=parallel()
        router=BalancedRouter(16,16,top_k=4,router_dim=8,sinkhorn_iterations=3,
            config=initial,autotune='cache',autotune_cache_dir=tmp_path)
        spec=router._selection_spec
        parsed=startup.parse_spec(spec)
        parsed={**parsed,'workload':{**parsed['workload'],'real_tokens':65}}
        key,root,_=cache_contract(x,parsed,'train')
        save_runtime_winner(key,chosen.to_dict(),root=root,config_id='parallel-test',median_ms=1.)
        startup._clear_runtime_caches();selection.clear_selection_cells()
        calls=[];resolve=startup._resolve
        def counted(*a,**kw):calls.append(1);return resolve(*a,**kw)
        monkeypatch.setattr(startup,'_resolve',counted)
        fn=torch.compile(lambda scores:startup.dispatch_sinkhorn(scores,4,3,spec,True,65)[0],
            fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
        cold=fn(x);hot=fn(x)
        assert len(calls)==1
        expected=sinkhorn(x,4,3,tuple(chosen.to_token()))[0]
        torch.testing.assert_close(cold,expected,rtol=0,atol=0)
        torch.testing.assert_close(hot,expected,rtol=0,atol=0)
        # Updating future selection must not change either outstanding adjoint.
        startup.publish_selected_config(x,4,3,spec,True,65,initial)
        wanted=torch.autograd.grad(expected,x,upstream)[0]
        for output in (hot,cold):
            torch.testing.assert_close(torch.autograd.grad(output,x,upstream)[0],wanted,rtol=0,atol=0)
        startup._clear_runtime_caches()
        restored=fn(x)
        torch.testing.assert_close(restored,expected,rtol=0,atol=0)
        assert len(calls)==2
        monkeypatch.setattr(startup,'_resolve',lambda *a,**kw:pytest.fail('retuned during capture'))
        # Autograd nodes, warmup and capture use the same stream.
        for _ in range(2):
            out=fn(x);grad=torch.autograd.grad(out,x,upstream)[0]
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            out=fn(x);grad=torch.autograd.grad(out,x,upstream)[0]
        for _ in range(3):graph.replay()
        stream.synchronize()
        torch.testing.assert_close(out,expected,rtol=0,atol=0)
        torch.testing.assert_close(grad,wanted,rtol=0,atol=0)
        graph.reset()
    startup._clear_runtime_caches()


@cuda
def test_actual_first_hit_records_unique_parallel_candidates_and_reuses(tmp_path,monkeypatch):
    import json
    from rtx import _routing_autotune as startup,_moe_selection as selection
    from rtx.routing import BalancedRouter
    from rtx.autotune.effort import AutotunePolicy
    from rtx._routing_execution import sinkhorn
    policy=AutotunePolicy.preset('quick',max_seconds=120,confirmation_repeats=0,
        samples=3,calls_per_sample=1,warmup=1)
    initial=RoutingKernelConfig(qb_backend='native_column',metadata_backend='cute',quota_threshold=0)
    router=BalancedRouter(16,16,top_k=4,config=initial,autotune='balanced',
                         tuning_policy=policy,autotune_cache_dir=tmp_path)
    x=torch.nn.functional.logsigmoid(torch.randn(4096,16,device='cuda')).requires_grad_()
    spec=router._selection_spec;selection.clear_selection_cells();startup._clear_runtime_caches()
    fn=torch.compile(lambda t:startup.dispatch_sinkhorn(t,4,5,spec,True)[0],
        fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    value=fn(x)
    key=selection.bound_config(*startup.binding_identity(x,4,5,spec,True))
    chosen=RoutingKernelConfig.from_token(json.loads(key))
    expected=sinkhorn(x,4,5,tuple(chosen.to_token()))[0]
    torch.testing.assert_close(value,expected,rtol=0,atol=0)
    upstream=torch.randn_like(x)
    torch.testing.assert_close(torch.autograd.grad(value,x,upstream)[0],
        torch.autograd.grad(expected,x,upstream)[0],rtol=0,atol=0)
    paths=list(tmp_path.rglob('observations.jsonl'))
    records=[json.loads(line) for p in paths for line in p.read_text().splitlines()]
    assert len(records)==len({r['config_id'] for r in records})==8
    assert any(r['config'].get('flat_column_shards_forward',0)>0 for r in records)
    assert any(r['config'].get('flat_row_backend')=='warp' for r in records)
    before={str(p):p.read_bytes() for p in paths}
    monkeypatch.setattr(startup,'_resolve',lambda *a,**kw:pytest.fail('repeat call selected again'))
    for _ in range(3):fn(x)
    assert before=={str(p):p.read_bytes() for p in paths}
    startup._clear_runtime_caches()
