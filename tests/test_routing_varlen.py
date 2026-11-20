"""Packed pools match independent examples; GPU paths never read boundaries."""
from dataclasses import replace

import pytest
import torch

from rtx.routing import BalancedRouter,RoutingKernelConfig


cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='assigned RTX CUDA required')


@pytest.fixture(autouse=True)
def isolate_independent_cuda_compile_experiments(request):
    # These cases deliberately compile different root modules/shapes. They
    # are independent experiments, not one application's eight-specialization
    # budget. Reset only the test process, never a production runtime limit.
    enabled=request.node.name.startswith('test_cuda_') and torch.cuda.is_available()
    if enabled:torch.compiler.reset()
    yield
    if enabled:torch.compiler.reset()


def config():
    return RoutingKernelConfig(qb_backend='native_column',metadata_backend='cute',quota_threshold=0)


def router(device='cpu',*,e=8,k=2):
    value=BalancedRouter(16,e,top_k=k,router_dim=8,config=config(),backend='auto',
                         device=device,dtype=torch.float32)
    with torch.no_grad():value.projection_weight.normal_(std=.05)
    return value


def test_varlen_workspace_bound_and_token_abi():
    from rtx._routing_varlen import layout,_flat,_decode_config
    for e,k in ((24,6),(32,4),(64,8),(48,6),(7,2),(4,4)):
        for lengths in ((512,2000,768,4096),(0,1,0,3),(13,14,15)):
            t=sum(lengths);key=layout(t,len(lengths),e,k,5,tuple(config().to_token()))
            multiple=e//__import__('math').gcd(e,k)
            ns=[(v+multiple-1)//multiple*multiple for v in lengths]
            capacity=sum(((v*k//e+31)//32)*32 for v in ns)
            assert sum(ns)<=key[4] and capacity<=key[5]
            assert key[5]%32==0 and _decode_config(tuple(_flat(key)))==key
    with pytest.raises(ValueError):layout(1,1,256,4,5,tuple(config().to_token()))


def test_packed_execution_options_are_optional_revisioned_tuner_choices():
    from rtx._routing_varlen import layout,_flat,_decode_config
    from rtx.autotune.varlen_routing import make_adapter,packed_structural_seeds
    from rtx.autotune.outcomes import TrialOutcome
    old=config();token=old.to_token()
    assert RoutingKernelConfig.from_token(token[:15])==old
    tuned=replace(old,packed_column_cache_items=32,packed_row_backend='warp')
    assert RoutingKernelConfig.from_token(tuned.to_token())==tuned
    assert RoutingKernelConfig.from_token(tuned.to_token()[:17])==tuned
    sharded=replace(tuned,packed_column_shards_forward=8,packed_column_shards_backward=16)
    assert RoutingKernelConfig.from_token(sharded.to_token())==sharded
    assert RoutingKernelConfig.from_token(sharded.to_token()[:19])==sharded
    expert_keys=replace(sharded,packed_qb_column_backend='expert_keys')
    assert RoutingKernelConfig.from_token(expert_keys.to_token())==expert_keys
    key=layout(7376,4,48,6,5,tuple(tuned.to_token()))
    assert _decode_config(tuple(_flat(key)))==key
    a=make_adapter(7376,4,48,6,5,tuned,lambda _:TrialOutcome('ok',median_ms=1.))
    assert set(a.axes['packed_column_cache_items'])=={0,4,8,16,32,64}
    assert set(a.axes['packed_row_backend'])=={'cta','warp'}
    assert a.initial_config.forward.row_groups==a.initial_config.backward.row_groups==1
    f=a.features(tuned)
    assert f['derived.forward_column_ctas']==4*48//tuned.forward.column_groups
    assert f['derived.forward_row_ctas']==__import__('math').ceil(key[4]/4)
    s=a.features(sharded)
    assert s['derived.forward_column_ctas']==4*8*3 and s['derived.backward_column_ctas']==4*16*3
    assert s['derived.forward_column_partial_bytes']==4*48*8*4
    assert s['derived.backward_column_partial_bytes']==4*48*16*4
    assert s['derived.column_launches']==4*5
    assert a.features(expert_keys)['derived.qb_ordered_key_bytes']==4*key[4]*48
    assert a.config_id(expert_keys)!=a.config_id(sharded)
    assert _decode_config(tuple(_flat(key))[:18])==key
    fwd=make_adapter(7376,4,48,6,5,sharded,lambda _:TrialOutcome('ok',median_ms=1.),training=False)
    assert 'packed_column_shards_backward' not in fwd.axes
    assert fwd.config_id(sharded)==fwd.config_id(replace(sharded,packed_column_shards_backward=0))
    assert any(c.packed_row_backend=='warp' and c.forward.column_groups==c.backward.column_groups==1
               for c in list(packed_structural_seeds(old))[:3])
    assert any(c.packed_column_shards_forward and c.packed_column_shards_backward
               for c in list(packed_structural_seeds(old))[:4])
    assert {c.packed_column_cache_items for c in list(packed_structural_seeds(old))
            if c.packed_qb_column_backend=='expert_keys'}=={0,16,64}
    assert a.config_id(tuned)!=a.config_id(old)
    torch_qb=replace(tuned,qb_backend='torch')
    assert a.config_id(torch_qb)==a.config_id(replace(torch_qb,packed_column_cache_items=0))
    assert a.config_id(torch_qb)==a.config_id(replace(torch_qb,packed_qb_column_backend='expert_keys'))
    for kwargs in (dict(packed_column_cache_items=3),dict(packed_column_cache_items=True),
                   dict(packed_row_backend='bad'),dict(packed_column_shards_forward=True),
                   dict(packed_column_shards_backward=64),dict(packed_qb_column_backend='bad')):
        with pytest.raises(ValueError):replace(old,**kwargs)


@pytest.mark.parametrize('mix',['legacy','mixed'])
@pytest.mark.parametrize('trials,longest',[(8,4096),(24,16384)])
def test_packed_search_times_both_column_families_under_normal_budget(tmp_path,mix,trials,longest):
    from rtx.autotune.effort import AutotunePolicy,make_effort_autotuner
    from rtx.autotune.outcomes import TrialOutcome
    from rtx.autotune.store import ResidualTuningStore
    from rtx.autotune.varlen_routing import make_adapter,packed_structural_seeds
    seen=[]
    def measure(c):
        seen.append(c)
        return TrialOutcome('ok',median_ms=1.,timings_ms=[1.])
    adapter=make_adapter(7376 if longest==4096 else 29504,4,32,4,5,config(),measure)
    seeds=tuple(packed_structural_seeds(adapter.initial_config,max_scheduled_length=longest))
    opening,expert_keys=seeds[:2]
    assert opening.packed_column_shards_forward==opening.packed_column_shards_backward==32
    assert opening.packed_column_cache_items==(16 if longest==4096 else 64)
    assert expert_keys==replace(opening,packed_qb_column_backend='expert_keys')
    oversize=next(packed_structural_seeds(config(),max_scheduled_length=16392))
    assert oversize.packed_column_cache_items==0
    policy=AutotunePolicy(max_trials=trials,max_seconds=15,candidate_mix=mix,confirmation_repeats=0)
    make_effort_autotuner(adapter,ResidualTuningStore(tmp_path),policy,candidates=seeds).tune()
    assert len(seen)==trials and seen[0]==adapter.initial_config
    assert seen[1:3]==[opening,expert_keys]


def test_packed_search_progressive_structure_opening_resumes_exact_prefix(tmp_path):
    from rtx.autotune.effort import AutotunePolicy,make_effort_autotuner
    from rtx.autotune.outcomes import TrialOutcome
    from rtx.autotune.store import ResidualTuningStore
    from rtx.autotune.varlen_routing import make_adapter,packed_structural_seeds
    seen=[]
    def measure(c):
        seen.append(c)
        value=0.9 if c.packed_column_shards_forward else 1.
        return TrialOutcome('ok',median_ms=value,timings_ms=[value])
    adapter=make_adapter(7376,4,32,4,5,config(),measure)
    seeds=tuple(packed_structural_seeds(adapter.initial_config,max_scheduled_length=4096))
    policy=AutotunePolicy(max_trials=8,max_seconds=15,candidate_mix='mixed',
        search_schedule='progressive',confirmation_repeats=0)
    def engine(root,trials):
        return make_effort_autotuner(adapter,ResidualTuningStore(root),replace(policy,max_trials=trials),candidates=seeds)
    initial=engine(tmp_path/'resumed',8);initial.tune()
    assert len(seen)==8 and all(s in seen for s in seeds[:2])
    resumed=engine(tmp_path/'resumed',16)
    assert resumed.checkpoint_key==initial.checkpoint_key
    resumed.tune();prefix=tuple(seen)
    assert len(prefix)==len({adapter.config_id(c) for c in prefix})==16
    seen.clear();engine(tmp_path/'one-shot',16).tune()
    assert tuple(seen)==prefix


def test_packed_bounds_cover_every_small_composition_and_are_tight_when_applicable():
    from itertools import combinations_with_replacement
    from math import gcd
    from rtx._routing_varlen import workspace_bounds
    for e,k in ((24,6),(7,2),(7,3),(4,4),(6,4),(5,4),(128,31)):
        a=e//gcd(e,k);q=k//gcd(e,k)
        for t in range(1,16):
            for b in range(1,5):
                n,c=workspace_bounds(t,b,e,k);max_n=max_c=0
                for cuts in combinations_with_replacement(range(t+1),b-1):
                    boundaries=(0,*cuts,t)
                    scheduled=[((z-v+a-1)//a)*a for v,z in zip(boundaries,boundaries[1:])]
                    physical=sum(((v*k//e+31)//32)*32 for v in scheduled)
                    max_n=max(max_n,sum(scheduled));max_c=max(max_c,physical)
                assert max_n==n and max_c<=c,(t,b,e,k,max_n,n,max_c,c)
                if 32%q==0:assert max_c==c
    assert workspace_bounds(19,4,8,2)[1]==128
    assert workspace_bounds(7376,4,48,6)[1]==1024
    assert workspace_bounds(3,200,8,2)==(12,96)


@pytest.mark.parametrize('lengths',[(5,0,11,3),(8,16),(1,1,1)])
def test_cpu_router_packed_matches_separate_pools_and_gradients(lengths):
    torch.manual_seed(1201)
    model=router();x=torch.randn(sum(lengths),16,requires_grad=True)
    cu=torch.tensor([0,*torch.tensor(lengths).cumsum(0).tolist()],dtype=torch.int32)
    actual=model(x,cu_seqlens=cu);capacity=actual.route_weights.shape[1]
    upstream=torch.randn_like(actual.route_weights)
    loss=(actual.route_weights*upstream).sum()
    expected_loss=0.;offset=0;position=0
    for length in lengths:
        if not length:continue
        expected=model(x[position:position+length]);c=expected.route_weights.shape[1]
        torch.testing.assert_close(actual.expert_ids[:,offset:offset+c],expected.expert_ids+position,rtol=0,atol=0)
        torch.testing.assert_close(actual.route_weights[:,offset:offset+c],expected.route_weights)
        transformed=expected.inverse//c*capacity+expected.inverse%c+offset
        torch.testing.assert_close(actual.inverse[position:position+length],transformed,rtol=0,atol=0)
        expected_loss=expected_loss+(expected.route_weights*upstream[:,offset:offset+c]).sum()
        offset+=(c+31)//32*32;position+=length
    a=torch.autograd.grad(loss,(x,*model.parameters()),retain_graph=True)
    b=torch.autograd.grad(expected_loss,(x,*model.parameters()))
    for v,r in zip(a,b):torch.testing.assert_close(v,r,rtol=3e-5,atol=2e-6)


def test_invalid_boundary_metadata_and_strict_padding_fail_clearly():
    value=router();x=torch.randn(5,16)
    for cu in (torch.tensor([0,5]),torch.tensor([0,3,2,5],dtype=torch.int32),
               torch.tensor([1,5],dtype=torch.int32),torch.tensor([0,4],dtype=torch.int32)):
        with pytest.raises(ValueError):value(x,cu_seqlens=cu)
    value.padding='error'
    with pytest.raises(ValueError):value(x,cu_seqlens=torch.tensor([0,5],dtype=torch.int32))


def test_cpu_full_moe_varlen_and_all_parameters_against_separate_calls():
    from rtx.moe import BalancedMoE
    torch.manual_seed(1129)
    model=BalancedMoE(16,8,16,top_k=2,shared_hidden_features=16,router_dim=8,
                      backend='torch',autotune='off')
    with torch.no_grad():model.router.projection_weight.normal_(std=.05)
    x=torch.randn(12,16,dtype=torch.bfloat16,requires_grad=True)
    actual=model(x,cu_seqlens=torch.tensor([0,5,5,12],dtype=torch.int32))
    expected=torch.cat([model(x[:5]),model(x[5:])])
    torch.testing.assert_close(actual,expected,rtol=.01,atol=2e-7)
    dy=torch.randn_like(actual)
    a=torch.autograd.grad(actual,(x,*model.parameters()),dy)
    b=torch.autograd.grad(expected,(x,*model.parameters()),dy)
    for index,(v,r) in enumerate(zip(a,b)):
        # Separate BF16 backwards round each per-sequence Parameter gradient
        # before summing; batched GEMMs reduce first and round once. Gate the
        # whole derivative rather than relative error at near-zero entries.
        delta=(v.float()-r.float());nrms=delta.norm()/r.float().norm().clamp_min(1e-12)
        assert nrms<.01 or delta.abs().max()<2e-7,(index,float(nrms),float(delta.abs().max()))


def test_fake_fullgraph_packed_router_has_visible_buffers_and_no_boundary_read():
    from torch._subclasses.fake_tensor import FakeTensorMode,FakeTensor
    def fake(mode,shape,dtype,grad=False):
        return FakeTensor(mode,torch.empty(shape,device='meta',dtype=dtype,requires_grad=grad),torch.device('cuda:0'))
    model=router();graphs=[]
    def backend(graph,_):graphs.append(graph);return graph.forward
    with FakeTensorMode() as mode:
        for name,p in list(model.named_parameters()):
            setattr(model,name,torch.nn.Parameter(fake(mode,p.shape,p.dtype,True)))
        x=fake(mode,(19,16),torch.float32,True);cu=fake(mode,(5,),torch.int32)
        actual=torch.compile(model,backend=backend,fullgraph=True,dynamic=False)(x,cu_seqlens=cu)
        assert actual.inverse.shape==(19,2) and actual.route_weights.requires_grad
    code='\n'.join(g.code for g in graphs[0].modules() if isinstance(g,torch.fx.GraphModule))
    for op in ('varlen_prepare','varlen_sinkhorn_execute','varlen_hard','varlen_metadata'):assert op in code
    assert '_local_scalar_dense' not in code and '.item(' not in code


def test_varlen_shared_adapter_preserves_geometry_and_length_context():
    from rtx.autotune.varlen_routing import make_adapter
    from rtx.autotune.outcomes import TrialOutcome
    adapter=make_adapter(7376,4,48,6,5,config(),lambda _:TrialOutcome('ok',median_ms=1.))
    assert adapter.context.workload['sequences']==4
    assert 'metadata_threads' in adapter.axes and 'backward.row_groups' in adapter.axes
    different_budget=make_adapter(7376,4,48,6,5,config(),lambda _:TrialOutcome('ok',median_ms=1.),
                                 dual_iterations=3,repair_rounds=32)
    assert adapter.context.identifier!=different_budget.context.identifier
    for _,_,value in adapter.neighbors(adapter.initial_config):
        assert adapter.deserialize(adapter.serialize(value))==value
    # Mapping/packing also uses metadata_threads even in the Torch metadata
    # alternative, so it must not be canonicalized as an inactive axis.
    torch_meta=replace(adapter.initial_config,metadata_backend='torch')
    assert adapter.config_id(torch_meta)!=adapter.config_id(replace(torch_meta,metadata_threads=512))
    large=make_adapter(4096,32,64,8,5,config(),lambda _:TrialOutcome('ok',median_ms=1.))
    assert large.rejection(replace(large.initial_config,quota_cache_items=8))


def test_private_packed_launch_abis_are_mutation_only():
    import rtx._routing_varlen
    for name,argument in (('varlen_prepare','buffers'),('varlen_sinkhorn_execute','buffers'),
                          ('varlen_hard','out'),('varlen_metadata','buffers'),
                          ('varlen_quantile_expert_keys','keys')):
        op=getattr(torch.ops.rtx_routing,name).default
        assert not op._schema.returns
        assert next(a for a in op._schema.arguments if a.name==argument).alias_info.is_write
        assert torch.Tag.cudagraph_unsafe not in op.tags


def test_packed_plan_participates_in_automatic_resource_accounting():
    from rtx._routing_varlen import HardPlan
    from rtx.runtime import tensor_storage_bytes
    from rtx._resources import _stream_context
    plan=HardPlan.__new__(HardPlan);plan.graph=None
    for name in ('scores','scheduled','sequence','local','current','load_caps','phase'):
        setattr(plan,name,torch.empty(3))
    assert tensor_storage_bytes(plan)==7*3*4
    assert _stream_context((0,12345,481,('shape',),5,64,('policy',)))==(0,12345,1)


def test_cold_context_keeps_only_used_normalization_state():
    from rtx._routing_varlen_autotune import _setup,selected_route
    class Context:
        def mark_non_differentiable(self,*args):self.nondiff=args
        def save_for_backward(self,*args):self.saved=args
    outputs=(torch.empty(8,128),torch.empty(8,128,dtype=torch.int64),
        torch.empty(19,2,dtype=torch.int64),torch.empty(2*5*28*8),
        torch.empty(5,dtype=torch.int32),torch.empty(28,dtype=torch.int64),
        torch.empty(28,dtype=torch.int32),torch.empty(15,dtype=torch.int64))
    ctx=Context();_setup(ctx,(None,None,2,5,5,64,'spec',True,False,0),outputs)
    assert len(torch.ops.rtx_routing.selected_varlen_route.default._schema.returns)==8 and len(ctx.saved)==7
    assert all(t.shape!=(28,8) for t in ctx.saved)


def test_packed_expert_identity_is_separate_from_flat_and_batch_count():
    from rtx.moe import DEFAULT_MOE_CONFIG,_selection_spec,_parse_spec,_packed_expert_spec
    original=_selection_spec(DEFAULT_MOE_CONFIG,'cache',None,None)
    packed=_packed_expert_spec(original,4)
    assert _parse_spec(packed)['batch_contract']==dict(layout='independent_varlen',sequences=4,segment_alignment=32)
    assert 'batch_contract' not in _parse_spec(original)
    assert packed!=_packed_expert_spec(original,5)


def test_fake_cold_packed_selection_transitions_to_static_without_cpu_outputs():
    from torch._subclasses.fake_tensor import FakeTensorMode,FakeTensor
    from torch._dynamo.testing import CompileCounterWithBackend
    from rtx import _moe_selection as selections
    from rtx import _routing_varlen_autotune as startup
    model=router();model.autotune='cache';model._refresh_selection_spec()
    counter=CompileCounterWithBackend('eager')
    spec=model._selection_spec
    call=torch.compile(lambda scores,cu:startup.dispatch(scores,cu,2,5,5,64,spec,True,False,4729),
                       backend=counter,fullgraph=True,dynamic=False)
    selections.clear_selection_cells()
    try:
        with FakeTensorMode() as mode:
            def fake(shape,dtype):return FakeTensor(mode,torch.empty(shape,device='meta',dtype=dtype),torch.device('cuda:0'))
            scores=fake((19,8),torch.float32);cu=fake((5,),torch.int32)
            call(scores,cu);assert counter.frame_count==1
            startup.publish(scores,cu,2,5,spec,True,False,config())
            call(scores,cu);assert counter.frame_count==2
            call(scores,cu);assert counter.frame_count==2
        assert 'selected_varlen_route' in counter.graphs[0].code
        assert 'selected_varlen_route' not in counter.graphs[1].code
    finally:startup._clear_runtime_caches()


@cuda
@pytest.mark.parametrize('lengths,e,k',[( (5,0,11,3),8,2),((14,7,1),7,2),((4,3),4,4),((65,0,129),64,8)])
def test_cuda_native_pools_all_gradients_and_sequence_isolation(lengths,e,k):
    torch.manual_seed(1739)
    model=router('cuda',e=e,k=k)
    cpu=router(e=e,k=k);cpu.load_state_dict(model.state_dict())
    x=torch.randn(sum(lengths),16,device='cuda',requires_grad=True)
    cu=torch.tensor([0,*torch.tensor(lengths).cumsum(0).tolist()],device='cuda',dtype=torch.int32)
    native=model(x,cu_seqlens=cu)
    # Use actual discrete choices with independent double normalization; never
    # bless a different Sinkhorn derivative because QB made a different choice.
    from rtx._routing_reference import log_sinkhorn
    oracle=x.detach().cpu().double().requires_grad_()
    params=[p.detach().cpu().double().requires_grad_() for p in model.parameters()]
    from rtx._routing_reference import sigmoid_router_log_affinities
    scores=sigmoid_router_log_affinities(oracle,*params,rms_epsilon=model.rms_epsilon)
    expected_parts=[];real=0;cap=native.route_weights.shape[1]
    selected=(native.inverse//cap).cpu();dy=torch.randn_like(native.route_weights)
    for length in lengths:
        if length:
            n=(length+e//__import__('math').gcd(e,k)-1)//(e//__import__('math').gcd(e,k))*(e//__import__('math').gcd(e,k))
            v=torch.nn.functional.pad(scores[real:real+length].double(),(0,0,0,n-length),value=-__import__('math').log(2))
            for _ in range(model.sinkhorn_iterations):
                v=v-torch.logsumexp(v,1,keepdim=True)+__import__('math').log(k)
                v=v-torch.logsumexp(v,0,keepdim=True)+__import__('math').log(n*k//e)
            expected_parts.append(v[:length].gather(1,selected[real:real+length]).softmax(1))
        real+=length
    expected=torch.cat(expected_parts)
    chosen=native.route_weights.flatten()[native.inverse]
    torch.testing.assert_close(chosen.cpu().double(),expected,rtol=5e-5,atol=1e-6)
    expected_dy=dy.flatten()[native.inverse].cpu().double()
    a=torch.autograd.grad(native.route_weights,(x,*model.parameters()),dy)
    b=torch.autograd.grad(expected,(oracle,*params),expected_dy)
    for value,reference in zip(a,b):
        torch.testing.assert_close(value.cpu().double(),reference,rtol=.003,atol=2e-5)
    # A companion sequence's values must not alter this sequence's routes.
    if len(lengths)>1 and lengths[0]>0:
        mutated=x.detach().clone()
        # Pure amplitude changes are nearly erased by router RMS normalization;
        # change directions so this genuinely perturbs companion affinities.
        mutated[lengths[0]:]=torch.randn_like(mutated[lengths[0]:])*3
        before=model.log_affinities(x).detach();after=model.log_affinities(mutated).detach()
        assert (before[lengths[0]:]-after[lengths[0]:]).abs().max()>.001
        result=model(mutated,cu_seqlens=cu)
        torch.testing.assert_close(result.inverse[:lengths[0]],native.inverse[:lengths[0]],rtol=0,atol=0)
        torch.testing.assert_close(result.route_weights.flatten()[result.inverse[:lengths[0]]],
                                   chosen[:lengths[0]],rtol=0,atol=0)


@cuda
@pytest.mark.parametrize('backend',['torch','native','native_warp'])
def test_cuda_alternative_qb_and_metadata_are_the_same_policy(backend):
    torch.manual_seed(4175)
    base=router('cuda');x=torch.randn(19,16,device='cuda',requires_grad=True)
    cu=torch.tensor([0,5,5,16,19],device='cuda',dtype=torch.int32)
    expected=base(x,cu_seqlens=cu)
    other=router('cuda');other.load_state_dict(base.state_dict())
    other.config=replace(config(),qb_backend=backend,metadata_backend='torch',quota_cache_items=8,
                         quota_early_exit=True,quota_threads=64)
    other._refresh_selection_spec()
    result=other(x,cu_seqlens=cu)
    for a,b in zip(result,expected):torch.testing.assert_close(a,b,rtol=3e-5,atol=5e-7)
    dy=torch.randn_like(result.route_weights)
    a=torch.autograd.grad(result.route_weights,(x,*other.parameters()),dy)
    b=torch.autograd.grad(expected.route_weights,(x,*base.parameters()),dy)
    for v,r in zip(a,b):torch.testing.assert_close(v,r,rtol=.001,atol=8e-6)


@cuda
@pytest.mark.parametrize('cache_items',[0,4,16,32,64])
@pytest.mark.parametrize('expert_keys',[False,True])
def test_cuda_cached_packed_columns_exact_with_ties_empty_and_streaming_fallback(cache_items,expert_keys):
    torch.manual_seed(19331)
    lengths=[0,17,65,1029];cu=torch.tensor([0,0,17,82,1111],device='cuda',dtype=torch.int32)
    n=sum(lengths);e=7;k=2
    # These lengths are already scheduled for the direct primitive. Values
    # deliberately include equal and negative thresholds.
    values=torch.randn(n,e,device='cuda');values[::3]=.5;values[1::7]=-2
    values[17:82,0]=-0.  # canonical float ordering for signed zeros
    # Compile only the quantile to keep this exact-order-statistic test small.
    from rtx.kernels.routing_varlen import compile_varlen_quantile
    def select(items,expert_keys=False):
        fn=compile_varlen_quantile(n,4,e,k,64,items,expert_keys)
        out=torch.empty(4*e,device='cuda')
        workspace=[torch.empty(n*e,dtype=torch.uint32,device='cuda')] if expert_keys else []
        fn(values.flatten(),cu,out,*workspace)
        return out.reshape(4,e)
    baseline=select(0);actual=select(cache_items,expert_keys)
    torch.testing.assert_close(actual.view(torch.int32),baseline.view(torch.int32),rtol=0,atol=0)
    offset=0
    for s,length in enumerate(lengths):
        if length:
            expected=values[offset:offset+length].sort(0,descending=True).values[length*k//e]
            torch.testing.assert_close(actual[s],expected,rtol=0,atol=0)
        else:assert bool((actual[s]==0).all())
        offset+=length


@cuda
def test_cuda_expert_key_layout_preserves_hard_routes_and_all_gradients():
    import copy
    torch.manual_seed(19332)
    a=router('cuda',e=48,k=6)
    c=replace(a.config,qb_backend='native_warp',qb_threads=64,packed_column_cache_items=4,
        packed_row_backend='warp',packed_column_shards_forward=8,packed_column_shards_backward=8)
    a.config=c;a._refresh_selection_spec()
    b=copy.deepcopy(a);b.config=replace(c,packed_qb_column_backend='expert_keys');b._refresh_selection_spec()
    x=torch.randn(211,16,device='cuda',requires_grad=True)
    cu=torch.tensor([0,65,65,81,211],device='cuda',dtype=torch.int32)
    ref=a(x,cu_seqlens=cu);act=b(x,cu_seqlens=cu)
    for v,r in zip(act,ref):torch.testing.assert_close(v,r,rtol=0,atol=0)
    dy=torch.randn_like(ref.route_weights)
    expected_grad=torch.autograd.grad(ref.route_weights,(x,*a.parameters()),dy)
    actual_grad=torch.autograd.grad(act.route_weights,(x,*b.parameters()),dy)
    for v,r in zip(actual_grad,expected_grad):torch.testing.assert_close(v,r,rtol=0,atol=0)


@cuda
def test_cuda_packing_validates_finite_scores_and_resets_status_without_poisoning_context():
    from rtx._routing_varlen import compiled,layout
    torch.manual_seed(19339)
    source=torch.randn(19,8,device='cuda');cu=torch.tensor([0,5,5,16,19],device='cuda',dtype=torch.int32)
    key=layout(19,4,8,2,5,tuple(config().to_token()));n=key[4]
    scheduled=torch.empty(5,device='cuda',dtype=torch.int32);physical=torch.empty_like(scheduled)
    sequence=torch.empty(n,device='cuda',dtype=torch.int64);local=torch.empty_like(sequence)
    rowmap=torch.empty(n,device='cuda',dtype=torch.int32);packed=torch.empty(n*8,device='cuda')
    status=torch.empty(1,device='cuda',dtype=torch.int32)
    prefix,pack,*_=compiled(key,source.device.index)
    def validate():
        args=(source.flatten(),cu,scheduled,physical,sequence,local,rowmap,packed,status)
        prefix(*args);pack(*args);return int(status.item())
    assert validate()==1
    # Inspect the status primitive, not an intentional CUDA device assertion:
    # a bad-input test must not poison this process's other GPU experiments.
    for bad in (float('nan'),float('inf'),-float('inf')):
        source[17,7]=bad;assert validate()==0
        source[17,7]=.25;assert validate()==1


@cuda
@pytest.mark.parametrize('e,k,lengths,geometry',[
    (7,2,[5,0,14],(0,0,1,128)),(24,6,[65,3,129],(0,0,1,128)),
    (64,8,[65,0,129],(0,0,1,128)),(4,4,[4,3],(0,0,1,128)),
    (7,2,[5,0,14],(8,16,4,128)),(24,6,[65,3,129],(4,0,2,64)),
    (64,8,[65,0,129],(0,8,8,256)),(4,4,[4,3],(32,32,1,64)),
    (32,4,[512,2000,768,4096],(16,16,4,128)),
    (48,6,[2048,8000,3072,16384],(32,32,8,128)),
])
def test_cuda_warp_rows_preserve_transport_and_all_score_gradients(e,k,lengths,geometry):
    from rtx._routing_varlen import SavedTransport
    import math
    torch.manual_seed(19345+e)
    n=sum(lengths);cu=torch.tensor([0,*torch.tensor(lengths).cumsum(0).tolist()],device='cuda',dtype=torch.int32)
    source=torch.nn.functional.logsigmoid(torch.randn(n,e,device='cuda')*1.5).requires_grad_()
    f,b,g,threads=geometry
    c=replace(config(),packed_row_backend='warp',forward=replace(config().forward,column_groups=g,threads=threads),
              backward=replace(config().backward,column_groups=g,threads=threads),
              packed_column_shards_forward=f,packed_column_shards_backward=b)
    result=SavedTransport.apply(source,cu,(k,*c.to_token()),5,False)
    out,scheduled,_,_,_,rowmap,_=result
    upstream=torch.randn(out.shape[1],out.shape[0],device=out.device).T
    gradient=torch.autograd.grad(out,source,upstream)[0]
    oracle=source.detach().double().requires_grad_();pieces=[];offset=0;virtual=0
    objectives=[]
    for length in lengths:
        if not length:continue
        multiple=e//math.gcd(e,k);ns=(length+multiple-1)//multiple*multiple
        value=torch.nn.functional.pad(oracle[offset:offset+length],(0,0,0,ns-length),value=-math.log(2))
        for _ in range(5):
            value=value-torch.logsumexp(value,1,keepdim=True)+math.log(k)
            value=value-torch.logsumexp(value,0,keepdim=True)+math.log(ns*k//e)
        pieces.append(value);objectives.append((value*upstream[virtual:virtual+ns].double()).sum())
        offset+=length;virtual+=ns
    expected=torch.cat(pieces)
    ref_grad=torch.autograd.grad(sum(objectives),oracle)[0]
    torch.testing.assert_close(out[:virtual].double(),expected,rtol=5e-5,atol=2e-6)
    torch.testing.assert_close(gradient.double(),ref_grad,rtol=5e-4,atol=8e-6)
    assert bool(out.isfinite().all()) and bool(gradient.isfinite().all())


@cuda
@pytest.mark.parametrize('shards',[0,16])
def test_cuda_compiled_graph_accepts_mutable_boundaries_without_recompile(shards):
    from torch._dynamo.testing import CompileCounterWithBackend
    torch.manual_seed(5121)
    model=router('cuda');x=torch.randn(19,16,device='cuda',requires_grad=True)
    model.config=replace(model.config,packed_column_shards_forward=shards,packed_column_shards_backward=shards,
        packed_qb_column_backend='expert_keys' if shards else 'row_major')
    model._refresh_selection_spec()
    cu=torch.tensor([0,5,5,16,19],device='cuda',dtype=torch.int32)
    from rtx._routing_varlen import workspace_bounds
    dy=torch.randn(8,workspace_bounds(19,4,8,2)[1],device='cuda')
    counter=CompileCounterWithBackend('inductor')
    fn=torch.compile(model,backend=counter,fullgraph=True,dynamic=False)
    def step():
        route=fn(x,cu_seqlens=cu)
        return route,torch.autograd.grad(route.route_weights,(x,*model.parameters()),dy)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):step()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):result,gradients=step()
    torch.cuda.current_stream().wait_stream(stream)
    try:
        for boundaries in ([0,0,7,10,19],[0,3,11,11,19]):
            with torch.no_grad():
                cu.copy_(torch.tensor(boundaries,device='cuda',dtype=torch.int32));x.add_(.01)
            graph.replay();torch.cuda.synchronize()
            expected=model(x,cu_seqlens=cu)
            for a,b in zip(result,expected):torch.testing.assert_close(a,b,rtol=5e-5,atol=2e-6)
            expected_grad=torch.autograd.grad(expected.route_weights,(x,*model.parameters()),dy)
            for a,b in zip(gradients,expected_grad):torch.testing.assert_close(a,b,rtol=.003,atol=1e-5)
        assert counter.frame_count==1
    finally:graph.reset()


@cuda
def test_cuda_packed_first_hit_search_cache_and_backward(tmp_path,monkeypatch):
    from rtx.autotune import AutotunePolicy
    import rtx.autotune.varlen_routing as tuning
    import rtx._routing_varlen_autotune as startup
    torch.manual_seed(7462)
    model=router('cuda');model.autotune='balanced';model.autotune_cache_dir=tmp_path
    model.tuning_policy=AutotunePolicy(max_trials=2,max_seconds=90,samples=3,calls_per_sample=1,
        warmup=1,confirmation_repeats=0,search_schedule='progressive',candidate_mix='mixed')
    model._refresh_selection_spec()
    x=torch.randn(19,16,device='cuda',requires_grad=True)
    cu=torch.tensor([0,5,5,16,19],device='cuda',dtype=torch.int32)
    fn=torch.compile(model,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    result=fn(x,cu_seqlens=cu)
    assert len(list(tmp_path.glob('runtime_winners/balanced_varlen_routing/*/*.json')))==1
    startup._clear_runtime_caches()
    def forbidden(*a,**kw):raise AssertionError('Backward/cache hit unexpectedly tuned again')
    monkeypatch.setattr(tuning,'tune',forbidden)
    result.route_weights.square().sum().backward()
    assert x.grad is not None and all(p.grad is not None for p in model.parameters())
    for _ in range(2):
        actual=fn(x,cu_seqlens=cu)
        torch.testing.assert_close(actual.inverse,result.inverse,rtol=0,atol=0)
    startup._clear_runtime_caches()


@cuda
def test_cuda_realistic_mixed_lengths_no_sequence_competition():
    from rtx._routing_varlen import layout
    torch.manual_seed(7719)
    model=router('cuda',e=48,k=6)
    lengths=[512,2000,768,4096];total=sum(lengths)
    x=torch.randn(total,16,device='cuda',requires_grad=True)
    cu=torch.tensor([0,512,2512,3280,7376],device='cuda',dtype=torch.int32)
    result=model(x,cu_seqlens=cu);c=result.route_weights.shape[1]
    local=result.inverse%c;expert=result.inverse//c
    physical=0;start=0
    for length in lengths:
        capacity=((length+7)//8*8)*6//48
        assert bool(((local[start:start+length]>=physical)&(local[start:start+length]<physical+capacity)).all())
        assert not bool((expert[start:start+length].sort(1).values.diff(dim=1)==0).any())
        physical+=(capacity+31)//32*32;start+=length
    loss=result.route_weights.square().sum()
    grads=torch.autograd.grad(loss,(x,*model.parameters()))
    assert all(bool(v.isfinite().all()) for v in grads)
    assert layout(total,4,48,6,5,tuple(config().to_token()))[4]<4*max(lengths)


@cuda
def test_cuda_tiny_mostly_empty_batch_uses_nonempty_workspace_bound():
    from rtx._routing_varlen import workspace_bounds
    torch.manual_seed(19359)
    model=router('cuda');model.config=replace(config(),packed_row_backend='warp',packed_column_cache_items=4)
    model._refresh_selection_spec()
    x=torch.randn(3,16,device='cuda',requires_grad=True)
    cu=torch.tensor([0,*([1]*8),*([2]*6),*([3]*6)],device='cuda',dtype=torch.int32)
    assert workspace_bounds(3,20,8,2)==(12,96)
    fn=torch.compile(model,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    route=fn(x,cu_seqlens=cu)
    assert route.expert_ids.shape==(8,96)
    torch.testing.assert_close(route.inverse%96,
        torch.tensor([[0,0],[32,32],[64,64]],device='cuda'),rtol=0,atol=0)
    torch.testing.assert_close(route.route_weights.flatten()[route.inverse].sum(1),torch.ones(3,device='cuda'))
    gradients=torch.autograd.grad(route.route_weights.square().sum(),(x,*model.parameters()))
    assert all(bool(v.isfinite().all()) for v in gradients)


@cuda
@pytest.mark.parametrize('d,shared,e,k,h',[(384,256,24,6,128),(384,256,32,4,128),
                                        (512,384,64,8,128),(768,512,48,6,256)])
@pytest.mark.parametrize('packed_execution',['original','warp_cached','fused_route','hybrid','fused_dy','fused_all'])
def test_cuda_whole_packed_moe_all_four_designs_compiled_all_gradients(d,shared,e,k,h,packed_execution):
    from rtx.moe import BalancedMoE,DEFAULT_MOE_CONFIG
    torch.manual_seed(8717+d+e)
    rc=config() if packed_execution=='original' else replace(config(),packed_row_backend='warp',packed_column_cache_items=32)
    expert_config=DEFAULT_MOE_CONFIG
    if packed_execution=='fused_route':
        p=expert_config.prepared
        expert_config=replace(expert_config,prepared=replace(p,backward_preparation='joint',
            route_gradient_epilogue='fused',transpose_mode='warp',input_preparation='indexed_transpose',
            recompute=replace(p.recompute,glu_grad_store='exchange')))
    if packed_execution=='hybrid':
        if h!=128:pytest.skip('On-chip forward currently supports H=128')
        from rtx.autotune.moe import make_moe_adapter
        a=make_moe_adapter((194,d,e,96,k,h),None)
        expert_config=a.update_fn(a.initial_config,'prepared.forward_engine','onchip')
        expert_config=a.update_fn(expert_config,'prepared.input_preparation','indexed_transpose')
    if packed_execution in ('fused_dy','fused_all'):
        from rtx.autotune.moe import make_moe_adapter
        a=make_moe_adapter((194,d,e,96,k,h),None)
        expert_config=a.update_fn(a.initial_config,'prepared.dy_preparation',
            'fused_all' if packed_execution=='fused_all' else 'fused')
    model=BalancedMoE(d,e,h,top_k=k,shared_hidden_features=shared,router_dim=16,
        device='cuda',config=expert_config,autotune='off',router_options={'config':rc})
    with torch.no_grad():model.router.projection_weight.normal_(std=.05)
    x=(torch.randn(194,d,device='cuda',dtype=torch.bfloat16)*.3).requires_grad_()
    cu=torch.tensor([0,65,65,194],device='cuda',dtype=torch.int32)
    dy=torch.randn_like(x)
    # Hard routing is discontinuous: eager/Inductor affinity rounding can
    # change a near-tie repair decision. Compare expert/compiler derivatives
    # with exactly the SAME differentiable route, not two different graphs.
    # Keep the original router Parameters in the derivative set; their saved
    # Sinkhorn backward still executes through the shared route tensor.
    parameters=tuple(model.parameters())
    original_router=model.router
    frozen=original_router(x,cu_seqlens=cu)
    class FrozenRoute(torch.nn.Module):
        def forward(self,value,*,cu_seqlens=None):return frozen
    model.router=FrozenRoute()
    expected=model(x,cu_seqlens=cu)
    ref_grads=torch.autograd.grad(expected,(x,*parameters),dy,retain_graph=True)
    fn=torch.compile(model,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    actual=fn(x,cu_seqlens=cu)
    gradients=torch.autograd.grad(actual,(x,*parameters),dy)
    for index,(value,reference) in enumerate(zip((actual,*gradients),(expected,*ref_grads))):
        assert bool(value.isfinite().all()),index
        delta=value.float()-reference.float()
        nrms=delta.norm()/reference.float().norm().clamp_min(1e-12)
        assert nrms<.01 or delta.abs().max()<2e-7,(index,float(nrms),float(delta.abs().max()))


@cuda
@pytest.mark.parametrize('shards',[0,16])
def test_cuda_packed_selected_opcheck(shards):
    from rtx._routing_varlen_autotune import selected_route
    model=router('cuda');model.config=replace(config(),packed_row_backend='warp',packed_column_cache_items=32,
        packed_column_shards_forward=shards,packed_column_shards_backward=shards,
        packed_qb_column_backend='expert_keys' if shards else 'row_major')
    model._refresh_selection_spec()
    scores=torch.nn.functional.logsigmoid(torch.randn(19,8,device='cuda')).requires_grad_()
    cu=torch.tensor([0,5,5,16,19],device='cuda',dtype=torch.int32)
    results=torch.library.opcheck(selected_route,(scores,cu,2,5,5,64,model._selection_spec,True,False,9812))
    assert set(results.values())=={'SUCCESS'}


@cuda
def test_cuda_whole_packed_moe_user_side_automatic_graphs(monkeypatch,tmp_path,record_property):
    from rtx.moe import BalancedMoE,DEFAULT_MOE_CONFIG
    torch.manual_seed(9129)
    model=BalancedMoE(128,8,128,top_k=2,shared_hidden_features=128,router_dim=16,
        device='cuda',config=DEFAULT_MOE_CONFIG,autotune='cache',autotune_cache_dir=tmp_path,
        router_options={'config':replace(config(),packed_row_backend='warp',packed_column_cache_items=32),'autotune':'cache'})
    with torch.no_grad():model.router.projection_weight.normal_(std=.05)
    x=(torch.randn(68,128,device='cuda',dtype=torch.bfloat16)*.3).requires_grad_()
    cu=torch.tensor([0,29,29,68],device='cuda',dtype=torch.int32);dy=torch.randn_like(x)
    replays=[];original=torch.cuda.CUDAGraph.replay
    def replay(graph):replays.append(id(graph));return original(graph)
    monkeypatch.setattr(torch.cuda.CUDAGraph,'replay',replay)
    actual_fn=torch.compile(model,fullgraph=True,dynamic=False,options={'triton.cudagraphs':True})
    reference_fn=torch.compile(model,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    forward=backward=0;parameters=tuple(model.parameters())
    for iteration in range(6):
        torch.compiler.cudagraph_mark_step_begin()
        with torch.no_grad():
            x.add_(.001)
            if iteration%2:cu.copy_(torch.tensor([0,0,37,68],device='cuda',dtype=torch.int32))
            else:cu.copy_(torch.tensor([0,29,29,68],device='cuda',dtype=torch.int32))
        before=len(replays);out=actual_fn(x,cu_seqlens=cu);forward+=len(replays)-before
        saved=out.detach().clone()
        before=len(replays);gradients=torch.autograd.grad(out,(x,*parameters),dy);backward+=len(replays)-before
        values=[saved,*[v.detach().clone() for v in gradients]]
        del out,gradients
        if iteration==5:
            expected=reference_fn(x,cu_seqlens=cu)
            refs=[expected,*torch.autograd.grad(expected,(x,*parameters),dy)]
            for index,(value,ref) in enumerate(zip(values,refs)):
                delta=value.float()-ref.float();nrms=delta.norm()/ref.float().norm().clamp_min(1e-12)
                assert bool(value.isfinite().all()) and (nrms<.01 or delta.abs().max()<2e-7),(index,float(nrms))
        del values
    record_property('forward_graph_replays',forward);record_property('backward_graph_replays',backward)
    assert forward>0 and backward>0


@cuda
@pytest.mark.parametrize('shards',[0,16])
def test_cuda_packed_live_memory_and_plan_count_stabilize(record_property,shards):
    from rtx._routing_varlen import _PLANS
    torch.manual_seed(9418)
    model=router('cuda');x=torch.randn(19,16,device='cuda',requires_grad=True)
    model.config=replace(model.config,packed_column_shards_forward=shards,packed_column_shards_backward=shards)
    model._refresh_selection_spec()
    cu=torch.tensor([0,5,5,16,19],device='cuda',dtype=torch.int32)
    fn=torch.compile(model,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    def step():
        route=fn(x,cu_seqlens=cu)
        values=torch.autograd.grad(route.route_weights.square().sum(),(x,*model.parameters()))
        assert all(bool(v.isfinite().all()) for v in values)
    for _ in range(4):step()
    torch.cuda.synchronize();before=len(_PLANS);live=[]
    for iteration in range(8):
        cu.copy_(torch.tensor([0,0,7,10,19] if iteration%2 else [0,5,5,16,19],device='cuda',dtype=torch.int32))
        step();torch.cuda.synchronize();live.append(torch.cuda.memory_allocated())
    record_property('live_allocation_growth_bytes',max(live)-min(live))
    assert max(live)==min(live) and len(_PLANS)==before


@cuda
def test_cuda_compiled_packed_optimizer_steps_are_finite_and_learn(record_property):
    from rtx.moe import BalancedMoE,DEFAULT_MOE_CONFIG
    torch.manual_seed(19931)
    c=replace(config(),packed_row_backend='warp',packed_column_cache_items=32)
    model=BalancedMoE(128,8,128,top_k=2,shared_hidden_features=128,router_dim=16,
        device='cuda',config=DEFAULT_MOE_CONFIG,autotune='off',router_options={'config':c})
    with torch.no_grad():model.router.projection_weight.normal_(std=.05)
    x=torch.randn(68,128,device='cuda',dtype=torch.bfloat16)*.3
    cu=torch.tensor([0,29,29,68],device='cuda',dtype=torch.int32)
    fn=torch.compile(model,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    # A well-conditioned zero-target smoke task. Scaling the loss makes SGD
    # updates representable in the intentionally BF16 Parameter storage.
    optimizer=torch.optim.SGD(model.parameters(),lr=.5)
    losses=[];before=[p.detach().clone() for p in model.parameters()]
    for _ in range(6):
        optimizer.zero_grad(set_to_none=True)
        output=fn(x,cu_seqlens=cu);loss=output.float().square().mean()*1e6
        loss.backward()
        assert bool(loss.isfinite()) and all(p.grad is not None and bool(p.grad.isfinite().all()) for p in model.parameters())
        losses.append(float(loss.detach()));optimizer.step()
    record_property('synthetic_training_losses',str(losses))
    assert losses[-1]<losses[0],losses
    assert any(not torch.equal(a,b) for a,b in zip(before,model.parameters()))
