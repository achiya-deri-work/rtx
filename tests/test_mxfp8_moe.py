"""Grouped expert numerical contracts; no quantizer crosses an expert edge."""
import pytest
import torch


def quantized_reference(value):
    """Independent FP32 round-to-infinity MXFP8 dequantization oracle."""
    width = value.shape[-1]
    padded = torch.nn.functional.pad(value.float(), (0, (-width) % 32))
    blocks = padded.reshape(*padded.shape[:-1], -1, 32)
    maximum = blocks.abs().amax(-1, keepdim=True)
    exponent = torch.ceil(torch.log2(maximum.clamp_min(2.**-119)/448)).clamp(-127, 127)
    scale = torch.exp2(exponent)
    return ((blocks/scale).to(torch.float8_e4m3fn).float()*scale).reshape_as(padded)[..., :width]


def operational_reference(x,gu,down,route,dy=None):
    """Separate batched Torch forward and explicit quantized STE adjoints."""
    ids,weights,inverse=route
    n,d=x.shape
    e,c=ids.shape
    h=down.shape[2]
    slot=torch.arange(e*c,device=x.device).reshape(e,c)
    valid=inverse[ids].eq(slot[...,None]).any(-1)
    gathered=x[ids]*valid[...,None]
    proj=torch.bmm(quantized_reference(gathered),quantized_reference(gu).transpose(1,2))
    gate,up=proj[...,:h],proj[...,h:]
    sig=1/(1+torch.exp(-gate))
    act=gate/(1+torch.exp(-gate))
    hidden=quantized_reference(act*up)
    outputs=torch.bmm(hidden,quantized_reference(down).transpose(1,2)).bfloat16()
    result=(outputs.flatten(0,1)[inverse].float()*weights.flatten()[inverse][...,None]).sum(1)
    if dy is None:return result
    slot_dy=(dy[ids]*weights[...,None]*valid[...,None]).bfloat16()
    dw=(dy[ids]*outputs.float()*valid[...,None]).sum(-1)
    dh=torch.bmm(quantized_reference(slot_dy),quantized_reference(down.transpose(1,2)).transpose(1,2)).bfloat16()
    dp=torch.cat((dh.float()*up*sig*(1+gate*(1-sig)),dh.float()*act),-1).bfloat16()
    ddown=torch.bmm(quantized_reference(slot_dy.transpose(1,2)),
                    quantized_reference(hidden.bfloat16().transpose(1,2)).transpose(1,2)).bfloat16()
    dgu=torch.bmm(quantized_reference(dp.transpose(1,2)),
                  quantized_reference(gathered.transpose(1,2)).transpose(1,2)).bfloat16()
    dxg=torch.bmm(quantized_reference(dp),quantized_reference(gu.transpose(1,2)).transpose(1,2)).bfloat16()
    dx=dxg.flatten(0,1)[inverse].float().sum(1).bfloat16()
    return result,(dx,dgu,ddown,dw)


def make_route(n,e,k,device):
    from rtx._routing_reference import padded_token_count,routing_metadata
    from rtx.routing import BalancedRouting
    ns=padded_token_count(n,e,k)
    assignments=(torch.arange(ns*k,device=device)%e).reshape(ns,k)
    # Random relabeling retains balance but defeats an expert-ID assumption.
    assignments=torch.randperm(e,device=device)[assignments]
    return BalancedRouting(*routing_metadata(torch.randn(ns,e,device=device),assignments,num_real_tokens=n))


def test_moe_cpu_reference_parameters_and_fullgraph():
    from rtx.moe import BalancedMoE,MXFP8MoEConfig
    assert MXFP8MoEConfig.from_dict(MXFP8MoEConfig().to_dict())==MXFP8MoEConfig()
    model=BalancedMoE(16,8,16,top_k=2,shared_hidden_features=16,backend='torch',router_dim=4)
    assert all(p.ndim==2 for p in model.gate_up.parameters())
    assert len(list(model.gate_up.parameters()))==16
    assert all(p.dtype==torch.bfloat16 for p in model.gate_up.parameters())
    assert len(model.shared_gate_up)==len(model.shared_down)==1
    fn=torch.compile(model,backend='aot_eager',fullgraph=True,dynamic=False)
    x=torch.randn(13,16,dtype=torch.bfloat16,requires_grad=True)
    y=fn(x)
    assert y.shape==x.shape and y.dtype==torch.bfloat16
    y.float().square().sum().backward()
    assert x.grad is not None and bool(x.grad.isfinite().all())
    assert all(p.grad is not None for p in model.parameters())


def test_moe_shared_tuner_adapter_resumption_and_no_policy_axes():
    from dataclasses import replace
    from rtx.autotune.moe import make_moe_adapter
    from rtx.autotune.effort import AutotunePolicy,make_effort_autotuner
    from rtx.autotune.outcomes import TrialOutcome
    from rtx.autotune.store import InMemoryTuningStore
    a=make_moe_adapter((128,384,24,32,6,128),
        lambda c:TrialOutcome('ok',median_ms=1+c.gate_up.tile_m*.001))
    assert not any('top_k' in name or 'rounds' in name or 'experts' in name for name in a.axes)
    assert 'gate_dw.transport' in a.axes and 'down_dx.transport' in a.axes
    assert 'gate_up.tile_m' in a.axes and 'movement_threads' in a.axes
    same=replace(a.initial_config,gate_up=replace(a.initial_config.gate_up,glu_grad_load='prefetch'))
    assert a.config_id(same)==a.config_id(a.initial_config)
    for _,_,c in a.neighbors(a.initial_config):assert a.deserialize(a.serialize(c))==c
    store=InMemoryTuningStore()
    policy=AutotunePolicy(max_trials=4,max_seconds=20,confirmation_repeats=0,
                          search_schedule='progressive',candidate_mix='mixed')
    first=make_effort_autotuner(a,store,policy).tune()
    second=make_effort_autotuner(a,store,replace(policy,max_trials=8)).tune()
    assert len(store.observations)==8 and second.median_ms<=first.median_ms


def test_unsafe_dynamic_register_role_budget_is_rejected():
    from rtx.kernels.mxfp8 import normalize_fwd_config,MXFP8Problem
    cfg=normalize_fwd_config(tile_m=64,tile_n=256,load_engine='tma',schedule='three_role',
        bf16_tile_k=32,bf16_swizzle='none',bf16_stages=1,consumer_registers=232)
    assert 'register' in cfg.implementation_rejection(MXFP8Problem(128,256,384))
    cfg=normalize_fwd_config(cfg,consumer_registers=160)
    assert cfg.implementation_rejection(MXFP8Problem(128,256,384)) is None


def test_grouped_grid_and_weight_address_bounds_without_allocation():
    from rtx.moe import _validate
    def operands(e,d,h):
        bf=lambda *s:torch.empty(s,device='meta',dtype=torch.bfloat16)
        ids=torch.empty(e,1,device='meta',dtype=torch.int64)
        return (bf(1,d),bf(e,2*h,d),bf(e,d,h),ids,
                torch.empty(e,1,device='meta',dtype=torch.float32),
                torch.empty(1,1,device='meta',dtype=torch.int64))
    with pytest.raises(ValueError,match='65535'):_validate(*operands(65536,1,1))
    with pytest.raises(ValueError,match='signed index'):_validate(*operands(60000,2048,2048))


@pytest.mark.parametrize('shape',[(512,384,24,128,6,128),(2000,768,48,250,6,256)])
def test_complete_vector_training_family_is_reachable(shape):
    from rtx.autotune.moe import structural_seeds,rejection
    seeds=list(structural_seeds(shape))
    vector=next(c for c in seeds if c.family=='grouped_staged' and c.gate_up.quant_vec==8)
    assert rejection(vector,shape) is None
    for stage in ('down_dx','down_dw','gate_dx','gate_dw'):
        config=getattr(vector,stage)
        assert config.quant_vec==8 and config.quant_store_bits==16
        assert config.load_engine in ('cpasync','tma') and config.bf16_swizzle=='none'
    fused=[c for c in seeds if c.family=='grouped_onchip']
    assert bool(fused)==(shape[-1]==128)
    assert all(rejection(c,shape) is None for c in fused)


def test_pipeline_seeds_are_proposals_not_only_pool_metadata():
    from rtx.autotune.moe import make_moe_adapter,make_moe_tuner
    from rtx.autotune.effort import AutotunePolicy
    from rtx.autotune.store import InMemoryTuningStore
    from rtx.autotune.outcomes import TrialOutcome
    adapter=make_moe_adapter((512,384,24,128,6,128),lambda c:TrialOutcome('ok',median_ms=1.))
    policy=AutotunePolicy(max_trials=16,search_schedule='progressive',candidate_mix='mixed')
    tuner=make_moe_tuner(adapter,InMemoryTuningStore(),policy)
    pool=tuner.strategies['structural'].pool
    kinds={(c.family,c.gate_up.load_engine,c.gate_up.schedule) for c in pool[:8]}
    assert ('grouped_onchip','scalar','cooperative') in kinds
    assert ('grouped_staged','cpasync','cooperative') in kinds
    assert ('grouped_staged','tma','warp_specialized') in kinds
    assert ('grouped_staged','tma','three_role') in kinds
    assert any(c.family=='grouped_prepared' and c.prepared.forward_engine=='onchip' for c in pool[:8])
    assert all(adapter.rejection(c) is None for c in pool[:8])
    onchip=next(c for c in pool if c.family=='grouped_onchip')
    from dataclasses import replace
    ignored=replace(onchip,down=replace(onchip.down,maxrregcount=96,persistent=True))
    assert adapter.config_id(ignored)==adapter.config_id(onchip)
    light=make_moe_adapter((512,384,24,128,6,128),None,allow_onchip=False)
    quick=make_moe_tuner(light,InMemoryTuningStore(),AutotunePolicy.preset('quick'))
    assert light.axes['family']==('grouped_staged','grouped_prepared')
    assert all(c.family!='grouped_onchip' for c in quick.strategies['structural'].pool)
    assert light.axes['prepared.forward_engine']==('packed',)
    assert all(c.prepared.forward_engine=='packed' for c in quick.strategies['structural'].pool)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='RTX CUDA required')


@cuda
@pytest.mark.parametrize('a_orientation,b_orientation', [('row','row'), ('row','transpose'), ('transpose','transpose')])
@pytest.mark.parametrize('engine', ['scalar','tma'])
@pytest.mark.parametrize('m,k',[(32,46),(64,128)])
def test_grouped_core_matches_independent_quantized_gemm(a_orientation,b_orientation,engine,m,k):
    from rtx.kernels.mxfp8 import MXFP8Problem, normalize_fwd_config
    from rtx.kernels.mxfp8_moe import compile_grouped_linear
    torch.manual_seed(815)
    torch.set_float32_matmul_precision('highest')
    e,n=3,64
    ashape=(m,k) if a_orientation=='row' else (k,m)
    bshape=(n,k) if b_orientation=='row' else (k,n)
    if engine=='tma' and ((ashape[1]%8) or (bshape[1]%8)):
        pytest.skip('TMA physical row strides require 16-byte alignment')
    a=torch.randn(e,*ashape,device='cuda',dtype=torch.bfloat16)*.1
    b=torch.randn(e,*bshape,device='cuda',dtype=torch.bfloat16)*.1
    cfg=normalize_fwd_config(tile_m=32,tile_n=64,load_engine=engine,
                            bf16_tile_k=32 if engine=='tma' else 128,bf16_stages=2 if engine=='tma' else 1)
    out=torch.empty(e,m,n,device='cuda',dtype=torch.bfloat16)
    fn=compile_grouped_linear(e,MXFP8Problem(m,n,k),cfg,a_orientation,b_orientation)
    af,bf,outf=a.flatten(0,1),b.flatten(0,1),out.flatten(0,1)
    fn(af,bf,outf,outf,outf,outf,outf)
    al=a if a_orientation=='row' else a.transpose(1,2)
    bl=b if b_orientation=='row' else b.transpose(1,2)
    expected=torch.bmm(quantized_reference(al),quantized_reference(bl).transpose(1,2)).bfloat16()
    torch.testing.assert_close(out,expected,rtol=.008,atol=2e-4)
    assert bool(out.isfinite().all())


@cuda
@pytest.mark.parametrize('n,e,k,d,h',[(16,4,2,64,64),(13,8,2,64,64),(13,5,2,35,49)])
def test_routed_forward_and_all_backward_parts(n,e,k,d,h):
    from rtx.moe import mxfp8_routed_experts
    torch.manual_seed(816)
    torch.set_float32_matmul_precision('highest')
    x=(torch.randn(n,d,device='cuda',dtype=torch.bfloat16)*.3).requires_grad_()
    gu=(torch.randn(e,2*h,d,device='cuda',dtype=torch.bfloat16)*.15).requires_grad_()
    down=(torch.randn(e,d,h,device='cuda',dtype=torch.bfloat16)*.15).requires_grad_()
    route=make_route(n,e,k,'cuda')
    route=type(route)(route.expert_ids,route.route_weights.detach().requires_grad_(),route.inverse)
    saved=[]
    with torch.autograd.graph.saved_tensors_hooks(lambda t:(saved.append(t),t)[1],lambda t:t):
        actual=mxfp8_routed_experts(x,gu,down,route)
    # Apart from the immutable CPU plan, only caller-owned operands/metadata.
    pointers={t.data_ptr() for t in (x,gu,down,*route)}
    assert all(t.device.type=='cpu' or t.data_ptr() in pointers for t in saved)
    # Noncontiguous upstream gradients are a normal autograd input contract.
    dy=torch.randn(actual.shape[1],actual.shape[0],device='cuda',dtype=actual.dtype).T
    grads=torch.autograd.grad(actual,(x,gu,down,route.route_weights),dy)
    with torch.no_grad():expected,expected_grads=operational_reference(x,gu,down,route,dy)
    torch.testing.assert_close(actual,expected,rtol=.01,atol=2e-4)
    for name,a,b in zip(('dx','dgu','ddown','dweights'),grads,expected_grads):
        assert bool(a.isfinite().all()),name
        nrms=(a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)
        assert nrms<.006,(name,float(nrms))


@cuda
@pytest.mark.parametrize('prepared',['none','separate','joint','tma_outputs','expert_grid','portfolio_grid'])
def test_routed_opcheck_fullgraph_changed_inputs_and_immutable_plan(tmp_path,prepared):
    from rtx.moe import _experts_op,_selection_spec,DEFAULT_MOE_CONFIG,mxfp8_routed_experts
    from rtx.autotune.moe import _choices
    torch.manual_seed(817)
    x=torch.randn(13,64,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    gu=(torch.randn(8,128,64,device='cuda',dtype=torch.bfloat16)*.02).requires_grad_()
    down=(torch.randn(8,64,64,device='cuda',dtype=torch.bfloat16)*.02).requires_grad_()
    route=make_route(13,8,2,'cuda')
    config=DEFAULT_MOE_CONFIG
    if prepared!='none':
        from dataclasses import replace
        config=replace(config,family='grouped_prepared',prepared=replace(config.prepared,
            backward_preparation='joint' if prepared=='joint' else 'separate',recompute=replace(config.prepared.recompute,
                glu_grad_store='exchange' if prepared=='joint' else 'direct')))
        if prepared in ('expert_grid','portfolio_grid'):
            config=replace(config,prepared=replace(config.prepared,quant_grid=prepared.removesuffix('_grid')))
        if prepared=='tma_outputs':
            from rtx.autotune.moe import make_moe_adapter
            a=make_moe_adapter((13,64,8,route.expert_ids.shape[1],2,64),None,initial=config)
            config=a.update_fn(config,'prepared.bf16_outputs','tma_all')
    spec=_selection_spec(config,'off',None,None)
    assert set(torch.library.opcheck(_experts_op,(x,gu,down,*route,spec,True)).values())=={'SUCCESS'}
    fn=torch.compile(lambda x,gu,down:mxfp8_routed_experts(x,gu,down,route,config=config),
                     fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    y=fn(x,gu,down)
    _choices.clear()
    dy=torch.randn_like(y)
    grads=torch.autograd.grad(y,(x,gu,down),dy)
    with torch.no_grad():expected=operational_reference(x,gu,down,route,dy)
    torch.testing.assert_close(y,expected[0],rtol=.015,atol=2e-4)
    for a,b in zip(grads,expected[1][:3]):
        assert float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))<.006
    with torch.no_grad():x.add_(.125);gu.mul_(.9)
    changed=fn(x,gu,down)
    torch.testing.assert_close(changed,operational_reference(x,gu,down,route),rtol=.015,atol=2e-4)
    assert not torch.equal(changed,y)


@cuda
def test_complete_balanced_moe_default_compile_backward_and_optimizer():
    from rtx.moe import BalancedMoE
    torch.manual_seed(818)
    model=BalancedMoE(384,24,128,top_k=6,shared_hidden_features=256,
                      device='cuda',autotune='off')
    with torch.no_grad():model.router.projection_weight.normal_(std=.02)
    fn=torch.compile(model,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,foreach=False)
    x=torch.randn(128,384,device='cuda',dtype=torch.bfloat16,requires_grad=True)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True);x.grad=None
        y=fn(x)
        assert y.shape==x.shape and y.dtype==torch.bfloat16 and bool(y.isfinite().all())
        y.float().square().mean().backward()
        assert x.grad is not None and bool(x.grad.isfinite().all())
        for p in model.parameters():assert p.grad is not None and bool(p.grad.isfinite().all())
        optimizer.step()
        model.gate_up.validate();model.down.validate()
        assert all(p.ndim==2 for p in model.gate_up.parameters())


@cuda
@pytest.mark.parametrize('stages,schedule',[(1,'serial'),(1,'warp_specialized'),(2,'warp_specialized'),(3,'warp_specialized'),(4,'warp_specialized')])
def test_onchip_hidden_and_buffered_down_match_staged(stages,schedule):
    from dataclasses import replace
    from rtx.moe import DEFAULT_MOE_CONFIG,_run_forward
    from rtx.configs.glu import normalize_glu_config
    from rtx.kernels.mxfp8 import normalize_fwd_config
    torch.manual_seed(820)
    n,d,e,k,h=130,384,2,1,128
    x=torch.randn(n,d,device='cuda',dtype=torch.bfloat16)*.3
    gu=torch.randn(e,2*h,d,device='cuda',dtype=torch.bfloat16)*.05
    down=torch.randn(e,d,h,device='cuda',dtype=torch.bfloat16)*.05
    route=make_route(n,e,k,'cuda')
    gate=normalize_glu_config(DEFAULT_MOE_CONFIG.gate_up,tile_n=256,glu_scratch='dedicated')
    dc=normalize_fwd_config(DEFAULT_MOE_CONFIG.down,quant_vec=8,quant_load_bits=128,quant_store_bits=32)
    staged=replace(DEFAULT_MOE_CONFIG,gate_up=gate,down=dc)
    fused=replace(staged,family='grouped_onchip',onchip_down_stages=stages,onchip_pipeline=schedule)
    expected=_run_forward(x,gu,down,*route,staged)
    for _ in range(3):
        actual=_run_forward(x,gu,down,*route,fused)
        torch.testing.assert_close(actual,expected,rtol=.008,atol=2e-5)
        x.add_(.001)
        expected=_run_forward(x,gu,down,*route,staged)


@cuda
def test_expert_first_hit_under_fullgraph_and_backward_plan_survives_cache_clear(tmp_path,monkeypatch):
    from rtx.moe import mxfp8_routed_experts
    from rtx.autotune.effort import AutotunePolicy
    import rtx.autotune.moe as tuning
    torch.manual_seed(821)
    x=(torch.randn(13,64,device='cuda',dtype=torch.bfloat16)*.3).requires_grad_()
    gu=(torch.randn(8,128,64,device='cuda',dtype=torch.bfloat16)*.02).requires_grad_()
    down=(torch.randn(8,64,64,device='cuda',dtype=torch.bfloat16)*.02).requires_grad_()
    route=make_route(13,8,2,'cuda')
    policy=AutotunePolicy(max_trials=4,max_seconds=60,samples=3,calls_per_sample=1,warmup=1,
                          confirmation_repeats=0,search_schedule='progressive',candidate_mix='mixed')
    fn=torch.compile(lambda x,gu,down:mxfp8_routed_experts(x,gu,down,route,autotune='balanced',
        tuning_policy=policy,autotune_cache_dir=str(tmp_path)),fullgraph=True,dynamic=False)
    y=fn(x,gu,down)
    assert list((tmp_path/'moe_effort').rglob('*.jsonl'))
    tuning._choices.clear()
    def forbidden(*a,**kw):raise AssertionError('Backward or a disk cache hit unexpectedly tuned')
    monkeypatch.setattr(tuning,'tune_moe',forbidden)
    gradients=torch.autograd.grad(y,(x,gu,down),torch.randn_like(y))
    assert all(bool(g.isfinite().all()) for g in gradients)
    winners=list((tmp_path/'runtime_winners'/'mxfp8_balanced_experts').rglob('*.json'))
    if winners:
        assert len(winners)==1
        assert bool(fn(x,gu,down).isfinite().all())


@cuda
@pytest.mark.parametrize('prepared',['none','separate','joint','native','cooperative','tma_outputs','expert_grid','portfolio_grid'])
def test_routed_manual_graph_changed_weights_and_bounded_live_workspace(prepared):
    from rtx.moe import DEFAULT_MOE_CONFIG,_run_forward,_run_backward
    torch.manual_seed(822)
    n,e,k=(2000,48,6) if prepared in ('expert_grid','portfolio_grid') else (70,4,2)
    x=torch.randn(n,128,device='cuda',dtype=torch.bfloat16)*.3
    gu=torch.randn(e,256,128,device='cuda',dtype=torch.bfloat16)*.02
    down=torch.randn(e,128,128,device='cuda',dtype=torch.bfloat16)*.02
    route=make_route(n,e,k,'cuda')
    dy=torch.randn(x.shape,device='cuda',dtype=torch.float32)
    config=DEFAULT_MOE_CONFIG
    if prepared!='none':
        from dataclasses import replace
        p=config.prepared
        if prepared=='native':p=p.with_scale_layout('mma128')
        elif prepared=='joint':p=replace(p,backward_preparation='joint',recompute=replace(p.recompute,glu_grad_store='exchange'))
        elif prepared in ('expert_grid','portfolio_grid'):p=replace(p,quant_grid=prepared.removesuffix('_grid'))
        config=replace(config,family='grouped_prepared',prepared=p)
        if prepared in ('cooperative','tma_outputs'):
            from rtx.autotune.moe import make_moe_adapter
            a=make_moe_adapter((70,128,4,route.expert_ids.shape[1],2,128),None,initial=config)
            config=a.update_fn(config,'prepared.linear_transport','cooperative') if prepared=='cooperative' else a.update_fn(config,'prepared.bf16_outputs','tma_all')
    def execute():
        return (_run_forward(x,gu,down,*route,config),
                *_run_backward(x,gu,down,*route,dy,config))
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):execute()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):captured=execute()
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(2):
        x.add_(.01);gu.mul_(.99);down.mul_(1.01)
        graph.replay()
        torch.cuda.synchronize()
        expected=operational_reference(x,gu,down,route,dy)
        for actual,reference in zip(captured,(expected[0],*expected[1])):
            error=(actual.float()-reference.float()).norm()/reference.float().norm().clamp_min(1e-12)
            assert float(error)<.006
    del expected
    allocations=[]
    for _ in range(8):
        outputs=execute()
        del outputs
        torch.cuda.synchronize()
        allocations.append(torch.cuda.memory_allocated())
    assert max(allocations)-min(allocations)<=65536,allocations
