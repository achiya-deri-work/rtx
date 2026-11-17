"""Quantize-once batch boundaries and independent expert recipe checks."""
import pytest
import torch

cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='SM120 CUDA required')


@cuda
@pytest.mark.parametrize('transpose_mode',['direct','staged'])
@pytest.mark.parametrize('n,d,e,k,h',[(16,64,4,2,64),(13,64,5,2,64),(130,384,2,1,128),(13,35,5,2,49)])
def test_prepared_experts_match_independent_all_gradients(n,d,e,k,h,transpose_mode):
    from rtx.moe import DEFAULT_MOE_CONFIG
    from rtx._moe_prepared import forward,backward
    from rtx._moe_reference import operational_reference
    from tests.test_mxfp8_moe import make_route
    from dataclasses import replace
    torch.manual_seed(831)
    x=torch.randn(n,d,device='cuda',dtype=torch.bfloat16)*.3
    gu=torch.randn(e,2*h,d,device='cuda',dtype=torch.bfloat16)*.05
    down=torch.randn(e,d,h,device='cuda',dtype=torch.bfloat16)*.05
    route=make_route(n,e,k,'cuda')
    dy=torch.randn(x.shape,device='cuda',dtype=torch.float32)
    expected=operational_reference(x,gu,down,*route,dy)
    config=replace(DEFAULT_MOE_CONFIG,gather_mode='warp')
    actual=(forward(x,gu,down,*route,config),
            *backward(x,gu,down,*route,dy,config,transpose_mode=transpose_mode))
    for name,a,b in zip(('out','dx','dgu','ddown','dweights'),actual,expected):
        assert bool(a.isfinite().all()),name
        nrms=float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
        assert nrms<.006 or float((a.float()-b.float()).abs().max())<2e-7,(name,nrms)


def test_prepared_adapter_is_independent_and_canonical():
    from dataclasses import replace
    from rtx.moe import DEFAULT_MOE_CONFIG,MXFP8MoEConfig
    from rtx.autotune.moe import make_moe_adapter,structural_seeds
    config=replace(DEFAULT_MOE_CONFIG,family='grouped_prepared',gather_mode='warp')
    assert MXFP8MoEConfig.from_dict(config.to_dict())==config
    a=make_moe_adapter((2000,768,48,250,6,256),None,initial=config)
    assert a.rejection(a.initial_config) is None
    assert 'prepared.down_dw.geometry' in a.axes and 'prepared.recompute.stages' in a.axes
    ignored=replace(config,gate_up=replace(config.gate_up,maxrregcount=96),onchip_down_stages=4)
    assert a.config_id(ignored)==a.config_id(config)
    changed=a.update_fn(config,'prepared.down_dw.stages',3)
    assert a.config_id(changed)!=a.config_id(config)
    assert changed.prepared.down_dx==config.prepared.down_dx
    seeds=[c for c in structural_seeds((2000,768,48,250,6,256)) if c.family=='grouped_prepared']
    assert len(seeds)>=4 and all(a.rejection(c) is None for c in seeds)


def test_padding_waste_gets_a_complete_compact_family_in_opening_seeds():
    from dataclasses import replace
    from rtx.moe import DEFAULT_MOE_CONFIG
    from rtx.autotune.moe import structural_seeds,rejection
    initial=replace(DEFAULT_MOE_CONFIG,prepared=replace(DEFAULT_MOE_CONFIG.prepared,capacity_multiple=128))
    shape=(128,768,48,16,6,256)
    seeds=list(structural_seeds(shape,initial))
    assert seeds[0].prepared.physical_capacity(16)==128
    assert seeds[1].family=='grouped_prepared'
    assert seeds[1].prepared.physical_capacity(16)==16
    assert seeds[2].prepared.input_preparation=='quantize_once'
    assert all(rejection(c,shape) is None for c in seeds[:3])


def test_prepared_component_arm_uses_shared_budget_and_resumes():
    from dataclasses import replace
    from rtx.moe import DEFAULT_MOE_CONFIG
    from rtx.autotune.moe import make_moe_adapter,make_moe_tuner
    from rtx.autotune.effort import AutotunePolicy
    from rtx.autotune.outcomes import TrialOutcome
    from rtx.autotune.store import InMemoryTuningStore
    initial=replace(DEFAULT_MOE_CONFIG,family='grouped_prepared')
    a=make_moe_adapter((512,384,24,128,6,128),lambda c:TrialOutcome('ok',median_ms=1.),initial=initial)
    store=InMemoryTuningStore()
    policy=AutotunePolicy(max_trials=8,max_seconds=60,confirmation_repeats=0,
                          search_schedule='progressive',candidate_mix='mixed',pool_size=16,structural_proposals=32)
    t=make_moe_tuner(a,store,policy)
    first=t.tune()
    assert first.evaluated_trials==8
    assert all(r.config.family!='grouped_onchip' for r in store.observations)
    second=make_moe_tuner(a,store,replace(policy,max_trials=16)).tune()
    assert len(store.observations)==16 and second.evaluated_trials==8
    assert any(r.strategy=='prepared_components' for r in store.observations)


@cuda
@pytest.mark.parametrize('m,n,k',[(64,128,128),(250,768,512),(17,35,64),(130,64,256)])
def test_cooperative_packed_gemm_against_independent_recipe(m,n,k):
    from dataclasses import replace
    from rtx._moe_prepared import prepare,packed_gemm
    from rtx.configs.moe import DEFAULT_PREPARED_MOE_CONFIG
    from rtx._moe_reference import quantize_decode
    torch.manual_seed(834)
    a=torch.randn(3,m,k,device='cuda',dtype=torch.bfloat16)*.3
    b=torch.randn(3,n,k,device='cuda',dtype=torch.bfloat16)*.05
    qa,qb=prepare(((a,'row',False),(b,'row',False)))
    out=torch.empty((3,m,n),device='cuda',dtype=torch.bfloat16)
    config=replace(DEFAULT_PREPARED_MOE_CONFIG.down,stages=1,scale_role='producer')
    packed_gemm(qa,qb,out,config,transport='cooperative')
    expected=torch.bmm(quantize_decode(a).double(),quantize_decode(b).double().transpose(1,2)).bfloat16()
    nrms=float((out.float()-expected.float()).norm()/expected.float().norm())
    assert nrms<.003,nrms


@cuda
@pytest.mark.parametrize('vec,stages',[(1,1),(2,1),(4,1),(4,2)])
def test_expert_tma_epilogue_does_not_cross_experts(vec,stages):
    from dataclasses import replace
    from rtx._moe_prepared import prepare,packed_gemm
    from rtx.configs.moe import DEFAULT_PREPARED_MOE_CONFIG
    from rtx._moe_reference import quantize_decode
    torch.manual_seed(835)
    a=torch.randn(3,128,256,device='cuda',dtype=torch.bfloat16)*.3
    b=torch.randn(3,384,256,device='cuda',dtype=torch.bfloat16)*.05
    qa,qb=prepare(((a,'row',False),(b,'row',False)))
    out=torch.empty((3,128,384),device='cuda',dtype=torch.bfloat16)
    config=replace(DEFAULT_PREPARED_MOE_CONFIG.down,epilogue='tma',store_vec=vec,epilogue_stages=stages)
    packed_gemm(qa,qb,out,config)
    expected=torch.bmm(quantize_decode(a).double(),quantize_decode(b).double().transpose(1,2)).bfloat16()
    nrms=float((out.float()-expected.float()).norm()/expected.float().norm())
    assert nrms<.003,nrms


@cuda
@pytest.mark.parametrize('grid_mode',['expert','portfolio'])
def test_batch_quant_grid_preserves_bytes_and_scales(grid_mode):
    from rtx._moe_prepared import prepare
    torch.manual_seed(836)
    x=torch.randn(48,257,384,device='cuda',dtype=torch.bfloat16)*.3
    gu=torch.randn(48,256,384,device='cuda',dtype=torch.bfloat16)*.05
    down=torch.randn(48,384,128,device='cuda',dtype=torch.bfloat16)*.05
    operands=((x,'row',False),(gu,'row',True),(down,'row',False),
              (x,'transpose',False),(down,'transpose',False))
    expected=prepare(operands,grid_mode='operand')
    actual=prepare(operands,grid_mode=grid_mode)
    for a,b in zip(actual,expected):
        for data,reference in zip(a,b):
            assert torch.equal(data.view(torch.uint8),reference.view(torch.uint8))


@cuda
@pytest.mark.parametrize('n,e',[(128,4),(130,2)])
def test_native_scale_transport_and_capacity_padding(n,e):
    from dataclasses import replace
    from rtx.moe import DEFAULT_MOE_CONFIG,_run_forward,_run_backward
    from rtx._moe_reference import operational_reference
    from tests.test_mxfp8_moe import make_route
    torch.manual_seed(832)
    d,h,k=384,128,1
    x=torch.randn(n,d,device='cuda',dtype=torch.bfloat16)*.3
    gu=torch.randn(e,2*h,d,device='cuda',dtype=torch.bfloat16)*.05
    down=torch.randn(e,d,h,device='cuda',dtype=torch.bfloat16)*.05
    route=make_route(n,e,k,'cuda');dy=torch.randn(x.shape,device='cuda',dtype=torch.float32)
    config=replace(DEFAULT_MOE_CONFIG,family='grouped_prepared',gather_mode='warp',
                   prepared=DEFAULT_MOE_CONFIG.prepared.with_scale_layout('mma128'))
    actual=(_run_forward(x,gu,down,*route,config),*_run_backward(x,gu,down,*route,dy,config))
    expected=operational_reference(x,gu,down,*route,dy)
    for name,a,b in zip(('out','dx','dgu','ddown','dweights'),actual,expected):
        assert bool(a.isfinite().all()),name
        nrms=float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
        assert nrms<.006 or float((a.float()-b.float()).abs().max())<2e-7,(name,nrms)


@cuda
@pytest.mark.parametrize('n,d,e,k,h',[(16,64,4,2,64),(130,384,2,1,128),(256,768,4,2,256),(13,35,5,2,64)])
@pytest.mark.parametrize('vec,math',[(1,'division'),(4,'shared_reciprocal')])
def test_joint_backward_matches_independent_recipe(n,d,e,k,h,vec,math):
    from dataclasses import replace
    from rtx.moe import DEFAULT_MOE_CONFIG,_run_forward,_run_backward
    from rtx._moe_reference import operational_reference
    from tests.test_mxfp8_moe import make_route
    torch.manual_seed(833)
    x=torch.randn(n,d,device='cuda',dtype=torch.bfloat16)*.3
    gu=torch.randn(e,2*h,d,device='cuda',dtype=torch.bfloat16)*.05
    down=torch.randn(e,d,h,device='cuda',dtype=torch.bfloat16)*.05
    route=make_route(n,e,k,'cuda');dy=torch.randn(x.shape,device='cuda',dtype=torch.float32)
    p=DEFAULT_MOE_CONFIG.prepared
    p=replace(p,backward_preparation='joint',gate_up=replace(p.gate_up,glu_quant_vec=vec,glu_silu_math=math),
              recompute=replace(p.recompute,glu_grad_store='exchange',glu_quant_vec=vec,glu_silu_math=math))
    config=replace(DEFAULT_MOE_CONFIG,family='grouped_prepared',gather_mode='warp',prepared=p)
    actual=(_run_forward(x,gu,down,*route,config),*_run_backward(x,gu,down,*route,dy,config))
    expected=operational_reference(x,gu,down,*route,dy)
    for name,a,b in zip(('out','dx','dgu','ddown','dweights'),actual,expected):
        assert bool(a.isfinite().all()),name
        nrms=float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12))
        assert nrms<.006 or float((a.float()-b.float()).abs().max())<2e-7,(name,nrms)
