"""Independent byte/order/scale checks for warp-local logical transpose."""
from dataclasses import replace
import math

import pytest
import torch

cuda=pytest.mark.skipif(not torch.cuda.is_available(),reason='assigned RTX CUDA required')


@pytest.mark.parametrize('mode',['fused','fused_all'])
def test_fused_dy_is_a_complete_reversible_tuner_choice_with_visible_storage(mode):
    from rtx.moe import DEFAULT_MOE_CONFIG,MXFP8MoEConfig
    from rtx.autotune.moe import make_moe_adapter,make_moe_tuner
    from rtx.autotune.effort import AutotunePolicy
    from rtx.autotune.store import InMemoryTuningStore
    from rtx._moe_execution import Builder
    shape=(7376,384,32,1024,4,128);a=make_moe_adapter(shape,None)
    fused=a.update_fn(a.initial_config,'prepared.dy_preparation',mode)
    assert a.rejection(fused) is None
    assert fused.prepared.input_preparation=='indexed_transpose'
    assert MXFP8MoEConfig.from_dict(fused.to_dict())==fused
    old=a.update_fn(fused,'prepared.dy_preparation','materialized')
    before=Builder(shape,old,'backward').build();after=Builder(shape,fused,'backward').build()
    assert len(after.steps)==len(before.steps)-(2 if mode=='fused_all' else 1)
    assert any(s[1]==('compile_routed_backward_quant' if mode=='fused_all' else 'compile_routed_dy_quant') for s in after.steps)
    assert not any(s[1]=='compile_movement' and s[2][6]=='slot_dy' for s in after.steps)
    size=lambda plan:sum(math.prod(b.shape)*b.dtype.itemsize for b in plan.buffers)
    assert size(before)-size(after)==shape[2]*shape[3]*shape[1]*2
    for name,value in (('transposed_tile_rows',64),('transposed_tile_k',64),('transposed_smem_padding',4)):
        changed=a.update_fn(fused,'prepared.quant.'+name,value)
        assert a.rejection(changed) is None and a.config_id(changed)!=a.config_id(fused)
        assert a.config_id(a.update_fn(old,'prepared.quant.'+name,value))==a.config_id(old)
    for name,value in (('prepared.input_preparation','quantize_once'),('prepared.backward_preparation','separate'),
                       ('prepared.scale_transport','mma128')):
        changed=a.update_fn(fused,name,value)
        assert changed.prepared.dy_preparation=='materialized' and a.rejection(changed) is None
    forward=make_moe_adapter(shape,None,phase='forward',initial=fused)
    assert forward.initial_config.prepared.dy_preparation=='materialized'
    tuner=make_moe_tuner(a,InMemoryTuningStore(),AutotunePolicy(max_trials=16,candidate_mix='mixed'))
    assert any(c.prepared.dy_preparation=='fused' for c in tuner.strategies['structural'].pool[:8])
    with pytest.raises(ValueError):replace(DEFAULT_MOE_CONFIG.prepared,dy_preparation='invalid')


@cuda
@pytest.mark.parametrize('n,d,h',[(519,384,192),(2003,768,384)])
@pytest.mark.parametrize('mode',['fused','fused_all'])
def test_fused_dy_full_expert_gradients_match_independent_fp64(n,d,h,mode):
    from rtx.autotune.moe import make_moe_adapter
    from rtx.moe import _run_forward,_run_backward
    from rtx._moe_reference import operational_reference
    from tests.test_mxfp8_moe import make_route
    torch.manual_seed(20381+n)
    e,k=8,2;route=make_route(n,e,k,'cuda');c=route[0].shape[1]
    x=torch.randn(n,d,device='cuda',dtype=torch.bfloat16)*.3
    gu=torch.randn(e,2*h,d,device='cuda',dtype=torch.bfloat16)*.02
    down=torch.randn(e,d,h,device='cuda',dtype=torch.bfloat16)*.02
    dy=torch.randn_like(x,dtype=torch.float32);operands=(x,gu,down,*route)
    a=make_moe_adapter((n,d,e,c,k,h),None)
    config=a.update_fn(a.initial_config,'prepared.dy_preparation',mode)
    expected=operational_reference(*operands,dy)
    actual=(_run_forward(*operands,config),*_run_backward(*operands,dy,config))
    for index,(value,reference) in enumerate(zip(actual,expected)):
        assert bool(value.isfinite().all()),index
        delta=value.float()-reference.float();nrms=delta.norm()/reference.float().norm().clamp_min(1e-12)
        assert nrms<.006 or delta.abs().max()<2e-7,(index,float(nrms))


def test_tuning_recovers_tail_safe_initial_without_changing_explicit_config():
    from rtx.moe import DEFAULT_MOE_CONFIG
    from rtx.autotune.moe import legal_tuning_initial,rejection
    shape=(29504,768,48,3808,6,256)
    base=replace(DEFAULT_MOE_CONFIG,family='grouped_prepared')
    requested=replace(base,prepared=replace(base.prepared,
        down=replace(base.prepared.down,tile_m=128,epilogue='tma',store_vec=4)))
    assert 'full M/N tiles' in rejection(requested,shape)
    actual,reason=legal_tuning_initial(shape,requested)
    assert reason and rejection(actual,shape) is None
    assert actual.prepared.down.epilogue=='direct'
    assert requested.prepared.down.epilogue=='tma'
    same,reason=legal_tuning_initial(shape,actual)
    assert same==actual and reason is None


def test_small_budget_defers_both_fusion_families_but_explicit_hybrid_is_honored():
    from types import SimpleNamespace
    from rtx.autotune.core import Proposal
    from rtx.autotune.moe import make_moe_adapter,MoEFamilySearch
    a=make_moe_adapter((512,384,24,128,6,128),None)
    packed=a.initial_config
    hybrid=a.update_fn(packed,'prepared.forward_engine','onchip')
    old=a.update_fn(packed,'family','grouped_onchip')
    class Fixed:
        name='fixed'
        def propose(self,*args):return [Proposal(c,'fixed') for c in (hybrid,old,packed)]
    search=MoEFamilySearch(Fixed())
    assert [p.config for p in search.propose(a,SimpleNamespace(current=[None]*7),None,3)]==[packed]
    assert [p.config for p in search.propose(a,SimpleNamespace(current=[None]*8),None,3)]==[hybrid,old,packed]
    explicit=make_moe_adapter((512,384,24,128,6,128),None,initial=hybrid)
    assert len(search.propose(explicit,SimpleNamespace(current=[]),None,3))==3


def test_warp_transpose_config_roundtrip_and_active_tuner_coordinates():
    from rtx.moe import DEFAULT_MOE_CONFIG,MXFP8MoEConfig
    from rtx.autotune.moe import make_moe_adapter,structural_seeds
    c=replace(DEFAULT_MOE_CONFIG,prepared=replace(DEFAULT_MOE_CONFIG.prepared,
        transpose_mode='warp',transpose_warp_tokens=4,transpose_warp_raster='tokens'))
    assert MXFP8MoEConfig.from_dict(c.to_dict())==c
    a=make_moe_adapter((7376,768,48,1024,6,256),None,initial=c)
    assert set(a.axes['prepared.transpose_mode'])=={'direct','staged','warp'}
    assert set(a.axes['prepared.transpose_warp_tokens'])=={2,4,8}
    assert a.rejection(a.initial_config) is None
    assert a.config_id(c)!=a.config_id(replace(c,prepared=replace(c.prepared,transpose_warp_tokens=8)))
    plain=replace(c,prepared=replace(c.prepared,transpose_mode='staged'))
    assert a.config_id(plain)==a.config_id(replace(plain,prepared=replace(plain.prepared,transpose_warp_tokens=2)))
    assert any(s.prepared.transpose_mode=='warp' for s in list(structural_seeds((7376,768,48,1024,6,256)))[:8])
    for kwargs in (dict(transpose_warp_tokens=1),dict(transpose_warp_tokens=True),dict(transpose_warp_raster='bad')):
        with pytest.raises(ValueError):replace(c.prepared,**kwargs)


@pytest.mark.parametrize('lanes',[2,4,8])
def test_byte_permutation_model_forms_contiguous_token_words(lanes):
    def prmt(a,b,selector):
        data=a.to_bytes(4,'little')+b.to_bytes(4,'little')
        return int.from_bytes(bytes(data[(selector>>(4*j))&7] for j in range(4)),'little')
    output=[None]*32
    for word in range(8//lanes):
        values=[int.from_bytes(bytes(t+(word*4+j)*lanes for j in range(4)),'little') for t in range(lanes)]
        first=[prmt(values[t],values[t^1],0x5140) if not t&1 else prmt(values[t^1],values[t],0x7362)
               for t in range(lanes)]
        if lanes>=4:
            values=[prmt(first[t],first[t^2],0x5410) if not t&2 else prmt(first[t^2],first[t],0x7632)
                    for t in range(lanes)]
        else:values=first
        for t,value in enumerate(values):
            position=4*t if lanes==2 else (((t&1)<<1)|((t>>1)&1))*lanes+(t//4)*4
            start=word*4*lanes+position
            output[start:start+4]=value.to_bytes(4,'little')
    assert output==list(range(32))


def test_expert_split_reduction_is_legal_normalized_and_compiler_visible():
    from rtx.moe import DEFAULT_MOE_CONFIG,MXFP8MoEConfig
    from rtx.autotune.moe import make_moe_adapter
    from rtx._moe_execution import Builder
    shape=(32768,384,32,4096,4,128)
    adapter=make_moe_adapter(shape,None)
    c=adapter.update_fn(adapter.initial_config,'prepared.gate_dw_splits',4)
    c=adapter.update_fn(c,'prepared.down_dw_splits',2)
    assert c.prepared.gate_dw.epilogue=='direct' and c.prepared.gate_dw.tiles_per_cta==1
    assert c.prepared.gate_dw.persistent_waves==0
    assert adapter.rejection(c) is None
    assert MXFP8MoEConfig.from_dict(c.to_dict())==c
    plan=Builder(shape,c,'backward').build()
    reductions=[s for s in plan.steps if s[1]=='compile_expert_dw_reduce']
    assert len(reductions)==2
    for _,_,(_,m,n,split,_,_),arguments in reductions:
        workspace,output=(plan.buffers[i] for i in arguments)
        assert workspace.dtype==torch.float32 and workspace.shape==(shape[2]*split*m,n)
        assert output.dtype==torch.bfloat16
    plain=DEFAULT_MOE_CONFIG
    assert adapter.config_id(plain)==adapter.config_id(replace(plain,prepared=replace(plain.prepared,dw_reduce_vector=8)))
    for kwargs in (dict(gate_dw_splits=True),dict(down_dw_splits=3),dict(dw_reduce_vector=3)):
        with pytest.raises(ValueError):replace(c.prepared,**kwargs)
    short=make_moe_adapter((128,128,8,32,2,128),None)
    assert short.rejection(short.update_fn(short.initial_config,'prepared.gate_dw_splits',2)) is not None


@cuda
@pytest.mark.parametrize('split,threads,vector',[(2,64,1),(4,128,4),(8,256,8)])
def test_grouped_split_reduction_has_no_cross_expert_alias(split,threads,vector):
    from rtx.kernels.mxfp8_moe_prepared import compile_expert_dw_reduce
    e,m,n=3,35,49
    torch.manual_seed(20043)
    workspace=torch.randn(e,split,m,n,device='cuda')
    actual=torch.empty(e*m,n,device='cuda',dtype=torch.bfloat16)
    compile_expert_dw_reduce(e,m,n,split,threads,vector)(workspace.view(e*split*m,n),actual)
    # Independent FP64 sum also verifies FP32 reduction rather than BF16
    # partial accumulation, and ragged output tiles exercise masks.
    torch.testing.assert_close(actual.view(e,m,n),workspace.double().sum(1).bfloat16(),rtol=.008,atol=.0001)


@cuda
def test_grouped_split_gemms_against_independent_fp64():
    from rtx.moe import DEFAULT_MOE_CONFIG,_run_forward,_run_backward
    from rtx.autotune.moe import make_moe_adapter
    from rtx._moe_reference import operational_reference
    torch.manual_seed(20047)
    n,d,e,k,h=2048,128,8,2,128;c=n*k//e
    ids=torch.arange(e*c,device='cuda').view(e,c)%n
    slots=torch.arange(e*c,device='cuda');inv=torch.stack((slots[:n],slots[n:]),1)
    weights=torch.full((e,c),.5,device='cuda')
    x=torch.randn(n,d,device='cuda',dtype=torch.bfloat16)*.3
    gu=torch.randn(e,2*h,d,device='cuda',dtype=torch.bfloat16)*.02
    down=torch.randn(e,d,h,device='cuda',dtype=torch.bfloat16)*.02
    dy=torch.randn_like(x,dtype=torch.float32)
    operands=(x,gu,down,ids,weights,inv)
    expected=operational_reference(*operands,dy)
    a=make_moe_adapter((n,d,e,c,k,h),None)
    config=a.update_fn(a.initial_config,'prepared.down_dw_splits',2)
    config=a.update_fn(config,'prepared.gate_dw_splits',4)
    config=replace(config,prepared=replace(config.prepared,transpose_mode='warp'))
    actual=(_run_forward(*operands,config),*_run_backward(*operands,dy,config))
    for index,(value,ref) in enumerate(zip(actual,expected)):
        delta=value.float()-ref.float();nrms=delta.norm()/ref.float().norm().clamp_min(1e-12)
        assert bool(value.isfinite().all()) and (nrms<.006 or delta.abs().max()<2e-7),(index,float(nrms))


def test_fused_route_epilogue_is_a_complete_tuner_choice():
    from rtx.autotune.moe import make_moe_adapter
    from rtx._moe_execution import Builder
    shape=(519,128,8,130,2,128)
    adapter=make_moe_adapter(shape,None)
    c=adapter.update_fn(adapter.initial_config,'prepared.route_gradient_epilogue','fused')
    assert adapter.rejection(c) is None
    assert c.prepared.backward_preparation=='joint'
    plan=Builder(shape,c,'backward').build()
    assert any(s[1]=='compile_expert_route_gradient' for s in plan.steps)
    assert not any(s[1]=='compile_movement' and s[2][6]=='route_grad' for s in plan.steps)
    broken=replace(c,prepared=replace(c.prepared,backward_preparation='separate'))
    assert adapter.rejection(broken) is not None


def test_low_budget_opening_covers_memory_traffic_families_and_index_map_is_visible():
    from rtx.autotune.moe import make_moe_adapter,make_moe_tuner
    from rtx.autotune.effort import AutotunePolicy
    from rtx.autotune.store import InMemoryTuningStore
    from rtx._moe_execution import Builder
    a=make_moe_adapter((7376,384,24,2048,6,128),None)
    t=make_moe_tuner(a,InMemoryTuningStore(),AutotunePolicy(max_trials=8,max_seconds=60))
    opening=[p.config for p in t.coverage_proposals]
    assert any(c.prepared.input_preparation=='indexed_transpose' for c in opening)
    indexed=a.update_fn(a.initial_config,'prepared.input_preparation','indexed_transpose')
    plan=Builder((7376,384,24,2048,6,128),indexed,'backward').build()
    assert any(s[1]=='compile_indexed_expert_transpose' for s in plan.steps)
    assert any(b.shape==(24*2048,1) and b.dtype==torch.int32 for b in plan.buffers)
    materialized=a.update_fn(indexed,'prepared.input_preparation','quantize_once')
    ordinary=Builder((7376,384,24,2048,6,128),materialized,'backward').build()
    assert sum(b.shape==(24*2048,384) and b.dtype==torch.bfloat16 for b in ordinary.buffers)==1+sum(
        b.shape==(24*2048,384) and b.dtype==torch.bfloat16 for b in plan.buffers)


@pytest.mark.parametrize('capacity',[16,1024,3808])
def test_alignment_aliases_do_not_consume_trials_or_change_execution(capacity):
    from rtx.moe import DEFAULT_MOE_CONFIG
    from rtx.autotune.moe import make_moe_adapter
    from rtx._moe_execution import Builder
    shape=(capacity*8,384,32,capacity,4,128)
    a=make_moe_adapter(shape,None)
    groups={}
    for multiple in (1,32,64,128,256):
        c=replace(DEFAULT_MOE_CONFIG,prepared=replace(DEFAULT_MOE_CONFIG.prepared,capacity_multiple=multiple))
        physical=c.prepared.physical_capacity(capacity)
        groups.setdefault(physical,[]).append(c)
    for configs in groups.values():
        assert len({a.config_id(c) for c in configs})==1
        plans=[Builder(shape,c,'backward').build() for c in configs]
        assert all(p.steps==plans[0].steps and p.buffers==plans[0].buffers for p in plans)


def test_short_reduction_transport_seeds_do_not_couple_long_weight_gradients():
    from rtx.moe import DEFAULT_MOE_CONFIG
    from rtx.autotune.moe import make_moe_adapter,structural_seeds
    shape=(7376,384,32,1024,4,128)
    a=make_moe_adapter(shape,None)
    seeds=list(structural_seeds(shape))
    for stage in ('down','down_dx','gate_dx'):
        index=('gate_up','down','recompute','down_dx','down_dw','gate_dx','gate_dw').index(stage)
        matches=[c for c in seeds if c.family=='grouped_prepared' and c.prepared.transports[index]=='cooperative'
            and c.prepared.down_dw==DEFAULT_MOE_CONFIG.prepared.down_dw
            and c.prepared.gate_dw==DEFAULT_MOE_CONFIG.prepared.gate_dw
            and sum(v=='cooperative' for v in c.prepared.transports)==1]
        assert matches,stage
        assert all(a.rejection(c) is None for c in matches)


def test_onchip_forward_has_prepared_backward_and_all_controls_stay_active():
    from rtx.autotune.moe import make_moe_adapter,structural_seeds
    from rtx._moe_execution import Builder
    shape=(519,384,32,96,4,128)
    a=make_moe_adapter(shape,None)
    c=a.update_fn(a.initial_config,'prepared.forward_engine','onchip')
    assert a.rejection(c) is None
    forward=Builder(shape,c,'forward').build();backward=Builder(shape,c,'backward').build()
    assert any(s[1]=='compile_onchip_experts' for s in forward.steps)
    assert not any(s[1]=='compile_expert_packed_gemm' for s in forward.steps)
    assert any(s[1]=='compile_expert_packed_gemm' for s in backward.steps)
    assert a.config_id(c)!=a.config_id(a.update_fn(c,'onchip_pipeline','warp_specialized'))
    assert any(s.prepared.forward_engine=='onchip' and s.gate_up.tile_m==64 for s in structural_seeds(shape))
    other=make_moe_adapter((519,768,48,96,6,256),None)
    assert other.rejection(other.update_fn(other.initial_config,'prepared.forward_engine','onchip')) is not None


@cuda
@pytest.mark.parametrize('rows,pipeline',[(32,'serial'),(64,'warp_specialized')])
def test_hybrid_forward_and_prepared_backward_match_fp64(rows,pipeline):
    from rtx.autotune.moe import make_moe_adapter
    from rtx.moe import _run_forward,_run_backward
    from rtx._moe_reference import operational_reference
    from tests.test_mxfp8_moe import make_route
    torch.manual_seed(20071)
    n,d,e,k,h=519,384,8,2,128
    route=make_route(n,e,k,'cuda');c=route[0].shape[1]
    x=torch.randn(n,d,device='cuda',dtype=torch.bfloat16)*.3
    gu=torch.randn(e,2*h,d,device='cuda',dtype=torch.bfloat16)*.02
    down=torch.randn(e,d,h,device='cuda',dtype=torch.bfloat16)*.02
    dy=torch.randn_like(x,dtype=torch.float32);operands=(x,gu,down,*route)
    a=make_moe_adapter((n,d,e,c,k,h),None)
    config=a.update_fn(a.initial_config,'prepared.forward_engine','onchip')
    config=a.update_fn(config,'gate_up.tile_m',rows)
    config=a.update_fn(config,'onchip_pipeline',pipeline)
    config=a.update_fn(config,'prepared.input_preparation','indexed_transpose')
    expected=operational_reference(*operands,dy)
    actual=(_run_forward(*operands,config),*_run_backward(*operands,dy,config))
    for index,(value,ref) in enumerate(zip(actual,expected)):
        delta=value.float()-ref.float();nrms=delta.norm()/ref.float().norm().clamp_min(1e-12)
        assert bool(value.isfinite().all()) and (nrms<.006 or delta.abs().max()<2e-7),(index,float(nrms))


@cuda
@pytest.mark.parametrize('input_preparation',['gather_bf16','indexed_transpose'])
@pytest.mark.parametrize('m,n,am,tiles,locality',[(64,64,2,1,'raster'),(128,128,4,1,'raster'),
                                               (64,64,2,4,'serpentine_b')])
def test_fused_route_gradient_epilogue_against_quantized_fp64(m,n,am,tiles,locality,input_preparation):
    from rtx.moe import _run_forward,_run_backward
    from rtx.autotune.moe import make_moe_adapter
    from rtx._moe_reference import operational_reference
    from tests.test_mxfp8_moe import make_route
    torch.manual_seed(20053)
    tokens,d,e,k,h=519,128,8,2,128
    route=make_route(tokens,e,k,'cuda');c=route[0].shape[1]
    x=torch.randn(tokens,d,device='cuda',dtype=torch.bfloat16)*.3
    gu=torch.randn(e,2*h,d,device='cuda',dtype=torch.bfloat16)*.02
    down=torch.randn(e,d,h,device='cuda',dtype=torch.bfloat16)*.02
    dy=torch.randn_like(x,dtype=torch.float32)
    operands=(x,gu,down,*route)
    expected=operational_reference(*operands,dy)
    a=make_moe_adapter((tokens,d,e,c,k,h),None)
    config=a.update_fn(a.initial_config,'prepared.route_gradient_epilogue','fused')
    config=a.update_fn(config,'prepared.input_preparation',input_preparation)
    config=replace(config,prepared=replace(config.prepared,down=replace(config.prepared.down,
        tile_m=m,tile_n=n,atom_layout_m=am,tiles_per_cta=tiles,tile_locality=locality)))
    assert a.rejection(config) is None
    actual=(_run_forward(*operands,config),*_run_backward(*operands,dy,config))
    for index,(value,ref) in enumerate(zip(actual,expected)):
        delta=value.float()-ref.float();nrms=delta.norm()/ref.float().norm().clamp_min(1e-12)
        assert bool(value.isfinite().all()) and (nrms<.006 or delta.abs().max()<2e-7),(index,float(nrms))


@cuda
@pytest.mark.parametrize('lanes',[2,4,8])
def test_indexed_transpose_matches_materialized_gather_exactly(lanes):
    from rtx.kernels.mxfp8_moe_warp_quant import compile_indexed_expert_transpose
    from rtx.kernels.mxfp8_moe_prepared import compile_expert_quant
    from rtx.configs.moe import DEFAULT_PREPARED_MOE_CONFIG
    torch.manual_seed(20059)
    n,d,e,cs=257,35,3,96
    raw=torch.randn(n*d+1,device='cuda',dtype=torch.bfloat16)*.3
    x=raw[1:].view(n,d)
    mapping=torch.randint(n,(e*cs,1),device='cuda',dtype=torch.int32)
    mapping[::7]=-1;mapping[32:64]=-1
    # The production gather writes positive zero for dummy slots. Multiplying
    # a negative value by zero would instead create a different -0 byte.
    gathered=torch.where(mapping.ge(0),x[mapping.flatten().clamp_min(0).long()],0)
    config=DEFAULT_PREPARED_MOE_CONFIG.quant
    q=torch.empty(e*d,cs,device='cuda',dtype=torch.float8_e4m3fn)
    sf=torch.empty(e*d,cs//32,device='cuda',dtype=torch.uint8)
    compile_indexed_expert_transpose(n,d,e,cs,config,lanes,'tokens','expert')(x,mapping,q,sf)
    expected_q=torch.empty_like(q);expected_sf=torch.empty_like(sf)
    compile_expert_quant(e,((d,cs,'transpose',False),),config,'warp','expert',lanes,'tokens',
        _byte_scales=True)(gathered,(gathered,),(expected_q,),(expected_sf,))
    torch.testing.assert_close(q.view(torch.uint8),expected_q.view(torch.uint8),rtol=0,atol=0)
    torch.testing.assert_close(sf,expected_sf,rtol=0,atol=0)


@cuda
@pytest.mark.parametrize('lanes,raster,math',[(2,'features','fp32'),(4,'tokens','fp32'),
                                         (8,'features','fp32'),(8,'tokens','bf16x2')])
@pytest.mark.parametrize('rows,k,layout',[(35,49,'row_major'),(128,256,'mma128')])
def test_warp_transpose_packed_values_and_scales_are_exact(lanes,raster,math,rows,k,layout):
    from rtx.kernels.mxfp8_moe_prepared import compile_expert_quant
    from rtx.configs.moe import DEFAULT_PREPARED_MOE_CONFIG
    torch.manual_seed(20031)
    e=3;raw=torch.randn(e*rows*k+1,device='cuda',dtype=torch.bfloat16)*.3
    source=raw[1:].view(e*k,rows)
    config=replace(DEFAULT_PREPARED_MOE_CONFIG.quant,scale_layout=layout,quant_math=math)
    width=(k+31)//32*32;specs=((rows,k,'transpose',False),)
    def run(mode):
        q=torch.empty(e*rows,width,device='cuda',dtype=torch.float8_e4m3fn)
        sf=torch.empty(e*rows,width//32,device='cuda',dtype=torch.float8_e8m0fnu)
        # Baseline takes an aligned copy; the new kernel must additionally
        # support the deliberately 2-byte-offset source.
        value=source if mode=='warp' else source.clone()
        compile_expert_quant(e,specs,config,mode,'operand',lanes,raster)(value,(value,),(q,),(sf,))
        return q.view(torch.uint8),sf.view(torch.uint8)
    expected=run('direct');actual=run('warp')
    for a,b in zip(actual,expected):torch.testing.assert_close(a,b,rtol=0,atol=0)
    # Do not accept a coherent bug shared by two layouts/implementations.
    if layout=='row_major':
        from rtx._moe_reference import quantize_decode
        decoded=actual[0].view(torch.float8_e4m3fn).float()*actual[1].view(torch.float8_e8m0fnu).float().repeat_interleave(32,-1)
        oracle=quantize_decode(source.view(e,k,rows).transpose(1,2)).reshape(e*rows,k)
        torch.testing.assert_close(decoded[:,:k],oracle,rtol=0,atol=0)


@cuda
def test_warp_transpose_full_prepared_forward_backward_against_fp64():
    from rtx.moe import DEFAULT_MOE_CONFIG,_run_forward,_run_backward
    from rtx._moe_reference import operational_reference
    torch.manual_seed(20037)
    n,d,e,k,h=128,128,8,2,128;c=n*k//e
    ids=torch.arange(e*c,device='cuda').view(e,c)%n
    slots=torch.arange(e*c,device='cuda');inv=torch.stack((slots[:n],slots[n:]),1)
    weights=torch.full((e,c),.5,device='cuda')
    x=torch.randn(n,d,device='cuda',dtype=torch.bfloat16)*.3
    gu=torch.randn(e,2*h,d,device='cuda',dtype=torch.bfloat16)*.02
    down=torch.randn(e,d,h,device='cuda',dtype=torch.bfloat16)*.02
    dy=torch.randn_like(x,dtype=torch.float32)
    operands=(x,gu,down,ids,weights,inv)
    expected=operational_reference(*operands,dy)
    config=replace(DEFAULT_MOE_CONFIG,prepared=replace(DEFAULT_MOE_CONFIG.prepared,
        transpose_mode='warp',input_preparation='quantize_once'))
    actual=(_run_forward(*operands,config),*_run_backward(*operands,dy,config))
    for index,(a,b) in enumerate(zip(actual,expected)):
        delta=a.float()-b.float();nrms=delta.norm()/b.float().norm().clamp_min(1e-12)
        assert bool(a.isfinite().all()) and (nrms<.006 or delta.abs().max()<2e-7),(index,float(nrms))
