"""CPU selection and compiler-visible allocation contracts for new adjoints."""
from dataclasses import replace
import json
import math

import pytest
import torch

from rtx._moe_execution import Builder
from rtx.autotune.moe import make_moe_adapter
from rtx.configs.moe import MXFP8PreparedMoEConfig
from rtx.configs.moe_backward import persistent_dw_rejection,route_gradient_scratch_bytes
from rtx.kernels.mxfp8 import MXFP8Problem


SHAPE=(12320,768,48,1600,6,256)


def test_new_backward_choices_roundtrip_and_forward_inactive_fields_canonicalize():
    adapter=make_moe_adapter(SHAPE,None)
    c=adapter.update_fn(adapter.initial_config,'prepared.route_gradient_loader','warp_vec2')
    c=adapter.update_fn(c,'prepared.route_gradient_gemm.geometry',(64,128,128,2,2))
    c=adapter.update_fn(c,'prepared.gate_dw_engine','persistent')
    c=adapter.update_fn(c,'prepared.gate_dw_reuse_tiles',3)
    assert adapter.rejection(c) is None
    assert adapter.deserialize(json.loads(json.dumps(c.to_dict())))==c
    assert c.prepared.route_gradient_gemm.tile_n==128
    assert c.prepared.down.tile_n==adapter.initial_config.prepared.down.tile_n
    fwd=make_moe_adapter(SHAPE,None,phase='forward',initial=c)
    p=fwd.initial_config.prepared
    assert p.gate_dw_engine==p.down_dw_engine=='packed'
    assert p.route_gradient_loader=='accumulator' and p.route_gradient_gemm is None


def test_indexed_consumer_removes_global_qxt_pair_instead_of_hiding_it():
    adapter=make_moe_adapter(SHAPE,None)
    baseline=adapter.update_fn(adapter.initial_config,'prepared.input_preparation','indexed_transpose')
    fused=adapter.update_fn(baseline,'prepared.gate_dw_engine','indexed')
    assert adapter.rejection(fused) is None
    builders=[Builder(SHAPE,c,'backward').build() for c in (baseline,fused)]
    symbols=[[s for _,s,_,_ in b.steps] for b in builders]
    assert 'compile_indexed_expert_transpose' in symbols[0]
    assert 'compile_indexed_expert_transpose' not in symbols[1]
    assert symbols[1].count('compile_indexed_gate_dw')==1
    sizes=[sum(math.prod(x.shape)*x.dtype.itemsize for x in b.buffers) for b in builders]
    _,d,e,cs,_,_=SHAPE
    assert sizes[0]-sizes[1]==e*d*cs*33//32
    assert [builders[0].buffers[i].shape for i in builders[0].outputs]==[
        builders[1].buffers[i].shape for i in builders[1].outputs]


@pytest.mark.parametrize('stage',['gate_dw','down_dw'])
def test_persistent_stage_owns_one_launch_and_no_split_workspace(stage):
    adapter=make_moe_adapter(SHAPE,None)
    c=adapter.update_fn(adapter.initial_config,f'prepared.{stage}_engine','persistent')
    c=adapter.update_fn(c,f'prepared.{stage}_reuse_tiles',3)
    assert adapter.rejection(c) is None
    builder=Builder(SHAPE,c,'backward').build()
    steps=[s for s in builder.steps if s[1]=='compile_persistent_dw']
    assert len(steps)==1 and len(steps[0][3])==5
    assert not any(b.dtype==torch.float32 and len(b.shape)==3 for b in builder.buffers)
    assert adapter.rejection(replace(c,prepared=replace(c.prepared,**{stage+'_splits':2}))) is not None


def test_loader_scratch_and_singleton_guard_are_cpu_only():
    p=MXFP8PreparedMoEConfig();g=replace(p.down,scale_load_vec=1)
    assert route_gradient_scratch_bytes(g,'warp_vec2')==route_gradient_scratch_bytes(g,'accumulator')
    assert route_gradient_scratch_bytes(g,'shared_vec4')-route_gradient_scratch_bytes(g,'accumulator')==4*g.tile_m*(g.tile_n+1)
    assert 'nonsingleton' in persistent_dw_rejection(MXFP8Problem(70,1,32),g,'m',2)


def test_inactive_fields_do_not_burn_configuration_identity():
    a=make_moe_adapter(SHAPE,None);base=a.initial_config
    noisy=replace(base,prepared=replace(base.prepared,gate_dw_reuse_axis='m',gate_dw_reuse_tiles=3,
        indexed_dw_token_lanes=2,indexed_dw_producer_warps=4,route_gradient_loader='shared_vec4'))
    assert a.config_id(base)==a.config_id(noisy)


def test_eight_trial_opening_reaches_combined_ownership_on_unseen_long_pack():
    from rtx.autotune.effort import AutotunePolicy
    from rtx.autotune.moe import make_moe_tuner
    from rtx.autotune.outcomes import TrialOutcome
    from rtx.autotune.store import InMemoryTuningStore
    adapter=make_moe_adapter((15728,640,40,1984,5,192),None)
    calls=[]
    def evaluate(c):
        calls.append(c)
        return TrialOutcome('ok',median_ms=1.,timings_ms=[1.,1.,1.])
    adapter.evaluator=evaluate
    policy=AutotunePolicy(max_trials=8,max_seconds=120,pool_size=64,structural_proposals=64,
                         candidate_mix='mixed',confirmation_repeats=0)
    make_moe_tuner(adapter,InMemoryTuningStore(),policy).tune()
    assert len(calls)==8 and len({adapter.config_id(c) for c in calls})==8
    assert calls[0]==adapter.initial_config
    selected=[c.prepared for c in calls if c.prepared.route_gradient_loader=='warp_vec2'
              and c.prepared.gate_dw_engine==c.prepared.down_dw_engine=='persistent']
    assert selected
    assert selected[0].gate_dw_reuse_axis=='m' and selected[0].down_dw_reuse_axis=='n'
    assert not any(c.prepared.gate_dw_engine=='indexed' for c in calls[:7])


@pytest.mark.parametrize('stage',['gate_dw','down_dw'])
def test_explicit_alternate_policies_reject_before_allocating_scratch(stage):
    adapter=make_moe_adapter(SHAPE,None)
    c=adapter.update_fn(adapter.initial_config,f'prepared.{stage}_engine','persistent')
    for fields in ({stage+'_splits':2},{'active_prefix':True}):
        b=Builder(SHAPE,replace(c,prepared=replace(c.prepared,**fields)),'backward')
        with pytest.raises(ValueError,match='Illegal backward ownership plan'):b.build()
        assert b.buffers==[] and b.steps==[]


@pytest.mark.parametrize('fields',[dict(input_preparation='quantize_once'),
    dict(dy_preparation='fused_all'),dict(active_prefix=True),dict(backward_preparation='separate')])
def test_explicit_invalid_indexed_inputs_never_reach_the_kernel(fields):
    a=make_moe_adapter(SHAPE,None)
    c=a.update_fn(a.initial_config,'prepared.gate_dw_engine','indexed')
    b=Builder(SHAPE,replace(c,prepared=replace(c.prepared,**fields)),'backward')
    with pytest.raises(ValueError,match='Illegal backward ownership plan'):b.build()
    assert b.buffers==[] and b.steps==[]


def test_cooperative_indexed_inactive_stage_and_role_settings_deduplicate():
    a=make_moe_adapter(SHAPE,None)
    c=a.update_fn(a.initial_config,'prepared.gate_dw_engine','indexed')
    p=c.prepared;transports=list(p.transports);transports[-1]='cooperative'
    alias=replace(c,prepared=replace(p,transports=tuple(transports),gate_dw=replace(p.gate_dw,
        stages=3,scale_role='producer',maxrregcount=128,mma_schedule='preload')))
    assert a.config_id(c)==a.config_id(alias)


def test_independent_route_schedule_checks_active_prefix_and_initial_register_roles():
    a=make_moe_adapter(SHAPE,None)
    c=a.update_fn(a.initial_config,'prepared.route_gradient_loader','warp_vec2')
    g=c.prepared.down
    bad=replace(c,prepared=replace(c.prepared,active_prefix=True,
        route_gradient_gemm=replace(g,tiles_per_cta=2)))
    assert 'Route-gradient active prefix' in a.rejection(bad)[1]
    for consumer,producer in ((192,48),(64,128)):
        bad=replace(c,prepared=replace(c.prepared,route_gradient_gemm=replace(g,
            maxrregcount=96,consumer_registers=consumer,producer_registers=producer)))
        assert a.rejection(bad) is not None
        b=Builder(SHAPE,bad,'backward')
        with pytest.raises(ValueError,match='Illegal backward ownership plan'):b.build()
        assert b.buffers==[]


def test_iteration2_single_launch_weight_preparation_and_inactive_fields():
    a=make_moe_adapter(SHAPE,None)
    c=a.update_fn(a.initial_config,'prepared.input_preparation','indexed_transpose')
    c=a.update_fn(c,'prepared.weight_preparation','dual_orientation')
    assert a.rejection(c) is None
    b=Builder(SHAPE,c,'backward').build()
    steps=[s for s in b.steps if s[1]=='compile_backward_weight_quant']
    assert len(steps)==1 and len(steps[0][3])==13
    assert a.deserialize(json.loads(json.dumps(c.to_dict())))==c
    before=Builder(SHAPE,replace(c,prepared=replace(c.prepared,weight_preparation='separate')),'forward').build()
    after=Builder(SHAPE,c,'forward').build()
    assert before.steps==after.steps and before.buffers==after.buffers
    p=replace(a.initial_config.prepared,weight_tile_cols=128,weight_load_values=8,weight_raster='rows',
              joint_recompute_gradient_warps=2)
    assert a.config_id(replace(a.initial_config,prepared=p))==a.config_id(a.initial_config)


@pytest.mark.parametrize('field,value',[
    ('forward_engine','onchip'),
])
def test_fragment_forward_explicit_contracts_fail_before_scratch(field,value):
    a=make_moe_adapter(SHAPE,None)
    c=a.update_fn(a.initial_config,'prepared.forward_glu_engine','fragment_packed')
    b=Builder(SHAPE,replace(c,prepared=replace(c.prepared,**{field:value})),'forward')
    with pytest.raises(ValueError,match='Illegal forward GLU plan'):b.build()
    assert b.buffers==[]


def test_fragment_forward_active_prefix_is_preserved_by_engine_switch():
    a=make_moe_adapter(SHAPE,None)
    c=a.update_fn(a.initial_config,'prepared.active_prefix',True)
    c=a.update_fn(c,'prepared.forward_glu_engine','fragment_packed')
    assert c.prepared.active_prefix and a.rejection(c) is None
    builder=Builder(SHAPE,c,'forward').build()
    step=next(step for step in builder.steps if step[1]=='compile_expert_packed_gemm' and step[2][3]=='glu')
    assert step[2][-3]>0 and step[2][-1]=='fragment_packed'
    assert len(step[3])==7 and step[3][-1]==builder.active_capacity


def test_fragment_physical_k_and_split_team_raw_guards():
    from rtx.configs.moe_backward import fragment_glu_rejection
    from rtx.configs.moe import joint_recompute_rejection
    a=make_moe_adapter(SHAPE,None)
    c=a.update_fn(a.initial_config,'prepared.forward_glu_engine','fragment_packed')
    assert 'K32' in fragment_glu_rejection(MXFP8Problem(64,128,129),c.prepared.gate_up,'fragment_packed')
    c=a.update_fn(a.initial_config,'prepared.joint_recompute_engine','split_warp')
    assert a.rejection(c) is None
    assert joint_recompute_rejection(MXFP8Problem(1600,512,768),c.prepared.recompute,
        'split_warp',c.prepared.recompute.num_threads//32) is not None
    bad=replace(c,prepared=replace(c.prepared,active_prefix=True))
    b=Builder(SHAPE,bad,'backward')
    with pytest.raises(ValueError):b.build()
    assert b.buffers==[]


def test_fragment_legality_uses_its_own_scratch_not_the_legacy_product_plane():
    from rtx.configs.moe_backward import fragment_glu_rejection
    a=make_moe_adapter(SHAPE,None)
    g=replace(a.initial_config.prepared.gate_up,tile_m=128,tile_n=128,atom_layout_m=8,
              atom_layout_n=2,stages=2,consumer_registers=112,glu_epilogue='registers')
    c=replace(a.initial_config,prepared=replace(a.initial_config.prepared,
        forward_glu_engine='fragment_packed',gate_up=g))
    assert fragment_glu_rejection(MXFP8Problem(1600,512,768),g,'fragment_packed') is None
    assert a.rejection(c) is None
    assert fragment_glu_rejection(MXFP8Problem(1600,512,768),replace(g,consumer_registers=192),
                                  'fragment_packed') is not None


def test_paired_weight_proposal_checks_actual_padded_global_x_extent():
    a=make_moe_adapter(SHAPE,None)
    c=a.update_fn(a.initial_config,'prepared.weight_preparation','dual_orientation')
    from rtx.autotune.moe import rejection
    assert 'signed index' in rejection(c,(30000000,65,48,625000,1,32))
