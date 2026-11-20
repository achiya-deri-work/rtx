"""Allocation/ownership contracts for bounded preparation and compact forward."""
from dataclasses import replace

import pytest

from rtx._moe_execution import Builder
from rtx.autotune.moe import make_moe_adapter
from rtx.moe import DEFAULT_MOE_CONFIG, MXFP8MoEConfig


SHAPE=(8192,384,32,2016,4,128)


def schedules():
    a=make_moe_adapter(SHAPE,None,tags={'batch_contract':{
        'layout':'independent_varlen','sequences':32,'segment_alignment':32}})
    c=a.initial_config
    for name,value in (('input_preparation','indexed_transpose'),('dy_preparation','fused'),
                       ('active_prefix',True),('joint_recompute_engine','dual_operand'),
                       ('forward_glu_engine','fragment_packed'),('down_dw_engine','persistent'),
                       ('gate_dw_engine','persistent')):
        c=a.update_fn(c,'prepared.'+name,value)
    assert a.rejection(c) is None
    return a,c


def test_old_configuration_roundtrip_and_independent_bounded_coordinates():
    old=DEFAULT_MOE_CONFIG.to_dict()
    for name in ('bounded_gather','bounded_dy','bounded_xt','forward_row_layout','compact_pool_count'):
        old['prepared'].pop(name)
    assert MXFP8MoEConfig.from_dict(old)==DEFAULT_MOE_CONFIG
    a,c=schedules()
    for name in ('bounded_gather','bounded_dy','bounded_xt'):
        p=a.update_fn(c,'prepared.'+name,True)
        assert a.rejection(p) is None
        assert sum(getattr(p.prepared,k) for k in ('bounded_gather','bounded_dy','bounded_xt'))==1
        assert a.deserialize(a.serialize(p))==p


@pytest.mark.parametrize('phase',['forward','backward'])
def test_bounded_producers_reuse_only_existing_prefix_and_storage(phase):
    a,c=schedules()
    p=replace(c.prepared,bounded_gather=True,bounded_dy=True,bounded_xt=True)
    old=Builder(SHAPE,c,phase).build()
    new=Builder(SHAPE,replace(c,prepared=p),phase).build()
    assert old.buffers==new.buffers and old.outputs==new.outputs
    symbols=('compile_packed_input_gather','compile_routed_dy_quant','compile_indexed_expert_transpose')
    for step in new.steps:
        if step[1] in symbols:
            assert step[3][-1]==new.active_capacity
    gather=next(s for s in new.steps if s[1]=='compile_packed_input_gather')
    assert gather[2][-4:]==((p.down.tile_m,p.gate_up.tile_m,(),False) if phase=='forward' else
                           (new.recompute_alignment(),p.recompute.tile_m,(p.down_dw.tile_k,p.gate_dw.tile_k),False))
    if phase=='backward':
        dy=next(s for s in new.steps if s[1]=='compile_routed_dy_quant')
        assert dy[2][-3:]==(new.recompute_alignment(),p.down_dw.tile_k,p.down_dx.tile_m)


def test_compact_forward_is_private_and_backward_plan_is_identical():
    a,c=schedules()
    compact=a.update_fn(c,'prepared.forward_row_layout','compact')
    assert compact.prepared.compact_pool_count==32 and a.rejection(compact) is None
    old=Builder(SHAPE,c,'forward').build()
    new=Builder(SHAPE,compact,'forward').build()
    names=[s[1] for s in new.steps]
    assert names.count('compile_compact_route')==names.count('compile_compact_gather')==1
    assert names.count('compile_compact_combine')==1 and 'compile_active_prefix' not in names
    assert 'compile_packed_input_gather' not in names
    assert len(new.outputs)==1
    gemms=[s for s in new.steps if s[1]=='compile_expert_packed_gemm']
    from rtx._moe_compact_layout import compact_forward_capacity
    import math
    cc=compact_forward_capacity(SHAPE[0],32,SHAPE[2],SHAPE[4],c.prepared.physical_capacity(SHAPE[3]),
                                math.lcm(32,c.prepared.gate_up.tile_m,c.prepared.down.tile_m))
    assert len(gemms)==2 and all(s[2][1].m==cc for s in gemms)
    assert old.buffers[old.outputs[0]].shape==new.buffers[new.outputs[0]].shape
    old_backward=Builder(SHAPE,c,'backward').build()
    new_backward=Builder(SHAPE,compact,'backward').build()
    assert old_backward.buffers==new_backward.buffers
    assert old_backward.steps==new_backward.steps and old_backward.outputs==new_backward.outputs


def test_invalid_explicit_plans_reject_before_allocations():
    a,c=schedules()
    invalid=(replace(c.prepared,active_prefix=False,bounded_gather=True),
             replace(c.prepared,bounded_dy=True,dy_preparation='fused_all'),
             replace(c.prepared,forward_row_layout='compact',compact_pool_count=0))
    for p in invalid:
        b=Builder(SHAPE,replace(c,prepared=p),'forward' if p.forward_row_layout=='compact' else 'backward')
        with pytest.raises(ValueError,match='Illegal'):b.build()
        assert b.buffers==b.steps==[]


def test_pool_bound_is_private_shape_metadata_not_a_search_coordinate():
    from rtx.moe import _packed_expert_spec,_regular_expert_spec,_selection_spec,_parse_spec
    a,c=schedules()
    c=a.update_fn(c,'prepared.forward_row_layout','compact')
    spec=_selection_spec(c,'off',None,None)
    assert _parse_spec(spec)['initial']['prepared']['compact_pool_count']==0
    assert _parse_spec(_packed_expert_spec(spec,17))['initial']['prepared']['compact_pool_count']==17
    assert _parse_spec(_regular_expert_spec(spec))['initial']['prepared']['compact_pool_count']==1
    assert 'prepared.compact_pool_count' not in a.axes
    opaque=make_moe_adapter(SHAPE,None)
    assert 'prepared.forward_row_layout' not in opaque.axes
    assert opaque.rejection(opaque.update_fn(c,'prepared.forward_row_layout','compact')) is not None


def test_actual_eight_trials_measure_additive_layout_and_producer_recipe():
    from rtx.autotune.effort import AutotunePolicy
    from rtx.autotune.moe import make_moe_tuner
    from rtx.autotune.outcomes import TrialOutcome
    from rtx.autotune.store import InMemoryTuningStore
    a,c=schedules()
    c=replace(c,prepared=replace(c.prepared,active_prefix=False))
    observed=[]
    def evaluate(candidate):
        observed.append(candidate)
        p=candidate.prepared
        # A synthetic oracle proves admissions/composition, not performance.
        score=1.-.1*p.active_prefix-.03*(candidate.dx_reduce_engine=='token')
        score-=.08*(p.bounded_gather and p.bounded_dy and p.bounded_xt)
        score-=.08*(p.forward_row_layout=='compact')
        return TrialOutcome('ok',median_ms=score,timings_ms=[score]*3)
    adapter=make_moe_adapter(SHAPE,evaluate,initial=c,tags={'batch_contract':{
        'layout':'independent_varlen','sequences':32,'segment_alignment':32}})
    policy=AutotunePolicy(max_trials=8,max_seconds=120,search_schedule='progressive',
        candidate_mix='mixed',confirmation_repeats=0,pool_size=32,structural_proposals=32)
    selected=make_moe_tuner(adapter,InMemoryTuningStore(),policy).tune().config
    assert len(observed)==len({adapter.config_id(c) for c in observed})==8
    assert all(adapter.rejection(c) is None for c in observed)
    assert observed[3].prepared.bounded_gather and observed[3].prepared.bounded_dy
    assert observed[4].prepared.forward_row_layout=='compact'
    assert selected.prepared.forward_row_layout=='compact'
    assert selected.prepared.bounded_gather and selected.prepared.bounded_dy and selected.prepared.bounded_xt


@pytest.mark.parametrize('phase',['forward','backward'])
def test_independent_read_write_audit_covers_new_composed_plans(phase):
    # Pytest imports this non-package test directory by basename. Importing
    # it again as tests.test_moe_plan re-registers its mock custom operator.
    from test_moe_plan import test_every_scratch_read_has_a_preceding_producer
    a,c=schedules()
    for name in ('bounded_dy','bounded_xt','bounded_gather'):
        c=a.update_fn(c,'prepared.'+name,True)
    for compact in (False,True):
        cfg=a.update_fn(c,'prepared.forward_row_layout','compact') if compact else c
        test_every_scratch_read_has_a_preceding_producer('iter4',SHAPE,cfg,phase)
