"""Exercise real bounded orchestration, not merely membership in its pool."""
from dataclasses import replace
from types import SimpleNamespace
import random

import pytest

from rtx.autotune.effort import AutotunePolicy
from rtx.autotune.moe import make_moe_adapter,make_moe_tuner
from rtx.autotune.outcomes import TrialOutcome
from rtx.autotune.store import InMemoryTuningStore


def _adapter(calls):
    adapter=make_moe_adapter((2000,384,32,352,4,128),None,tags={
        'batch_contract':{'layout':'independent_varlen','sequences':4,'segment_alignment':32}})
    def evaluate(config):
        calls.append(config)
        return TrialOutcome('ok',median_ms=1.,timings_ms=[1.,1.,1.])
    adapter.evaluator=evaluate
    return adapter


@pytest.mark.parametrize('mix',['legacy','mixed'])
def test_eight_real_trials_measure_new_families_and_keep_legacy(mix):
    calls=[];adapter=_adapter(calls)
    policy=AutotunePolicy(max_trials=8,max_seconds=120,candidate_mix=mix,
        confirmation_repeats=0,pool_size=64,structural_proposals=64)
    make_moe_tuner(adapter,InMemoryTuningStore(),policy).tune()
    assert len(calls)==8 and calls[0]==adapter.initial_config
    assert any(c.prepared.active_prefix for c in calls)
    assert any(c.prepared.joint_recompute_engine!='legacy' for c in calls)
    assert any(c.prepared.input_preparation=='indexed_transpose' for c in calls)
    assert any(c.prepared.input_preparation=='quantize_once' for c in calls)
    assert any(c.prepared.down!=adapter.initial_config.prepared.down for c in calls)
    assert calls[1].prepared.down!=adapter.initial_config.prepared.down
    assert calls[1].prepared.joint_recompute_engine=='dual_operand'
    assert calls[1].prepared.dy_preparation=='fused'
    assert calls[2].prepared.dy_preparation=='fused'
    assert calls[2].prepared.down.epilogue=='direct'
    assert calls[1].prepared.down.epilogue in ('tma','hybrid')
    assert any(c.prepared.dy_preparation=='materialized' for c in calls)
    assert all(adapter.rejection(c) is None for c in calls)
    assert len({adapter.config_id(c) for c in calls})==8


def test_incompatible_prior_keeps_normal_opening_coverage(monkeypatch):
    import rtx.autotune.moe as module
    shared=module.make_effort_autotuner
    # Model loading/compatibility itself is tested by the shared engine.
    # Simulate its documented fallback: no frozen ranker was installed.
    def without_compatible_head(adapter,store,policy,**kwargs):
        return shared(adapter,store,replace(policy,prior_artifact=None),**kwargs)
    monkeypatch.setattr(module,'make_effort_autotuner',without_compatible_head)
    calls=[];adapter=_adapter(calls)
    policy=AutotunePolicy(max_trials=8,max_seconds=120,prior_artifact='different-family.json',
        confirmation_repeats=0,pool_size=64,structural_proposals=64)
    make_moe_tuner(adapter,InMemoryTuningStore(),policy).tune()
    assert len(calls)==8
    assert any(c.prepared.joint_recompute_engine!='legacy' for c in calls)
    assert any(c.prepared.active_prefix for c in calls)


def test_per_search_identity_cache_is_bounded_without_mutable_serialization_aliases():
    calls=[];adapter=_adapter(calls);config=adapter.initial_config
    expected=adapter.config_id(config)
    document=adapter.serialize(config)
    document['gather_mode']='not-a-real-mode'
    assert adapter.config_id(config)==expected
    assert adapter.serialize(config)['gather_mode']==config.gather_mode
    assert adapter.config_id_fn.cache_info().maxsize==4096
    assert adapter.config_id_fn.cache_info().hits>0


@pytest.mark.parametrize('shape',[(8192,384,32,2016,4,128),(4096,512,40,1152,5,192)])
def test_padded_eight_trial_opening_preserves_fragment_and_persistent_wins(shape):
    initial=make_moe_adapter(shape,None)
    config=initial.update_fn(initial.initial_config,'prepared.backward_preparation','joint')
    for name,value in (('forward_glu_engine','fragment_packed'),
                       ('down_dw_engine','persistent'),('gate_dw_engine','persistent')):
        config=initial.update_fn(config,'prepared.'+name,value)
    assert initial.rejection(config) is None
    calls=[]
    def evaluate(candidate):
        calls.append(candidate)
        return TrialOutcome('ok',median_ms=1.,timings_ms=[1.]*3)
    adapter=make_moe_adapter(shape,evaluate,initial=config,tags={
        'batch_contract':{'layout':'independent_varlen','sequences':16,'segment_alignment':32}})
    policy=AutotunePolicy(max_trials=8,max_seconds=120,search_schedule='progressive',
        candidate_mix='mixed',confirmation_repeats=0,pool_size=32,structural_proposals=32)
    make_moe_tuner(adapter,InMemoryTuningStore(),policy).tune()
    assert len(calls)==8 and calls[0]==adapter.initial_config
    assert calls[1].prepared.active_prefix
    assert calls[1].prepared.forward_glu_engine=='fragment_packed'
    assert calls[1].prepared.down_dw_engine==calls[1].prepared.gate_dw_engine=='persistent'
    assert calls[2].prepared==calls[1].prepared
    assert calls[2].combine_reduce_engine==calls[2].dx_reduce_engine=='token'
    assert any(c.dx_reduce_engine=='token' and c.combine_reduce_engine=='token' for c in calls)
    assert any(c.prepared.input_preparation=='quantize_once' for c in calls)
    assert len({adapter.config_id(c) for c in calls})==8


def test_geometry_seed_grows_real_work_and_preserves_register_and_numeric_policy():
    from rtx.autotune.moe import geometry_growth_seeds
    adapter=_adapter([])
    initial=adapter.update_fn(adapter.initial_config,'prepared.input_preparation','indexed_transpose')
    candidate=next(geometry_growth_seeds(adapter,initial))
    a=initial.prepared.down;b=candidate.prepared.down
    assert (b.tile_m,b.tile_n)!=(a.tile_m,a.tile_n)
    assert b.stages==1 and b.mma_schedule=='preload'
    for field in ('maxrregcount','producer_registers','consumer_registers','scale_layout'):
        assert getattr(a,field)==getattr(b,field)
    assert candidate.prepared.input_preparation==initial.prepared.input_preparation
    assert candidate.prepared.quant==initial.prepared.quant
    assert adapter.rejection(candidate) is None


def test_geometry_component_turn_rotates_to_gate_dx_after_down():
    from rtx.autotune.moe import PreparedComponentSearch
    adapter=_adapter([])
    base=adapter.initial_config;identifier=adapter.config_id(base)
    observation=SimpleNamespace(config=base,config_id=identifier,score=1.,
        outcome=TrialOutcome('ok',median_ms=1.),strategy='initial')
    history=SimpleNamespace(successful=[observation],seen_ids={identifier},
        observations=[SimpleNamespace(coordinate='prepared.row_gemm.geometry_recipe',strategy='prepared_components')])
    proposals=PreparedComponentSearch().propose(adapter,history,random.Random(42),4)
    geometry=next(p.config for p in proposals if p.coordinate=='prepared.row_gemm.geometry_recipe')
    assert geometry.prepared.gate_dx!=base.prepared.gate_dx
    assert geometry.prepared.down==base.prepared.down


def test_progressive_eight_to_sixteen_preserves_prefix_without_repeating_trials(monkeypatch):
    # The oracle below already fixes every candidate's GPU latency. Fix the
    # other measured input to UCB as well: CPU proposal/evaluation jitter can
    # otherwise break tied rewards differently across two synthetic runs.
    monkeypatch.setattr('rtx.autotune.bandit.observation_strategy_cost_s',lambda observation:.001)
    first=[];adapter=_adapter(first);store=InMemoryTuningStore()
    policy=AutotunePolicy(max_trials=8,max_seconds=120,search_schedule='progressive',
        candidate_mix='mixed',confirmation_repeats=0,pool_size=64,structural_proposals=64)
    make_moe_tuner(adapter,store,policy).tune()
    assert len(first)==8
    resumed=[];continued=_adapter(resumed)
    make_moe_tuner(continued,store,replace(policy,max_trials=16)).tune()
    fresh=[];full=_adapter(fresh)
    make_moe_tuner(full,InMemoryTuningStore(),replace(policy,max_trials=16)).tune()
    assert len(resumed)==8 and len(fresh)==16
    identifiers=lambda values:[adapter.config_id(c) for c in values]
    assert identifiers(first)==identifiers(fresh[:8])
    assert identifiers(first+resumed)==identifiers(fresh)


def test_legacy_refinement_does_not_invent_a_strict_progressive_checkpoint():
    calls=[];adapter=_adapter(calls);store=InMemoryTuningStore()
    policy=AutotunePolicy(max_trials=8,max_seconds=120,confirmation_repeats=0,
                         pool_size=16,structural_proposals=16)
    first=make_moe_tuner(adapter,store,policy)
    assert first.checkpoint_key is None
    first.tune()
    resumed=make_moe_tuner(adapter,store,replace(policy,max_trials=16))
    assert resumed.checkpoint_key is None
    assert resumed.tune().evaluated_trials==8
    assert len({adapter.config_id(c) for c in calls})==len(calls)==16


@pytest.mark.parametrize('mode',['quantize_once','indexed_transpose'])
def test_component_search_rebases_compact_recipe_without_losing_traffic_wins(mode):
    from rtx.autotune.moe import PreparedComponentSearch
    calls=[];adapter=_adapter(calls)
    incumbent=adapter.update_fn(adapter.initial_config,'prepared.input_preparation',mode)
    incumbent=adapter.update_fn(incumbent,'prepared.backward_preparation','joint')
    incumbent=adapter.update_fn(incumbent,'prepared.route_gradient_epilogue','fused')
    assert adapter.rejection(incumbent) is None
    identifier=adapter.config_id(incumbent)
    observation=SimpleNamespace(config=incumbent,config_id=identifier,score=1.,
        outcome=TrialOutcome('ok',median_ms=1.),strategy='initial')
    history=SimpleNamespace(successful=[observation],seen_ids={identifier},observations=[observation])
    proposals=PreparedComponentSearch().propose(adapter,history,random.Random(42),4)
    assert len(proposals)==4
    compact=proposals[0]
    assert compact.coordinate=='prepared.recompute.recipe'
    assert compact.parent_config_id==identifier
    assert compact.config.prepared.joint_recompute_engine=='dual_operand'
    assert compact.config.prepared.recompute.tile_n==64
    assert compact.config.prepared.recompute.stages==1
    assert compact.config.prepared.input_preparation==mode
    assert compact.config.prepared.route_gradient_epilogue=='fused'
    assert not compact.config.prepared.active_prefix
    assert adapter.rejection(compact.config) is None
    assert len({adapter.config_id(p.config) for p in proposals})==4
    # Do not propose the same recipe again or starve normal coordinates.
    history.seen_ids.add(adapter.config_id(compact.config))
    again=PreparedComponentSearch().propose(adapter,history,random.Random(42),4)
    assert adapter.config_id(compact.config) not in {adapter.config_id(p.config) for p in again}
    assert any(p.coordinate!='prepared.recompute.recipe' for p in again)


@pytest.mark.parametrize('budget',[8,16])
def test_bounded_search_combines_independently_measured_traffic_and_recompute_wins(budget):
    calls=[];adapter=_adapter(calls)
    def measure(config):
        calls.append(config)
        # Independent synthetic costs: this tests orchestration, not a GPU
        # heuristic or a claim about which family is fastest on real hardware.
        ms=1.-.1*(config.prepared.input_preparation=='indexed_transpose')
        ms-=.2*(config.prepared.joint_recompute_engine=='dual_operand')
        return TrialOutcome('ok',median_ms=ms,timings_ms=[ms]*3)
    adapter.evaluator=measure
    policy=AutotunePolicy(max_trials=budget,max_seconds=120,candidate_mix='mixed',
        confirmation_repeats=0,pool_size=64,structural_proposals=64)
    winner=make_moe_tuner(adapter,InMemoryTuningStore(),policy).tune().config
    assert winner.prepared.input_preparation=='indexed_transpose'
    assert winner.prepared.joint_recompute_engine=='dual_operand'
    assert len({adapter.config_id(c) for c in calls})==len(calls)==budget
