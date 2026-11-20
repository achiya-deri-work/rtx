"""Compound legality and actual bounded search, with no GPU timing claims."""
from dataclasses import replace

import pytest

from rtx.autotune.effort import AutotunePolicy
from rtx.autotune.moe import make_moe_adapter,make_moe_tuner,row_store_seeds
from rtx.autotune.output_transport import row_store_candidates
from rtx.autotune.outcomes import TrialOutcome
from rtx.autotune.store import InMemoryTuningStore
from rtx.configs.mxfp8 import MXFP8GemmConfig
from rtx.kernels.mxfp8 import MXFP8Problem


@pytest.mark.parametrize('rows',[12480,13312])
def test_compound_store_crosses_geometry_buffer_barrier_without_changing_math(rows):
    g=MXFP8GemmConfig(tile_m=128,tile_n=128,atom_layout_m=4,stages=1,
                     epilogue='direct',store_vec=1)
    p=MXFP8Problem(rows,384,256)
    c=next(row_store_candidates(g,p))
    assert c.epilogue=='tma' and c.rejection(p) is None
    if rows%128:
        assert c.tile_m==64 and c.atom_layout_m==2 and c.stages==2
    else:assert c.tile_m==128 and c.stages==1
    assert replace(c,tile_m=g.tile_m,tile_n=g.tile_n,atom_layout_m=g.atom_layout_m,
                   atom_layout_n=g.atom_layout_n,stages=g.stages,epilogue='direct',store_vec=1)==g


@pytest.mark.parametrize('m,n',[(12384,386),(12480,63),(7,128)])
def test_ineligible_outputs_never_pad_or_bypass_tma_stride_contract(m,n):
    assert not list(row_store_candidates(MXFP8GemmConfig(),MXFP8Problem(m,n,256)))


@pytest.mark.parametrize('capacity',[1568,1600,1664])
def test_physical_expert_capacity_controls_stores_not_total_tokens(capacity):
    a=make_moe_adapter((12480,384,32,capacity,4,128),None)
    candidates=list(row_store_seeds(a,a.initial_config))
    assert candidates
    assert candidates[0].prepared.down.epilogue==('hybrid' if capacity%64 else 'tma')
    assert all(a.rejection(c) is None for c in candidates)
    for c in candidates:
        assert c.prepared.capacity_multiple==a.initial_config.prepared.capacity_multiple
        assert c.prepared.quant==a.initial_config.prepared.quant
        assert c.prepared.down_dw==a.initial_config.prepared.down_dw
        assert c.prepared.gate_dw==a.initial_config.prepared.gate_dw


@pytest.mark.parametrize('tma_wins',[True,False])
@pytest.mark.parametrize('phase',['train','forward'])
def test_eight_trials_actually_compare_store_basin_and_keep_direct_when_faster(tma_wins,phase):
    calls=[]
    def evaluate(c):
        calls.append(c)
        ms=(.5 if tma_wins else 2.) if c.prepared.down.epilogue=='tma' else 1.
        return TrialOutcome('ok',median_ms=ms,timings_ms=[ms]*3)
    a=make_moe_adapter((12480,384,32,1600,4,128),evaluate,phase=phase)
    p=AutotunePolicy(max_trials=8,max_seconds=120,confirmation_repeats=0,
                     pool_size=32,structural_proposals=32,candidate_mix='mixed')
    result=make_moe_tuner(a,InMemoryTuningStore(),p).tune()
    assert len(calls)==len({a.config_id(c) for c in calls})==8
    assert calls[0].prepared.down.epilogue=='direct'
    assert calls[1].prepared.down.epilogue=='tma'
    assert (result.config.prepared.down.epilogue=='tma')==tma_wins


def test_cold_onchip_shared_also_reaches_stores_within_eight(monkeypatch):
    from tests.test_mlp_moe_seed_coverage import _adapter
    from rtx.autotune.mlp import make_mlp_tuner
    a,seeds=_adapter(monkeypatch,(12480,384,256,384),prepared=False)
    calls=[]
    def evaluate(c):
        calls.append(c)
        ms=.5 if c.down_backend=='prequant' and c.down_gemm.epilogue=='tma' else 1.
        return TrialOutcome('ok',median_ms=ms,timings_ms=[ms]*3)
    a.evaluator=evaluate
    p=AutotunePolicy(max_trials=8,max_seconds=120,confirmation_repeats=0,
                    candidate_mix='mixed',pool_size=32,structural_proposals=32)
    result=make_mlp_tuner(a,InMemoryTuningStore(),p,candidates=seeds).tune()
    assert len(calls)==8 and calls[0].down_backend=='onchip'
    assert calls[1].down_backend=='prequant' and calls[1].down_gemm.epilogue=='tma'
    assert result.config.down_gemm.epilogue=='tma'


@pytest.mark.parametrize('prepared',[True,False])
def test_forward_only_shared_store_recipe_preserves_unused_backward(monkeypatch,prepared):
    from tests.test_mlp_moe_seed_coverage import _adapter
    a,seeds=_adapter(monkeypatch,(12480,384,256,384),prepared=prepared,phase='forward')
    assert seeds[0].down_backend=='prequant' and seeds[0].down_gemm.epilogue=='tma'
    assert seeds[0].gate_backward==a.initial_config.gate_backward
    assert seeds[0].down_backward==a.initial_config.down_backward
    assert a.rejection(seeds[0]) is None


def test_shared_progressive_eight_to_sixteen_reuses_exact_prefix(monkeypatch):
    from tests.test_mlp_moe_seed_coverage import _adapter
    from rtx.autotune.mlp import make_mlp_tuner
    monkeypatch.setattr('rtx.autotune.bandit.observation_strategy_cost_s',lambda observation:.001)
    a,seeds=_adapter(monkeypatch,(12480,384,256,384),prepared=False)
    calls=[]
    def evaluate(c):
        calls.append(a.config_id(c))
        return TrialOutcome('ok',median_ms=1.,timings_ms=[1.]*3)
    a.evaluator=evaluate
    policy=AutotunePolicy(max_trials=8,max_seconds=120,confirmation_repeats=0,
        search_schedule='progressive',candidate_mix='mixed',pool_size=32,structural_proposals=32)
    store=InMemoryTuningStore()
    tuner=make_mlp_tuner(a,store,policy,candidates=seeds)
    assert len(tuner.coverage_proposals)==6
    assert all(p.strategy=='structural' for p in tuner.coverage_proposals)
    tuner.tune()
    assert len(calls)==8
    make_mlp_tuner(a,store,replace(policy,max_trials=16),candidates=seeds).tune()
    continued=list(calls);calls.clear()
    make_mlp_tuner(a,InMemoryTuningStore(),replace(policy,max_trials=16),candidates=seeds).tune()
    assert len(continued)==len(calls)==len(set(calls))==16
    assert calls==continued


@pytest.mark.parametrize('reservation',[dict(exploration_trials=3),dict(baseline_trials=1),
                                      dict(initial_neighborhood_trials=2)])
def test_explicit_user_reservations_are_not_overridden(monkeypatch,reservation):
    from tests.test_mlp_moe_seed_coverage import _adapter
    from rtx.autotune.mlp import make_mlp_tuner
    from rtx.autotune.effort import make_effort_autotuner
    a,seeds=_adapter(monkeypatch,(12480,384,256,384),prepared=False)
    policy=AutotunePolicy(max_trials=8,max_seconds=120,confirmation_repeats=0,
        search_schedule='progressive',candidate_mix='mixed',pool_size=32,structural_proposals=32,
        **reservation)
    ordinary=make_effort_autotuner(a,InMemoryTuningStore(),policy,candidates=seeds)
    selected=make_mlp_tuner(a,InMemoryTuningStore(),policy,candidates=seeds)
    assert selected.coverage_proposals==ordinary.coverage_proposals


def test_native128_initial_reaches_legal_store_anchor_without_fused_preparation():
    a=make_moe_adapter((13312,384,32,1664,4,128),None)
    base=a.update_fn(a.initial_config,'prepared.scale_transport','mma128')
    assert a.rejection(base) is None
    a=make_moe_adapter((13312,384,32,1664,4,128),None,initial=base)
    policy=AutotunePolicy(max_trials=8,max_seconds=120,confirmation_repeats=0,
                         pool_size=32,structural_proposals=32)
    tuner=make_moe_tuner(a,InMemoryTuningStore(),policy)
    assert tuner.coverage_proposals[0].config.prepared.down.epilogue=='tma'
    assert a.rejection(tuner.coverage_proposals[0].config) is None


def test_onchip_forward_never_proposes_an_inactive_prepared_store():
    a=make_moe_adapter((13312,384,32,1664,4,128),None,phase='forward')
    onchip=a.update_fn(a.initial_config,'prepared.forward_engine','onchip')
    assert a.rejection(onchip) is None
    assert not list(row_store_seeds(a,onchip))
    a=make_moe_adapter((13312,384,32,1664,4,128),None,phase='forward',initial=onchip)
    policy=AutotunePolicy(max_trials=8,max_seconds=120,confirmation_repeats=0,
                         pool_size=32,structural_proposals=32)
    c=make_moe_tuner(a,InMemoryTuningStore(),policy).coverage_proposals[0].config
    assert c.prepared.forward_engine=='packed' and c.prepared.down.epilogue=='tma'
    assert a.rejection(c) is None


def test_store_prior_crosses_dw_partition_plans_without_transplanting_quant_or_registers(monkeypatch):
    from tests.test_mlp_moe_seed_coverage import _adapter
    from rtx.autotune.mlp import make_mlp_tuner
    a,seeds=_adapter(monkeypatch,(12480,384,256,384),prepared=False)
    left,right=seeds[1:3]
    restored=right
    for group in ('gate_backward','down_backward'):
        before=getattr(left,group);after=getattr(right,group)
        assert before.dw.quant_a==after.dw.quant_a and before.dw.quant_b==after.dw.quant_b
        assert before.dw.gemm==after.dw.gemm
        restored=replace(restored,**{group:replace(after,dw=before.dw)})
    assert restored==left and left!=right
    calls=[]
    def measure(c):
        calls.append(c)
        ms=.2 if a.config_id(c)==a.config_id(right) else 1.
        return TrialOutcome('ok',median_ms=ms,timings_ms=[ms]*3)
    a.evaluator=measure
    policy=AutotunePolicy(max_trials=8,max_seconds=120,confirmation_repeats=0,
        search_schedule='progressive',candidate_mix='mixed',pool_size=32,structural_proposals=32)
    result=make_mlp_tuner(a,InMemoryTuningStore(),policy,candidates=seeds).tune()
    assert len(calls)==8 and calls[3]==right and result.config==right
