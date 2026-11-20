"""Metadata-only bounded-search coverage for already-prepared shared MLPs.

No CUDA execution or numerical/performance claim is made by these tests.
The real adapter's legality and proposal transforms run on meta input shapes.
"""
from dataclasses import replace

import pytest
import torch


def _adapter(monkeypatch,shape,*,prepared=True,fused=False,phase='train'):
    import rtx.autotune.mlp as tuning
    from rtx.mlp import DEFAULT_MLP_CONFIG
    from rtx.fp8_bwd import _default_bwd_config
    from rtx.kernels.mxfp8 import MXFP8Problem
    from tests.test_mlp_dual_quant import policy
    m,k,h,o=shape
    if prepared:
        base=policy(m,k,h,o)
        if not fused:
            base=replace(base,gradient_quantization='separate',hidden_quantization='separate',
                         joint_output_storage='packed',recompute=replace(base.recompute,
                             gemm=replace(base.recompute.gemm,glu_grad_store='direct')))
    else:
        base=replace(DEFAULT_MLP_CONFIG,
                     gate_backward=_default_bwd_config(MXFP8Problem(m,2*h,k)),
                     down_backward=_default_bwd_config(MXFP8Problem(m,o,h)))
    monkeypatch.setattr(tuning,'_validate',lambda *args:None)
    monkeypatch.setattr(tuning,'_mlp_hardware',lambda _:dict(multiprocessor_count=170))
    monkeypatch.setattr(torch.cuda,'Stream',lambda **kwargs:object())
    values=[torch.empty(*size,device='meta',dtype=torch.bfloat16)
            for size in ((m,k),(2*h,k),(o,h))]
    adapter=tuning.make_mlp_adapter(*values,initial=base,phase=phase,
        _reference_cache={'values':(torch.empty(m,o,device='meta',dtype=torch.bfloat16),None)})
    return adapter,tuning.mlp_structural_seeds(adapter,'mxfp8',phase)


@pytest.mark.parametrize('shape',[(2000,768,512,768),(1536,384,256,384),(129,128,128,128)])
def test_prepared_row_fusion_is_a_protected_legal_opening_without_policy_change(monkeypatch,shape):
    adapter,seeds=_adapter(monkeypatch,shape)
    row=[c for c in seeds[:6] if c.gradient_quantization=='recompute']
    assert row,'Legal row-only fusion was lost behind the structural tail'
    c=row[0];base=adapter.initial_config
    assert adapter.rejection(c) is None
    assert c.joint_output_storage=='separate' and c.recompute.gemm.glu_grad_store=='exchange'
    assert c.hidden_quantization==base.hidden_quantization=='separate'
    assert c.gate_up.glu_silu_math==base.gate_up.glu_silu_math
    assert c.recompute.gemm.glu_hidden_decode==base.recompute.gemm.glu_hidden_decode
    for side in ('gate_backward','down_backward'):
        for leg in ('dx','dw'):
            before=getattr(getattr(base,side),leg).quant_a
            after=getattr(getattr(c,side),leg).quant_a
            assert (after.quant_math,after.quant_amax,after.scale_rounding)==(
                before.quant_math,before.quant_amax,before.scale_rounding)


def test_half_block_rows_keep_legal_row_fusion_but_not_invalid_full_fusion(monkeypatch):
    adapter,seeds=_adapter(monkeypatch,(2000,768,512,768))
    assert any(c.gradient_quantization=='recompute' for c in seeds[:6])
    assert not any(c.gradient_quantization=='recompute_both' or c.hidden_quantization=='recompute' for c in seeds)
    assert all(adapter.rejection(c) is None for c in seeds)


def test_independent_neighbor_split_is_reachable_inside_small_portfolio(monkeypatch):
    adapter,seeds=_adapter(monkeypatch,(2000,768,512,768))
    base=adapter.initial_config
    assert base.gate_backward.dw.split_reduction==base.down_backward.dw.split_reduction==4
    candidates=[c for c in seeds[:8] if c.gate_backward==base.gate_backward
                and c.down_backward.dw.split_reduction==8]
    assert candidates,'A 4/4 incumbent must reach 4/8 without a coupled split change'
    assert all(adapter.rejection(c) is None for c in candidates)
    assert len(seeds)==len(set(seeds)) and base not in seeds


def test_existing_onchip_short_family_opening_stays_4_then_8(monkeypatch):
    _,seeds=_adapter(monkeypatch,(11296,256,768,256),prepared=False)
    assert {c.gate_backward.dw.split_reduction for c in seeds[:6]} >= {4,8}
    assert all(c.gate_backend==c.down_backend==c.recompute_backend=='prequant' for c in seeds[:6])
    assert seeds[0].down_gemm.epilogue=='hybrid'


def test_existing_long_dual_quant_swizzle_opening_is_not_displaced(monkeypatch):
    _,seeds=_adapter(monkeypatch,(40000,128,256,128),fused=True)
    assert any(c.gradient_quantization=='recompute_both' for c in seeds[:8])
    assert {c.recompute.gemm.glu_scratch_swizzle for c in seeds[:8]}=={'none','xor_16b'}


@pytest.mark.parametrize('rows',[1536,29504])
def test_aligned_prepared_shared_compares_forward_and_combined_stores_first(monkeypatch,rows):
    adapter,seeds=_adapter(monkeypatch,(rows,384,256,384))
    assert seeds[0].down_gemm.epilogue=='tma'
    assert seeds[0].down_backward==adapter.initial_config.down_backward
    assert seeds[1].down_backward.dx.gemm.epilogue=='tma'
    assert seeds[1].gate_backward.dx.gemm.epilogue=='tma'
    assert all(adapter.rejection(c) is None for c in seeds[:2])
    assert seeds[2].down_gemm.epilogue=='tma'
    assert seeds[2].gate_backward.dw.split_reduction!=seeds[1].gate_backward.dw.split_reduction


@pytest.mark.parametrize('schedule',['legacy','progressive'])
def test_real_shared_eight_trial_opening_precedes_generic_neighbors(monkeypatch,schedule):
    from rtx.autotune.mlp import make_mlp_tuner
    from rtx.autotune.effort import AutotunePolicy
    from rtx.autotune.outcomes import TrialOutcome
    from rtx.autotune.store import InMemoryTuningStore
    adapter,seeds=_adapter(monkeypatch,(29504,384,256,384))
    calls=[]
    def evaluate(config):
        calls.append(config)
        return TrialOutcome('ok',median_ms=1.,timings_ms=[1.]*3)
    adapter.evaluator=evaluate
    policy=AutotunePolicy(max_trials=8,max_seconds=120,confirmation_repeats=0,
        candidate_mix='mixed',search_schedule=schedule,structural_proposals=64)
    tuner=make_mlp_tuner(adapter,InMemoryTuningStore(),policy,candidates=seeds)
    assert (tuner.checkpoint_key is None)==(schedule=='legacy')
    tuner.tune()
    assert len(calls)==8 and calls[0]==adapter.initial_config
    assert calls[1]==seeds[0] and calls[2]==seeds[1]
    assert calls[1].down_gemm.epilogue=='tma'
    assert calls[2].gate_backward.dx.gemm.epilogue=='tma'
    assert all(adapter.rejection(c) is None for c in calls)


def test_seed_change_invalidates_disk_winners_missing_transport_coverage():
    from rtx.autotune.mlp import MLP_AUTOTUNE_REVISION
    assert MLP_AUTOTUNE_REVISION==50


@pytest.mark.parametrize('schedule',['legacy','progressive'])
@pytest.mark.parametrize('shape',[(128,384,256,384),(173,576,320,576)])
def test_actual_small_shared_admissions_pair_only_forward(monkeypatch,shape,schedule):
    from rtx.autotune.mlp import make_mlp_tuner, _small_forward_opening
    from rtx.autotune.effort import AutotunePolicy
    from rtx.autotune.outcomes import TrialOutcome
    from rtx.autotune.store import InMemoryTuningStore
    adapter,seeds=_adapter(monkeypatch,shape,prepared=False)
    calls=[]
    def evaluate(config):
        calls.append(config)
        return TrialOutcome('ok',median_ms=1.,timings_ms=[1.]*3)
    adapter.evaluator=evaluate
    policy=AutotunePolicy(max_trials=8,max_seconds=120,confirmation_repeats=0,
        candidate_mix='mixed',search_schedule=schedule,structural_proposals=64)
    store=InMemoryTuningStore()
    tuner=make_mlp_tuner(adapter,store,policy,candidates=seeds)
    tuner.tune()
    assert len(calls)==8 and calls[0]==adapter.initial_config
    expected=_small_forward_opening(adapter)
    assert len(expected)>=3
    first=0 if expected[0]==adapter.initial_config else 1
    assert tuple(calls[first:first+3])==expected[:3]
    onchip,gate_only,both=expected[:3]
    assert gate_only==replace(onchip,gate_backend='prequant')
    assert both==replace(gate_only,down_backend='prequant')
    assert all(adapter.rejection(c) is None for c in calls)
    assert all(getattr(c,group).dw.split_reduction==1 for c in expected
               for group in ('gate_backward','down_backward'))
    if shape[0]==128:
        assert len(expected)==6 and tuple(calls[1:7])==expected
        for c in expected[3:]:
            assert c.gate_backward.dw.gemm.epilogue==c.down_backward.dw.gemm.epilogue=='direct'
    if schedule=='progressive':
        initial_calls=tuple(calls)
        continuing=make_mlp_tuner(adapter,store,replace(policy,max_trials=16),candidates=seeds)
        assert continuing.checkpoint_key==tuner.checkpoint_key
        continuing.tune()
        assert tuple(calls[:8])==initial_calls
        assert len(calls)==len({adapter.config_id(c) for c in calls})==16


def test_enough_reduction_panels_retains_existing_opening(monkeypatch):
    from rtx.autotune.mlp import _small_forward_opening
    adapter,seeds=_adapter(monkeypatch,(512,384,256,384),prepared=False)
    assert _small_forward_opening(adapter)==()
    assert seeds[0].gate_backend==seeds[0].down_backend==seeds[0].recompute_backend=='prequant'


@pytest.mark.parametrize('shape',[(16384,384,192,384),(12288,384,256,384),
                                  (16384,384,320,384),(40000,128,256,128)])
@pytest.mark.parametrize('schedule',['legacy','progressive'])
def test_cold_narrow_opening_compares_independent_splits_same_stores(monkeypatch,shape,schedule):
    from rtx.autotune.mlp import make_mlp_tuner, mlp_independent_dw_seeds
    from rtx.autotune.effort import AutotunePolicy
    from rtx.autotune.outcomes import TrialOutcome
    from rtx.autotune.store import InMemoryTuningStore
    adapter,seeds=_adapter(monkeypatch,shape,prepared=False)
    configs=[]
    def evaluate(c):
        configs.append(c)
        return TrialOutcome('ok',median_ms=1.,timings_ms=[1.]*3)
    adapter.evaluator=evaluate
    policy=AutotunePolicy(max_trials=8,max_seconds=120,confirmation_repeats=0,
        candidate_mix='mixed', search_schedule=schedule,structural_proposals=64)
    store=InMemoryTuningStore()
    tuner=make_mlp_tuner(adapter,store,policy,candidates=seeds)
    tuner.tune()
    assert len(configs)==len(set(configs))==8
    incumbent,narrow,strong,down,gate,direct,reduced,exploration=configs
    assert incumbent==adapter.initial_config
    groups=('gate_backward','down_backward')
    assert narrow.down_gemm.epilogue==strong.down_gemm.epilogue=='tma'
    assert tuple(getattr(narrow,g).dw.split_reduction for g in groups)==(4,4)
    assert tuple(getattr(strong,g).dw.split_reduction for g in groups)==(8,8)
    assert (down,gate)==mlp_independent_dw_seeds(adapter,strong)
    for g,c in zip(reversed(groups),(down,gate)):
        assert getattr(c,g).dw.split_reduction==2*getattr(strong,g).dw.split_reduction
        b=getattr(c,g);base=getattr(strong,g).dw
        assert replace(c,**{g:replace(b,dw=replace(b.dw,split_reduction=base.split_reduction,
                                                  reduction_tile=base.reduction_tile))})==strong
        assert adapter.rejection(c) is None
    assert direct.down_gemm.epilogue==reduced.down_gemm.epilogue=='direct'
    assert tuple(getattr(direct,g).dw.split_reduction for g in groups)==(4,4)
    assert tuple(getattr(reduced,g).dw.split_reduction for g in groups)==(8,8)
    assert exploration not in [p.config for p in tuner.coverage_proposals]
    assert adapter.context.tags['independent_dw_opening_revision']==1
    if schedule=='progressive':
        initial_calls=tuple(configs)
        resumed=make_mlp_tuner(adapter,store,replace(policy,max_trials=16),candidates=seeds)
        assert resumed.checkpoint_key==tuner.checkpoint_key
        resumed.tune()
        assert tuple(configs[:8])==initial_calls
        assert len(configs)==len({adapter.config_id(c) for c in configs})==16


@pytest.mark.parametrize('shape,phase,changed', [
    ((16384,384,192,384),'train',True),
    ((12288,384,256,384),'train',True),
    ((40000,128,256,128),'train',True),
    ((16384,512,512,512),'train',False),
    ((128,384,256,384),'train',False),
    ((12288,384,256,384),'forward',False)])
def test_shared_runtime_revision_only_changes_affected_training(monkeypatch,shape,phase,changed):
    from rtx._mlp_autotune import _mlp_runtime_winner_key, _runtime_variant
    from rtx.autotune import winners
    from rtx.kernels.mxfp8 import MXFP8Problem
    adapter,_=_adapter(monkeypatch,shape,prepared=False,phase=phase)
    monkeypatch.setattr(winners,'runtime_winner_key',lambda *a,**kw:kw)
    m,k,h,o=shape
    base=adapter.initial_config.to_dict();train=phase=='train'
    value=_mlp_runtime_winner_key(MXFP8Problem(m,2*h,k),o,'silu',train,base,False,None,device='meta')
    assert (value['variant']!=_runtime_variant(o,'silu',train,base,False))==changed
    assert ('independent_dw_opening_revision' in adapter.context.tags)==changed


def test_independent_dw_opening_rejects_excess_partial_workspace(monkeypatch):
    from rtx.autotune.mlp import mlp_independent_dw_seeds
    adapter,seeds=_adapter(monkeypatch,(16384,448,320,448),prepared=False)
    anchor=seeds[1]
    assert anchor.down_gemm.epilogue=='tma'
    assert anchor.gate_backward.dw.split_reduction==anchor.down_backward.dw.split_reduction==8
    down,=mlp_independent_dw_seeds(adapter,anchor)
    assert down.gate_backward==anchor.gate_backward
    assert down.down_backward.dw.split_reduction==16
    assert down==seeds[2]


def test_measured_long_shared_strict_runtime_cache(tmp_path,monkeypatch):
    """Opt-in integration with the independently measured T12288 sidecar."""
    import hashlib
    import json
    import os
    from pathlib import Path
    import rtx
    from benchmarks.benchmark_mlp_dw_opening import metrics, oracle
    from rtx._mlp_autotune import SELECTIONS, _mlp_runtime_winner_key
    from rtx.autotune import mlp as tuning, winners
    from rtx.autotune.core import stable_id
    from rtx.kernels.mxfp8 import MXFP8Problem

    sidecar=os.environ.get('RTX_TEST_MLP_REUSE_SIDECAR')
    if not sidecar or not torch.cuda.is_available():
        pytest.skip('requires CUDA and a measured RTX_TEST_MLP_REUSE_SIDECAR')
    raw=Path(sidecar).read_bytes()
    side=json.loads(raw)
    evidence=json.loads(Path(side['evidence']).read_text())
    shape=(12288,384,256,384)
    m,k,h,o=shape
    assert sum(evidence['lengths'])==m and evidence['case']['d_model']==k
    assert evidence['case']['shared_hidden_dim']==h and evidence['status']=='complete'
    row=evidence['variants'][side['label']]
    assert row['stationary'] and row['median_ms']==side['median_ms']
    assert row['config']==evidence['search']['winner']==side['config']
    assert len(side['paired_relative_speedups'])==4 and min(side['paired_relative_speedups'])>.01
    measured=rtx.MXFP8MLPConfig.from_dict(side['config'])
    device=torch.device('cuda:0')
    request=tmp_path/'request'
    cache=tmp_path/'cache'
    layer=rtx.MXFP8MLP(k,h,o,device=device,config=rtx.MXFP8MLPConfig(),
        autotune='balanced',autotune_cache_dir=cache).require_cached_()
    spec=layer.export_tuning_spec()
    initial=spec['initial']
    problem=MXFP8Problem(m,2*h,k)
    key=_mlp_runtime_winner_key(problem,o,'silu',True,initial,False,None,device=device)
    evidence_id=hashlib.sha256(raw).hexdigest()
    winner=winners.save_runtime_winner(key,measured.to_dict(),root=cache,
        config_id=stable_id(measured.to_dict()),median_ms=side['median_ms'],
        metadata={'test_only_sidecar_sha256':evidence_id})
    contents=winner.read_bytes()
    loads=[]
    original_load=winners.load_runtime_winner
    def read(k,*args,**kwargs):
        loads.append(k)
        return original_load(k,*args,**kwargs)
    monkeypatch.setattr(winners,'load_runtime_winner',read)
    monkeypatch.setattr(tuning,'tune_mxfp8_mlp',lambda *a,**kw:pytest.fail('strict cache retuned'))

    rng=torch.Generator(device=device).manual_seed(47)
    x=(torch.randn((m,k),device=device,dtype=torch.bfloat16,generator=rng)*.3).requires_grad_()
    dy=torch.randn((m,o),device=device,dtype=torch.bfloat16,generator=rng)*.1
    leaves=(x,*layer.parameters())
    compiled=torch.compile(layer,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    def run():
        y=compiled(x)
        return y,*torch.autograd.grad(y,leaves,dy)
    SELECTIONS.clear()
    actual=run()
    assert loads==[key] and [json.loads(v) for v in SELECTIONS.values()]==[measured.to_dict()]
    replay=run()
    assert loads==[key]
    errors=metrics(actual,oracle(*[v.detach() for v in leaves],dy))
    metrics(replay,actual,exact=('output','dx','dgate_up','ddown'))

    # Exercise the real deployment cache reader with validated measurement
    # evidence at its inspection boundary; this is not a preflight worker test.
    candidate=rtx.MLPCandidate(shape,layer.config)
    candidate.prepare(request,cache_dir=cache)
    inspected=dict(current_sources_match=True,selected_plan={'config':measured.to_dict()},
        environment=dict(torch=str(torch.__version__),cuda=torch.version.cuda,
                         gpu=torch.cuda.get_device_name(device)),contract={'shape':list(shape)})
    monkeypatch.setattr(rtx.MLPCandidate,'inspect',lambda *a,**kw:inspected)
    coverage=candidate.check_cache(request,device=device)
    assert loads==[key,key] and coverage['status']=='covered'
    assert coverage['selected_config']==measured.to_dict() and Path(coverage['winner_path'])==winner
    assert winner.read_bytes()==contents and not list(cache.rglob('observations.jsonl'))
    print(json.dumps(dict(sidecar_sha256=evidence_id,variant=key.variant,
        lookup_count=len(loads),coverage=coverage['status'],gradient_checks=errors)))


@pytest.mark.parametrize('shape,expected_rows',[
    ((16140,576,320,576),((64,64),(64,64))),
    ((15728,640,192,640),((128,128),(64,64))),
])
def test_half_aligned_features_reach_complete_prequant_before_local_search(monkeypatch,shape,expected_rows):
    adapter,seeds=_adapter(monkeypatch,shape,prepared=False)
    prepared=[c for c in seeds[:6]
              if c.gate_backend==c.down_backend==c.recompute_backend=='prequant']
    assert prepared,'A fixed transpose tile must not filter the whole prepared family'
    c=prepared[0]
    assert adapter.rejection(c) is None
    for group,rows in zip(('gate_backward','down_backward'),expected_rows):
        b=getattr(c,group)
        assert b.dx.backend==b.dw.backend=='decomposed'
        assert b.dx.quant_a.transposed_tile_rows==rows[0]
        assert b.dw.quant_a.transposed_tile_rows==rows[1]
        before=getattr(adapter.initial_config,group)
        for leg in ('dx','dw'):
            old=getattr(before,leg).quant_a;new=getattr(b,leg).quant_a
            assert (new.quant_math,new.quant_amax,new.scale_rounding)==(
                old.quant_math,old.quant_amax,old.scale_rounding)
