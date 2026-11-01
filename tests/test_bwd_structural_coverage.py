from dataclasses import replace
import pytest

from rtx._bwd_search import backward_structural_seeds,backward_search_space,active_backward_dict
from rtx.bwd_autotune import BWD_SEARCH_SPACE,update_bwd_config,bwd_config_id
from rtx.kernels.mxfp8 import MXFP8Problem
from rtx.kernels.mxfp8_bwd import DEFAULT_FUSED_MXFP8_BWD_CONFIG as BASE


@pytest.mark.parametrize('rows,n,k',[(8500,256,256),(8512,256,256),(8192,64,256),(136000,256,256),(1024,128,64)])
def test_shape_families_are_legal_and_include_split(rows,n,k):
    p=MXFP8Problem(rows,n,k)
    seeds=backward_structural_seeds(p,BASE)
    assert seeds and all(c.implementation_rejection(p) is None for c in seeds)
    assert any(c.dw.split_reduction>1 for c in seeds)
    assert any(c.dw.backend=='decomposed' and c.dw.split_reduction>1 for c in seeds)
    axes=backward_search_space(p,BWD_SEARCH_SPACE)
    if rows==8500:
        assert any(v['dw'].get('reduction_tile')==2176 for v in axes['dw_reduction'])


def test_compound_backend_reaches_narrow_shape():
    p=MXFP8Problem(8512,64,256)
    axes=backward_search_space(p,BWD_SEARCH_SPACE)
    candidate=update_bwd_config(BASE,axes['dw_backend'][-1])
    assert candidate.implementation_rejection(p) is None
    assert candidate.dw.quant_a.transposed_tile_rows==64


def test_shape_space_is_idempotent_and_cache_lookup_uses_same_domain():
    from unittest.mock import patch
    from rtx.bwd_autotune import load_cached_mxfp8_bwd_config
    p=MXFP8Problem(8500,256,256)
    axes=backward_search_space(p,BWD_SEARCH_SPACE)
    assert backward_search_space(p,axes)==axes
    with patch('rtx.bwd_autotune.DeviceFingerprint.current',return_value=object()), \
         patch('rtx.bwd_autotune.BwdJsonTuningDatabase') as database:
        database.return_value.best.return_value=None
        assert load_cached_mxfp8_bwd_config(p) is None
        assert database.call_args.args[-1]==axes


def test_explicit_fused_unsplit_domain_is_not_expanded():
    p=MXFP8Problem(8500,256,256)
    axes={'dw_backend':({'dw':{'backend':'fused'}},),
          'dw_reduction':({'dw':{'reduction':'full_fp32'}},)}
    assert backward_search_space(p,axes)==axes


def test_fused_inactive_quant_and_gemm_do_not_spend_trials():
    changed=replace(BASE,dw=replace(BASE.dw,quant_launches='dual',
        quant_a=replace(BASE.dw.quant_a,persistent_waves=3),
        gemm=replace(BASE.dw.gemm,stages=3)))
    assert active_backward_dict(BASE)==active_backward_dict(changed)
    assert bwd_config_id(BASE)==bwd_config_id(changed)
    changed=replace(BASE,dw=replace(BASE.dw,fused=replace(BASE.dw.fused,quant_amax='bf16_bits')))
    assert bwd_config_id(BASE)!=bwd_config_id(changed)


@pytest.mark.parametrize('trials',[3,8,16])
def test_opening_reaches_complete_families_and_resumes(trials):
    from rtx.autotune.adapters import make_mxfp8_bwd_adapter
    from rtx.autotune.effort import make_effort_autotuner,AutotunePolicy
    from rtx.autotune.outcomes import TrialOutcome
    from rtx.autotune.store import InMemoryTuningStore
    p=MXFP8Problem(8512,256,256)
    measured=[]
    adapter=make_mxfp8_bwd_adapter(p,lambda c: measured.append(c) or TrialOutcome('ok',median_ms=1.),initial=BASE,device={})
    store=InMemoryTuningStore()
    policy=AutotunePolicy(max_trials=trials,max_seconds=30,confirmation_repeats=0,
                         search_schedule='progressive',pool_size=8)
    first=make_effort_autotuner(adapter,store,policy).tune()
    assert len(measured)==trials
    assert {c.dw.backend for c in measured[:3]}=={'fused','decomposed'}
    assert any(c.dw.split_reduction>1 for c in measured[:3])
    coverage=first.tuning_evidence['structural_coverage']['dw']
    assert {r['backend'] for r in coverage}=={'fused','decomposed'}
    assert {r['grid_ctas'] for r in coverage}>={4,32}
    second=make_effort_autotuner(adapter,store,policy).tune()
    assert second.evaluated_trials==0 and len(measured)==trials
