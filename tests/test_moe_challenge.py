import math
from types import SimpleNamespace
from pathlib import Path

import pytest
import torch

from benchmarks.profile_moe_challenge import contexts,command
from benchmarks.benchmark_moe_varlen import profile_input,discrepancy
from benchmarks.analyze_moe_amortization import kernel_group


def test_challenge_is_broad_controlled_and_has_equal_token_packing():
    rows=contexts();by={r['id']:r for r in rows}
    assert len(rows)==len(by)==38
    assert {128,512,2048,8192,32768,65536}<={sum(r['lengths']) for r in rows}
    for r in rows:
        assert 1<=r['case']['top_k']<=r['case']['num_experts']
        assert r['case']['d_model']%32==r['case']['expert_hidden_dim']%32==0
        assert min(r['lengths'])>0
    for name in ('packed_2','packed_8','packed_32','packed_uneven'):
        assert sum(by[name]['lengths'])==8192
        assert by[name]['case']==by['anchor_8192']['case']
    a,b=by['equal_active_hidden']['case'],by['anchor_8192']['case']
    assert a['top_k']*a['expert_hidden_dim']==b['top_k']*b['expert_hidden_dim']
    assert len({r['input_profile'] for r in rows})==4


def test_command_preserves_half_alignment_and_all_comparison_contracts():
    row=next(r for r in contexts() if r['id']=='half_aligned')
    args=SimpleNamespace(expert_trials=8,shared_trials=8,router_trials=4,
        tune_seconds=90,rounds=2,samples=7,no_traces=False)
    cmd=command(Path('/source'),Path('/result.json'),row,args)
    assert cmd[cmd.index('--hidden-alignment')+1]=='32'
    assert '--max-autotune' in cmd and '--oracle' in cmd and '--bf16-numerics' in cmd
    assert {'bf16_saved_step','bf16_checkpoint_step','mxfp8_step'}<=set(cmd)


def test_cold_input_profiles_are_finite_reproducible_and_preserve_default():
    torch.manual_seed(3);x=torch.randn(128,256,dtype=torch.bfloat16)*.3
    assert profile_input(x,'normal') is x
    uniform=profile_input(x,'uniform')
    assert float(uniform.abs().max())<=.53
    assert abs(float(uniform.float().square().mean().sqrt())-.3)<.02
    outliers=profile_input(x,'outliers')
    assert torch.equal(outliers.flatten()[::1024],x.flatten()[::1024]*16)
    correlated=profile_input(x,'correlated')
    assert float(torch.nn.functional.cosine_similarity(correlated[0].float(),correlated[1].float(),dim=0))>.98
    for profile in ('normal','uniform','outliers','correlated'):
        a=profile_input(x,profile)
        assert a.dtype==x.dtype and a.shape==x.shape and bool(a.isfinite().all())
        assert torch.equal(a,profile_input(x,profile))
    with pytest.raises(ValueError):profile_input(x,'invalid')


def test_numerical_report_is_not_a_loosened_oracle_gate():
    a=torch.tensor([1.,2.]);r=torch.tensor([2.,4.])
    error=discrepancy(a,r)
    assert error['nrms']==.5 and error['all_finite']
    assert math.isclose(error['cosine'],1.,rel_tol=1e-6)
    with pytest.raises(AssertionError):discrepancy(torch.tensor([float('nan')]),torch.ones(1))
    assert kernel_group('ExpertFragmentGLU')=='gate_up_gemm_activation_quant'
    assert kernel_group('BackwardWeightQuant')=='quantization_preparation'


def test_postprocessor_knows_panel_gemms_and_never_adds_concurrent_coverage(tmp_path):
    import json
    from benchmarks.report_moe_challenge import group,trace_detail
    assert group('PersistentExpertDW')=='gemm'
    assert group('direct_copy_kernel_cuda')=='framework_copy'
    path=tmp_path/'trace.json'
    path.write_text(json.dumps(dict(traceEvents=[
        dict(cat='kernel',name='PersistentExpertDW',ts=0,dur=4),
        dict(cat='kernel',name='BackwardWeightQuant',ts=2,dur=4),
        dict(cat='cuda_runtime',name='cudaGraphLaunch',ts=0,dur=1)])))
    result=trace_detail(path)
    assert result['kernel_sum_us']==8 and result['span_us']==6 and result['coverage']==1
    assert result['groups']['gemm']['share']==.5
    assert result['cuda_api_counts']=={'cudaGraphLaunch':1}
