import json

import pytest

from benchmarks.compare_nvfp4_backward_policies import load_schedule


@pytest.mark.parametrize('fault',[None,'shape','phase','source','revision','ambiguous','family','policy','payload'])
def test_selected_backward_schedule_contract(tmp_path,fault):
    context=dict(family='ms_eden_preparation_research',kernel_revision=7,workload=dict(m=128,n=256,k=384),
                 tags=dict(phase='dx',source='original_bf16',packed_reader=False,layout='mma128',
                           policy='shared_sign_hadamard128_e8m4_rtz_eden_scale_sr'))
    if fault=='shape': context['workload']['m']=256
    if fault=='phase': context['tags']['phase']='dw'
    if fault=='source': context['tags']['source']='forward_current_roundtrip_bf16'
    if fault=='revision': context['kernel_revision']=1
    if fault=='family': context['family']='other'
    if fault=='policy': context['tags']['policy']='other'
    value=dict(context=context,config={})
    if fault=='payload': value['config']['reader_mode']='staged'
    (tmp_path/'selection-one.json').write_text(json.dumps(value))
    if fault=='ambiguous':
        (tmp_path/'selection-two.json').write_text(json.dumps(value))
    if fault:
        with pytest.raises(ValueError):
            load_schedule(tmp_path,(128,256,384),'dx')
    else:
        assert load_schedule(tmp_path,(128,256,384),'dx').gemm.scale_layout=='mma128'


def test_focused_refinement_preserves_other_phase():
    from dataclasses import replace
    from benchmarks.tune_ms_eden_preparation import focused_candidates
    from rtx.autotune.posthoc import make_posthoc_preparation_adapter
    adapter=make_posthoc_preparation_adapter((2048,768,1536),'dw','mma128',lambda _:None)
    initial=replace(adapter.initial_config,producer_blocks=704,producer_load='coalesced')
    prep=focused_candidates(adapter,initial,'preparation')
    gemm=focused_candidates(adapter,initial,'gemm')
    assert prep[0]==gemm[0]==initial
    assert all(c.gemm==initial.gemm for c in prep)
    assert all(replace(c,gemm=initial.gemm)==initial for c in gemm)
    assert any(c.amax_reduction=='staged' for c in prep[:9])
    assert len({c.gemm for c in gemm[:12]})>4


@pytest.mark.parametrize('kwargs',[{'steps':0},{'width':129},{'forward_scaling':'delayed'}])
def test_training_probe_rejects_invalid_contract_before_cuda(kwargs):
    from benchmarks.train_ms_eden_probe import run
    with pytest.raises(ValueError):
        run(**kwargs)
