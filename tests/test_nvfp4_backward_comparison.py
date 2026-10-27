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
    assert all(c.gemm_split_reduction==initial.gemm_split_reduction for c in prep)
    assert any(c.gemm_split_reduction!=initial.gemm_split_reduction for c in gemm)
    from dataclasses import fields
    preparation_fields=[f.name for f in fields(initial)
                        if f.name!='gemm' and not f.name.startswith('gemm_')]
    assert all(all(getattr(c,name)==getattr(initial,name) for name in preparation_fields)
               for c in gemm)
    assert any(c.amax_reduction=='staged' for c in prep[:9])
    assert len({c.gemm for c in gemm[:12]})>4


@pytest.mark.parametrize('kwargs',[{'steps':0},{'width':129},{'forward_scaling':'delayed'}])
def test_training_probe_rejects_invalid_contract_before_cuda(kwargs):
    from benchmarks.train_ms_eden_probe import run
    with pytest.raises(ValueError):
        run(**kwargs)


def test_public_rng_tuning_uses_actual_public_phase_evaluator(tmp_path):
    import json
    from types import SimpleNamespace
    from benchmarks.tune_ms_eden_preparation import evaluation_command
    from rtx.autotune.posthoc import PosthocPreparationSchedule
    args=SimpleNamespace(public_rng=True,shape="4096,2048,512",phase="dw",
                         source_policy="original_bf16",packed_reader=False)
    config=PosthocPreparationSchedule(rng_mode="philox",producer_blocks=256)
    command=evaluation_command(args,config,tmp_path)
    assert command[2]=="benchmarks.compare_ms_eden_amortization"
    assert command[command.index("--evaluate-phase")+1]=="dw"
    assert json.loads(command[command.index("--schedule-json")+1])["rng_mode"]=="philox"
    args.public_rng=False
    command=evaluation_command(args,PosthocPreparationSchedule(),tmp_path)
    assert command[2]=="benchmarks.study_ms_eden_backward"
    assert command[command.index("--schedule-json")+1]==str(tmp_path/"schedule.json")


@pytest.mark.parametrize("rng,signs",[("explicit","reload"),("philox","register")])
def test_public_seed_portfolio_covers_both_rng_and_reuse_choices(rng,signs):
    from benchmarks.tune_ms_eden_preparation import public_rng_seeds
    from rtx.autotune.posthoc import PosthocPreparationSchedule
    initial=PosthocPreparationSchedule(rng_mode=rng,producer_signs=signs,producer_blocks=256)
    proposals=public_rng_seeds(initial)
    assert proposals[0]==initial
    assert {(c.rng_mode,c.producer_signs) for c in proposals}=={
        (r,s) for r in ("explicit","philox") for s in ("reload","register")}
    assert all(c.producer_blocks==256 for c in proposals)
