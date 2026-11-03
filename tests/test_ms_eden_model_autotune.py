from dataclasses import asdict,replace

import pytest

from rtx.autotune.ms_eden_design import MSEdenDesign
from rtx.autotune.ms_eden_model import MSEdenModelSchedule,make_ms_eden_model_adapter
from rtx.autotune.outcomes import TrialOutcome


def adapter(batch=8):
    design=MSEdenDesign()
    design=replace(design,schedule=replace(design.schedule,backward_pipeline='paired'))
    bases={f'{batch*128},128,128':asdict(design),f'{batch*128},256,128':asdict(design)}
    seen=[]
    result=make_ms_eden_model_adapter(bases,lambda value:(seen.append(value) or TrialOutcome('ok',median_ms=1)),
        batch=batch,sequence=128,model_context={'model':'test'},device={'uuid':'gpu-a'})
    return result,seen


def test_model_schedule_roundtrip_identity_and_real_expansion():
    a,seen=adapter()
    base=a.initial_config
    coordinate=a.coordinates()[0]
    changed=a.update_fn(base,coordinate,'paired_dw_first')
    assert a.config_id(changed)!=a.config_id(base)
    assert a.deserialize(a.serialize(changed))==changed
    assert a.rejection(changed) is None
    a.evaluate(changed)
    assert sum(v['schedule']['backward_pipeline']=='paired_dw_first' for v in seen[0].values())==1
    assert a.context.tags['objective']=='complete_compiled_model'
    assert not a.context.tags['automatic_promotion']
    b,_=adapter(batch=16)
    assert a.context.identifier!=b.context.identifier


def test_model_schedule_contract_failures():
    a,_=adapter()
    with pytest.raises(ValueError):MSEdenModelSchedule(('unknown',))
    with pytest.raises(ValueError):a.deserialize({'pipelines':{}})
    with pytest.raises(ValueError):a.serialize(MSEdenModelSchedule(('paired',)))
    with pytest.raises(ValueError,match='batch and sequence'):
        make_ms_eden_model_adapter({'128,128,128':asdict(MSEdenDesign())},lambda _:None,
            batch=2,sequence=128,model_context={'model':'test'})


def test_coupled_owner_choice_has_priority_seed():
    from benchmarks.tune_ms_eden_design import ownership_seeds
    from benchmarks.tune_ms_eden_pipeline import candidates
    from rtx.autotune.ms_eden_pipeline import make_ms_eden_pipeline_adapter
    seeds=ownership_seeds(MSEdenDesign())
    assert [v.schedule.backward_pipeline for v in seeds]==['paired','paired_dw_first']
    a=make_ms_eden_pipeline_adapter((512,512,512),lambda _:None)
    assert any(v.backward_pipeline=='paired_dw_first' for v in candidates(a,8))
