from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from benchmarks.profile_backward_subparts import resolve_schedules
from rtx.autotune.ms_eden_pipeline import MSEdenPipelineSchedule


def test_manifest_pins_both_schedules_and_pipeline(tmp_path):
    config=MSEdenPipelineSchedule(backward_pipeline='paired')
    path=tmp_path/'candidate.json'
    path.write_text(json.dumps({'candidate':asdict(config)}))
    args=SimpleNamespace(mode='public',schedule_manifest=path,backward_pipeline='serial')
    schedules,pipeline=resolve_schedules(args)
    assert pipeline=='paired'
    for phase in ('dx','dw'):
        assert json.loads(schedules[phase])==asdict(getattr(config,phase))
    args.mode='fp8'
    with pytest.raises(ValueError,match='public or fp4'):resolve_schedules(args)
    args.mode='fp4'
    path.write_text(json.dumps({'candidate':{'dx':{}}}))
    with pytest.raises(ValueError,match='explicit candidate'):resolve_schedules(args)


def test_legacy_default_is_unchanged():
    schedules,pipeline=resolve_schedules(SimpleNamespace(backward_pipeline='serial'))
    assert pipeline=='serial'
    assert all(json.loads(value)['producer_signs']=='register' for value in schedules.values())


def test_completed_comparison_pins_actual_overrides(tmp_path):
    config=MSEdenPipelineSchedule(backward_pipeline='operand_paired')
    schedules={phase:asdict(getattr(config,phase)) for phase in ('dx','dw')}
    schedules['dx']['producer_values_per_lane']=16
    for value in schedules.values():
        value['producer_grouped_store_bits']=64
    path=tmp_path/'comparison.json'
    path.write_text(json.dumps({'candidate_pipeline':'operand_paired',
        'schedules':{'candidate':{k:json.dumps(v) for k,v in schedules.items()}}}))
    actual,pipeline=resolve_schedules(SimpleNamespace(mode='public',schedule_manifest=path))
    assert pipeline=='operand_paired'
    assert {k:json.loads(v) for k,v in actual.items()}==schedules
def test_exact_design_selection_reaches_profile_and_records_dispatch(monkeypatch,tmp_path):
    import json
    import sys
    from dataclasses import asdict
    from contextlib import contextmanager
    from benchmarks import profile_backward_subparts as driver
    from benchmarks import ms_eden_design_dispatch
    from rtx.autotune.ms_eden_design import MSEdenDesign
    design=MSEdenDesign()
    selection=tmp_path/'selection.json'
    selection.write_text(json.dumps({'candidate':asdict(design)}))
    output=tmp_path/'result.json'
    argv=['profile','--mode','public','--design-selection',str(selection),
          '--shape','512,512,512','--output',str(output)]
    monkeypatch.setattr(sys,'argv',argv)
    events=[]
    @contextmanager
    def dispatch(actual):
        assert actual==design
        events.append('enter')
        try:yield []
        finally:events.append('exit')
    monkeypatch.setattr(ms_eden_design_dispatch,'dispatch',dispatch)
    def worker(args):
        schedules,pipeline=driver.resolve_schedules(args)
        assert pipeline==design.schedule.backward_pipeline
        assert json.loads(schedules['dx'])==asdict(design.schedule.dx)
        return {'tested':True}
    monkeypatch.setattr(driver,'worker',worker)
    driver.main()
    result=json.loads(output.read_text())
    assert result['design']==asdict(design)
    assert len(result['design_source_sha256'])==7
    assert 'rtx/kernels/ms_eden_bank_layout.py' in result['design_source_sha256']
    assert events==['enter','exit']
    monkeypatch.setattr(sys,'argv',argv+['--preparation-variant','tma_xor5_1'])
    import pytest
    with pytest.raises(ValueError,match='without other schedule overrides'):driver.main()
