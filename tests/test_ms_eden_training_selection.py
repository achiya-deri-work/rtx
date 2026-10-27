from dataclasses import asdict, replace

import pytest

from benchmarks.select_ms_eden_training_designs import MODEL_SHAPES, choose_common
from rtx.autotune.ms_eden_design import MSEdenDesign


def records(mode, costs, labels=MODEL_SHAPES):
    base=MSEdenDesign()
    design=replace(base,row_transform=mode,schedule=replace(base.schedule,
        dx=replace(base.schedule.dx,producer_values_per_lane=16)))
    return [dict(shape=label,design=asdict(design),median_ms=cost,path=f'{mode}/{label}')
            for label,cost in zip(labels,costs)]


def test_selection_does_not_mix_per_shape_implementation_winners():
    result=choose_common(records('native',[1,1,1,10])+records('fma',[4,4,4,4]))
    assert result['estimated_layer_backward_ms']==13
    assert result['implementation'][-1]=='native'
    assert all(r['design']['row_transform']=='native' for r in result['selected'].values())
    assert result['requires_independent_confirmation']
    assert not result['automatic_promotion']


def test_incomplete_fast_family_cannot_win():
    result=choose_common(records('native',[3]*4)+records('fma',[.1]*3))
    assert result['implementation'][-1]=='native'
    with pytest.raises(ValueError,match='covers all'):
        choose_common(records('fma',[.1]*3))


def test_best_candidate_per_shape_within_family_and_deterministic_ties():
    rows=records('native',[3]*4)
    better=dict(rows[0],median_ms=1,path='a')
    tie=dict(better,path='z')
    result=choose_common([tie,*rows,better])
    assert result['estimated_layer_backward_ms']==10
    assert result['selected'][MODEL_SHAPES[0]]['path']=='a'


def test_empty_search_is_not_a_training_recipe():
    with pytest.raises(ValueError,match='no timing-eligible'):
        choose_common([])


def test_collector_rejects_missing_timing_proof_and_mixed_devices(tmp_path,monkeypatch):
    import json
    from types import SimpleNamespace
    from benchmarks import select_ms_eden_training_designs as module
    from rtx.autotune.outcomes import TrialOutcome
    def adapter(shape,evaluator,**kwargs):
        def evaluate(design):
            raw=evaluator(design)
            return TrialOutcome('ok',median_ms=1,metadata={'sampling':{'collection':
                {'stationary':raw['stationary']}}})
        return SimpleNamespace(evaluate=evaluate)
    monkeypatch.setattr(module,'make_ms_eden_design_adapter',adapter)
    def write(index,device,stationary):
        path=tmp_path/f'search-{MODEL_SHAPES[0]}'/str(index)/'result.json'
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps(dict(design=asdict(MSEdenDesign()),
            device=device,source_sha256={'kernel':'same'},stationary=stationary)))
    write(0,'gpu-a',None)
    write(1,'gpu-a',False)
    rows,excluded,identities=module.collect(tmp_path)
    assert not rows and len(excluded)==2 and not identities
    write(2,'gpu-a',True)
    assert len(module.collect(tmp_path)[0])==1
    write(3,'gpu-b',True)
    with pytest.raises(ValueError,match='mixed device'):
        module.collect(tmp_path)
