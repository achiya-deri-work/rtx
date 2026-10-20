import pytest
import copy

from rtx.autotune.mlp_preflight import live_memory_gate


@pytest.mark.parametrize('phase',['train','forward'])
@pytest.mark.parametrize('damage',[None,'missing_gradient','missing_seed','nonfinite','inconsistent','too_large','near_zero'])
def test_numerical_gate_fails_closed(phase,damage):
    from rtx.autotune.mlp_preflight import require_numerical_gate
    names=('output','dx','dgate','ddown') if phase=='train' else ('output',)
    metrics={name:dict(finite=True,max_abs=.001,rms_error=.001,
                      reference_rms=1.,normalized_rms=.001) for name in names}
    report=dict(status='ok',phase=phase,oracle='independent_mxfp8_recipe_v1',
                cases=[dict(seed=s,metrics=copy.deepcopy(metrics)) for s in (932,933)])
    target=report['cases'][0]['metrics']['output']
    if damage=='missing_gradient': del report['cases'][0]['metrics'][names[-1]]
    if damage=='missing_seed': report['cases'].pop()
    if damage=='nonfinite': target['finite']=False
    if damage=='inconsistent': target['normalized_rms']=0.
    if damage=='too_large': target.update(rms_error=.1,normalized_rms=.1)
    if damage=='near_zero': target.update(reference_rms=0.,normalized_rms=1e9)
    if damage is None:
        require_numerical_gate(report)
    else:
        with pytest.raises(RuntimeError,match='numerical gate failed'):
            require_numerical_gate(report)


def test_cold_sample_does_not_define_warm_stability():
    gate=live_memory_gate([800,100,100,100])
    assert gate['passed'] and gate['spread_bytes']==0
    assert gate['warmup_pairs']==1 and gate['measured_pairs']==3


@pytest.mark.parametrize('samples',[[100,100,101,102],[100,101,100,101]])
def test_growth_and_unstable_residency_fail(samples):
    assert not live_memory_gate(samples)['passed']


@pytest.mark.parametrize('samples',[[],[1,1,1],[1,1,1,-1],[1,1,1,True],[1,1,1,1.0]])
def test_invalid_memory_evidence_rejected(samples):
    with pytest.raises(ValueError):
        live_memory_gate(samples)
from rtx.autotune.mlp_preflight import capacity_gate


def test_capacity_completion_does_not_hide_recovered_oom():
    gate = capacity_gate('ok', {'num_ooms': 1, 'num_alloc_retries': 1}, [{}])
    assert gate == dict(completed=True, oom_free=False, passed=False,
                        oom_count=1, allocation_retries=1)


def test_capacity_retries_are_not_ooms():
    assert capacity_gate('ok', {'num_alloc_retries': 2}, [])['passed']


def test_capacity_observer_and_status_are_both_checked():
    assert not capacity_gate('ok', {}, [{}])['passed']
    assert not capacity_gate('error', {}, [])['passed']


def valid_report():
    return dict(status='ok', allocator_event_deltas=dict(num_ooms=0, num_alloc_retries=0),
                oom_events=[], live_allocated=[4, 2, 2, 2],
                capacity_gate=capacity_gate('ok', {}, []),
                live_memory_gate=live_memory_gate([4, 2, 2, 2]))


@pytest.mark.parametrize('change', ['oom', 'live', 'missing', 'negative', 'status'])
def test_parent_gate_rejects_inconsistent_evidence(change):
    from rtx.autotune.mlp_preflight import require_preflight_gates
    r = valid_report()
    if change == 'oom': r['allocator_event_deltas']['num_ooms'] = 1
    if change == 'live': r['live_allocated'][-1] = 3
    if change == 'missing': del r['capacity_gate']
    if change == 'negative': r['allocator_event_deltas']['num_ooms'] = -1
    if change == 'status': r['status'] = 'error'
    with pytest.raises(RuntimeError, match='deployment gate failed'):
        require_preflight_gates(r)


def test_parent_stops_after_successful_worker_with_failed_capacity(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import rtx.autotune.mlp_preflight as module
    root = tmp_path/'reports'
    monkeypatch.setattr('sys.argv', ['preflight', '--shape', '128,128,128,128',
        '--spec', str(tmp_path/'spec.json'), '--output', str(root)])
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        r = valid_report()
        r['allocator_event_deltas']['num_ooms'] = 1
        r['capacity_gate'] = capacity_gate('ok', r['allocator_event_deltas'], [])
        (root/'cold'/'report.json').write_text(json.dumps(r))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='deployment gate failed'):
        module.main()
    assert len(calls) == 1 and not (root/'reuse').exists()
    assert (root/'cold'/'report.json').exists()


def test_parent_accepts_valid_gates():
    from rtx.autotune.mlp_preflight import require_preflight_gates
    require_preflight_gates(valid_report())


def test_parent_rejects_successful_numerical_worker_without_evidence(tmp_path, monkeypatch):
    import json
    import hashlib
    from rtx.mlp import MXFP8MLPConfig
    from types import SimpleNamespace
    import rtx.autotune.mlp_preflight as module
    root=tmp_path/'reports'
    monkeypatch.setattr('sys.argv',['preflight','--shape','128,128,128,128',
        '--spec',str(tmp_path/'spec.json'),'--output',str(root),'--check-numerics'])
    sources={'mlp.py':'fixed'}
    monkeypatch.setattr(module,'source_manifest',lambda: sources)
    config = MXFP8MLPConfig().to_dict()
    plan = dict(config=config, sha256=hashlib.sha256(
        json.dumps(config, sort_keys=True).encode()).hexdigest())
    calls=[]
    def run(command,**kwargs):
        calls.append(command)
        phase=('cold','reuse','numerics')[len(calls)-1]
        report=valid_report()
        report.update(source_hashes_before=sources,source_hashes_after=sources,
                      selected_plan=plan)
        (root/phase/'report.json').write_text(json.dumps(report))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess,'run',run)
    with pytest.raises(RuntimeError,match='numerical gate failed'):
        module.main()
    assert len(calls)==3 and '--numerical-worker' in calls[-1]


@pytest.mark.parametrize('change', ['none', 'missing', 'before', 'after', 'current'])
def test_preflight_source_identity(change, monkeypatch):
    import rtx.autotune.mlp_preflight as module
    expected = {'mlp.py': 'original'}
    report = dict(source_hashes_before=dict(expected), source_hashes_after=dict(expected))
    current = dict(expected)
    if change == 'missing': del report['source_hashes_before']
    if change == 'before': report['source_hashes_before']['mlp.py'] = 'changed'
    if change == 'after': report['source_hashes_after']['mlp.py'] = 'changed'
    if change == 'current': current['mlp.py'] = 'changed'
    monkeypatch.setattr(module, 'source_manifest', lambda: current)
    if change == 'none':
        module.require_preflight_sources(report, expected)
    else:
        with pytest.raises(RuntimeError, match='source identity'):
            module.require_preflight_sources(report, expected)


def test_parent_stops_before_reuse_when_worker_source_differs(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace
    import rtx.autotune.mlp_preflight as module
    root = tmp_path/'reports'
    monkeypatch.setattr('sys.argv', ['preflight', '--shape', '128,128,128,128',
        '--spec', str(tmp_path/'spec.json'), '--output', str(root)])
    monkeypatch.setattr(module, 'source_manifest', lambda: {'mlp.py': 'original'})
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        report = valid_report()
        report.update(source_hashes_before={'mlp.py': 'changed'},
                      source_hashes_after={'mlp.py': 'changed'})
        (root/'cold'/'report.json').write_text(json.dumps(report))
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, 'run', run)
    with pytest.raises(RuntimeError, match='source identity'):
        module.main()
    assert len(calls) == 1 and not (root/'reuse').exists()
    assert (root/'cold'/'report.json').exists()
