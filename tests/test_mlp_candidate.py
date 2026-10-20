import hashlib
import json
import subprocess
from pathlib import Path

import pytest
import torch
import rtx


def candidate(phase='train'):
    return rtx.MLPCandidate((128,128,128,128),rtx.MXFP8MLPConfig(),phase=phase)


def test_bundle_roundtrip_detached_proposal_and_no_overwrite(tmp_path):
    c=candidate()
    path=tmp_path/'bundle.json'
    c.save(path)
    assert rtx.MLPCandidate.load(path)==c
    values=c.to_dict()
    values['config']['gate_up']['tile_m']=987
    assert c.to_dict()['config']['gate_up']['tile_m']!=987
    with pytest.raises(FileExistsError): c.save(path)
    values=json.loads(path.read_text())
    values['status']='qualified'
    path.write_text(json.dumps(values))
    with pytest.raises(ValueError): rtx.MLPCandidate.load(path)


def test_prepare_cpu_only_exact_contract_and_user_budget(tmp_path,monkeypatch):
    c=candidate('forward')
    monkeypatch.setattr(torch.cuda,'_lazy_init',lambda:pytest.fail('preparation initialized CUDA'))
    policy=rtx.AutotunePolicy(max_trials=1,max_seconds=30,warmup=1,samples=3,
        calls_per_sample=1,confirmation_repeats=0,finalist_rounds=0)
    request=c.prepare(tmp_path/'request',policy=policy)
    assert request['requires_qualification'] and not request['installs_runtime_winner']
    assert not request['source_matches_evidence']
    assert request['cache_is_request_local']
    spec=(tmp_path/'request'/'spec.json').read_text()
    assert hashlib.sha256(spec.encode()).hexdigest()==request['spec_sha256']
    assert json.loads(spec)['policy']['max_trials']==1
    command=request['command']
    assert command[command.index('--phase')+1]=='forward'
    assert command[command.index('--shape')+1]=='128,128,128,128'
    assert '--check-numerics' in command
    with pytest.raises(FileExistsError): c.prepare(tmp_path/'request')


def test_import_keeps_evidence_without_promoting(tmp_path):
    from rtx.autotune.replacement_benchmark import source_manifest
    path=tmp_path/'paired.json'
    data=dict(shape=[128]*4,phase='train',mode='selected_direct',
        arguments={'mode':'selected_direct'},records=[{'outcome':{'status':'ok'}}],
        challenger=rtx.MXFP8MLPConfig().to_dict(),rtx_source_sha256=source_manifest(),
        decision={'promoted':True})
    path.write_text(json.dumps(data))
    c=rtx.MLPCandidate.from_qualification(path)
    assert c.to_dict()['status']=='proposal'
    assert c.to_dict()['evidence']['sha256']==hashlib.sha256(path.read_bytes()).hexdigest()
    assert c.prepare(tmp_path/'request')['source_matches_evidence']
    data['mode']='outer_capture'
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError): rtx.MLPCandidate.from_qualification(path)


@pytest.mark.parametrize('shape',[(0,1,1,1),(True,1,1,1),(1,2,3),[1,2,3,4]])
def test_bad_shape_rejected(shape):
    with pytest.raises(ValueError): rtx.MLPCandidate(shape,rtx.MXFP8MLPConfig())


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
def test_public_candidate_preflight_end_to_end(tmp_path, monkeypatch):
    c=candidate()
    policy=rtx.AutotunePolicy(max_trials=1,max_seconds=30,warmup=1,samples=3,
        calls_per_sample=1,confirmation_repeats=0,finalist_rounds=0)
    request=c.prepare(tmp_path/'request',policy=policy)
    result=subprocess.run(request['command'],capture_output=True,text=True,timeout=180,
                          cwd=Path(__file__).resolve().parents[1])
    assert result.returncode==0,(result.stdout,result.stderr)
    root=tmp_path/'request'/'qualification'
    for phase in ('cold','reuse'):
        report=json.loads((root/phase/'report.json').read_text())
        assert report['capacity_gate']['passed'] and report['live_memory_gate']['passed']
        assert report['verify_cache']==(phase=='reuse')
    report=json.loads((root/'numerics'/'report.json').read_text())
    assert report['numerical_gate']['passed'] and report['verify_cache']
    assert json.loads((tmp_path/'request'/'candidate.json').read_text())['status']=='proposal'
    lazy_init = torch.cuda._lazy_init
    monkeypatch.setattr(torch.cuda, '_lazy_init', lambda: pytest.fail('inspection initialized CUDA'))
    inspected = c.inspect(tmp_path/'request')
    assert inspected['status'] == 'preflight_passed'
    assert inspected['current_sources_match']
    assert not inspected['requires_requalification']
    assert not inspected['installs_runtime_winner']
    assert inspected['selected_plan'] == report['selected_plan']
    # Shape-dependent backward defaults are resolved even with one trial.
    assert inspected['selected_plan']['config']['gate_backward'] is not None
    assert inspected['matches_initial_serialization'] is False
    assert inspected['gaps']
    # Do not trust exit status/pass bits or accept another context's evidence.
    for stage, key, value in (
        ('reuse', 'verify_cache', False),
        ('reuse', 'shape', [256,128,128,128]),
        ('numerics', 'phase', 'forward'),
        ('reuse', 'source_hashes_after', {}),
        ('reuse', 'torch', 'different'),
        ('cold', 'synthetic_reserve_bytes', 1),
        ('numerics', 'cases', []),
        ('reuse', 'selected_plan', {'config': {}, 'sha256': 'wrong'}),
    ):
        path = root/stage/'report.json'
        original = path.read_text()
        changed = json.loads(original)
        changed[key] = value
        path.write_text(json.dumps(changed))
        try:
            with pytest.raises((ValueError, RuntimeError)):
                c.inspect(tmp_path/'request')
        finally:
            path.write_text(original)
    monkeypatch.setattr(torch.cuda, '_lazy_init', lazy_init)
    import rtx._mlp_autotune as selection
    original_select = selection._select
    monkeypatch.setattr(selection, '_select', lambda *a, **kw: pytest.fail('coverage attempted selection'))
    coverage = c.check_cache(tmp_path/'request')
    assert coverage['status'] == 'covered' and coverage['point_in_time_only']
    winner_path = Path(coverage['winner_path'])
    original_winner = winner_path.read_text()
    winner = json.loads(original_winner)
    winner['config']['backward_overlap'] = not winner['config']['backward_overlap']
    winner_path.write_text(json.dumps(winner))
    assert c.check_cache(tmp_path/'request')['status'] == 'winner_changed'
    winner_path.write_text('{invalid')
    assert c.check_cache(tmp_path/'request')['status'] == 'missing_or_ineligible'
    winner_path.write_text(original_winner)
    assert c.check_cache(tmp_path/'request')['status'] == 'covered'
    layer = rtx.MXFP8MLP(128, 128, device='cuda', config=c.config, autotune='balanced',
        tuning_policy=policy, autotune_cache_dir=request['qualification_cache_dir'])
    ids = [id(p) for p in layer.parameters()]
    weights = [p.detach().clone() for p in layer.parameters()]
    before_spec = layer.export_tuning_spec()
    with pytest.raises(ValueError, match='duplicate'):
        rtx.require_qualified_mlp_cache_(layer, [tmp_path/'request', tmp_path/'request'])
    assert layer.export_tuning_spec() == before_spec
    frozen = rtx.require_qualified_mlp_cache_(layer, [tmp_path/'request'])
    assert len(frozen['contexts']) == 1
    assert layer.export_tuning_spec()['cache_miss'] == 'error'
    assert ids == [id(p) for p in layer.parameters()]
    assert all(torch.equal(p, old) for p, old in zip(layer.parameters(), weights))
    monkeypatch.setattr(selection, '_select', original_select)
    import rtx.autotune.mlp as tuning
    monkeypatch.setattr(tuning, 'tune_mxfp8_mlp', lambda *a, **kw: pytest.fail('strict module tuned'))
    compiled = torch.compile(layer, fullgraph=True, dynamic=False,
                             options={'triton.cudagraphs': False})
    x = torch.randn(128, 128, device='cuda', dtype=torch.bfloat16, requires_grad=True)*.1
    dy = torch.randn_like(x)*.1
    y = compiled(x)
    actual = (y, *torch.autograd.grad(y, (x, *layer.parameters()), dy))
    with torch.no_grad():
        expected = tuning.reference(x, *layer.parameters(), dy, 'silu')
    for a, b in zip(actual, expected):
        assert bool(a.isfinite().all())
        assert float((a.float()-b.float()).norm()/b.float().norm().clamp_min(1e-12)) < .025
    with torch.no_grad(), pytest.raises(RuntimeError, match='requires a cached winner'):
        compiled(x)
    from rtx.autotune import replacement_benchmark
    monkeypatch.setattr(replacement_benchmark, 'source_manifest', lambda: {'changed': 'source'})
    archived = c.inspect(tmp_path/'request')
    assert archived['requires_requalification'] and not archived['current_sources_match']
    with pytest.raises(RuntimeError, match='current-source'):
        c.check_cache(tmp_path/'request')


def test_inspection_rejects_missing_or_changed_request(tmp_path):
    c = candidate()
    c.prepare(tmp_path/'request')
    with pytest.raises(FileNotFoundError):
        c.inspect(tmp_path/'request')
    path = tmp_path/'request'/'spec.json'
    path.write_text(path.read_text() + ' ')
    with pytest.raises(ValueError, match='spec changed'):
        c.inspect(tmp_path/'request')


def test_selected_plan_gate_rejects_missing_corrupt_and_different():
    from rtx.autotune.mlp_preflight import require_selected_plan, record_selected_plan
    from rtx._execution_plan_token import PLAN_TOKEN_BYTES, write_plan_token
    from dataclasses import replace
    token = torch.empty(PLAN_TOKEN_BYTES, dtype=torch.uint8)
    config = rtx.MXFP8MLPConfig()
    write_plan_token(token, json.dumps(config.to_dict()))
    report = {}
    record_selected_plan(report, token)
    plan = require_selected_plan(report)
    assert plan['config'] == config.to_dict()
    with pytest.raises(RuntimeError): require_selected_plan({})
    with pytest.raises(RuntimeError): require_selected_plan(report, {'different': True})
    write_plan_token(token, json.dumps(replace(config, backward_overlap=True).to_dict()))
    with pytest.raises(RuntimeError, match='changed within'):
        record_selected_plan(report, token)
