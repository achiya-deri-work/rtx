import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

from rtx.autotune.collection_health import assert_healthy_path, first_device_fault
from rtx.autotune.pretrained import load_offline_observations


def test_marker_blocks_direct_nested_and_zip_ingestion(tmp_path):
    bundle = tmp_path/'bundle'
    journal = bundle/'context/residuals/observations.jsonl'
    journal.parent.mkdir(parents=True)
    journal.write_text('')
    (bundle/'collection_health.json').write_text(json.dumps({'evidence_eligible':False}))
    for path in (bundle, journal, tmp_path):
        with pytest.raises(ValueError, match='Quarantined'):
            load_offline_observations([path])
    archive = tmp_path/'bundle.zip'
    with zipfile.ZipFile(archive,'w') as output:
        for file in bundle.rglob('*'):
            if file.is_file():output.write(file,file.relative_to(tmp_path))
    with pytest.raises(ValueError, match='Quarantined'):
        load_offline_observations([archive])


def test_malformed_marker_fails_closed(tmp_path):
    marker=tmp_path/'collection_health.json'
    for text in ('not json','{}','[]'):
        marker.write_text(text)
        with pytest.raises(ValueError, match='Invalid'):
            assert_healthy_path(tmp_path)


def test_quarantine_also_blocks_previously_exported_models(tmp_path):
    from rtx.autotune.pretrained import (
        NormalizedCostModel, ContextRankingModel, ConditionalRuleSet, load_pretrained_family,
    )
    from rtx.autotune.prior import FrozenPrior
    (tmp_path/'collection_health.json').write_text('{"evidence_eligible":false}')
    model=tmp_path/'model.json'
    model.write_text('{}')
    for cls in (NormalizedCostModel,ContextRankingModel,ConditionalRuleSet,FrozenPrior):
        with pytest.raises(ValueError,match='Quarantined'):cls.load(model)
    with pytest.raises(ValueError,match='Quarantined'):
        load_pretrained_family(tmp_path,'mxfp8_balanced_experts',1)


def test_scan_faults_is_bounded_and_nonfatal_compile_errors_are_not_cuda_faults(tmp_path):
    log=tmp_path/'run.log'
    log.write_text('compile_error NVVM rejected tile\n')
    assert first_device_fault(log) is None
    log.write_text('compile_error NVVM rejected tile\nCUDA warning: an illegal memory access was encountered\n')
    fault=first_device_fault(log)
    assert fault['line']==2 and 'illegal memory access' in fault['diagnostic']


def study_args(tmp_path):
    root=tmp_path/'source'
    (root/'rtx').mkdir(parents=True)
    (root/'benchmarks').mkdir()
    for name in ('study_moe_priors.py','benchmark_moe_varlen.py'):
        (root/'benchmarks'/name).write_text('# frozen driver\n')
    case=dict(id='case-0',d_model=384,num_experts=32,top_k=4,
              expert_hidden_dim=128,shared_hidden_dim=256,num_shared_experts=1)
    (root/'benchmarks/balanced_moe_cases.json').write_text(json.dumps({'cases':[case]}))
    out=tmp_path/'study'
    out.mkdir()
    (out/'machine.json').write_text('{}')
    return SimpleNamespace(source=root,output=out,contexts=None,budgets=[8],shared_trials=0,
        tune_seconds=120,max_autotune=False,seeds=None,plan_only=False,
        max_seconds=60,max_visits=1,visit_timeout=5)


def test_zero_exit_hidden_cuda_fault_quarantines_and_stops_study(tmp_path,monkeypatch):
    import benchmarks.study_moe_priors as study
    args=study_args(tmp_path)
    calls=[]
    def command(root,out,row,budget,args):
        calls.append(row['id'])
        target=out/row['id']/f'effort-{budget}/result.json'
        source=study.source_digest(root)
        code=f"from pathlib import Path;Path({str(target)!r}).write_text({json.dumps(dict(status='complete',source_digest=source))!r});print('CUDA warning: an illegal memory access was encountered',flush=True)"
        return [sys.executable,'-c',code]
    monkeypatch.setattr(study,'build_command',command)
    with pytest.raises(RuntimeError,match='Fatal CUDA fault'):
        study.run(args)
    assert len(calls)==1
    health=json.loads((args.output/'case-0-tiny/effort-8/health.json').read_text())
    assert health['exit_code']==76 and health['evidence_eligible'] is False
    with pytest.raises(ValueError,match='Quarantined'):
        study.run(args)


def test_clean_study_completion_resumes_but_driver_drift_is_rejected(tmp_path,monkeypatch):
    import benchmarks.study_moe_priors as study
    args=study_args(tmp_path)
    def command(root,out,row,budget,args):
        target=out/row['id']/f'effort-{budget}/result.json'
        text=json.dumps(dict(status='complete',source_digest=study.source_digest(root)))
        return [sys.executable,'-c',f'from pathlib import Path;Path({str(target)!r}).write_text({text!r})']
    monkeypatch.setattr(study,'build_command',command)
    assert study.run(args)[0]['evidence_eligible'] is True
    # The first visit is skipped, then the next still-unmeasured context runs.
    monkeypatch.setattr(study,'build_command',lambda *a: (_ for _ in ()).throw(RuntimeError('next context')))
    with pytest.raises(RuntimeError,match='next context'):
        study.run(args)
    (args.source/'benchmarks/benchmark_moe_varlen.py').write_text('# changed driver\n')
    with pytest.raises(ValueError,match='Study identity changed'):
        study.run(args)


def test_cached_complete_result_with_fault_log_is_not_reused(tmp_path,monkeypatch):
    import benchmarks.study_moe_priors as study
    args=study_args(tmp_path)
    args.plan_only=True
    study.run(args)
    args.plan_only=False
    directory=args.output/'case-0-tiny/effort-8'
    directory.mkdir(parents=True)
    (directory/'result.json').write_text(json.dumps(dict(status='complete',
        source_digest=study.source_digest(args.source))))
    (directory/'health.json').write_text('{"evidence_eligible":true}')
    (directory/'run.log').write_text('CUDA warning: illegal memory access\n')
    monkeypatch.setattr(study,'build_command',lambda *a: pytest.fail('Must not start another worker'))
    with pytest.raises(RuntimeError,match='Cached visit contains'):
        study.run(args)
    assert json.loads((args.output/'collection_health.json').read_text())['evidence_eligible'] is False


def test_success_without_complete_result_is_not_valid_evidence(tmp_path,monkeypatch):
    import benchmarks.study_moe_priors as study
    args=study_args(tmp_path)
    monkeypatch.setattr(study,'build_command',lambda *args:[sys.executable,'-c','pass'])
    with pytest.raises(RuntimeError,match='did not publish'):
        study.run(args)
    health=json.loads((args.output/'case-0-tiny/effort-8/health.json').read_text())
    assert health['evidence_eligible'] is False


def test_audit_reports_quarantine_even_below_an_ancestor_marker(tmp_path):
    from rtx.autotune.audit import audit_bundles
    (tmp_path/'collection_health.json').write_text('{"evidence_eligible":false}')
    bundle=tmp_path/'study'
    bundle.mkdir()
    (bundle/'machine.json').write_text('{}')
    (bundle/'manifest.json').write_text('{}')
    result=audit_bundles([bundle])
    assert 'Quarantined' in str(result)


def test_benchmark_keeps_max_autotune_and_nested_isolation_after_storage_fix():
    from benchmarks.benchmark_moe_varlen import compile_options
    for effort in (False,True):
        options=compile_options(effort)
        assert options['fullgraph'] is True and options['dynamic'] is False
        assert options['options']['max_autotune'] is effort
        assert options['options']['autotune_in_subproc'] is effort
        assert 'max_autotune_gemm_backends' not in options['options']


def test_log_audit_does_not_mistake_clean_logs_for_correctness_qualification(tmp_path):
    from benchmarks.audit_collection_health import audit_logs
    (tmp_path/'healthy.log').write_text('all done\n')
    report=audit_logs(tmp_path)
    assert report['evidence_eligible'] is True and report['faulted_logs']==0
    (tmp_path/'fault.log').write_text('CUDA error: illegal memory access\n')
    report=audit_logs(tmp_path)
    assert report['evidence_eligible'] is False and report['faulted_logs']==1
