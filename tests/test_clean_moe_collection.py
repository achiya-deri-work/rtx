import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.collect_moe_clean import validate_result, kernel_faults, collect


def record():
    return dict(status='complete',source_digest='source',
        inductor_max_autotune=True,inductor_autotune_in_subproc=True,
        validation=dict(all_finite=True,
            independent_quantized_fp64_experts={k:dict(nrms=.0001,max_abs=.0001)
                for k in ('output','input','gate_dw','down_dw','route_dw')},
            fresh_gradient_replay={k:[dict(max_nrms=0.,max_abs=0.)]*2 for k in
                ('mxfp8_step','bf16_checkpoint_step','expert_step','expert_bf16_checkpoint_step')}),
        memory=dict(live_bytes=[123]*5,live_growth_bytes=0))


def test_accepts_only_complete_numerical_replay_memory_gates(tmp_path):
    target=tmp_path/'result.json'
    target.write_text(json.dumps(record()))
    assert validate_result(target)['live_bytes']==123
    bad=[]
    r=record();r['validation']['all_finite']=False;bad.append(r)
    r=record();del r['validation']['independent_quantized_fp64_experts']['gate_dw'];bad.append(r)
    r=record();r['validation']['fresh_gradient_replay']['mxfp8_step']=[];bad.append(r)
    r=record();r['memory']['live_growth_bytes']=1;bad.append(r)
    r=record();r['validation']['independent_quantized_fp64_experts']['input']['nrms']=float('nan');bad.append(r)
    for r in bad:
        target.write_text(json.dumps(r))
        with pytest.raises(ValueError):validate_result(target)


def test_kernel_log_gate_filters_gpu_and_fails_closed_on_access_error(monkeypatch):
    import benchmarks.collect_moe_clean as module
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,
        stderr='',stdout='Xid (PCI:0000:01:00): 31\nXid (PCI:0000:03:00): 31'))
    assert kernel_faults(123.,'0000:01:00')==['Xid (PCI:0000:01:00): 31']
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,
        stderr='',stdout='-- No entries --\n'))
    assert kernel_faults(123.,'0000:01:00')==[]
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=1,
        stderr='',stdout=''))
    with pytest.raises(RuntimeError,match='Cannot qualify'):kernel_faults(123.,'0000:01:00')
    monkeypatch.setattr(module.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0,
        stderr='not seeing messages from other users',stdout=''))
    with pytest.raises(RuntimeError,match='Cannot qualify'):kernel_faults(123.,'0000:01:00')


def test_failed_result_stops_before_export_or_next_visit(tmp_path,monkeypatch):
    import benchmarks.collect_moe_clean as module
    result=tmp_path/'result.json';result.write_text('{"status":"running"}')
    calls=[]
    def visit(args):
        calls.append(1)
        return [dict(result=str(result),context='tiny',budget=8)]
    monkeypatch.setattr(module,'study_run',visit)
    monkeypatch.setattr(module,'analyze',lambda *a:pytest.fail('Invalid result exported'))
    args=SimpleNamespace(source=tmp_path,output=tmp_path/'new',max_seconds=60,
                         max_visits=3,shared_trials=8,pci='0000:01:00')
    with pytest.raises(ValueError,match='Incomplete'):collect(args)
    assert calls==[1]
    assert json.loads((args.output/'study/collection_health.json').read_text())['evidence_eligible'] is False


@pytest.mark.parametrize('budget,status,graceful',[(60,124,True),(1800,124,False),(60,76,False)])
def test_only_campaign_clipped_timeout_is_a_graceful_stop(tmp_path,monkeypatch,budget,status,graceful):
    import benchmarks.collect_moe_clean as module
    from benchmarks.study_moe_priors import StudyVisitError
    def visit(args):
        raise StudyVisitError('worker failed',dict(status=status,evidence_eligible=False))
    monkeypatch.setattr(module,'study_run',visit)
    args=SimpleNamespace(source=tmp_path,output=tmp_path/'run',max_seconds=budget,
                         max_visits=3,shared_trials=8,pci='0000:01:00')
    if graceful:
        assert collect(args)==[]
        stop=json.loads((args.output/'completion.json').read_text())
        assert stop['reason']=='deadline' and stop['last_qualified'] is None
        assert stop['unfinished_visit']['evidence_eligible'] is False
    else:
        with pytest.raises(StudyVisitError):collect(args)


def test_finalized_prefix_rejects_naive_and_late_timestamps():
    from benchmarks.finalize_moe_collection import finished_before
    assert finished_before({'finished_at':'1970-01-01T00:00:01+00:00'},2)
    assert not finished_before({'finished_at':'1970-01-01T00:00:03Z'},2)
    with pytest.raises(ValueError,match='Naive'):
        finished_before({'finished_at':'1970-01-01T00:00:01'},2)


def test_finalization_rejects_nonpositive_partitions_before_io(tmp_path):
    from benchmarks.finalize_moe_collection import finalize
    with pytest.raises(ValueError,match='positive'):
        finalize(tmp_path/'source',tmp_path/'output',partition_rows=0)
    assert not (tmp_path/'output').exists()
