import importlib.util
import io
import json
from pathlib import Path

import pytest

spec=importlib.util.spec_from_file_location('mlp_profile',Path(__file__).parents[1]/'tools/analyze_mlp_profile.py')
module=importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def event(cat,name,ts,dur,correlation=None,tid=1):
    return dict(ph='X',cat=cat,name=name,ts=ts,dur=dur,pid=0,tid=tid,
                args={} if correlation is None else dict(correlation=correlation,device=0))


def test_correlated_kernels_not_gpu_annotation_or_nested_aggregates():
    events=[event('user_annotation','rtx_mlp/tokenizer/forward/m20480',0,10),
        event('cuda_driver','launch',2,1,1),event('cuda_runtime','launch',5,1,2),
        event('kernel','a',20,4,1),event('kernel','b',22,4,2),
        event('gpu_user_annotation','rtx_mlp/tokenizer/forward/m20480',20,100),
        event('cpu_op','aggregate',0,100)]
    result=module.analyze(events)
    assert result['summed_kernel_us']==8
    assert result['busy_kernel_union_us_by_device']=={'0':6}
    row=result['categories']['rtx_mlp/tokenizer/forward/m20480']
    assert row['kernels']==2 and row['summed_kernel_time_fraction']==1


def test_cross_thread_missing_and_ambiguous_launches_are_not_attributed():
    events=[event('user_annotation','rtx_mlp/tokenizer/forward/m20480',0,10),
        event('cuda_driver','launch',2,1,1,tid=2),event('kernel','a',20,4,1),
        event('kernel','b',30,4,2),event('cuda_driver','launch',2,1,3),
        event('cuda_driver','launch',3,1,3),event('kernel','c',40,4,3)]
    result=module.analyze(events)
    assert set(result['categories'])=={'unlabelled','missing_correlation','ambiguous_correlation'}
    assert result['runtime_ranges']['rtx_mlp/tokenizer/forward/m20480']['kernels']==0


def test_nested_labels_assign_only_innermost_and_shared_boundary_excludes_old():
    events=[event('user_annotation','rtx_mlp/outer',0,20),
        event('user_annotation','rtx_mlp/inner',1,4),
        event('cuda_driver','launch',2,1,1),event('kernel','a',30,2,1),
        event('cuda_driver','launch',5,1,2),event('kernel','b',40,3,2)]
    result=module.analyze(events)
    assert result['categories']['rtx_mlp/inner']['kernel_us']==2
    assert result['categories']['rtx_mlp/outer']['kernel_us']==3


def test_negative_duration_refused():
    with pytest.raises(ValueError,match='negative'):
        module.analyze([event('kernel','a',0,-1)])


def test_recorded_owner_call_count_audit():
    report={'runtime_ranges':{'rtx_mlp/tokenizer/forward/m20480':{'calls':2}}}
    row=dict(phase='forward',x={'shape':[32,640,256]},gu={'shape':[1536,256]})
    assert module.audit_ownership_counts(report,[row,row])['passed']
    assert not module.audit_ownership_counts(report,[row])['passed']


@pytest.mark.parametrize('chunk_size',[1,2,7,1024])
def test_streamed_trace_handles_boundaries_escapes_and_metadata(chunk_size):
    events=[event('kernel','a\\"b',12345,4,1),event('kernel','\u05d0',14,3,2)]
    source=json.dumps(dict(before=123456789,traceEvents=events,after={'x':[True,None,1.2345]}))
    assert list(module.trace_events(io.StringIO(source),chunk_size))==events


@pytest.mark.parametrize('source',[
    '{}','{"traceEvents": [','{"traceEvents": [1]}',
    '{"traceEvents": [], "traceEvents": []}',
    '{"traceEvents": []} trailing',
])
def test_streamed_trace_rejects_invalid_or_ambiguous_input(source):
    with pytest.raises(ValueError):
        list(module.trace_events(io.StringIO(source),2))
