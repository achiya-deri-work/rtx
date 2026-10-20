from tools.analyze_workspace_snapshot import summarize


def test_residency_is_not_peak_or_pool_label():
    report = summarize(dict(segments=[dict(segment_pool_id=(0, 1), total_size=128,
        blocks=[dict(state='active_allocated', size=64, frames=[
            dict(filename='/project/rtx/rtx/fp8_bwd.py', line=416)]),
            dict(state='active_awaiting_free', size=32, frames=[]),
            dict(state='inactive', size=32, frames=[])])], device_traces=[[
                dict(action='alloc'), dict(action='free_requested')]]))
    assert report['active_allocation_sites'] == {'fp8_bwd.py:416': 64}
    assert report['pools']['(0, 1)']['reserved_bytes'] == 128
    assert report['pools']['(0, 1)']['active_awaiting_free_bytes'] == 32
    assert report['trace_summary'][0]['events'] == 2
    assert 'not peak' in report['scope']
