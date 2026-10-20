import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location('schedule_analysis', Path(__file__).parents[1]/'tools/analyze_mlp_schedule.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def row(phase, start, end, token=10, address=20, training=True):
    return dict(phase=phase, start_us=start, end_us=end, training=training,
                token={'address': token}, x={'address':address,'storage_bytes':100,'shape':[2,4]},
                gu={'address':30,'shape':[16,4]},down={'address':40,'shape':[4,8]})


def test_reused_token_storage_pair_is_consumed_once():
    result = analysis.analyze([row('forward',0,1),row('backward',3,4),
                               row('forward',5,6),row('backward',8,9)], 7)
    assert result['matched_backward_calls'] == 2
    assert result['pending_at_allocator_peak_bytes'] == 100
    assert result['pending_at_allocator_peak'][0]['forward_end_us'] == 6
    assert result['maximum_pending_input_bytes'] == 100


def test_shared_storage_deduplicated_and_no_grad_excluded():
    result = analysis.analyze([row('forward',0,1),row('forward',1,2,token=11),
        row('forward',2,3,token=12,address=21,training=False),
        row('backward',5,6,token=11),row('backward',7,8)],4)
    assert len(result['pending_at_allocator_peak']) == 2
    assert result['pending_at_allocator_peak_bytes'] == 100
    assert result['unconsumed_training_forwards'] == 0


def test_missing_forward_and_zero_interval():
    result = analysis.analyze([row('backward',0,1),row('forward',2,3),row('backward',3,4)],3)
    assert result['unmatched_backward_calls'] == 1
    assert result['matched_backward_calls'] == 1
    assert result['maximum_pending_input_bytes'] == 0
    assert result['pending_at_allocator_peak_bytes'] == 0
