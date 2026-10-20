import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')


@pytest.mark.parametrize('phase,activation', [('train','silu'), ('forward','gelu')])
def test_preflight_tunes_then_reuses_in_fresh_worker(tmp_path,phase,activation):
    import rtx
    policy=rtx.AutotunePolicy(max_trials=1,max_seconds=30,warmup=1,samples=3,
        calls_per_sample=1,confirmation_repeats=0,finalist_rounds=0)
    layer=rtx.MXFP8MLP(128,128,device='cpu',autotune='balanced',activation=activation,
        tuning_policy=policy,autotune_cache_dir=tmp_path/'cache')
    spec=tmp_path/'spec.json'
    layer.export_tuning_spec(spec)
    out=tmp_path/'preflight'
    result=subprocess.run([sys.executable,'-m','rtx.autotune.mlp_preflight',
        '--shape','128,128,128,128','--spec',str(spec),'--output',str(out),
        '--phase',phase,'--activation',activation,'--check-numerics'],
        capture_output=True,text=True,timeout=180)
    logs='\n'.join(p.read_text() for p in out.rglob('worker.log'))
    assert result.returncode==0,(result.stdout,result.stderr,logs)
    reports=[json.loads((out/phase/'report.json').read_text()) for phase in ('cold','reuse')]
    assert all(r['status']=='ok' and not r['oom_events'] for r in reports)
    assert reports[1]['verify_cache']
    assert reports[0]['spec_sha256']==reports[1]['spec_sha256']
    assert reports[0]['source_hashes_before']
    assert all(r['source_hashes_before'] == r['source_hashes_after'] ==
               reports[0]['source_hashes_before'] for r in reports)
    assert all(len(set(r['live_allocated'][1:]))==1 for r in reports)
    assert all(r['live_memory_gate']['passed'] for r in reports)
    assert all(r['capacity_gate']['passed'] and r['phase']==phase
               and r['activation']==activation for r in reports)
    assert len(list((tmp_path/'cache').rglob('observations.jsonl')))==1
    numerical=json.loads((out/'numerics'/'report.json').read_text())
    assert reports[0]['selected_plan'] == reports[1]['selected_plan'] == numerical['selected_plan']
    assert numerical['numerical_gate']['passed'] and numerical['verify_cache']
    assert numerical['capacity_evidence'] is False
    assert numerical['source_hashes_before']==numerical['source_hashes_after']==reports[0]['source_hashes_before']
    assert [c['seed'] for c in numerical['cases']]==[932,933]
    names={'output','dx','dgate','ddown'} if phase=='train' else {'output'}
    assert all(set(c['metrics'])==names for c in numerical['cases'])
    if phase=='train':
        assert all((out/worker/'before-first-backward.pickle').stat().st_size > 0
                   for worker in ('cold','reuse'))
    else:
        assert all(not (out/worker/'before-first-backward.pickle').exists()
                   for worker in ('cold','reuse'))


def test_oom_snapshot_precedes_stack_unwind(tmp_path):
    # Impossible single allocation fails without occupying the device. Keep a
    # known live tensor in the throwing frame; the observer must see it before
    # the exception handler gets control.
    code='''
import json,sys,torch
from rtx.autotune.memory_diagnostics import install_oom_recorder
events=install_oom_recorder(sys.argv[1],history_entries=1000)
def fail():
    live=torch.empty(1024*1024,device="cuda",dtype=torch.uint8)
    torch.empty(torch.cuda.get_device_properties(0).total_memory*2,device="cuda",dtype=torch.uint8)
try:
    fail()
except torch.OutOfMemoryError:
    assert events and events[0]["allocated"]>=1024*1024,events
    assert "diagnostic_error" not in events[0],events
else:
    raise AssertionError("allocation unexpectedly succeeded")
'''
    result=subprocess.run([sys.executable,'-c',code,str(tmp_path)],
        capture_output=True,text=True,timeout=60)
    if result.returncode and 'nvmlInit_v2_' in result.stderr and 'INTERNAL ASSERT FAILED' in result.stderr:
        pytest.xfail('PyTorch NVML initialization aborts the OOM path before observers; requires matched driver libraries')
    assert result.returncode==0,(result.stdout,result.stderr)
    event=json.loads((tmp_path/'oom-0000.json').read_text())
    assert event['stage']=='allocator_observer_before_python_unwind'
    assert Path(event['snapshot']).stat().st_size>0
