import json
from pathlib import Path
import subprocess
import sys

import pytest
import torch


@pytest.mark.parametrize('side', ['candidate','reference'])
@pytest.mark.parametrize('bad', [float('nan'),float('inf'),float('-inf')])
def test_recipe_gate_rejects_nonfinite_either_side(side,bad):
    from rtx.autotune.replacement_benchmark import error_metrics, require_recipe_metrics
    values = {'candidate':torch.ones(4), 'reference':torch.ones(4)}
    values[side][0] = bad
    metric = error_metrics(**values)
    assert not metric['finite']
    with pytest.raises(RuntimeError, match='nonfinite'):
        require_recipe_metrics({'output':metric}, ('output',))


def test_recipe_gate_near_zero_and_inconsistent_evidence():
    from rtx.autotune.replacement_benchmark import error_metrics, require_recipe_metrics
    ref = torch.zeros(4)
    passing = error_metrics(torch.full((4,),5e-8),ref)
    require_recipe_metrics({'output':passing}, ('output',))
    failing = error_metrics(torch.full((4,),2e-7),ref)
    with pytest.raises(RuntimeError,match='tolerance'):
        require_recipe_metrics({'output':failing}, ('output',))
    for key, val in [('rms',float('nan')), ('normalized_rms',0.), ('max_abs',-1.)]:
        with pytest.raises(RuntimeError):
            require_recipe_metrics({'output':{**passing,key:val}}, ('output',))
    with pytest.raises(RuntimeError,match='missing'):
        require_recipe_metrics({'output':passing}, ('output','dx'))


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('mode',['compiled_direct','outer_capture'])
def test_three_way_replacement_report(tmp_path,mode):
    output=tmp_path/'comparison'
    subprocess.run([sys.executable,'-m','rtx.autotune.replacement_benchmark',
                    '--shape','17,35,97,41','--mix','1,1,1','--samples','3',
                    '--mode',mode,'--prefer-memory','--max-slowdown','.25',
                    '--minimum-speedup','.02','--output',str(output)],check=True,timeout=300,
                   cwd=Path(__file__).resolve().parents[1])
    report=json.loads((output/'comparison.json').read_text())
    assert report['decision']['status']=='unqualified'
    assert report['decision']['selected'] is None
    assert report['decision']['policy']==dict(prefer_memory=True,max_slowdown=.25,minimum_speedup=.02)
    assert {r['backend'] for r in report['reports']}=={'bf16','bf16_recompute','mxfp8'}
    for r in report['reports']:
        assert r['status']=='ok'
        assert r['measurement_revision']==2
        assert r['contract']['numerical_gate_revision']==2
        cleanup=r['reference_workspace_cleanup']
        assert cleanup['allocated_before']-cleanup['allocated_after']==cleanup['released_bytes']
        assert cleanup['released_bytes']>=0
        assert set(r['numerics'])=={'output','dx','dgate_up','ddown'}
        assert all(v['finite'] and v['normalized_rms']<.025 for v in r['numerics'].values())
        assert r['inference_numerics']['output']['finite']
        assert r['inference_numerics']['output']['normalized_rms']<.025
        assert set(r['bf16_difference'])==set(r['numerics'])
        assert len(set(r['memory']['live_samples']))==1
        assert set(r['timings'])=={'inference_forward','training_forward','backward','training_pair','workload'}
        assert r['contract']['mix']['inference_forwards']==1
        assert r['source_hashes']
        assert 'kernels/mxfp8_mlp.py' in r['source_hashes']
        assert 'fp8_bwd.py' in r['source_hashes']
        assert 'runtime.py' in r['source_hashes']
        assert r['contract']['timing_modes']['workload']==mode
        assert bool(r['replay_validation'])==(mode=='outer_capture')
