"""Launcher safety checks; these do not launch CUDA or qualify performance."""
from pathlib import Path
import subprocess

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('name', ['campaign', 'training'])
def test_launcher_syntax_and_existing_output_guard(tmp_path, name):
    script = ROOT / f'tools/run_nvfp4_backward_{name}.sh'
    subprocess.run(['bash', '-n', str(script)], check=True)
    output = tmp_path / 'existing'
    output.mkdir()
    marker = output / 'keep'
    marker.write_text('original')
    args = ['/bin/false', 'selection', str(output)] if name == 'campaign' else [
        '/bin/false', 'tokens', 'designs', str(output)]
    result = subprocess.run(['bash', str(script), *args], capture_output=True, text=True)
    assert result.returncode == 2
    assert 'refusing to overwrite' in result.stderr
    assert marker.read_text() == 'original'


def test_training_stops_at_first_failed_confirmation(tmp_path):
    output = tmp_path / 'new'
    result = subprocess.run([
        'bash', str(ROOT / 'tools/run_nvfp4_backward_training.sh'),
        '/bin/false', 'tokens', 'designs', str(output),
    ], capture_output=True, text=True)
    assert result.returncode != 0
    assert 'TRAIN seed=' not in result.stdout
    assert sorted(p.name for p in output.iterdir()) == ['confirm-decoder1536.log']
