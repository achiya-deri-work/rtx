"""The production matrix must work as either a module or a direct script."""
import os
from pathlib import Path
import subprocess
import sys

import pytest


@pytest.mark.parametrize('entry', [
    ('-m', 'benchmarks.validate_production'),
    ('benchmarks/validate_production.py',),
])
def test_production_help_without_cuda(entry):
    root=Path(__file__).resolve().parents[1]
    result=subprocess.run([sys.executable,*entry,'--help'],cwd=root,
        env={**os.environ,'CUDA_VISIBLE_DEVICES':''},capture_output=True,text=True,timeout=30)
    assert result.returncode==0, result.stderr
    assert '--frontend' in result.stdout and '--quick' in result.stdout
