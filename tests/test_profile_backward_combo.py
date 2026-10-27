"""Compiler option isolation for the backward profiling driver."""
import json
import sys

import pytest


def test_combo_scoped_and_recorded(monkeypatch, tmp_path):
    import torch._inductor.config as config
    from benchmarks import profile_backward_subparts as driver
    output = tmp_path / 'result.json'
    monkeypatch.setattr(sys, 'argv', ['profile', '--mode', 'public', '--combo-kernels', '--output', str(output)])
    def worker(args):
        assert config.combo_kernels and config.combo_kernels_pointwise_only
        return {'tested': True}
    monkeypatch.setattr(driver, 'worker', worker)
    with config.patch(combo_kernels=False, combo_kernels_pointwise_only=False):
        driver.main()
        assert not config.combo_kernels and not config.combo_kernels_pointwise_only
    assert json.loads(output.read_text())['compiler_options'] == {
        'combo_kernels': True, 'combo_kernels_pointwise_only': True}


def test_combo_restored_on_failure(monkeypatch, tmp_path):
    import torch._inductor.config as config
    from benchmarks import profile_backward_subparts as driver
    monkeypatch.setattr(sys, 'argv', ['profile', '--mode', 'public', '--combo-kernels', '--output', str(tmp_path/'result.json')])
    def fail(args):
        raise RuntimeError('sentinel')
    monkeypatch.setattr(driver, 'worker', fail)
    with config.patch(combo_kernels=False):
        with pytest.raises(RuntimeError, match='sentinel'):
            driver.main()
        assert not config.combo_kernels


def test_combo_rejects_suite(monkeypatch, tmp_path):
    from benchmarks import profile_backward_subparts as driver
    monkeypatch.setattr(sys, 'argv', ['profile', '--combo-kernels', '--output', str(tmp_path/'result.json')])
    with pytest.raises(SystemExit) as error:
        driver.main()
    assert error.value.code == 2


def test_dual_finalizer_scoped_and_recorded(monkeypatch,tmp_path):
    from benchmarks import profile_backward_subparts as driver
    from rtx import experimental_ms_eden as eden
    original=eden._launch_operand_paired
    manifest=tmp_path/'schedule.json';manifest.write_text('{}')
    output=tmp_path/'result.json'
    monkeypatch.setattr(sys,'argv',['profile','--mode','public','--scale-finalizer','dual_both',
        '--schedule-manifest',str(manifest),'--output',str(output)])
    def worker(args):
        assert eden._launch_operand_paired is not original
        return {'tested':True}
    monkeypatch.setattr(driver,'worker',worker)
    driver.main()
    assert eden._launch_operand_paired is original
    result=json.loads(output.read_text())
    assert result['scale_finalizer']=='dual_both'
    assert len(result['finalizer_source_sha256'])==2
