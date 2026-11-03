"""Exhaustive finite-BF16 sign identity and complete-producer exactness."""
import os
import pytest


def test_all_finite_bf16_sign_bits_match_fp32_multiplication():
    import numpy as np
    words = np.arange(65536, dtype=np.uint32) << 16
    words = words[(words & np.uint32(0x7f800000)) != np.uint32(0x7f800000)]
    values = words.view(np.float32)
    for sign in (1., -1.):
        expected = (values * np.float32(sign)).view(np.uint32)
        actual = words ^ np.uint32(0 if sign > 0 else 0x80000000)
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.skipif(os.getenv('RTX_RUN_PACKED_SIGNS_CUDA') != '1', reason='assigned GPU only')
@pytest.mark.parametrize('shape', [(128, 128), (4096, 1536)])
@pytest.mark.parametrize('stages', [1, 2, 3])
@pytest.mark.parametrize('unroll', [True, False])
def test_packed_signs_exact_producer_and_changed_graph(monkeypatch, shape, stages, unroll):
    import test_ms_eden_bank_layout as original
    import rtx.kernels.ms_eden_bank_layout as module
    from rtx.kernels.ms_eden_packed_signs import compile_packed_rademacher
    monkeypatch.setattr(module, 'compile_bank_layout', compile_packed_rademacher)
    original.test_exact_outputs_and_current_buffer_graph(shape, stages, True, unroll)


@pytest.mark.skipif(os.getenv('RTX_RUN_PACKED_SIGNS_CUDA') != '1', reason='assigned GPU only')
@pytest.mark.parametrize('cap', [64, 72])
def test_packed_signs_register_ceiling(monkeypatch, cap):
    import test_ms_eden_bank_layout as original
    import rtx.kernels.ms_eden_bank_layout as module
    from rtx.kernels.ms_eden_packed_signs import compile_packed_rademacher
    def compiler(*args, **kwargs):
        return compile_packed_rademacher(*args, register_cap=cap, **kwargs)
    monkeypatch.setattr(module, 'compile_bank_layout', compiler)
    original.test_exact_outputs_and_current_buffer_graph((4096, 1536), 1, True, False)


@pytest.mark.skipif(os.getenv('RTX_RUN_PACKED_SIGNS_CUDA') != '1', reason='assigned GPU only')
@pytest.mark.parametrize('hint', [7, 8])
def test_packed_signs_occupancy_hint(monkeypatch, hint):
    import test_ms_eden_bank_layout as original
    import rtx.kernels.ms_eden_bank_layout as module
    from rtx.kernels.ms_eden_packed_signs import compile_packed_rademacher
    def compiler(*args, **kwargs):
        return compile_packed_rademacher(*args, min_blocks_hint=hint, **kwargs)
    monkeypatch.setattr(module, 'compile_bank_layout', compiler)
    original.test_exact_outputs_and_current_buffer_graph((4096, 1536), 1, True, False)
