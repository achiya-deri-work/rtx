"""Research VPL32 layout and exact producer qualification."""
import os
import pytest


def test_group32_pair_alignment_bijection_and_conflict_free_loads():
    from rtx.kernels.ms_eden_group32 import group32_offset as offset
    assert sorted(offset(r,c) for r in range(128) for c in range(32)) == list(range(4096))
    for r in range(128):
        for c in range(0,32,2):
            assert offset(r,c)%2 == 0 and offset(r,c+1) == offset(r,c)+1
        for c in range(32):
            assert offset(r,c) == offset((r//32)*32+r%2,c)+(r%32//2)*64
    for warp in range(4):
        for j in range(32):
            banks = [(offset((lane%4)*32+j,warp+(lane//4)*4)//2)%32 for lane in range(32)]
            assert len(set(banks)) == 32


def test_combined_store_ownership_matches_two_half_fragments():
    for tile in (0, 1, 31, 127):
        packed, scales = [], []
        for lane in range(4):
            start = tile*64 + lane*16
            scale_start = tile*8 + lane*2
            assert start % 16 == 0 and scale_start % 2 == 0
            old = [tile*64 + (lane*2+half)*8+j for half in range(2) for j in range(8)]
            assert old == list(range(start, start+16))
            packed.extend(old)
            scales.extend([scale_start, scale_start+1])
        assert packed == list(range(tile*64, (tile+1)*64))
        assert scales == list(range(tile*8, (tile+1)*8))


@pytest.mark.skipif(os.getenv('RTX_RUN_GROUP32_CUDA') != '1', reason='assigned GPU only')
@pytest.mark.parametrize('shape', [(128,128), (4096,1536)])
@pytest.mark.parametrize('stages', [1,2])
def test_group32_exact_source_signs_scales_partials_and_graph(monkeypatch,shape,stages):
    from rtx.kernels import ms_eden_bank_layout as bank
    from rtx.kernels.ms_eden_group32 import compile_group32
    from tests.test_ms_eden_bank_layout import test_exact_outputs_and_current_buffer_graph
    # Reuse the independently retained producer comparison, not its arithmetic.
    monkeypatch.setattr(bank, 'compile_bank_layout', compile_group32)
    test_exact_outputs_and_current_buffer_graph(shape,stages,True,False,wide=True)


@pytest.mark.skipif(os.getenv('RTX_RUN_GROUP32_CUDA') != '1', reason='assigned GPU only')
@pytest.mark.parametrize('shape', [(128,128), (4096,1536)])
def test_group32_reloaded_signs_exact(monkeypatch, shape):
    from rtx.kernels import ms_eden_bank_layout as bank
    from rtx.kernels.ms_eden_group32 import compile_group32
    from tests.test_ms_eden_bank_layout import test_exact_outputs_and_current_buffer_graph
    def reload(*args, **kwargs):
        kwargs['signs_mode'] = 'reload'
        return compile_group32(*args, **kwargs)
    monkeypatch.setattr(bank, 'compile_bank_layout', reload)
    test_exact_outputs_and_current_buffer_graph(shape,1,True,False,wide=True)


@pytest.mark.skipif(os.getenv('RTX_RUN_GROUP32_CUDA') != '1', reason='assigned GPU only')
@pytest.mark.parametrize('shape', [(128,128), (4096,1536)])
@pytest.mark.parametrize('signs_mode', ['register', 'reload'])
def test_group32_combined_stores_exact(monkeypatch, shape, signs_mode):
    from rtx.kernels import ms_eden_bank_layout as bank
    from rtx.kernels.ms_eden_group32 import compile_group32
    from tests.test_ms_eden_bank_layout import test_exact_outputs_and_current_buffer_graph
    def combined(*args, **kwargs):
        kwargs['signs_mode'] = signs_mode
        return compile_group32(*args, combined_stores=True, **kwargs)
    monkeypatch.setattr(bank, 'compile_bank_layout', combined)
    test_exact_outputs_and_current_buffer_graph(shape,1,True,False,wide=True)
