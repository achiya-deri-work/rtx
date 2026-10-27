"""Exact arithmetic and changed-buffer graph gates for research TMA staging."""
import os
import pytest


def test_tma_probe_contract_and_partial_ownership():
    from rtx.kernels.ms_eden_tma_preparation import TmaPreparationKernel
    from rtx.kernels.nvfp4_posthoc import HadamardPreparationKernel, PosthocScaleConfig
    kwargs = dict(load_mode='coalesced_x8',async_copy=True,
                  producer_transpose_values_per_lane=16,producer_unroll_rows=False)
    for stages in (1,2,3):
        baseline = HadamardPreparationKernel(4096,1536,PosthocScaleConfig(128,256),4,True,stages=stages,**kwargs)
        candidate = TmaPreparationKernel(4096,1536,PosthocScaleConfig(128,256),4,True,stages=stages,**kwargs)
        assert (candidate.tiles,candidate.blocks,candidate.partial_count) == (baseline.tiles,baseline.blocks,baseline.partial_count)
    with pytest.raises(ValueError,match='TMA probe'):
        TmaPreparationKernel(4096,1536,PosthocScaleConfig(128,256),4,False,**kwargs)
    with pytest.raises(ValueError,match='swizzle'):
        TmaPreparationKernel(4096,1536,PosthocScaleConfig(128,256),4,True,tma_swizzle='128b',**kwargs)


@pytest.mark.skipif(os.getenv('RTX_RUN_TMA_PREPARATION_CUDA') != '1', reason='assigned GPU only')
@pytest.mark.parametrize('shape', [(128,128),(4096,1536)])
@pytest.mark.parametrize('stages', [1,2,3])
def test_tma_exact_producer_and_graph(monkeypatch,shape,stages):
    from rtx.kernels import ms_eden_bank_layout as bank
    from rtx.kernels.ms_eden_tma_preparation import compile_tma_preparation
    from tests.test_ms_eden_bank_layout import test_exact_outputs_and_current_buffer_graph
    def tma(*args, **kwargs):
        kwargs.pop('address_hoist')
        kwargs.pop('min_blocks_hint')
        kwargs.pop('butterfly_fma')
        return compile_tma_preparation(*args,**kwargs)
    monkeypatch.setattr(bank,'compile_bank_layout',tma)
    test_exact_outputs_and_current_buffer_graph(shape,stages,True,False,wide=True)


@pytest.mark.parametrize('bits', [(2,3),(0,2),(0,1,2)])
def test_xor4_register_permutation_inverts_load_order(bits):
    for lane in range(8):
        original = list(range(lane*16,(lane+1)*16))
        permutation = sum(((lane>>i)&1)<<b for i,b in enumerate(bits))
        values = [original[j^permutation] for j in range(16)]
        for bit in range(len(bits)):
            distance = 1 << bits[bit]
            for group in range(16//(2*distance)):
                for j in range(distance):
                    lo = group*2*distance+j
                    if lane & (1 << bit):
                        values[lo],values[lo+distance] = values[lo+distance],values[lo]
        assert values == original


@pytest.mark.skipif(os.getenv('RTX_RUN_TMA_PREPARATION_CUDA') != '1', reason='assigned GPU only')
@pytest.mark.parametrize('shape', [(128,128),(4096,1536)])
@pytest.mark.parametrize('stages', [1,2])
@pytest.mark.parametrize('consumer_order', ['xor4','xor5'])
def test_tma_xor4_exact(monkeypatch,shape,stages,consumer_order):
    from rtx.kernels import ms_eden_tma_preparation as tma
    original = tma.compile_tma_preparation
    def permuted(*args,**kwargs):
        return original(*args,consumer_order=consumer_order,**kwargs)
    monkeypatch.setattr(tma,'compile_tma_preparation',permuted)
    test_tma_exact_producer_and_graph(monkeypatch,shape,stages)


def test_sw64_byte_bank_model_predicts_permutation_conflicts():
    from collections import Counter
    for bits,expected in (((),8),((2,3),4),((0,2),2),((0,1,2),2)):
        for warp in range(4):
            for base in (0,4):
                for j in range(16):
                    banks=[]
                    for lane in range(32):
                        local=lane%8
                        permutation=sum(((local>>i)&1)<<b for i,b in enumerate(bits))
                        row=local*16+(j^permutation)
                        column=warp+(base+lane//8)*4
                        byte=2*(row*32+column)
                        physical=byte^((byte>>3)&0x30)
                        banks.append((physical//4)%32)
                    assert max(Counter(banks).values()) == expected


def test_row_permutation_cannot_remove_sw64_two_way_conflict_floor():
    # Current warp ownership fixes bank bit0 for every possible source row.
    # At most16 distinct banks serve32 distinct BF16 words. Adding another
    # row permutation bit cannot improve XOR5's two-way conflict floor.
    for warp in range(4):
        banks = set()
        for subrow in range(8):
            column = warp + subrow*4
            for row in range(128):
                byte = 2*(row*32+column)
                physical = byte ^ ((byte>>3)&0x30)
                bank = (physical//4)%32
                assert bank%2 == warp//2
                banks.add(bank)
        assert len(banks) == 16


def test_column_ownership_bijection_and_conflict_free_model():
    columns = [w%2+2*(s%4)+8*(w//2)+16*(s//4) for w in range(4) for s in range(8)]
    assert sorted(columns) == list(range(32))
    for warp in range(4):
        for base in (0,4):
            for j in range(16):
                banks=[]
                for lane in range(32):
                    subrow=base+lane//8
                    column=warp%2+2*(subrow%4)+8*(warp//2)+16*(subrow//4)
                    row=(lane%8)*16+(j^(lane%8))
                    byte=2*(row*32+column)
                    banks.append(((byte^((byte>>3)&0x30))//4)%32)
                assert len(set(banks)) == 32


@pytest.mark.skipif(os.getenv('RTX_RUN_TMA_PREPARATION_CUDA') != '1', reason='assigned GPU only')
@pytest.mark.parametrize('shape', [(128,128),(4096,1536)])
@pytest.mark.parametrize('stages', [1,2])
def test_tma_columns_exact(monkeypatch,shape,stages):
    from rtx.kernels import ms_eden_bank_layout as bank
    from rtx.kernels.ms_eden_tma_preparation import compile_tma_preparation
    from tests.test_ms_eden_bank_layout import test_exact_outputs_and_current_buffer_graph
    def tma(*args,**kwargs):
        for key in ('address_hoist','min_blocks_hint','butterfly_fma'):
            kwargs.pop(key)
        return compile_tma_preparation(*args,consumer_order='columns',**kwargs)
    monkeypatch.setattr(bank,'compile_bank_layout',tma)
    test_exact_outputs_and_current_buffer_graph(shape,stages,True,False,wide=True,
                                               regrouped_warp_partials=True)
