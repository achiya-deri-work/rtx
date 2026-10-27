"""Larger TMA transfers retain byte-exact per-block preparation arithmetic."""
from dataclasses import replace
import os

import pytest


def test_tile_geometry_contract_and_design_identity():
    from rtx.kernels.ms_eden_tma_preparation import TmaPreparationKernel
    from rtx.kernels.nvfp4_posthoc import PosthocScaleConfig
    from rtx.autotune.ms_eden_design import make_ms_eden_design_adapter
    kwargs=dict(load_mode='coalesced_x8',async_copy=True,
                producer_transpose_values_per_lane=16,producer_unroll_rows=False)
    for rows in (32,64,128):
        kernel=TmaPreparationKernel(512,384,PosthocScaleConfig(128,8),4,True,
            consumer_order='xor4',tma_tile_rows=rows,**kwargs)
        assert kernel.tma_tile_rows==rows
    for rows in (16,256,True,32.0):
        with pytest.raises(ValueError,match='tile rows'):
            TmaPreparationKernel(512,384,PosthocScaleConfig(128,8),4,True,tma_tile_rows=rows,**kwargs)
    a=make_ms_eden_design_adapter((4096,1536,512),lambda _:None)
    assert a.axes['tma_tile_rows']==(32,64,128)
    assert a.config_id(a.initial_config)!=a.config_id(replace(a.initial_config,tma_tile_rows=64))
    assert a.rejection(replace(a.initial_config,tma_tile_rows=64)) is not None


@pytest.mark.skipif(os.getenv('RTX_RUN_TMA_TILES_CUDA')!='1',reason='assigned GPU only')
@pytest.mark.parametrize('shape',[(128,128),(4096,1536)])
@pytest.mark.parametrize('tile_rows',[32,64,128])
@pytest.mark.parametrize('stages',[1,2,3])
def test_tiled_producer_bytes_maximum_and_replay(shape,tile_rows,stages):
    import torch
    from rtx.kernels.ms_eden_tma_preparation import compile_tma_preparation
    from rtx.kernels.nvfp4_posthoc import compile_hadamard_preparation,PosthocScaleConfig
    rows,cols=shape
    torch.manual_seed(947)
    source=torch.randn(cols,rows,device='cuda',dtype=torch.bfloat16)*.25
    signs=(torch.randint(2,(128,),device='cuda')*2-1).float()
    kwargs=dict(load_mode='coalesced_x8',signs_mode='register',stages=stages,
        async_copy=True,producer_transpose_values_per_lane=16,producer_unroll_rows=False)
    config=PosthocScaleConfig(128,8)
    reference,count=compile_hadamard_preparation(rows,cols,config,4,True,**kwargs)
    candidate,actual_count=compile_tma_preparation(rows,cols,config,4,True,
        consumer_order='xor4',tma_tile_rows=tile_rows,**kwargs)
    assert actual_count==count
    def outputs():
        return (torch.empty(rows*cols//2,device='cuda',dtype=torch.uint8),
                torch.empty(rows*cols//16,device='cuda',dtype=torch.bfloat16),
                torch.empty(count,device='cuda'))
    expected,actual=outputs(),outputs()
    def check():
        reference(source.flatten(),signs,*expected)
        for a,b in zip(actual[:2],expected[:2]):
            torch.testing.assert_close(a.view(torch.uint8),b.view(torch.uint8),rtol=0,atol=0)
        # Tile ownership changes partial grouping, not the tensor maximum.
        torch.testing.assert_close(actual[2].amax(),expected[2].amax(),rtol=0,atol=0)
        assert torch.isfinite(actual[1]).all() and torch.isfinite(actual[2]).all()
    for _ in range(3):candidate(source.flatten(),signs,*actual)
    check()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):candidate(source.flatten(),signs,*actual)
    source.mul_(.5);signs.neg_()
    graph.replay()
    check()
