"""Instruction-only row butterfly variant must preserve every stored bit."""
import os
import pytest


@pytest.mark.skipif(os.getenv('RTX_RUN_ROW_FMA_CUDA') != '1', reason='assigned GPU only')
@pytest.mark.parametrize('shape', [(128,128), (4096,1536)])
@pytest.mark.parametrize('values', [8,16])
def test_row_fma_exact_and_changed_input_graph(shape, values):
    import torch
    from rtx.kernels.ms_eden_bank_layout import compile_row_fma
    from rtx.kernels.nvfp4_posthoc import compile_hadamard_preparation, PosthocScaleConfig
    rows,cols=shape
    torch.manual_seed(953)
    source=torch.randn(rows*cols,device='cuda',dtype=torch.bfloat16)*.25
    signs=(torch.randint(2,(128,),device='cuda')*2-1).float()
    kw=dict(producer_values_per_lane=values, correction_decode='native',
            signs_mode='register', producer_grouped_store_bits=64)
    config=PosthocScaleConfig(128,256)
    ref,count=compile_hadamard_preparation(rows,cols,config,4,False,**kw)
    run,n=compile_row_fma(rows,cols,config,4,False,**kw)
    assert count==n
    def outputs():
        return (torch.empty(rows*cols//2,device='cuda',dtype=torch.uint8),
                torch.empty(rows*cols//16,device='cuda',dtype=torch.bfloat16),
                torch.empty(count,device='cuda'))
    actual,expected=outputs(),outputs()
    def check():
        ref(source,signs,*expected)
        for a,b in zip(actual,expected):
            torch.testing.assert_close(a.view(torch.uint8),b.view(torch.uint8),rtol=0,atol=0)
        assert torch.isfinite(actual[1]).all() and torch.isfinite(actual[2]).all()
    run(source,signs,*actual)
    check()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):run(source,signs,*actual)
    for mode in ('zero','outliers','exponents'):
        if mode=='zero': source.zero_()
        elif mode=='outliers':
            source.normal_().mul_(.25);source[::127]=32
        else:
            source.copy_(torch.ldexp(torch.randn_like(source).float(),
                torch.randint(-40,41,source.shape,device='cuda')).bfloat16())
        signs.neg_()
        graph.replay()
        check()
