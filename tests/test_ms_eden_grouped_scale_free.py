"""Grouped scale-free correction: exact codes, independent FP64 scale oracle."""
import os
import pytest
import torch


@pytest.mark.skipif(os.getenv('RTX_RUN_GROUPED_SCALE_FREE_CUDA')!='1',reason='assigned GPU only')
@pytest.mark.parametrize('transpose',[False,True])
@pytest.mark.parametrize('vpl',[8,16])
@pytest.mark.parametrize('streamed',[False,True])
def test_grouped_scale_free(transpose,vpl,streamed):
    from rtx.kernels.ms_eden_grouped_refine import compile_grouped_refine
    from rtx.kernels.nvfp4_posthoc import compile_hadamard_preparation,PosthocScaleConfig
    from tests.test_ms_eden_threshold_scale_free import oracle_bf16_bits
    torch.manual_seed(981)
    shape=(256,384)
    source=torch.zeros(shape,device='cuda',dtype=torch.bfloat16)
    signs=torch.ones(128,device='cuda')
    kwargs=dict(transpose=transpose,producer_values_per_lane=vpl,producer_transpose_values_per_lane=vpl,
        load_mode='coalesced_x8',async_copy=transpose,stages=2 if transpose else 1,
        signs_mode='register',correction_decode='native',producer_grouped_store_bits=32)
    baseline,count=compile_hadamard_preparation(*shape,PosthocScaleConfig(128,8),4,**kwargs)
    candidate,other=compile_grouped_refine(*shape,PosthocScaleConfig(128,8),4,scale_free=True,scale_free_streamed=streamed,**kwargs)
    fragment,fragment_count=compile_grouped_refine(*shape,PosthocScaleConfig(128,8),4,scale_free=True,**kwargs)
    assert count==other
    assert fragment_count==count
    def buffers():return (torch.empty(source.numel()//2,device='cuda',dtype=torch.uint8),
        torch.empty(source.numel()//16,device='cuda',dtype=torch.bfloat16),torch.empty(count,device='cuda'))
    expected,actual=buffers(),buffers()
    fragment_output=buffers()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):candidate(source.flatten(),signs,*actual)
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):candidate(source.flatten(),signs,*actual)
    for distribution in ('zero','gaussian','outliers','sparse','tiny','large'):
        source.normal_(0,.25)
        if distribution=='zero':source.zero_()
        elif distribution=='outliers':source.flatten()[::97].mul_(32)
        elif distribution=='sparse':source.mul_((torch.rand(shape,device='cuda')<.05).to(source.dtype))
        elif distribution=='tiny':source.mul_(1e-30)
        elif distribution=='large':source.mul_(1e30)
        signs.random_(0,2).mul_(2).sub_(1)
        baseline(source.flatten(),signs,*expected);graph.replay();torch.cuda.synchronize()
        fragment(source.flatten(),signs,*fragment_output);torch.cuda.synchronize()
        for lhs,rhs in zip(actual,fragment_output):
            torch.testing.assert_close(lhs.view(torch.uint8),rhs.view(torch.uint8),rtol=0,atol=0)
        torch.testing.assert_close(actual[0],expected[0],rtol=0,atol=0)
        for value in actual[1:]:assert torch.isfinite(value.float()).all()
        assert (actual[1].view(torch.int16).int()-expected[1].view(torch.int16).int()).abs().max()<=1
        # Independent double-precision H128 and EDEN scale, not GPU tree reuse.
        x=source.cpu().double()
        if transpose:x=x.T.contiguous()
        x=x.reshape(-1,128)*signs.cpu().double()
        for distance in (1,2,4,8,16,32,64):
            v=x.reshape(-1,128//(2*distance),2,distance)
            x=torch.stack((v[:,:,0]+v[:,:,1],v[:,:,0]-v[:,:,1]),dim=2).reshape(-1,128)
        x=x*(128**-.5)
        bits=actual[0].cpu().long()
        codes=torch.stack((bits&15,bits>>4),dim=1).flatten()
        levels=torch.tensor([0,.5,1,1.5,2,3,4,6],dtype=torch.float64)
        q=(levels[codes&7]*torch.where((codes&8)!=0,-1.,1.)).reshape(-1,16)
        values=x.reshape(-1,16)
        denominator=(values*q).sum(1)
        oracle=torch.where(denominator>0,(values*values).sum(1)/denominator,0.)
        oracle_words=torch.from_numpy(oracle_bf16_bits(oracle.numpy()).astype('int32'))
        actual_words=actual[1].view(torch.int16).cpu().int()
        assert (actual_words-oracle_words).abs().max()<=1
    before=torch.cuda.memory_allocated()
    for _ in range(16):graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated()==before
