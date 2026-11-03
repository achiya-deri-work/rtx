"""CTA cooperative max preserves final scales and outer factors exactly."""
import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA required')
@pytest.mark.parametrize('threads',[64,128,256,512])
@pytest.mark.parametrize('count',[1,3,1024])
def test_exact_current_buffers(threads,count):
    from rtx.kernels.nvfp4_posthoc import compile_posthoc_scales,PosthocScaleConfig
    from rtx.kernels.ms_eden_cta_finalizer import compile_cta_scales
    config=PosthocScaleConfig(threads,3,'mma128')
    src=torch.rand(128*96,device='cuda').to(torch.bfloat16)
    maximum=torch.rand(count,device='cuda')*448
    uniform=torch.rand(src.numel(),device='cuda')
    outputs=[torch.empty_like(src,dtype=torch.float8_e4m3fn) for _ in range(2)]
    factors=[torch.empty(1,device='cuda') for _ in range(2)]
    runners=[compile_posthoc_scales(128,96,config,count),compile_cta_scales(128,96,config,count)]
    runners[1](src,maximum,uniform,outputs[1],factors[1])
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runners[1](src,maximum,uniform,outputs[1],factors[1])
    for mode in ('random','tail','zero','range'):
        src.uniform_()
        uniform.uniform_()
        maximum.uniform_(0,448)
        if mode=='tail':maximum[-1]=1000
        if mode=='zero':maximum.zero_();src.zero_()
        if mode=='range':
            maximum.copy_(torch.logspace(-20,20,count,device='cuda'))
            src.mul_(1e20)
        runners[0](src,maximum,uniform,outputs[0],factors[0])
        graph.replay()
        assert torch.equal(outputs[0].view(torch.uint8),outputs[1].view(torch.uint8))
        assert torch.equal(factors[0].view(torch.int32),factors[1].view(torch.int32))
        expected=torch.where(maximum.max()>0,maximum.max()*(1./448.),torch.ones_like(maximum[0]))
        assert torch.equal(factors[1],expected.reshape(1))
