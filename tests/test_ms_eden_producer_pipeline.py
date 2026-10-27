"""Opt-in exact qualification of async transpose-producer schedules."""
import os
import pytest
import torch


@pytest.mark.skipif(os.environ.get('RTX_RUN_PAIRED_CUDA')!='1',reason='explicit remote GPU opt-in')
@pytest.mark.parametrize('threads,load',[(64,'coalesced'),(128,'coalesced_wide'),(256,'coalesced_x4'),(512,'coalesced_x8')])
@pytest.mark.parametrize('stages',[1,2,3])
def test_buffered_producer_exact_across_geometry(threads,load,stages):
    from rtx.kernels.nvfp4_posthoc import compile_hadamard_preparation,PosthocScaleConfig
    source=torch.randn(512,512,device='cuda',dtype=torch.bfloat16)
    signs=torch.ones(128,device='cuda')
    config=PosthocScaleConfig(threads,256)
    baseline,count=compile_hadamard_preparation(512,512,config,4,True,'reduction_major',load,'register')
    candidate,other_count=compile_hadamard_preparation(512,512,config,4,True,'reduction_major',load,'register',stages,True)
    assert count==other_count
    def buffers():
        return (torch.empty(source.numel()//2,device='cuda',dtype=torch.uint8),
                torch.empty(source.numel()//16,device='cuda',dtype=torch.bfloat16),
                torch.empty(count,device='cuda'))
    expected,actual=buffers(),buffers()
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        candidate(source.flatten(),signs,*actual)
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        candidate(source.flatten(),signs,*actual)
    for zero in (False,True):
        source.normal_(0,.25)
        if zero: source.zero_()
        signs.random_(0,2).mul_(2).sub_(1)
        baseline(source.flatten(),signs,*expected)
        graph.replay(); torch.cuda.synchronize()
        for a,b in zip(expected,actual):
            torch.testing.assert_close(a.view(torch.uint8),b.view(torch.uint8),atol=0,rtol=0)
