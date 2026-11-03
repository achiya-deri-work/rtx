"""Byte-exact packed conversion, format-boundary oracle and graph replay."""
import math

import pytest
import torch


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('layout', ['row_major','mma128'])
@pytest.mark.parametrize('partial_count', [1,1024])
@pytest.mark.parametrize('probability_math', ['divide','power2'])
def test_boundary_oracle(layout, partial_count, probability_math):
    from rtx.kernels.nvfp4_posthoc import PosthocScaleConfig, compile_posthoc_scales
    from rtx.kernels.ms_eden_paired_finalizer import compile_paired_scales
    from tests.test_ms_eden_probability_math import finalizer_inputs
    source, probabilities, pairs = finalizer_inputs()
    source, probabilities = source.cuda(), probabilities.cuda()
    maximum = torch.zeros(partial_count, device='cuda')
    maximum[0] = 448.
    uniforms = torch.zeros_like(probabilities)
    config = PosthocScaleConfig(128,128,layout,probability_math)
    runners = [compile_posthoc_scales(128,8,config,partial_count),
               compile_paired_scales(128,8,config,partial_count)]
    outputs = [torch.empty(1024,device='cuda',dtype=torch.float8_e4m3fn) for _ in runners]
    factors = [torch.empty(1,device='cuda') for _ in runners]
    runners[1](source,maximum,uniforms,outputs[1],factors[1])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        runners[1](source,maximum,uniforms,outputs[1],factors[1])
    for direction in (-math.inf,0.,math.inf):
        target = probabilities if direction==0 else torch.nextafter(probabilities,torch.full_like(probabilities,direction))
        uniforms.copy_(target.clamp(0.,1.-2.**-24))
        runners[0](source,maximum,uniforms,outputs[0],factors[0])
        graph.replay()
        assert torch.equal(outputs[0].view(torch.uint8),outputs[1].view(torch.uint8))
        assert torch.equal(factors[0],factors[1])
        logical = outputs[1].view(torch.uint8)
        if layout=='mma128':
            logical=logical.reshape(1,1,2,32,4,4).permute(0,4,3,1,2,5).contiguous().reshape(-1)
        expected=torch.tensor([pair[int(draw<p)] if pair is not None else 0
            for pair,draw,p in zip(pairs,uniforms.cpu().tolist(),probabilities.cpu().tolist())],
            dtype=torch.uint8,device='cuda')
        eligible=torch.tensor([pair is not None for pair in pairs],device='cuda')
        assert torch.equal(logical[eligible],expected[eligible])
    source.zero_()
    maximum.zero_()
    graph.replay()
    assert (outputs[1].view(torch.uint8)==0).all() and factors[1].item()==1.


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('shape', [(129,10),(4096,96)])
def test_grid_stride_and_current_buffers(shape):
    from rtx.kernels.nvfp4_posthoc import PosthocScaleConfig, compile_posthoc_scales
    from rtx.kernels.ms_eden_paired_finalizer import compile_paired_scales
    rows, groups = shape
    config = PosthocScaleConfig(128,3,'row_major')
    src = torch.rand(rows*groups,device='cuda').to(torch.bfloat16)
    maximum = torch.tensor([2.],device='cuda')
    uniform = torch.rand(rows*groups,device='cuda')
    outputs = [torch.empty_like(src,dtype=torch.float8_e4m3fn) for _ in range(2)]
    factors = [torch.empty(1,device='cuda') for _ in range(2)]
    runners = [compile_posthoc_scales(rows,groups,config),compile_paired_scales(rows,groups,config)]
    for seed in (941,942):
        torch.manual_seed(seed)
        src.uniform_(0,2)
        uniform.uniform_()
        for fn,out,factor in zip(runners,outputs,factors):
            fn(src,maximum,uniform,out,factor)
        assert torch.equal(outputs[0].view(torch.uint8),outputs[1].view(torch.uint8))
        assert torch.equal(factors[0],factors[1])
