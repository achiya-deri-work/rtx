"""One input tile, two independent exact quantizations; no transpose buffer."""
import os
import pytest
import torch


@pytest.mark.skipif(os.getenv('RTX_RUN_SHARED_DY_CUDA')!='1',reason='assigned GPU only')
@pytest.mark.parametrize('shape',[(128,128),(384,256)])
@pytest.mark.parametrize('threads',[128,256])
@pytest.mark.parametrize('vpl',[8,16])
@pytest.mark.parametrize('row_vector_bits,swizzle',[(16,0),(128,0),(128,1)])
@pytest.mark.parametrize('warp_roles',[False,True])
def test_shared_dy_exact(shape,threads,vpl,row_vector_bits,swizzle,warp_roles,register_cap=255):
    from rtx.kernels.ms_eden_shared_dy import compile_shared_dy
    from rtx.kernels.nvfp4_posthoc import compile_hadamard_preparation,PosthocScaleConfig
    torch.manual_seed(984)
    source=torch.zeros(shape,device='cuda',dtype=torch.bfloat16)
    signs=[torch.ones(128,device='cuda') for _ in range(2)]
    runner,count=compile_shared_dy(*shape,threads,3,vpl,row_vector_bits,swizzle,warp_roles=warp_roles,register_cap=register_cap)
    def buffers(n):return (torch.empty(source.numel()//2,device='cuda',dtype=torch.uint8),
        torch.empty(source.numel()//16,device='cuda',dtype=torch.bfloat16),torch.empty(n,device='cuda'))
    actual=[buffers(count) for _ in range(2)]
    expected=[];references=[]
    for transpose in (False,True):
        ref,n=compile_hadamard_preparation(*shape,PosthocScaleConfig(threads,3),4,transpose,
            signs_mode='register',correction_decode='native',load_mode='coalesced_x8',
            producer_values_per_lane=vpl,producer_transpose_values_per_lane=vpl,
            producer_grouped_store_bits=32,async_copy=transpose,stages=1)
        expected.append(buffers(n));references.append(ref)
    args=(source.flatten(),signs[0],*actual[0],signs[1],*actual[1])
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):runner(*args)
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):runner(*args)
    for distribution in ('gaussian','outliers','tiny','zero'):
        source.normal_(0,.25)
        if distribution=='outliers':source.flatten()[::97]*=32
        if distribution=='tiny':source.mul_(1e-30)
        if distribution=='zero':source.zero_()
        for sign,ref,buffers_ref in zip(signs,references,expected):
            sign.random_(0,2).mul_(2).sub_(1)
            ref(source.flatten(),sign,*buffers_ref)
        graph.replay();torch.cuda.synchronize()
        for lhs,rhs in zip(actual,expected):
            for a,b in zip(lhs[:2],rhs[:2]):
                torch.testing.assert_close(a.view(torch.uint8),b.view(torch.uint8),rtol=0,atol=0)
            assert torch.isfinite(lhs[2]).all()
            torch.testing.assert_close(lhs[2].amax(),rhs[2].amax(),rtol=0,atol=0)
            torch.testing.assert_close(lhs[2].amax(),lhs[1].float().amax(),rtol=0,atol=0)
    before=torch.cuda.memory_allocated()
    for _ in range(16):graph.replay()
    torch.cuda.synchronize();assert torch.cuda.memory_allocated()==before


@pytest.mark.skipif(os.getenv('RTX_RUN_SHARED_DY_CUDA')!='1',reason='assigned GPU only')
@pytest.mark.parametrize('threads',[128,256])
@pytest.mark.parametrize('cap',[80,96])
def test_shared_roles_register_cap_exact(threads,cap):
    test_shared_dy_exact((384,256),threads,16,128,0,True,register_cap=cap)
