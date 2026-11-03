"""Inline stochastic scales match materialization with identical uniforms."""
import os
import pytest
import torch


@pytest.mark.skipif(os.getenv('RTX_RUN_INLINE_SCALES_CUDA')!='1',reason='assigned GPU only')
@pytest.mark.parametrize('shape',[(128,128,128),(256,128,384)])
@pytest.mark.parametrize('role',['producer','consumers'])
@pytest.mark.parametrize('probability',['divide','power2'])
def test_inline_scales(shape,role,probability):
    from rtx.configs.nvfp4 import NVFP4Problem,NVFP4GemmConfig
    from rtx.kernels.ms_eden_inline_scales import compile_inline_scale_gemm
    from rtx.kernels.nvfp4_posthoc import compile_posthoc_scales,PosthocScaleConfig
    from rtx.kernels.nvfp4_gemm import compile_nvfp4_gemm
    from rtx.fp4 import _packed_fp4_view
    m,n,k=shape
    config=NVFP4GemmConfig(scale_role=role,scale_layout='row_major',scale_load_vec=1)
    torch.manual_seed(1007)
    packed=[torch.randint(0,256,(rows,k//2),device='cuda',dtype=torch.uint8) for rows in (m,n)]
    temp=[torch.rand(rows,k//16,device='cuda').bfloat16() for rows in (m,n)]
    maxima=[torch.empty(1,device='cuda') for _ in range(2)]
    uniforms=[torch.rand(rows*k//16,device='cuda') for rows in (m,n)]
    scales=[torch.empty_like(t,dtype=torch.float8_e4m3fn) for t in temp]
    factors=torch.empty(2,device='cuda')
    expected=torch.empty(m,n,device='cuda',dtype=torch.bfloat16)
    actual=torch.empty_like(expected)
    finalizers=[compile_posthoc_scales(rows,k//16,PosthocScaleConfig(probability_math=probability)) for rows in (m,n)]
    reference=compile_nvfp4_gemm(NVFP4Problem(*shape),config,outer_factor_mode='pair')
    run=compile_inline_scale_gemm(NVFP4Problem(*shape),config,probability)
    q=[_packed_fp4_view(p) for p in packed]
    args=(*q,*temp,actual,(*maxima,*uniforms))
    for t,mx in zip(temp,maxima):mx.copy_(t.float().amax().reshape(1))
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):run(*args)
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):run(*args)
    for iteration in range(4):
        for index,(p,t,mx,u,s,finalize) in enumerate(zip(packed,temp,maxima,uniforms,scales,finalizers)):
            p.random_(0,256);t.uniform_(0,1);u.uniform_()
            if iteration==1:t.flatten()[::97]*=32
            if iteration==2:t.mul_(1e-30)
            if iteration==3:t.zero_()
            mx.copy_(t.float().amax().reshape(1))
            finalize(t.flatten(),mx,u,s.flatten(),factors[index:index+1])
        reference(*q,*scales,expected,factors)
        graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        assert torch.isfinite(actual).all()
        # Independent packed-format decode and FP64 GEMM, not the kernel's MMA.
        decoded=[]
        for p,s in zip(packed,scales):
            codes=torch.stack((p&15,p>>4),dim=-1).flatten(-2).long()
            exp,man=(codes>>1)&3,codes&1
            value=torch.where(exp==0,man.double()*.5,(1+man.double()*.5)*torch.exp2(exp.double()-1))
            value=torch.where((codes&8)!=0,-value,value)
            decoded.append(value*s.double().repeat_interleave(16,dim=1))
        factor=(factors[0]*factors[1]).double()
        oracle=(decoded[0]@decoded[1].T)*factor
        error=(actual.double()-oracle).square().mean().sqrt()
        assert error<=.005*oracle.square().mean().sqrt()+1e-30
    before=torch.cuda.memory_allocated()
    for _ in range(16):graph.replay()
    torch.cuda.synchronize();assert torch.cuda.memory_allocated()==before
