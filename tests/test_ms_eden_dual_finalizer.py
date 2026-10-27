"""Disjoint paired finalization must match the existing independent kernels."""
import os
import pytest
import torch


@pytest.mark.skipif(os.getenv('RTX_RUN_DUAL_FINALIZER_CUDA')!='1',reason='assigned GPU only')
@pytest.mark.parametrize('layout',['row_major','mma128'])
@pytest.mark.parametrize('probability',['divide','power2'])
@pytest.mark.parametrize('counts',[(1,3),(1031,77)])
@pytest.mark.parametrize('inline_max,scope',[(False,'warp'),(True,'warp'),(True,'cta')])
@pytest.mark.parametrize('pair_values,bit_brackets',[(False,False),(True,False),(False,True)])
def test_dual_finalizer(layout,probability,counts,inline_max,scope,pair_values,bit_brackets):
    from rtx.kernels.ms_eden_dual_finalizer import compile_dual_finalizer
    from rtx.kernels.nvfp4_posthoc import compile_posthoc_scales,compile_partial_maximum,PosthocScaleConfig
    shapes=((129,17),(97,9)) if layout=='row_major' else ((256,16),(128,8))
    configs=[PosthocScaleConfig(128,blocks,layout,probability) for blocks in (7,3)]
    torch.manual_seed(982)
    operands=[];references=[];kernels=[]
    for (rows,groups),count,config in zip(shapes,counts,configs):
        source=torch.rand(rows*groups,device='cuda').bfloat16()
        partials=torch.rand(count,device='cuda');partials[-1]=1.
        maximum=torch.empty(1,device='cuda')
        uniforms=torch.rand(rows*groups,device='cuda')
        scales=torch.empty(rows*groups,device='cuda',dtype=torch.float8_e4m3fn)
        outer=torch.empty(1,device='cuda')
        operands.append((source,partials,maximum,uniforms,scales,outer))
        references.append((torch.empty_like(maximum),torch.empty_like(scales),torch.empty_like(outer)))
        kernels.append((compile_partial_maximum(count),compile_posthoc_scales(rows,groups,config,1)))
    run=compile_dual_finalizer(*shapes[0],*shapes[1],*configs,*counts,inline_max=inline_max,inline_scope=scope,pair_values=pair_values,bit_brackets=bit_brackets)
    args=(*operands[0],*operands[1])
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):run(*args)
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):run(*args)
    for iteration in range(3):
        for (src,part,maximum,uniforms,scales,outer),(refmax,refscale,refouter),(reduce,finalize) in zip(operands,references,kernels):
            src.mul_(.5);part.mul_(.5);uniforms.uniform_()
            if iteration==2:src.zero_();part.zero_()
            reduce(part,refmax);finalize(src,refmax,uniforms,refscale,refouter)
        graph.replay();torch.cuda.synchronize()
        for (_,_,maximum,_,scales,outer),reference in zip(operands,references):
            for actual,expected in zip((maximum,scales,outer),reference):
                torch.testing.assert_close(actual.view(torch.uint8),expected.view(torch.uint8),rtol=0,atol=0)
    before=torch.cuda.memory_allocated()
    for _ in range(16):graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated()==before


def test_inline_max_is_opt_in_and_strictly_boolean():
    from rtx.kernels.ms_eden_dual_finalizer import DualFinalizer
    from rtx.kernels.nvfp4_posthoc import PosthocScaleConfig
    c=PosthocScaleConfig()
    assert not DualFinalizer(128,8,128,8,c,c,1024,256).inline_max
    assert DualFinalizer(128,8,128,8,c,c,1024,256,True).inline_max
    with pytest.raises(ValueError,match='bool'):
        DualFinalizer(128,8,128,8,c,c,1024,256,'true')
    with pytest.raises(ValueError,match='scope'):
        DualFinalizer(128,8,128,8,c,c,1024,256,True,'unknown')
    with pytest.raises(ValueError,match='requires'):
        DualFinalizer(128,8,128,8,c,c,1024,256,False,'cta')
    with pytest.raises(ValueError,match='separate'):
        DualFinalizer(128,8,128,8,c,c,1024,256,pair_values=True,bit_brackets=True)


@pytest.mark.skipif(os.getenv('RTX_RUN_DUAL_FINALIZER_CUDA')!='1',reason='assigned GPU only')
def test_bit_brackets_fp32_boundaries_exact():
    import cutlass.cute as cute
    from cutlass import Float32,Float8E4M3FN
    from rtx.kernels.ms_eden_dual_finalizer import FinalizerLeg
    from rtx.kernels.nvfp4_posthoc import PosthocScaleKernel,PosthocScaleConfig
    levels=torch.arange(127,dtype=torch.uint8,device='cuda').view(torch.float8_e4m3fn).float()
    edges=torch.cat((levels,(levels[:-1]+levels[1:])*.5))
    values=torch.cat((edges,torch.nextafter(edges,torch.full_like(edges,float('inf'))),
        torch.nextafter(edges,torch.full_like(edges,-float('inf'))))).clamp(0,448)
    values=torch.cat((values,torch.tensor([-0.],device='cuda')))
    size=values.numel();cfg=PosthocScaleConfig(128,7)
    def fake(dtype,n):return cute.runtime.make_fake_tensor(dtype,(n,),stride=(1,),assumed_align=4)
    inputs=(fake(Float32,size),fake(Float32,1),fake(Float32,size),
            fake(Float8E4M3FN,size),fake(Float32,1),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True))
    kernels=[cute.compile(k,*inputs,options='--enable-tvm-ffi --opt-level 3') for k in (
        PosthocScaleKernel(1,size,cfg,1),FinalizerLeg(1,size,cfg,1,bit_brackets=True))]
    maximum=torch.full((1,),448.,device='cuda')
    outputs=[(torch.empty(size,device='cuda',dtype=torch.float8_e4m3fn),
              torch.empty(1,device='cuda')) for _ in kernels]
    uniforms=torch.empty_like(values)
    for draw in (0.,.5,1.-2.**-24):
        uniforms.fill_(draw)
        for run,(out,outer) in zip(kernels,outputs):run(values,maximum,uniforms,out,outer)
        torch.cuda.synchronize()
        assert outputs[0][1].item()==1.
        assert torch.equal(outputs[0][0].view(torch.uint8),outputs[1][0].view(torch.uint8))
