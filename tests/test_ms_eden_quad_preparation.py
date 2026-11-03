"""Exact four-role producer outputs and compact CTA ownership."""
import os
import pytest


def test_compact_cta_bands():
    from rtx.kernels.ms_eden_quad_preparation import cta_bands
    for counts in ((1,1,1,1),(7,3,11,5),(256,16,128,64)):
        ownership=[]
        for start,end,lower,roles in cta_bands(counts):
            ownership.extend((roles[(p-start)%len(roles)],lower+(p-start)//len(roles))
                             for p in range(start,end))
        assert len(ownership)==sum(counts)
        assert sorted(ownership)==[(i,j) for i,n in enumerate(counts) for j in range(n)]


@pytest.mark.skipif(os.getenv('RTX_RUN_QUAD_PREPARATION_CUDA')!='1',reason='assigned GPU only')
@pytest.mark.parametrize('counts',[(7,3,11,5),(8,8,8,8)])
@pytest.mark.parametrize('vpl',[8,16])
def test_exact_quad(counts,vpl):
    import torch
    from rtx.kernels.ms_eden_quad_preparation import PreparationRole,compile_quad_preparation
    from rtx.kernels.nvfp4_posthoc import compile_hadamard_preparation,PosthocScaleConfig
    torch.manual_seed(983)
    shapes=((256,384),(384,128),(256,384),(256,128))
    roles=[];inputs=[];expected=[];actual=[];references=[]
    for i,((rows,cols),blocks) in enumerate(zip(shapes,counts)):
        options=dict(load_mode='coalesced_x8',async_copy=True,stages=1,
            tile_order='row_major' if i<2 else 'reduction_major',signs_mode='register',
            correction_decode='native',producer_values_per_lane=vpl,
            producer_transpose_values_per_lane=vpl,producer_grouped_store_bits=32)
        config=PosthocScaleConfig(128,blocks)
        role=PreparationRole(rows,cols,config,4,i!=0,**options);roles.append(role)
        run,count=compile_hadamard_preparation(rows,cols,config,4,i!=0,**options)
        assert count==role.partial_count
        source=torch.randn(rows*cols,device='cuda',dtype=torch.bfloat16)*.25
        signs=(torch.randint(2,(128,),device='cuda')*2-1).float()
        inputs.append((source,signs));references.append(run)
        def buffers():return (torch.empty(rows*cols//2,device='cuda',dtype=torch.uint8),
            torch.empty(rows*cols//16,device='cuda',dtype=torch.bfloat16),torch.empty(count,device='cuda'))
        actual.append(buffers());expected.append(buffers())
    quad=compile_quad_preparation(tuple(roles))
    args=tuple(v for pair,buffers in zip(inputs,actual) for v in (*pair,*buffers))
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):quad(*args)
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):quad(*args)
    for iteration in range(4):
        for pair,buffers,run in zip(inputs,expected,references):
            source,signs=pair
            source.normal_(0,.25);signs.random_(0,2).mul_(2).sub_(1)
            if iteration==1:source[::97]*=32
            if iteration==2:source.zero_()
            if iteration==3:source.mul_(1e-30)
            run(*pair,*buffers)
        graph.replay();torch.cuda.synchronize()
        for left,right in zip(actual,expected):
            for a,b in zip(left,right):
                torch.testing.assert_close(a.view(torch.uint8),b.view(torch.uint8),rtol=0,atol=0)
    before=torch.cuda.memory_allocated()
    for _ in range(16):graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated()==before
