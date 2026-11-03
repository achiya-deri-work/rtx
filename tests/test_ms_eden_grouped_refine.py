"""Exact full-producer contract for the grouped quotient experiment."""
import os
import pytest
import torch


@pytest.mark.skipif(os.getenv("RTX_RUN_GROUPED_REFINE_CUDA")!="1",reason="assigned GPU opt-in")
@pytest.mark.parametrize("transpose",[False,True])
@pytest.mark.parametrize("vpl",[8,16])
@pytest.mark.parametrize("distribution",["gaussian","outliers","zero","wide_range"])
@pytest.mark.parametrize("streamed",[False,True],ids=['fragment','streamed'])
def test_exact_producer_and_changed_graph(transpose,vpl,distribution,streamed):
    from rtx.kernels.ms_eden_grouped_refine import compile_grouped_refine
    from rtx.kernels.nvfp4_posthoc import compile_hadamard_preparation,PosthocScaleConfig
    torch.manual_seed(977)
    source=torch.randn(256,384,device="cuda",dtype=torch.bfloat16)*.25
    signs=(torch.randint(2,(128,),device="cuda")*2-1).float()
    if distribution=="outliers":source.flatten()[::97]*=32
    if distribution=="zero":source.zero_()
    if distribution=="wide_range":
        powers=torch.randint(-100,100,source.shape,device="cuda")
        source.copy_(torch.ldexp(source.float(),powers).bfloat16())
    options=dict(producer_values_per_lane=vpl,producer_transpose_values_per_lane=vpl,
        load_mode="coalesced_x8",stages=2,async_copy=True,signs_mode="register",correction_decode="native",
        producer_grouped_store_bits=32)
    outputs=[];graphs=[];src=source.flatten()
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    for compiler in (compile_hadamard_preparation,compile_grouped_refine):
        extra={'streamed':streamed} if compiler is compile_grouped_refine else {}
        run,count=compiler(256,384,PosthocScaleConfig(128,8),4,transpose,**options,**extra)
        buffers=(torch.empty(source.numel()//2,device="cuda",dtype=torch.uint8),
            torch.empty(source.numel()//16,device="cuda",dtype=torch.bfloat16),torch.empty(count,device="cuda"))
        with torch.cuda.stream(stream):
            for _ in range(3):run(src,signs,*buffers)
        stream.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):run(src,signs,*buffers)
        outputs.append(buffers);graphs.append(graph)
    for iteration in range(3):
        if iteration:
            source.mul_(.5);signs.neg_()
        for graph in graphs:graph.replay()
        torch.cuda.synchronize()
        for expected,actual in zip(*outputs):
            torch.testing.assert_close(actual.view(torch.uint8),expected.view(torch.uint8),rtol=0,atol=0)
    before=torch.cuda.memory_allocated()
    for _ in range(16):
        for graph in graphs:graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated()==before
