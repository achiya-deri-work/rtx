"""Fused preparation qualification, not complete-backward qualification."""
import os
import pytest
import torch

from rtx.kernels.ms_eden_h8_preparation import H8PreparationKernel
from rtx.kernels.nvfp4_posthoc import PosthocScaleConfig


def test_grouped_source_is_not_silently_accepted():
    with pytest.raises(ValueError,match="four values"):
        H8PreparationKernel(256,384,PosthocScaleConfig(),4,False,producer_values_per_lane=8)


@pytest.mark.parametrize("scope",["row","vpl4"])
def test_experiment_dispatch_and_restoration(monkeypatch,scope):
    import functools
    from rtx.kernels import nvfp4_posthoc as original_module
    from rtx.kernels import ms_eden_h8_preparation as candidate_module
    from benchmarks.experiment_ms_eden_h8_preparation import patched_preparation
    original=original_module.compile_hadamard_preparation
    @functools.wraps(original)
    def scalar(*args,**kwargs):return "scalar"
    monkeypatch.setattr(original_module,"compile_hadamard_preparation",scalar)
    monkeypatch.setattr(candidate_module,"compile_h8_preparation",lambda **kwargs:"h8")
    with pytest.raises(RuntimeError,match="sentinel"):
        with patched_preparation(scope) as records:
            run=original_module.compile_hadamard_preparation
            assert run(256,384)=="h8"
            assert run(256,384,transpose=True)==("h8" if scope=="vpl4" else "scalar")
            assert run(256,384,transpose=True,producer_transpose_values_per_lane=16)=="scalar"
            assert len(records)==3
            raise RuntimeError("sentinel")
    assert original_module.compile_hadamard_preparation is scalar


@pytest.mark.skipif(os.getenv("RTX_RUN_H8_PREPARATION_CUDA") != "1",reason="assigned GPU opt-in")
@pytest.mark.parametrize("transpose,load,stages",[(False,"direct",1),(True,"direct",1),
    (True,"coalesced_x8",1),(True,"coalesced_x8",2)])
@pytest.mark.parametrize("distribution",["gaussian","outliers","zero","arbitrary_signs"])
def test_fused_preparation_oracle_graph_and_memory(transpose,load,stages,distribution):
    from rtx.kernels.ms_eden_h8_preparation import compile_h8_preparation
    from rtx.kernels.nvfp4_posthoc import compile_hadamard_preparation
    from benchmarks.compare_ms_eden_tf32_hadamard import dense_fp64_oracle
    torch.manual_seed(973)
    source=torch.randn(256,384,device="cuda",dtype=torch.bfloat16)*.25
    signs=(torch.randint(2,(128,),device="cuda")*2-1).float()
    if distribution=="outliers": source.flatten()[::97]*=32
    if distribution=="zero": source.zero_()
    if distribution=="arbitrary_signs": signs.fill_(1.001)
    kwargs=dict(load_mode=load,stages=stages,async_copy=load!="direct",signs_mode="register",
                correction_decode="native",producer_row_load_bits=32)
    runners=[]
    outputs=[]
    src=source.flatten()
    for compile_fn in (compile_hadamard_preparation,compile_h8_preparation):
        run,count=compile_fn(256,384,PosthocScaleConfig(128,8),4,transpose,**kwargs)
        buffers=(torch.empty(source.numel()//2,device="cuda",dtype=torch.uint8),
                 torch.empty(source.numel()//16,device="cuda",dtype=torch.bfloat16),
                 torch.empty(count,device="cuda"))
        run(src,signs,*buffers)
        runners.append(run);outputs.append(buffers)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    graphs=[]
    for run,buffers in zip(runners,outputs):
        with torch.cuda.stream(stream):
            for _ in range(3):run(src,signs,*buffers)
        stream.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):run(src,signs,*buffers)
        graphs.append(graph)
    lut=torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],device="cuda")
    def check():
        oracle=dense_fp64_oracle(source.T.contiguous() if transpose else source,signs)
        decoded=[]
        for packed,scales,partials in outputs:
            assert torch.isfinite(scales).all() and torch.isfinite(partials).all()
            codes=torch.stack((packed&15,packed>>4),dim=1).flatten().long()
            decoded.append(lut[codes]*scales.float().repeat_interleave(16))
            torch.testing.assert_close(partials.max(),scales.float().max(),rtol=0,atol=0)
        denom=oracle.norm().clamp_min(1e-300)
        errors=[float((value.double()-oracle).norm()/denom) for value in decoded]
        difference=float((decoded[0].double()-decoded[1]).norm()/denom)
        assert errors[1] <= errors[0]+1e-4,errors
        assert difference <= .005,difference
        if distribution in ("zero","arbitrary_signs"):
            for a,b in zip(outputs[0],outputs[1]):
                torch.testing.assert_close(a.view(torch.uint8),b.view(torch.uint8),rtol=0,atol=0)
    check()
    source.mul_(.5);signs.neg_()
    for graph in graphs:graph.replay()
    torch.cuda.synchronize();check()
    before=torch.cuda.memory_allocated()
    for _ in range(16):
        for graph in graphs:graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated()==before
