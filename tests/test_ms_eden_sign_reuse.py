"""Exact sign-register reuse with changed inputs and CUDA graph replay."""
from dataclasses import replace

import pytest
import torch

pytestmark=pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


@pytest.mark.parametrize("transpose",[False,True])
@pytest.mark.parametrize("packed",[False,True])
@pytest.mark.parametrize("load",["direct","coalesced_wide"])
def test_sign_registers_preserve_changed_graph_inputs(transpose,packed,load):
    from benchmarks.study_ms_eden_backward import prepare
    from benchmarks.compare_nvfp4_packed_gemm import decode
    from rtx.autotune.posthoc import PosthocPreparationSchedule
    from types import SimpleNamespace
    torch.manual_seed(827)
    source=torch.randn(256,384,device="cuda",dtype=torch.bfloat16)*.2
    saved=None
    if packed:
        saved=SimpleNamespace(qdata=torch.randint(256,(256,192),device="cuda",dtype=torch.uint8),
            scale=(torch.rand(256,24,device="cuda")*.3).to(torch.float8_e4m3fn),
            per_tensor_scale=torch.tensor([.17],device="cuda"))
        source.copy_(decode(saved.qdata,saved.scale,saved.per_tensor_scale).bfloat16())
    config=PosthocPreparationSchedule(producer_load=load,producer_blocks=3)
    baseline=prepare(source,transpose,"mma128",config)
    candidate=prepare(source,transpose,"mma128",replace(config,producer_signs="register"),saved)
    candidate["uniforms"].copy_(baseline["uniforms"])
    signs=(torch.randint(2,(128,),device="cuda")*2-1).float()
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): candidate["run"](signs)
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream): candidate["run"](signs)
    torch.cuda.current_stream().wait_stream(stream)
    for zero in (False,True):
        signs.random_(0,2).mul_(2).sub_(1)
        if saved is not None:
            saved.qdata.random_(0,256)
            if zero: saved.qdata.zero_()
            source.copy_(decode(saved.qdata,saved.scale,saved.per_tensor_scale).bfloat16())
        else:
            source.normal_(0,.2)
            if zero: source.zero_()
        baseline["run"](signs)
        graph.replay()
        for name in ("packed","scales","factor"):
            torch.testing.assert_close(candidate[name].view(torch.uint8),
                baseline[name].view(torch.uint8),rtol=0,atol=0)
