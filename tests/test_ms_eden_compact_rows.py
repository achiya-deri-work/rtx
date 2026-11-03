"""Compact producer row loops: exact scheduling equivalence, not new arithmetic."""
from contextlib import nullcontext
from dataclasses import asdict,replace
import inspect
import json
import os
from types import SimpleNamespace

import pytest
import torch

from rtx.autotune.posthoc import (PosthocPreparationSchedule,make_posthoc_preparation_adapter,
    preparation_schedule_features)


def test_default_constexpr_and_opt_in_controls_are_validated():
    from rtx.kernels.nvfp4_posthoc import (HadamardPreparationKernel,PosthocScaleConfig,
        compile_hadamard_preparation,compile_packed_hadamard_preparation)
    assert PosthocPreparationSchedule().producer_unroll_rows is True
    for compiler in (compile_hadamard_preparation,compile_packed_hadamard_preparation):
        assert inspect.signature(compiler).parameters["producer_unroll_rows"].default is True
    for value in (0,1,None,"false"):
        with pytest.raises(ValueError,match="unroll rows"):
            PosthocPreparationSchedule(producer_unroll_rows=value)
        with pytest.raises(ValueError,match="unroll rows"):
            HadamardPreparationKernel(128,128,PosthocScaleConfig(),4,True,producer_unroll_rows=value)
    for value in (True,False):
        kernel=HadamardPreparationKernel(128,128,PosthocScaleConfig(),4,True,
            load_mode="coalesced_x8",producer_unroll_rows=value)
        assert kernel.producer_unroll_rows is value and kernel.rows_per_warp==8


def test_row_loop_axis_identity_features_and_resources():
    adapter=make_posthoc_preparation_adapter((4096,1536,512),"dx","mma128",lambda _:None)
    base=replace(adapter.initial_config,producer_load="coalesced_x8",producer_async_copy=True,producer_stages=2)
    compact=replace(base,producer_unroll_rows=False)
    assert adapter.axes["producer_unroll_rows"]==(True,False)
    assert adapter.rejection(compact) is None
    assert adapter.config_id(compact)!=adapter.config_id(base)
    assert adapter.deserialize(adapter.serialize(compact))==compact
    assert PosthocPreparationSchedule(**json.loads(json.dumps(asdict(compact))))==compact
    before,after=(preparation_schedule_features((4096,1536,512),"dx",c) for c in (base,compact))
    assert before["b_producer_rows_per_warp"]==after["b_producer_rows_per_warp"]==8
    assert before["b_producer_unroll_rows"]==1 and after["b_producer_unroll_rows"]==0
    for key in before:
        if not key.endswith("producer_unroll_rows"): assert before[key]==after[key],key


@pytest.mark.parametrize("packed",[False,True])
def test_runtime_and_study_route_compact_row_flag(monkeypatch,packed):
    import rtx.fp4 as fp4
    import rtx.experimental_ms_eden as eden
    import rtx.kernels.nvfp4_posthoc as kernels
    from benchmarks.study_ms_eden_backward import prepare
    seen=[]
    def producer(*args,**kwargs):
        seen.append(kwargs["producer_unroll_rows"])
        return (lambda *args:None),4
    monkeypatch.setattr(kernels,"compile_hadamard_preparation",producer)
    monkeypatch.setattr(kernels,"compile_packed_hadamard_preparation",producer)
    monkeypatch.setattr(kernels,"compile_posthoc_scales",lambda *a:lambda *a:None)
    monkeypatch.setattr(kernels,"compile_posthoc_scale_product",lambda:lambda *a:None)
    monkeypatch.setattr(fp4,"compile_nvfp4_gemm",lambda *a,**kw:lambda *a:None)
    monkeypatch.setattr(fp4,"_packed_fp4_view",lambda value:value)
    monkeypatch.setattr(fp4,"_check_sm12x",lambda _:None)
    monkeypatch.setattr(torch.cuda,"device",lambda _:nullcontext())
    source=torch.empty(128,128,dtype=torch.bfloat16);codes=torch.empty(128,64,dtype=torch.uint8)
    scales=torch.empty(1024,dtype=torch.float8_e4m3fn);outer=torch.ones(1)
    signs,uniforms=torch.ones(128),torch.zeros(1024)
    config=PosthocPreparationSchedule(producer_unroll_rows=False,producer_load="coalesced_x8")
    eden._build_gradient(source,codes if packed else source,signs,uniforms,uniforms,"dx",config,
        (128,128,128),scales if packed else None,outer if packed else None)
    saved=SimpleNamespace(qdata=codes,scale=scales,per_tensor_scale=outer) if packed else None
    prepare(source,True,"mma128",config,packed_source=saved)
    assert seen==[False]*3


run_cuda=pytest.mark.skipif(os.environ.get("RTX_RUN_COMPACT_ROWS_CUDA")!="1",
    reason="explicit assigned idle GPU qualification opt-in")


@run_cuda
@pytest.mark.parametrize("load",["coalesced_x4","coalesced_x8"])
@pytest.mark.parametrize("async_copy,stages,packed",[
    (False,1,False),(True,1,False),(True,2,False),(True,3,False),(False,1,True)])
@pytest.mark.parametrize("shape",[(128,128),(512,512)])
def test_compact_rows_same_draw_exact_changed_buffer_graph(load,async_copy,stages,packed,shape):
    from benchmarks.study_ms_eden_backward import prepare
    torch.manual_seed(967)
    source=torch.randn(*shape,device="cuda",dtype=torch.bfloat16)
    signs=torch.ones(128,device="cuda")
    saved=None
    if packed:
        saved=SimpleNamespace(qdata=torch.randint(256,(shape[0],shape[1]//2),device="cuda",dtype=torch.uint8),
            scale=torch.full((shape[0],shape[1]//16),.25,device="cuda",dtype=torch.float8_e4m3fn),
            per_tensor_scale=torch.ones(1,device="cuda"))
    config=PosthocPreparationSchedule(producer_load=load,producer_signs="register",producer_blocks=256,
        producer_order="reduction_major",producer_async_copy=async_copy,producer_stages=stages)
    expected=prepare(source,True,"mma128",config,packed_source=saved)
    actual=prepare(source,True,"mma128",replace(config,producer_unroll_rows=False),packed_source=saved)
    actual["run"](signs);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual["run"](signs)
    for mode in ("gaussian","sparse","outlier","zero"):
        source.normal_(0,.25)
        if mode=="sparse":source[::2].zero_()
        if mode=="outlier":source[0,0]=4096
        if mode=="zero":source.zero_()
        if packed:
            saved.qdata.random_(0,256)
            if mode=="zero":saved.qdata.zero_()
        signs.random_(0,2).mul_(2).sub_(1)
        expected["uniforms"].uniform_();actual["uniforms"].copy_(expected["uniforms"])
        expected["run"](signs);graph.replay();torch.cuda.synchronize()
        for key in ("packed","temporary_scales","partials","scales","factor"):
            assert torch.equal(expected[key].view(torch.uint8),actual[key].view(torch.uint8)),key
        assert torch.isfinite(actual["partials"]).all()
