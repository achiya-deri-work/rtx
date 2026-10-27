"""Opt-in exact integer producer scale integration; no default promotion."""
from contextlib import nullcontext
from dataclasses import asdict,replace
import inspect
import json
import os

import pytest
import torch
from rtx.autotune.posthoc import PosthocPreparationSchedule,make_posthoc_preparation_adapter


def test_scale_math_defaults_rejection_identity_features_and_compile_contract():
    from rtx.kernels.nvfp4_posthoc import (HadamardPreparationKernel,PosthocScaleConfig,
        compile_hadamard_preparation,compile_packed_hadamard_preparation)
    base=PosthocPreparationSchedule()
    candidate=replace(base,producer_scale_math="integer_rtz4")
    adapter=make_posthoc_preparation_adapter((512,512,128),"dx","mma128",lambda _:None)
    assert base.producer_scale_math=="divide"
    assert adapter.rejection(candidate) is None
    assert adapter.config_id(base)!=adapter.config_id(candidate)
    assert adapter.deserialize(adapter.serialize(candidate))==candidate
    assert adapter.extra_features_fn(candidate)["a_producer_scale_integer_rtz4"]==1.
    assert adapter.context.tags["schedule_features_revision"]==22
    assert adapter.axes["producer_scale_math"]==("divide","integer_rtz4")
    for compile_fn in (compile_hadamard_preparation,compile_packed_hadamard_preparation):
        assert inspect.signature(compile_fn).parameters["producer_scale_math"].default=="divide"
        with pytest.raises(ValueError,match="mantissa_bits=4"):
            compile_fn(128,128,mantissa_bits=3,producer_scale_math="integer_rtz4")
    for packed in (False,True):
        kernel=HadamardPreparationKernel(128,128,PosthocScaleConfig(),4,False,
            packed_source=packed,producer_scale_math="integer_rtz4")
        assert kernel.producer_scale_math=="integer_rtz4"
    with pytest.raises(ValueError,match="scale math"):
        replace(base,producer_scale_math="approx")


def test_runtime_and_study_forward_scale_math(monkeypatch):
    import rtx.experimental_ms_eden as eden
    import rtx.fp4 as fp4
    import rtx.kernels.nvfp4_posthoc as kernels
    from benchmarks.study_ms_eden_backward import prepare
    seen=[]
    def producer(*args,**kwargs):
        seen.append(kwargs["producer_scale_math"])
        return (lambda *a:None),4
    monkeypatch.setattr(kernels,"compile_hadamard_preparation",producer)
    monkeypatch.setattr(kernels,"compile_posthoc_scales",lambda *a:lambda *a:None)
    monkeypatch.setattr(kernels,"compile_posthoc_scale_product",lambda:lambda *a:None)
    monkeypatch.setattr(fp4,"compile_nvfp4_gemm",lambda *a,**kw:lambda *a:None)
    monkeypatch.setattr(fp4,"_packed_fp4_view",lambda x:x)
    monkeypatch.setattr(fp4,"_check_sm12x",lambda _:None)
    monkeypatch.setattr(torch.cuda,"device",lambda _:nullcontext())
    source=torch.empty(128,128,dtype=torch.bfloat16)
    config=PosthocPreparationSchedule(producer_scale_math="integer_rtz4")
    eden._build_gradient(source,source,torch.ones(128),torch.zeros(1024),torch.zeros(1024),
        "dx",config,(128,128,128))
    prepare(source,True,"mma128",config)
    assert seen==["integer_rtz4"]*3


run_cuda=pytest.mark.skipif(os.environ.get("RTX_RUN_PRODUCER_SCALE_MATH_CUDA")!="1",
    reason="explicit assigned idle GPU exact integration qualification")


def fill_source(source,mode):
    source.normal_(0,.25)
    if mode=="outlier":source[0,0]=4096
    if mode=="range":
        source.copy_(torch.ldexp(source.float(),torch.randint(-40,40,source.shape,device=source.device)).bfloat16())
    if mode=="zero":
        source.zero_();source.view(torch.int16)[0,0]=-32768


@run_cuda
@pytest.mark.parametrize("transpose",[False,True])
@pytest.mark.parametrize("vpl",[4,8,16])
def test_scale_math_all_prepared_arrays_exact_changed_input_graph(transpose,vpl):
    from benchmarks.study_ms_eden_backward import prepare
    torch.manual_seed(991)
    source=torch.randn(512,512,device="cuda",dtype=torch.bfloat16)
    signs=torch.ones(128,device="cuda")
    base=PosthocPreparationSchedule(producer_load="coalesced_x8",producer_blocks=3,
        producer_async_copy=True,producer_stages=2,producer_signs="register",correction_decode="native",
        producer_values_per_lane=vpl,producer_transpose_values_per_lane=vpl)
    expected=prepare(source,transpose,"mma128",base)
    actual=prepare(source,transpose,"mma128",replace(base,producer_scale_math="integer_rtz4"))
    actual["run"](signs)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual["run"](signs)
    for mode in ("gaussian","outlier","range","zero"):
        fill_source(source,mode);signs.random_(0,2).mul_(2).sub_(1)
        expected["uniforms"].uniform_();actual["uniforms"].copy_(expected["uniforms"])
        expected["run"](signs);graph.replay();torch.cuda.synchronize()
        for key in ("packed","temporary_scales","partials","scales","factor"):
            assert torch.equal(expected[key].view(torch.uint8),actual[key].view(torch.uint8)),(mode,key)


@run_cuda
@pytest.mark.parametrize("phase",["dx","dw"])
@pytest.mark.parametrize("vpl",[4,8,16])
def test_scale_math_complete_gradient_exact(phase,vpl):
    from rtx.experimental_ms_eden import ms_eden_gradient
    torch.manual_seed(992)
    dy=torch.randn(512,256,device="cuda",dtype=torch.bfloat16)
    operand=torch.randn((256,128) if phase=="dx" else (512,128),device="cuda",dtype=torch.bfloat16)
    signs=torch.ones(128,device="cuda")
    ua=torch.rand(dy.numel()//16,device="cuda");ub=torch.rand(operand.numel()//16,device="cuda")
    base=PosthocPreparationSchedule(producer_load="coalesced_x8",producer_async_copy=True,
        producer_stages=2,correction_decode="native",outer_factor_mode="pair",producer_signs="register",
        producer_values_per_lane=vpl,producer_transpose_values_per_lane=vpl)
    def run(config):
        return ms_eden_gradient(dy,operand,signs,ua,ub,phase=phase,schedule_json=json.dumps(asdict(config)))
    candidate=replace(base,producer_scale_math="integer_rtz4")
    run(candidate);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=run(candidate)
    for mode in ("gaussian","outlier","range","zero"):
        fill_source(dy,mode);fill_source(operand,mode)
        signs.random_(0,2).mul_(2).sub_(1);ua.uniform_();ub.uniform_()
        expected=run(base);graph.replay();torch.cuda.synchronize()
        assert torch.equal(expected.view(torch.uint8),actual.view(torch.uint8)),mode
