"""Opt-in end-of-producer CTA maximum: exact semantics, unchanged tile mapping."""
from contextlib import nullcontext
from dataclasses import asdict, replace
import inspect
import json
import os
from types import SimpleNamespace

import pytest
import torch

from rtx.autotune.posthoc import (PosthocPreparationSchedule,
    make_posthoc_preparation_adapter, preparation_schedule_features)


@pytest.mark.parametrize("threads",[64,128,256,512])
def test_scope_keeps_work_mapping_and_reports_actual_output_count(threads):
    from rtx.kernels.nvfp4_posthoc import (HadamardPreparationKernel,
        PackedHadamardPreparationKernel,PosthocScaleConfig,
        compile_hadamard_preparation,compile_packed_hadamard_preparation)
    for compiler in (compile_hadamard_preparation,compile_packed_hadamard_preparation):
        assert inspect.signature(compiler).parameters["partial_max_scope"].default=="warp"
        with pytest.raises(ValueError,match="partial maximum scope"):
            compiler(128,128,partial_max_scope="atomic")
    for kernel_type in (HadamardPreparationKernel,PackedHadamardPreparationKernel):
        kwargs=dict(config=PosthocScaleConfig(threads,32),mantissa_bits=4,transpose=False)
        warp=kernel_type(1024,512,**kwargs)
        cta=kernel_type(1024,512,**kwargs,partial_max_scope="cta")
        assert warp.blocks==cta.blocks==32
        assert warp.work_warp_count==cta.work_warp_count==32*(threads//32)
        assert warp.partial_count==warp.work_warp_count
        assert cta.partial_count==cta.blocks
        assert cta.tiles==warp.tiles


def test_cta_scope_identity_features_and_geometry_are_consistent():
    adapter=make_posthoc_preparation_adapter((4096,512,512),"dx","mma128",lambda _:None)
    base=replace(adapter.initial_config,producer_load="coalesced_wide",producer_signs="register",
        a_geometry={"producer_threads":64},b_geometry={"producer_threads":256})
    cta=replace(base,partial_max_scope="cta")
    assert adapter.axes["partial_max_scope"]==("warp","cta")
    assert adapter.rejection(cta) is None
    assert adapter.config_id(cta)!=adapter.config_id(base)
    assert adapter.deserialize(adapter.serialize(cta))==cta
    assert PosthocPreparationSchedule(**json.loads(json.dumps(asdict(cta))))==cta
    a=preparation_schedule_features((4096,512,512),"dx",base)
    b=preparation_schedule_features((4096,512,512),"dx",cta)
    for operand,warps in (("a",2),("b",8)):
        assert b[f"{operand}_producer_partial_count"]*warps==a[f"{operand}_producer_partial_count"]
        assert b[f"{operand}_producer_work_warp_count"]==a[f"{operand}_producer_work_warp_count"]
        assert b[f"{operand}_logical_sign_reads"]==a[f"{operand}_logical_sign_reads"]
    assert b["a_producer_smem_bytes"]==8
    assert b["b_producer_smem_bytes"]==a["b_producer_smem_bytes"]+32
    assert b["preparation_scratch_bytes"]<a["preparation_scratch_bytes"]
    assert b["amax_logical_load_elements"]<a["amax_logical_load_elements"]
    assert b["preparation_launches"]==a["preparation_launches"]
    assert b["packed_payload_bytes"]==a["packed_payload_bytes"]
    with pytest.raises(ValueError,match="partial maximum scope"):
        PosthocPreparationSchedule(partial_max_scope="atomic")


@pytest.mark.parametrize("packed",[False,True])
def test_runtime_and_study_route_scope_and_compiler_partial_count(monkeypatch,packed):
    import rtx.fp4 as fp4
    import rtx.kernels.nvfp4_posthoc as kernels
    import rtx.experimental_ms_eden as eden
    from benchmarks.study_ms_eden_backward import prepare
    seen=[]
    counts=[]
    def producer(*args,**kwargs):
        seen.append(kwargs["partial_max_scope"])
        return (lambda *args:None),7
    def finalizer(*args):
        counts.append(args[3])
        return lambda *args:None
    monkeypatch.setattr(kernels,"compile_hadamard_preparation",producer)
    monkeypatch.setattr(kernels,"compile_packed_hadamard_preparation",producer)
    monkeypatch.setattr(kernels,"compile_posthoc_scales",finalizer)
    monkeypatch.setattr(kernels,"compile_posthoc_scale_product",lambda:lambda *args:None)
    monkeypatch.setattr(fp4,"compile_nvfp4_gemm",lambda *args:lambda *args:None)
    monkeypatch.setattr(fp4,"_packed_fp4_view",lambda value:value)
    monkeypatch.setattr(fp4,"_check_sm12x",lambda device:None)
    monkeypatch.setattr(torch.cuda,"device",lambda device:nullcontext())
    source=torch.empty(128,128,dtype=torch.bfloat16)
    codes=torch.empty(128,64,dtype=torch.uint8)
    scales=torch.empty(1024,dtype=torch.float8_e4m3fn)
    signs,uniforms,outer=torch.ones(128),torch.zeros(1024),torch.ones(1)
    config=PosthocPreparationSchedule(partial_max_scope="cta")
    eden._build_gradient(source,codes if packed else source,signs,uniforms,uniforms,
        "dx",config,(128,128,128),scales if packed else None,outer if packed else None)
    saved=SimpleNamespace(qdata=codes,scale=scales,per_tensor_scale=outer) if packed else None
    buffers=prepare(source,True,"mma128",config,packed_source=saved)
    assert seen==["cta"]*3 and counts==[7]*3
    assert buffers["partials"].shape==(7,)


run_cuda=pytest.mark.skipif(os.environ.get("RTX_RUN_CTA_MAXIMUM_CUDA")!="1",
    reason="explicit assigned idle GPU qualification opt-in")


@run_cuda
@pytest.mark.parametrize("rows,cols,threads,blocks,load,transpose,stages,packed",[
    (1024,512,64,32,"direct",False,1,False),
    (128,256,128,256,"direct",True,1,False),
    (128,128,128,256,"coalesced_wide",True,1,False),
    (512,512,256,256,"coalesced_x4",True,2,False),
    (128,128,512,256,"coalesced_x8",True,3,False),
    (128,256,128,32,"direct",False,1,True),
    (128,256,128,256,"coalesced_wide",True,1,True),
])
def test_cta_maximum_exact_payload_global_amax_and_changed_buffer_graph(
    rows,cols,threads,blocks,load,transpose,stages,packed,vpl=4):
    from benchmarks.study_ms_eden_backward import prepare
    torch.manual_seed(952)
    source=torch.randn(rows,cols,device="cuda",dtype=torch.bfloat16)
    signs=torch.empty(128,device="cuda").random_(0,2).mul_(2).sub_(1)
    saved=None
    if packed:
        saved=SimpleNamespace(qdata=torch.randint(256,(rows,cols//2),device="cuda",dtype=torch.uint8),
            scale=torch.full((rows,cols//16),.25,device="cuda",dtype=torch.float8_e4m3fn),
            per_tensor_scale=torch.ones(1,device="cuda"))
    config=PosthocPreparationSchedule(producer_threads=threads,producer_blocks=blocks,
        producer_load=load,producer_order="reduction_major" if transpose else "row_major",
        producer_signs="register",producer_stages=stages,producer_async_copy=stages>1 or (vpl>4 and transpose),
        producer_values_per_lane=vpl,producer_transpose_values_per_lane=vpl,
        producer_grouped_store_bits=32 if vpl>4 else 8,correction_decode='native')
    warp=prepare(source,transpose,"mma128",config,packed_source=saved)
    cta=prepare(source,transpose,"mma128",replace(config,partial_max_scope="cta"),packed_source=saved)
    assert warp["partials"].numel()==cta["partials"].numel()*(threads//32)
    cta["run"](signs)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        cta["run"](signs)
    for distribution in ("gaussian","sparse","outliers","zero"):
        source.normal_(0,.25)
        if distribution=="sparse": source[::2].zero_()
        if distribution=="outliers": source[0,0]=4096
        if distribution=="zero": source.zero_()
        if packed:
            saved.qdata.random_(0,256)
            if distribution=="zero": saved.qdata.zero_()
        signs.random_(0,2).mul_(2).sub_(1)
        warp["uniforms"].uniform_(); cta["uniforms"].copy_(warp["uniforms"])
        warp["run"](signs); graph.replay(); torch.cuda.synchronize()
        for key in ("packed","temporary_scales","scales","factor"):
            assert torch.equal(warp[key].view(torch.uint8),cta[key].view(torch.uint8)),key
        assert torch.isfinite(cta["partials"]).all()
        assert torch.equal(warp["partials"].amax(),cta["partials"].amax())
        assert torch.equal(cta["partials"].amax(),cta["temporary_scales"].float().amax())


@run_cuda
@pytest.mark.parametrize('vpl',[8,16])
@pytest.mark.parametrize('transpose',[False,True])
def test_grouped_cta_maximum_exact(vpl,transpose):
    test_cta_maximum_exact_payload_global_amax_and_changed_buffer_graph(
        512,384,128,64,'coalesced_x8',transpose,1,False,vpl=vpl)


@run_cuda
@pytest.mark.parametrize("phase",["dx","dw"])
@pytest.mark.parametrize("amax",["inline","staged"])
def test_cta_maximum_exact_complete_gradient_and_graph(phase,amax):
    from rtx.experimental_ms_eden import ms_eden_gradient
    torch.manual_seed(953)
    dy=torch.randn(512,256,device="cuda",dtype=torch.bfloat16)
    operand=torch.randn(256 if phase=="dx" else 512,128,device="cuda",dtype=torch.bfloat16)
    signs=torch.ones(128,device="cuda")
    ua=torch.rand(dy.numel()//16,device="cuda")
    ub=torch.rand(operand.numel()//16,device="cuda")
    base=PosthocPreparationSchedule(producer_load="coalesced_wide",producer_blocks=256,
        producer_order="row_major" if phase=="dx" else "reduction_major",amax_reduction=amax,
        a_geometry={"producer_threads":64},b_geometry={"producer_threads":256})
    cta=replace(base,partial_max_scope="cta")
    def gradient(config):
        return ms_eden_gradient(dy,operand,signs,ua,ub,phase=phase,schedule_json=json.dumps(asdict(config)))
    gradient(cta)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): captured=gradient(cta)
    for zero in (False,True):
        dy.normal_(); operand.normal_(); signs.random_(0,2).mul_(2).sub_(1)
        ua.uniform_(); ub.uniform_()
        if zero: dy.zero_(); operand.zero_()
        expected,actual=gradient(base),gradient(cta)
        graph.replay(); torch.cuda.synchronize()
        assert torch.isfinite(actual).all()
        assert torch.equal(expected,actual) and torch.equal(expected,captured)
