"""Explicit BF16 row-load widths; no arithmetic, ownership, or dtype changes."""
from contextlib import nullcontext
from dataclasses import asdict,replace
import inspect
import json
import os

import pytest
import torch

from rtx.autotune.posthoc import (PosthocPreparationSchedule,check_preparation_copy_alignment,
    make_posthoc_preparation_adapter,preparation_schedule_features)


def test_width_defaults_alignment_rejections_and_scope():
    from rtx.kernels.nvfp4_posthoc import compile_hadamard_preparation,compile_packed_hadamard_preparation
    assert inspect.signature(compile_hadamard_preparation).parameters["producer_row_load_bits"].default==16
    assert PosthocPreparationSchedule().producer_row_load_bits==16
    for value in (True,0,128,None,32.):
        with pytest.raises(ValueError,match="row load bits"):
            PosthocPreparationSchedule(producer_row_load_bits=value)
        with pytest.raises(ValueError,match="row load bits"):
            compile_hadamard_preparation(128,128,producer_row_load_bits=value)
    storage=torch.empty(128*128+8,dtype=torch.bfloat16)
    for width in (32,64):
        config=PosthocPreparationSchedule(producer_row_load_bits=width)
        check_preparation_copy_alignment(storage[:128*128].view(128,128),config,False)
        with pytest.raises(ValueError,match="vector row load width"):
            check_preparation_copy_alignment(storage[1:1+128*128].view(128,128),config,False)
        # The row option is inactive for a synchronous transposed operand.
        check_preparation_copy_alignment(storage[1:1+128*128].view(128,128),config,True)
        with pytest.raises(ValueError,match="original BF16"):
            compile_packed_hadamard_preparation(128,128,producer_row_load_bits=width)


def test_width_identity_features_and_json():
    adapter=make_posthoc_preparation_adapter((4096,1536,512),"dx","mma128",lambda _:None)
    base=adapter.initial_config
    before=preparation_schedule_features((4096,1536,512),"dx",base)
    assert adapter.axes["producer_row_load_bits"]==(16,32,64)
    for width in (32,64):
        config=replace(base,producer_row_load_bits=width)
        assert adapter.rejection(config) is None
        assert adapter.config_id(config)!=adapter.config_id(base)
        assert adapter.deserialize(adapter.serialize(config))==config
        assert PosthocPreparationSchedule(**json.loads(json.dumps(asdict(config))))==config
        after=preparation_schedule_features((4096,1536,512),"dx",config)
        assert after["a_producer_row_load_bits"]==width
        assert after["b_producer_row_load_bits"]==0
        assert after["a_producer_row_load_instructions"]*width==before["a_producer_row_load_instructions"]*16
        for key in ("preparation_scratch_bytes","packed_payload_bytes","amax_logical_load_elements",
                    "a_producer_partial_count","a_producer_smem_bytes","preparation_launches"):
            assert before[key]==after[key]


def test_runtime_and_study_forward_row_load_width(monkeypatch):
    import rtx.experimental_ms_eden as eden
    import rtx.fp4 as fp4
    import rtx.kernels.nvfp4_posthoc as kernels
    from benchmarks.study_ms_eden_backward import prepare
    seen=[]
    def producer(*args,**kwargs):
        seen.append(kwargs["producer_row_load_bits"])
        return (lambda *args:None),4
    monkeypatch.setattr(kernels,"compile_hadamard_preparation",producer)
    monkeypatch.setattr(kernels,"compile_posthoc_scales",lambda *a:lambda *a:None)
    monkeypatch.setattr(kernels,"compile_posthoc_scale_product",lambda:lambda *a:None)
    monkeypatch.setattr(fp4,"compile_nvfp4_gemm",lambda *a,**kw:lambda *a:None)
    monkeypatch.setattr(fp4,"_packed_fp4_view",lambda value:value)
    monkeypatch.setattr(fp4,"_check_sm12x",lambda _:None)
    monkeypatch.setattr(torch.cuda,"device",lambda _:nullcontext())
    source=torch.empty(128,128,dtype=torch.bfloat16)
    signs,uniforms=torch.ones(128),torch.zeros(1024)
    config=PosthocPreparationSchedule(producer_row_load_bits=64)
    eden._build_gradient(source,source,signs,uniforms,uniforms,"dx",config,(128,128,128))
    prepare(source,False,"mma128",config)
    assert seen==[64]*3


run_cuda=pytest.mark.skipif(os.environ.get("RTX_RUN_ROW_VECTOR_LOAD_CUDA")!="1",
    reason="explicit assigned idle GPU qualification opt-in")


@run_cuda
@pytest.mark.parametrize("width",[32,64])
@pytest.mark.parametrize("shape",[(128,128),(512,512)])
def test_row_vector_exact_payload_and_changed_buffer_graph(width,shape):
    from benchmarks.study_ms_eden_backward import prepare
    torch.manual_seed(974)
    # A nonzero but naturally aligned storage offset must remain supported.
    storage=torch.empty(shape[0]*shape[1]+8,device="cuda",dtype=torch.bfloat16)
    source=storage[4:4+shape[0]*shape[1]].view(shape)
    signs=torch.ones(128,device="cuda")
    base=PosthocPreparationSchedule(producer_blocks=96,producer_load="coalesced_x8",
        producer_async_copy=True,producer_stages=2,producer_signs="register",correction_decode="native",
        producer_order="reduction_major" if shape[0]>128 else "row_major")
    expected=prepare(source,False,"mma128",base)
    actual=prepare(source,False,"mma128",replace(base,producer_row_load_bits=width))
    actual["run"](signs);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual["run"](signs)
    for mode in ("gaussian","sparse","outlier","zero"):
        source.normal_(0,.25)
        if mode=="sparse":source[::2].zero_()
        if mode=="outlier":source[0,0]=4096
        if mode=="zero":source.zero_();source.view(torch.int16)[0,0]=-32768
        signs.random_(0,2).mul_(2).sub_(1)
        expected["uniforms"].uniform_();actual["uniforms"].copy_(expected["uniforms"])
        expected["run"](signs);graph.replay();torch.cuda.synchronize()
        for key in ("packed","temporary_scales","partials","scales","factor"):
            assert torch.equal(expected[key].view(torch.uint8),actual[key].view(torch.uint8)),key
        assert torch.isfinite(actual["partials"]).all()


@run_cuda
@pytest.mark.parametrize("width",[32,64])
def test_row_vector_exact_dx_gradient_graph(width):
    from rtx.experimental_ms_eden import ms_eden_gradient
    torch.manual_seed(975)
    dy=torch.randn(512,256,device="cuda",dtype=torch.bfloat16)
    weight=torch.randn(256,128,device="cuda",dtype=torch.bfloat16)
    signs=torch.ones(128,device="cuda")
    ua=torch.rand(dy.numel()//16,device="cuda");ub=torch.rand(weight.numel()//16,device="cuda")
    base=PosthocPreparationSchedule(producer_load="coalesced_x8",producer_async_copy=True,
        producer_stages=2,correction_decode="native",outer_factor_mode="pair")
    candidate=replace(base,producer_row_load_bits=width)
    def run(config):
        return ms_eden_gradient(dy,weight,signs,ua,ub,phase="dx",schedule_json=json.dumps(asdict(config)))
    run(candidate);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=run(candidate)
    for zero in (False,True):
        dy.normal_();weight.normal_();ua.uniform_();ub.uniform_();signs.random_(0,2).mul_(2).sub_(1)
        if zero:dy.zero_();weight.zero_()
        expected=run(base);graph.replay();torch.cuda.synchronize()
        assert torch.equal(expected.view(torch.uint8),actual.view(torch.uint8))
        assert torch.isfinite(actual).all()
