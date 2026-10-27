"""Row-only tile batching exposes ILP without changing per-warp tile ownership."""
from contextlib import nullcontext
from dataclasses import asdict,replace
import inspect
import json
import os

import pytest
import torch

from rtx.autotune.posthoc import (PosthocPreparationSchedule,make_posthoc_preparation_adapter,
    preparation_schedule_features)


def test_batch_mapping_preserves_every_warp_sequence_and_tail():
    from rtx.kernels.nvfp4_posthoc import HadamardPreparationKernel,PosthocScaleConfig
    for rows,cols,blocks in ((128,128,256),(512,512,96),(1024,512,32)):
        for batch in (1,2,4):
            kernel=HadamardPreparationKernel(rows,cols,PosthocScaleConfig(128,blocks),4,False,
                producer_row_tile_batch=batch)
            for warp in range(kernel.work_warp_count):
                expected=list(range(warp,kernel.tiles,kernel.work_warp_count))
                actual=[start+offset*kernel.work_warp_count
                    for start in range(warp,kernel.tiles,kernel.work_warp_count*batch)
                    for offset in range(batch) if start+offset*kernel.work_warp_count<kernel.tiles]
                assert expected==actual


def test_batch_defaults_validation_axes_resources_and_transpose_inactivity():
    from rtx.kernels.nvfp4_posthoc import compile_hadamard_preparation,compile_packed_hadamard_preparation
    assert inspect.signature(compile_hadamard_preparation).parameters["producer_row_tile_batch"].default==1
    adapter=make_posthoc_preparation_adapter((4096,1536,512),"dx","mma128",lambda _:None)
    base=replace(adapter.initial_config,producer_blocks=256,producer_threads=128)
    assert base.producer_row_tile_batch==1
    for batch in (2,4):
        config=replace(base,producer_row_tile_batch=batch)
        assert adapter.rejection(config) is None
        assert adapter.config_id(base)!=adapter.config_id(config)
        assert adapter.deserialize(adapter.serialize(config))==config
        assert PosthocPreparationSchedule(**json.loads(json.dumps(asdict(config))))==config
        f=preparation_schedule_features((4096,1536,512),"dx",config)
        assert f["a_producer_row_outer_iterations_max"]==48//batch
        assert f["a_producer_row_tile_batch"]==batch and f["b_producer_row_tile_batch"]==1
        assert f["a_producer_partial_count"]==1024 and f["a_producer_smem_bytes"]==0
        other=preparation_schedule_features((4096,1536,512),"dw",config)
        assert other["a_producer_row_tile_batch"]==other["b_producer_row_tile_batch"]==1
    assert adapter.axes["producer_row_tile_batch"]==(1,2,4)
    for value in (True,0,3,8,None):
        with pytest.raises(ValueError,match="row tile batch"):
            PosthocPreparationSchedule(producer_row_tile_batch=value)
        with pytest.raises(ValueError,match="row tile batch"):
            compile_hadamard_preparation(128,128,producer_row_tile_batch=value)
    with pytest.raises(ValueError,match="original BF16"):
        compile_packed_hadamard_preparation(128,128,producer_row_tile_batch=2)


def test_runtime_and_study_forward_row_batch(monkeypatch):
    import rtx.experimental_ms_eden as eden
    import rtx.fp4 as fp4
    import rtx.kernels.nvfp4_posthoc as kernels
    from benchmarks.study_ms_eden_backward import prepare
    seen=[]
    def producer(*args,**kwargs):
        seen.append(kwargs["producer_row_tile_batch"])
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
    config=PosthocPreparationSchedule(producer_row_tile_batch=4)
    eden._build_gradient(source,source,signs,uniforms,uniforms,"dx",config,(128,128,128))
    prepare(source,False,"mma128",config)
    assert seen==[4]*3


run_cuda=pytest.mark.skipif(os.environ.get("RTX_RUN_ROW_TILE_BATCH_CUDA")!="1",
    reason="explicit assigned idle GPU qualification opt-in")


@run_cuda
@pytest.mark.parametrize("batch",[2,4])
@pytest.mark.parametrize("order",["row_major","reduction_major"])
@pytest.mark.parametrize("shape,blocks",[((128,128),256),((512,512),96)])
def test_row_batch_exact_payload_and_changed_input_graph(batch,order,shape,blocks):
    from benchmarks.study_ms_eden_backward import prepare
    torch.manual_seed(970)
    source=torch.randn(*shape,device="cuda",dtype=torch.bfloat16)
    signs=torch.ones(128,device="cuda")
    base=PosthocPreparationSchedule(producer_blocks=blocks,producer_load="coalesced_x8",
        producer_stages=2,producer_async_copy=True,producer_order=order,producer_signs="register",
        partial_max_scope="cta" if order=="reduction_major" else "warp",correction_decode="native")
    expected=prepare(source,False,"mma128",base)
    actual=prepare(source,False,"mma128",replace(base,producer_row_tile_batch=batch))
    actual["run"](signs);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual["run"](signs)
    for mode in ("gaussian","sparse","outlier","zero"):
        source.normal_(0,.25)
        if mode=="sparse":source[::2].zero_()
        if mode=="outlier":source[0,0]=4096
        if mode=="zero":source.zero_()
        signs.random_(0,2).mul_(2).sub_(1)
        expected["uniforms"].uniform_();actual["uniforms"].copy_(expected["uniforms"])
        expected["run"](signs);graph.replay();torch.cuda.synchronize()
        for key in ("packed","temporary_scales","partials","scales","factor"):
            assert torch.equal(expected[key].view(torch.uint8),actual[key].view(torch.uint8)),key
        assert torch.isfinite(actual["partials"]).all()


@run_cuda
@pytest.mark.parametrize("batch",[2,4])
@pytest.mark.parametrize("phase",["dx","dw"])
def test_row_batch_exact_gradient_and_transposed_inactivity(batch,phase):
    from rtx.experimental_ms_eden import ms_eden_gradient
    torch.manual_seed(971)
    dy=torch.randn(512,256,device="cuda",dtype=torch.bfloat16)
    operand=torch.randn(256 if phase=="dx" else 512,128,device="cuda",dtype=torch.bfloat16)
    signs=torch.ones(128,device="cuda")
    ua=torch.rand(dy.numel()//16,device="cuda");ub=torch.rand(operand.numel()//16,device="cuda")
    base=PosthocPreparationSchedule(producer_load="coalesced_x8",producer_stages=2,producer_async_copy=True,
        correction_decode="native",outer_factor_mode="pair")
    candidate=replace(base,producer_row_tile_batch=batch)
    def run(config):
        return ms_eden_gradient(dy,operand,signs,ua,ub,phase=phase,schedule_json=json.dumps(asdict(config)))
    run(candidate);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=run(candidate)
    for zero in (False,True):
        dy.normal_();operand.normal_();ua.uniform_();ub.uniform_();signs.random_(0,2).mul_(2).sub_(1)
        if zero:dy.zero_();operand.zero_()
        expected=run(base);graph.replay();torch.cuda.synchronize()
        assert torch.equal(expected.view(torch.uint8),actual.view(torch.uint8))
        assert torch.isfinite(actual).all()
