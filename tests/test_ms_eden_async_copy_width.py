"""Explicit async-copy widths: structural CPU contracts and same-draw GPU tests."""
from contextlib import nullcontext
from dataclasses import asdict,replace
import inspect
import json
import os

import pytest
import torch

from rtx.autotune.posthoc import (PosthocPreparationSchedule,check_preparation_copy_alignment,
    make_posthoc_preparation_adapter,preparation_schedule_features)


@pytest.mark.parametrize("copy_bits",[32,64,128])
def test_vector_copy_indices_cover_tile_once_and_preserve_all_alignments(copy_bits):
    from rtx.kernels.nvfp4_posthoc import HadamardPreparationKernel,PosthocScaleConfig
    for threads,load in ((64,"coalesced_x4"),(128,"coalesced_wide"),(128,"coalesced_x8")):
        kernel=HadamardPreparationKernel(512,512,PosthocScaleConfig(threads,32),4,True,
            load_mode=load,async_copy=True,stages=3,producer_copy_bits=copy_bits)
        size=kernel.copy_elements
        assert kernel.async_row_stride%size==0
        vectors=128*kernel.source_tile_rows//size
        for stage in range(3):
            seen=[]
            for thread in range(threads):
                for j in range((vectors+threads-1)//threads):
                    vector=thread+j*threads
                    if vector>=vectors: continue
                    index=vector*size
                    row,column=divmod(index,kernel.source_tile_rows)
                    source=row*512+column
                    target=stage*128*kernel.async_row_stride+row*kernel.async_row_stride+column
                    assert source%size==target%size==0
                    assert column+size<=kernel.source_tile_rows
                    seen.extend(range(index,index+size))
            assert sorted(seen)==list(range(128*kernel.source_tile_rows))


def test_defaults_rejections_and_actual_source_pointer_alignment():
    from rtx.kernels.nvfp4_posthoc import (HadamardPreparationKernel,PosthocScaleConfig,
        compile_hadamard_preparation,compile_packed_hadamard_preparation)
    assert inspect.signature(compile_hadamard_preparation).parameters["producer_copy_bits"].default==32
    for bits in (False,16,256,32.):
        with pytest.raises(ValueError,match="copy bits"):
            PosthocPreparationSchedule(producer_copy_bits=bits)
        with pytest.raises(ValueError,match="copy bits"):
            HadamardPreparationKernel(128,128,PosthocScaleConfig(),4,True,producer_copy_bits=bits)
    with pytest.raises(ValueError,match="BF16 source"):
        compile_packed_hadamard_preparation(128,128,producer_copy_bits=128)
    with pytest.raises(ValueError,match="asynchronous coalesced"):
        HadamardPreparationKernel(128,128,PosthocScaleConfig(),4,True,producer_copy_bits=128)
    with pytest.raises(ValueError,match="aligned source row"):
        HadamardPreparationKernel(128,128,PosthocScaleConfig(64,32),4,True,
            load_mode="coalesced",async_copy=True,producer_copy_bits=128)
    source=torch.empty(128*128+8,dtype=torch.bfloat16)
    for bits in (32,64,128):
        schedule=PosthocPreparationSchedule(producer_load="coalesced_x8",producer_async_copy=True,
            producer_copy_bits=bits)
        check_preparation_copy_alignment(source[:128*128].view(128,128),schedule,True)
        with pytest.raises(ValueError,match="source base"):
            check_preparation_copy_alignment(source[1:1+128*128].view(128,128),schedule,True)
        check_preparation_copy_alignment(source[1:1+128*128].view(128,128),schedule,False)


def test_copy_width_identity_features_coupled_axes_and_geometry_rejection():
    adapter=make_posthoc_preparation_adapter((4096,512,512),"dx","mma128",lambda _:None)
    shared=replace(adapter.initial_config,producer_load="coalesced_x8",producer_async_copy=True,
        producer_stages=3)
    assert shared.producer_copy_bits==32
    before=preparation_schedule_features((4096,512,512),"dx",shared)
    for bits in (64,128):
        config=replace(shared,producer_copy_bits=bits)
        assert adapter.rejection(config) is None
        assert adapter.config_id(shared)!=adapter.config_id(config)
        assert adapter.deserialize(adapter.serialize(config))==config
        assert PosthocPreparationSchedule(**json.loads(json.dumps(asdict(config))))==config
        features=preparation_schedule_features((4096,512,512),"dx",config)
        assert features["a_producer_copy_bits"]==0
        assert features["a_producer_smem_bytes"]==0
        assert features["b_producer_copy_bits"]==bits
        assert features["b_producer_smem_bytes"]==2*(3*128*(32+bits//16)-bits//16)
        assert features["b_producer_async_copy_instructions"]==before["b_producer_async_copy_instructions"]*32//bits
        assert features["b_producer_partial_count"]==before["b_producer_partial_count"]
        proposed=adapter.update_fn(adapter.initial_config,"producer_copy_bits",bits)
        assert proposed.producer_async_copy and adapter.rejection(proposed) is None
    invalid=replace(shared,producer_threads=64,producer_load="coalesced",producer_copy_bits=128)
    assert adapter.rejection(invalid)[0]=="implementation_rejected"


def test_runtime_and_study_forward_copy_width(monkeypatch):
    import rtx.fp4 as fp4
    import rtx.experimental_ms_eden as eden
    import rtx.kernels.nvfp4_posthoc as kernels
    from benchmarks.study_ms_eden_backward import prepare
    seen=[]
    def producer(*args,**kwargs):
        seen.append(kwargs["producer_copy_bits"])
        return (lambda *args:None),4
    monkeypatch.setattr(kernels,"compile_hadamard_preparation",producer)
    monkeypatch.setattr(kernels,"compile_posthoc_scales",lambda *args:lambda *args:None)
    monkeypatch.setattr(kernels,"compile_posthoc_scale_product",lambda:lambda *args:None)
    monkeypatch.setattr(fp4,"compile_nvfp4_gemm",lambda *args:lambda *args:None)
    monkeypatch.setattr(fp4,"_packed_fp4_view",lambda value:value)
    monkeypatch.setattr(fp4,"_check_sm12x",lambda device:None)
    monkeypatch.setattr(torch.cuda,"device",lambda device:nullcontext())
    source=torch.empty(128,128,dtype=torch.bfloat16)
    signs,uniforms=torch.ones(128),torch.zeros(1024)
    config=PosthocPreparationSchedule(producer_copy_bits=128,producer_async_copy=True,
        producer_load="coalesced_x8")
    eden._build_gradient(source,source,signs,uniforms,uniforms,"dx",config,(128,128,128))
    prepare(source,True,"mma128",config)
    assert seen==[128]*3


run_cuda=pytest.mark.skipif(os.environ.get("RTX_RUN_ASYNC_COPY_WIDTH_CUDA")!="1",
    reason="explicit assigned idle GPU qualification opt-in")


@run_cuda
@pytest.mark.parametrize("copy_bits",[64,128])
@pytest.mark.parametrize("stages",[1,2,3])
@pytest.mark.parametrize("shape,threads,blocks,load",[
    ((128,128),128,256,"coalesced_x8"), # Many empty cooperative tail CTAs.
    ((512,512),128,32,"coalesced_x8"), # Multi-iteration and pipeline tails.
    ((128,128),64,256,"coalesced_x4"),
])
def test_copy_width_same_draw_exact_with_changed_buffer_graph(copy_bits,stages,shape,threads,blocks,load):
    from benchmarks.study_ms_eden_backward import prepare
    torch.manual_seed(958)
    source=torch.randn(*shape,device="cuda",dtype=torch.bfloat16)
    signs=torch.empty(128,device="cuda").random_(0,2).mul_(2).sub_(1)
    schedule=PosthocPreparationSchedule(producer_threads=threads,producer_blocks=blocks,
        producer_load=load,producer_order="reduction_major",producer_signs="register",
        producer_stages=stages,producer_async_copy=True)
    expected=prepare(source,True,"mma128",schedule)
    actual=prepare(source,True,"mma128",replace(schedule,producer_copy_bits=copy_bits))
    actual["run"](signs)
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph): actual["run"](signs)
    for distribution in ("gaussian","sparse","outlier","zero"):
        source.normal_(0,.25)
        if distribution=="sparse": source[::2].zero_()
        if distribution=="outlier": source[0,0]=4096
        if distribution=="zero": source.zero_()
        signs.random_(0,2).mul_(2).sub_(1)
        expected["uniforms"].uniform_();actual["uniforms"].copy_(expected["uniforms"])
        expected["run"](signs);graph.replay();torch.cuda.synchronize()
        for key in ("packed","temporary_scales","partials","scales","factor"):
            assert torch.equal(expected[key].view(torch.uint8),actual[key].view(torch.uint8)),key
        assert torch.isfinite(actual["partials"]).all()
