"""Grouped staged-transpose consumers: exact mapping, unchanged staging/rings."""
from contextlib import nullcontext
from dataclasses import asdict,replace
import inspect
import json
import os

import pytest
import torch

from rtx.autotune.posthoc import (PosthocPreparationSchedule,make_posthoc_preparation_adapter,
    preparation_schedule_features)


def test_transpose_subgroups_preserve_all_staged_coordinates_and_warp_tiles():
    for warps in (2,4,8,16):
        for vpl in (8,16):
            groups,lanes=vpl//4,128//vpl
            for warp in range(warps):
                old={(sub,element):(element,warp+sub*warps)for sub in range(8)for element in range(128)}
                actual={}
                for base in range(0,8,groups):
                    for lane in range(32):
                        sub=base+lane//lanes
                        for j in range(vpl):
                            element=(lane%lanes)*vpl+j
                            key=(sub,element)
                            assert key not in actual
                            actual[key]=(element,warp+sub*warps)
                assert actual==old
                assert {warp+sub*warps for sub in range(8)}=={column for _,column in actual.values()}


def test_transpose_vpl_defaults_restrictions_identity_and_resources():
    from rtx.kernels.nvfp4_posthoc import (HadamardPreparationKernel,PosthocScaleConfig,
        compile_hadamard_preparation,compile_packed_hadamard_preparation)
    assert inspect.signature(compile_hadamard_preparation).parameters["producer_transpose_values_per_lane"].default==4
    adapter=make_posthoc_preparation_adapter((4096,1536,512),"dx","mma128",lambda _:None)
    base=replace(adapter.initial_config,producer_load="coalesced_x8",producer_async_copy=True,producer_stages=2)
    before=preparation_schedule_features((4096,1536,512),"dx",base)
    assert base.producer_transpose_values_per_lane==4
    for vpl in (8,16):
        candidate=replace(base,producer_transpose_values_per_lane=vpl)
        assert adapter.rejection(candidate) is None
        assert adapter.config_id(base)!=adapter.config_id(candidate)
        assert adapter.deserialize(adapter.serialize(candidate))==candidate
        assert PosthocPreparationSchedule(**json.loads(json.dumps(asdict(candidate))))==candidate
        after=preparation_schedule_features((4096,1536,512),"dx",candidate)
        assert after["a_producer_values_per_lane"]==4 and after["b_producer_values_per_lane"]==vpl
        assert after["b_producer_transpose_consumer_iterations"]==8//(vpl//4)
        for operand in ("a","b"):
            for suffix in ("producer_partial_count","producer_smem_bytes","producer_stages","producer_ctas"):
                assert before[f"{operand}_{suffix}"]==after[f"{operand}_{suffix}"]
        kernel=HadamardPreparationKernel(512,512,PosthocScaleConfig(),4,True,
            load_mode="coalesced_x8",producer_transpose_values_per_lane=vpl)
        assert kernel.values_per_lane==vpl and kernel.rows_per_warp==8
        row=HadamardPreparationKernel(512,512,PosthocScaleConfig(),4,False,producer_transpose_values_per_lane=vpl)
        assert row.values_per_lane==4
        assert adapter.update_fn(adapter.initial_config,"producer_transpose_values_per_lane",vpl).producer_load=="coalesced_x8"
        for kwargs in (dict(load_mode="direct"),dict(load_mode="coalesced_x4"),
                       dict(load_mode="coalesced_x8",normalization_math="refine")):
            with pytest.raises(ValueError,match="grouped transpose"):
                compile_hadamard_preparation(128,128,transpose=True,producer_transpose_values_per_lane=vpl,**kwargs)
        with pytest.raises(ValueError,match="original BF16"):
            compile_packed_hadamard_preparation(128,128,transpose=True,load_mode="coalesced_x8",
                producer_transpose_values_per_lane=vpl)
    for invalid in (True,0,2,32,None):
        with pytest.raises(ValueError,match="transpose values per lane"):
            PosthocPreparationSchedule(producer_transpose_values_per_lane=invalid)


def test_runtime_and_study_forward_transpose_vpl_without_changing_row_vpl(monkeypatch):
    import rtx.experimental_ms_eden as eden
    import rtx.fp4 as fp4
    import rtx.kernels.nvfp4_posthoc as kernels
    from benchmarks.study_ms_eden_backward import prepare
    seen=[]
    def producer(*args,**kwargs):
        seen.append((kwargs["producer_values_per_lane"],kwargs["producer_transpose_values_per_lane"]))
        return (lambda *a:None),4
    monkeypatch.setattr(kernels,"compile_hadamard_preparation",producer)
    monkeypatch.setattr(kernels,"compile_posthoc_scales",lambda *a:lambda *a:None)
    monkeypatch.setattr(kernels,"compile_posthoc_scale_product",lambda:lambda *a:None)
    monkeypatch.setattr(fp4,"compile_nvfp4_gemm",lambda *a,**kw:lambda *a:None)
    monkeypatch.setattr(fp4,"_packed_fp4_view",lambda x:x)
    monkeypatch.setattr(fp4,"_check_sm12x",lambda _:None)
    monkeypatch.setattr(torch.cuda,"device",lambda _:nullcontext())
    source=torch.empty(128,128,dtype=torch.bfloat16);signs=torch.ones(128);uniforms=torch.zeros(1024)
    config=PosthocPreparationSchedule(producer_values_per_lane=8,producer_transpose_values_per_lane=16,
        producer_load="coalesced_x8")
    eden._build_gradient(source,source,signs,uniforms,uniforms,"dx",config,(128,128,128))
    prepare(source,True,"mma128",config)
    assert seen==[(8,16)]*3


run_cuda=pytest.mark.skipif(os.environ.get("RTX_RUN_TRANSPOSE_VALUES_PER_LANE_CUDA")!="1",
    reason="explicit assigned idle GPU qualification opt-in")


@run_cuda
@pytest.mark.parametrize("vpl",[8,16])
@pytest.mark.parametrize("async_copy,stages",[(False,1),(True,1),(True,2),(True,3)])
def test_transpose_vpl_exact_staging_tail_graph_and_payload(vpl,async_copy,stages):
    check_transpose_vpl(vpl,async_copy,stages)


@run_cuda
@pytest.mark.parametrize("vpl",[8,16])
@pytest.mark.parametrize("stages",[2,3])
def test_transpose_vpl_full_ring_reuse_tail_reload_cta_and_range(vpl,stages):
    check_transpose_vpl(vpl,True,stages,shape=(512,512),blocks=3,
        signs_mode="reload",maximum_scope="cta",extra_edges=True)


def check_transpose_vpl(vpl,async_copy,stages,*,shape=(128,128),blocks=256,
                        signs_mode="register",maximum_scope="warp",extra_edges=False):
    from benchmarks.study_ms_eden_backward import prepare
    torch.manual_seed(989)
    source=torch.randn(*shape,device="cuda",dtype=torch.bfloat16);signs=torch.ones(128,device="cuda")
    base=PosthocPreparationSchedule(producer_load="coalesced_x8",producer_blocks=blocks,
        producer_async_copy=async_copy,producer_stages=stages,producer_signs=signs_mode,correction_decode="native",
        producer_order="reduction_major",partial_max_scope=maximum_scope)
    expected=prepare(source,True,"mma128",base)
    actual=prepare(source,True,"mma128",replace(base,producer_transpose_values_per_lane=vpl))
    actual["run"](signs);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual["run"](signs)
    for mode in (("gaussian","outlier","range","zero") if extra_edges else ("gaussian","outlier","zero")):
        source.normal_(0,.25)
        if mode=="outlier":source[0,0]=4096
        if mode=="range":source.copy_(torch.ldexp(source.float(),torch.randint(-40,40,shape,device="cuda")).bfloat16())
        if mode=="zero":source.zero_();source.view(torch.int16)[0,0]=-32768
        signs.random_(0,2).mul_(2).sub_(1)
        expected["uniforms"].uniform_();actual["uniforms"].copy_(expected["uniforms"])
        expected["run"](signs);graph.replay();torch.cuda.synchronize()
        for key in ("packed","temporary_scales","partials","scales","factor"):
            assert torch.equal(expected[key].view(torch.uint8),actual[key].view(torch.uint8)),key
        assert torch.isfinite(actual["partials"]).all()
