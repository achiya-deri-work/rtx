"""Independent subgroup mapping/tree proofs and opt-in exact GPU qualification."""
from contextlib import nullcontext
from dataclasses import asdict,replace
import inspect
import json
import os

import pytest
import torch

from rtx.autotune.posthoc import (PosthocPreparationSchedule,make_posthoc_preparation_adapter,
    preparation_schedule_features)


def cpu_hadamard(value,vpl):
    lanes=128//vpl
    value=value.reshape(-1,lanes,vpl).clone()
    for level in range(vpl.bit_length()-1):
        distance=1 << level
        grouped=value.reshape(value.shape[0],lanes,-1,2,distance)
        a,b=grouped[...,0,:].clone(),grouped[...,1,:].clone()
        grouped[...,0,:],grouped[...,1,:]=a+b,a-b
    for level in range(lanes.bit_length()-1):
        shift=1 << level
        other=value[:,torch.arange(lanes)^shift,:]
        value=torch.where(((torch.arange(lanes)&shift)==0)[None,:,None],value+other,other-value)
    return value.reshape(-1,128)


def test_subgroup_hadamard_and_balanced_quad_associations_match_bits():
    generator=torch.Generator().manual_seed(979)
    value=torch.ldexp(torch.randn(256,128,generator=generator),
        torch.randint(-100,100,(256,128),generator=generator)).bfloat16().float()
    value[::3,::2]=0.
    expected=cpu_hadamard(value,4)
    for vpl in (8,16):
        assert torch.equal(expected.view(torch.int32),cpu_hadamard(value,vpl).view(torch.int32))
    # The four quarter subtotals retain the original +0/two-pair construction;
    # this proves only the subsequent association/commutation, not GPU FMA IR.
    quarters=torch.ldexp(torch.rand(4096,4,generator=generator),
        torch.randint(-100,100,(4096,4),generator=generator))
    old=quarters+quarters[:,torch.tensor([1,0,3,2])]
    old=old+old[:,torch.tensor([2,3,0,1])]
    pair0=quarters[:,0]+quarters[:,1]
    pair1=quarters[:,2]+quarters[:,3]
    canonical=pair0+pair1
    assert torch.equal(old.view(torch.int32),canonical[:,None].expand_as(old).contiguous().view(torch.int32))
    assert torch.equal(canonical.view(torch.int32),(pair1+pair0).view(torch.int32))


def test_grouped_warps_keep_original_tile_sets_and_uniform_tail_loop():
    for tiles,warps in ((128,128),(2048,384),(49152,1024)):
        for vpl in (8,16):
            groups=vpl//4
            for warp in range(warps):
                old=list(range(warp,tiles,warps))
                new=[first+g*warps for first in range(warp,tiles,warps*groups)
                    for g in range(groups) if first+g*warps<tiles]
                assert old==new


def test_vpl_defaults_rejections_identity_and_effective_features():
    from rtx.kernels.nvfp4_posthoc import (HadamardPreparationKernel,PosthocScaleConfig,
        compile_hadamard_preparation,compile_packed_hadamard_preparation)
    assert inspect.signature(compile_hadamard_preparation).parameters["producer_values_per_lane"].default==4
    adapter=make_posthoc_preparation_adapter((4096,1536,512),"dx","mma128",lambda _:None)
    base=replace(adapter.initial_config,producer_blocks=256,producer_signs="register")
    before=preparation_schedule_features((4096,1536,512),"dx",base)
    assert adapter.axes["producer_values_per_lane"]==(4,8,16)
    for vpl,shuffles in ((8,16),(16,12)):
        candidate=replace(base,producer_values_per_lane=vpl)
        assert adapter.rejection(candidate) is None
        assert adapter.config_id(base)!=adapter.config_id(candidate)
        assert adapter.deserialize(adapter.serialize(candidate))==candidate
        assert PosthocPreparationSchedule(**json.loads(json.dumps(asdict(candidate))))==candidate
        after=preparation_schedule_features((4096,1536,512),"dx",candidate)
        assert after["a_producer_values_per_lane"]==vpl and after["b_producer_values_per_lane"]==4
        assert after["a_producer_hadamard_shuffle_instructions_per_group"]==shuffles
        assert after["a_producer_partial_count"]==before["a_producer_partial_count"]
        assert after["a_producer_row_outer_iterations_max"]==48//(vpl//4)
        kernel=HadamardPreparationKernel(128,128,PosthocScaleConfig(),4,True,producer_values_per_lane=vpl)
        assert kernel.values_per_lane==4
        with pytest.raises(ValueError,match="original BF16"):
            compile_packed_hadamard_preparation(128,128,producer_values_per_lane=vpl)
        for override in (dict(normalization_math="refine"),dict(producer_row_tile_batch=2)):
            assert adapter.rejection(replace(candidate,**override))[0]=="implementation_rejected"
            with pytest.raises(ValueError,match="divide normalization"):
                compile_hadamard_preparation(128,128,producer_values_per_lane=vpl,**override)
    for bad in (True,0,2,32,None):
        with pytest.raises(ValueError,match="values per lane"):
            PosthocPreparationSchedule(producer_values_per_lane=bad)


def test_runtime_and_study_route_vpl(monkeypatch):
    import rtx.experimental_ms_eden as eden
    import rtx.fp4 as fp4
    import rtx.kernels.nvfp4_posthoc as kernels
    from benchmarks.study_ms_eden_backward import prepare
    seen=[]
    def producer(*a,**kw):seen.append(kw["producer_values_per_lane"]);return (lambda *a:None),4
    monkeypatch.setattr(kernels,"compile_hadamard_preparation",producer)
    monkeypatch.setattr(kernels,"compile_posthoc_scales",lambda *a:lambda *a:None)
    monkeypatch.setattr(kernels,"compile_posthoc_scale_product",lambda:lambda *a:None)
    monkeypatch.setattr(fp4,"compile_nvfp4_gemm",lambda *a,**kw:lambda *a:None)
    monkeypatch.setattr(fp4,"_packed_fp4_view",lambda x:x)
    monkeypatch.setattr(fp4,"_check_sm12x",lambda _:None)
    monkeypatch.setattr(torch.cuda,"device",lambda _:nullcontext())
    source=torch.empty(128,128,dtype=torch.bfloat16);signs=torch.ones(128);uniforms=torch.zeros(1024)
    config=PosthocPreparationSchedule(producer_values_per_lane=16)
    eden._build_gradient(source,source,signs,uniforms,uniforms,"dx",config,(128,128,128))
    prepare(source,False,"mma128",config)
    assert seen==[16]*3


run_cuda=pytest.mark.skipif(os.environ.get("RTX_RUN_VALUES_PER_LANE_CUDA")!="1",
    reason="explicit assigned idle GPU qualification opt-in")


@run_cuda
@pytest.mark.parametrize("vpl",[8,16])
@pytest.mark.parametrize("decode",["lut","native"])
@pytest.mark.parametrize("shape",[(128,128),(512,512)])
def test_vpl_exact_payload_partial_maxima_and_changed_buffer_graph(vpl,decode,shape):
    from benchmarks.study_ms_eden_backward import prepare
    torch.manual_seed(980)
    source=torch.randn(*shape,device="cuda",dtype=torch.bfloat16);signs=torch.ones(128,device="cuda")
    base=PosthocPreparationSchedule(producer_blocks=96,producer_load="coalesced_x8",
        producer_async_copy=True,producer_stages=2,correction_decode=decode,
        producer_signs="register" if shape[0]==128 else "reload",
        producer_order="row_major" if decode=="native" else "reduction_major",
        partial_max_scope="warp" if shape[0]==128 else "cta",
        producer_row_load_bits=16 if shape[0]==128 else 64)
    expected=prepare(source,False,"mma128",base)
    actual=prepare(source,False,"mma128",replace(base,producer_values_per_lane=vpl))
    actual["run"](signs);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual["run"](signs)
    for mode in ("gaussian","sparse","outlier","range","zero"):
        source.normal_(0,.25)
        if mode=="sparse":source[::2].zero_()
        if mode=="outlier":source[0,0]=4096
        if mode=="range":
            exponents=torch.randint(-40,40,shape,device="cuda")
            source.copy_(torch.ldexp(source.float(),exponents).bfloat16())
        if mode=="zero":source.zero_();source.view(torch.int16)[0,0]=-32768
        signs.random_(0,2).mul_(2).sub_(1)
        expected["uniforms"].uniform_();actual["uniforms"].copy_(expected["uniforms"])
        expected["run"](signs);graph.replay();torch.cuda.synchronize()
        for key in ("packed","temporary_scales","partials","scales","factor"):
            assert torch.equal(expected[key].view(torch.uint8),actual[key].view(torch.uint8)),key
        assert torch.isfinite(actual["partials"]).all()


@run_cuda
@pytest.mark.parametrize("vpl",[8,16])
def test_vpl_same_draw_complete_dx_gradient_graph(vpl):
    from rtx.experimental_ms_eden import ms_eden_gradient
    torch.manual_seed(981)
    dy=torch.randn(512,256,device="cuda",dtype=torch.bfloat16)
    weight=torch.randn(256,128,device="cuda",dtype=torch.bfloat16)
    signs=torch.ones(128,device="cuda")
    ua=torch.rand(dy.numel()//16,device="cuda");ub=torch.rand(weight.numel()//16,device="cuda")
    base=PosthocPreparationSchedule(producer_load="coalesced_x8",producer_async_copy=True,
        producer_stages=2,correction_decode="native",outer_factor_mode="pair",producer_signs="register")
    candidate=replace(base,producer_values_per_lane=vpl)
    def run(config):
        return ms_eden_gradient(dy,weight,signs,ua,ub,phase="dx",schedule_json=json.dumps(asdict(config)))
    run(candidate);graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):actual=run(candidate)
    for zero in (False,True):
        dy.normal_();weight.normal_();signs.random_(0,2).mul_(2).sub_(1);ua.uniform_();ub.uniform_()
        if zero:dy.zero_();weight.zero_()
        expected=run(base);graph.replay();torch.cuda.synchronize()
        assert torch.equal(expected.view(torch.uint8),actual.view(torch.uint8))
        assert torch.isfinite(actual).all()
