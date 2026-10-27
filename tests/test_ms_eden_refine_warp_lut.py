"""Exact register-only reciprocal LUT; no approximate/new numerical policy.

Each lane computes one RN reciprocal at startup; all lanes execute each lookup
before any eligibility branch. GPU checks are explicitly opt-in and exact.
"""
from dataclasses import replace
import inspect
import json
import os

import numpy as np
import pytest
import torch


def test_all_reciprocal_words_and_scale_indices_are_exact():
    lane=np.arange(32,dtype=np.uint32)
    denominator=(np.uint32(0x3f800000)|((lane&15)<<19)).view(np.float32)
    cached=np.divide(np.float32(1),denominator,dtype=np.float32)
    expected=(np.float64(16)/(16+(lane&15)).astype(np.float64)).astype(np.float32)
    np.testing.assert_array_equal(cached.view(np.uint32),expected.view(np.uint32))
    np.testing.assert_array_equal(cached[:16].view(np.uint32),cached[16:].view(np.uint32))
    for exponent in (1,2,37,127,217,253):
        scale=(np.uint32(exponent<<23)|(np.arange(16,dtype=np.uint32)<<19)).view(np.float32)
        np.testing.assert_array_equal((scale.view(np.uint32)>>19)&15,np.arange(16))


def test_mode_identity_vpl4_restriction_and_literal_operand_contract():
    from rtx.autotune.posthoc import PosthocPreparationSchedule,make_posthoc_preparation_adapter
    from rtx.kernels.nvfp4_posthoc import HadamardPreparationKernel,PosthocScaleConfig
    from benchmarks.compare_ms_eden_pipeline import schedule_pair
    base=PosthocPreparationSchedule()
    candidate=replace(base,normalization_math="refine_warp_lut")
    adapter=make_posthoc_preparation_adapter((512,512,128),"dx","mma128",lambda _:None)
    assert base.normalization_math=="divide"
    assert "refine_warp_lut" in adapter.axes["normalization_math"]
    assert adapter.rejection(candidate) is None
    assert adapter.config_id(candidate)!=adapter.config_id(replace(base,normalization_math="refine"))
    assert adapter.deserialize(adapter.serialize(candidate))==candidate
    assert adapter.extra_features_fn(candidate)["a_normalization_refine_warp_lut"]==1.
    assert adapter.context.tags["schedule_features_revision"]==22
    assert all(json.loads(value)["normalization_math"]=="refine_warp_lut"
               for value in schedule_pair({"normalization_math":"refine_warp_lut"}).values())
    for transpose in (False,True):
        grouped={"producer_transpose_values_per_lane":8,"load_mode":"coalesced_x8"} if transpose else {"producer_values_per_lane":8}
        with pytest.raises(ValueError):
            HadamardPreparationKernel(128,128,PosthocScaleConfig(),4,transpose,
                normalization_math="refine_warp_lut",**grouped)
    assert adapter.rejection(replace(candidate,producer_values_per_lane=8))[0]=="implementation_rejected"
    entry=inspect.getsource(HadamardPreparationKernel.warp_reciprocal_entry)
    assert "0f3f800000" in entry and "read_only_args=[denominator]" in entry


run_cuda=pytest.mark.skipif(os.environ.get("RTX_RUN_REFINE_WARP_LUT_CUDA")!="1",
                            reason="explicit assigned-GPU exact qualification")


@run_cuda
def test_exact_lookup_and_quotients_including_mixed_invalid_source_lanes():
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from cutlass import Float32
    from cutlass.experimental.primitives import nvvm_wrapper as nvvm
    from rtx.kernels.nvfp4_posthoc import HadamardPreparationKernel
    generator=np.random.default_rng(953)
    normalized=np.concatenate((generator.uniform(-8,8,32).astype(np.float32),
        np.array([0.,-0.,2.**-90,-2.**-90,2.**-91,16.,-16.,32.],dtype=np.float32)))
    values,scales=[],[]
    with np.errstate(all="ignore"):
        for exponent in (-126,-90,0,90,126,127):
            for mantissa in range(16,32):
                block=np.ldexp(normalized,exponent,dtype=np.float32)
                values.extend(block)
                scales.extend([np.ldexp(np.float32(mantissa/16),exponent,dtype=np.float32)]*len(block))
    # First halfwarp's OWN scales are invalid; the valid second halfwarp asks
    # for those first-half lanes' table words. Conditional shuffles are wrong.
    invalid=np.array([0,0x80000000,1,0x007fffff,0xbf800000,0x7f800000,0xff800000,0x7fc00000,
        0,0x80000000,1,0x007fffff,0xbf800000,0x7f800000,0xff800000,0x7fc00000],dtype=np.uint32).view(np.float32)
    valid=(np.uint32(0x3f800000)|(np.arange(16,dtype=np.uint32)<<19)).view(np.float32)
    values.extend(np.ones(32,dtype=np.float32));scales.extend(np.concatenate((invalid,valid)))
    source_edges=np.array([0,0x80000000,1,0x80000001,0x7f7fffff,0xff7fffff,
        0x7f800000,0xff800000,0x7fc00000],dtype=np.uint32).view(np.float32)
    for scale in (1.,1.000001,0.,-1.,np.inf,np.nan):
        values.extend(source_edges);scales.extend([np.float32(scale)]*len(source_edges))
    padding=(-len(values))%128
    values=np.pad(np.asarray(values,dtype=np.float32),(0,padding))
    scales=np.pad(np.asarray(scales,dtype=np.float32),(0,padding),constant_values=1)
    class Probe(HadamardPreparationKernel):
        def __init__(self,count):self.count=count
        @cute.jit
        def __call__(self,v:cute.Tensor,s:cute.Tensor,out:cute.Tensor,ref:cute.Tensor,
                     table:cute.Tensor,stream:cuda.CUstream):
            self.probe(v,s,out,ref,table).launch(grid=(self.count//128,1,1),block=(128,1,1),stream=stream)
        @cute.kernel
        def probe(self,v:cute.Tensor,s:cute.Tensor,out:cute.Tensor,ref:cute.Tensor,table:cute.Tensor):
            thread,_,_=cute.arch.thread_idx();block,_,_=cute.arch.block_idx();i=block*128+thread
            cached=self.warp_reciprocal_entry()
            reciprocal=self.warp_reciprocal_lookup(cached,s[i])
            eligible,m,power,unused=self.normalization_parameters(s[i],False)
            table[i]=reciprocal
            out[i]=self.normalize_refined(v[i],s[i],eligible,m,power,reciprocal)
            ref[i]=nvvm.inline_ptx_hl("div.rn.f32 {$w0}, {$r0}, {$r1};",
                write_only_types=[Float32],read_only_args=[v[i],s[i]])
    source,scale=(torch.tensor(value,device="cuda") for value in (values,scales))
    actual,expected,looked=(torch.empty_like(source) for _ in range(3))
    fake=lambda:cute.runtime.make_fake_tensor(Float32,(len(values),),stride=(1,),assumed_align=4)
    runner=cute.compile(Probe(len(values)),*(fake() for _ in range(5)),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options="--enable-tvm-ffi --opt-level 3")
    runner(source,scale,actual,expected,looked)
    torch.testing.assert_close(actual.view(torch.int32),expected.view(torch.int32),atol=0,rtol=0)
    index=(scales.view(np.uint32)>>19)&15
    table=np.divide(np.float32(1),(np.uint32(0x3f800000)|(np.arange(16,dtype=np.uint32)<<19)).view(np.float32),dtype=np.float32)
    np.testing.assert_array_equal(looked.cpu().numpy().view(np.uint32),table[index].view(np.uint32))


@run_cuda
@pytest.mark.parametrize("shape,transpose,buffered,mantissa_bits,batch",[
    ((128,128),False,False,3,1),((4096,512),False,False,4,2),((4096,512),True,True,4,1)])
def test_lut_same_draw_all_prepared_buffers_and_changed_graph_inputs(shape,transpose,buffered,mantissa_bits,batch):
    from rtx.kernels.nvfp4_posthoc import compile_hadamard_preparation,compile_posthoc_scales,PosthocScaleConfig
    torch.manual_seed(954)
    rows,cols=shape[::-1] if transpose else shape
    source=torch.empty(shape,device="cuda",dtype=torch.bfloat16)
    signs=torch.ones(128,device="cuda");uniforms=torch.empty(source.numel()//16,device="cuda")
    kwargs=dict(mantissa_bits=mantissa_bits,transpose=transpose,
        tile_order="reduction_major" if transpose else "row_major",load_mode="coalesced_x8",
        signs_mode="register",stages=2 if buffered else 1,async_copy=buffered,
        correction_decode="native",partial_max_scope="cta",producer_row_tile_batch=batch,
        producer_unroll_rows=False)
    baseline,count=compile_hadamard_preparation(*shape,PosthocScaleConfig(128,256),**kwargs)
    candidate,other=compile_hadamard_preparation(*shape,PosthocScaleConfig(128,256),normalization_math="refine_warp_lut",**kwargs)
    assert count==other
    finalizer=compile_posthoc_scales(rows,cols//16,PosthocScaleConfig(128,224,"mma128",probability_math="power2"),count)
    def buffers():return (torch.empty(source.numel()//2,device="cuda",dtype=torch.uint8),
        torch.empty(source.numel()//16,device="cuda",dtype=torch.bfloat16),torch.empty(count,device="cuda"),
        torch.empty(source.numel()//16,device="cuda",dtype=torch.float8_e4m3fn),torch.empty(1,device="cuda"))
    expected,actual=buffers(),buffers()
    def run(producer,buffers):
        packed,corrected,partials,scales,factor=buffers
        producer(source.flatten(),signs,packed,corrected,partials)
        finalizer(corrected,partials,uniforms,scales,factor)
    source.zero_();uniforms.fill_(.5)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):run(candidate,actual)
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):run(candidate,actual)
    for distribution in ("zero","gaussian","sparse","outliers","tiny"):
        source.normal_(0,.25)
        if distribution=="zero":source.zero_()
        elif distribution=="sparse":source.mul_((torch.rand(shape,device="cuda")<.05).to(source.dtype))
        elif distribution=="outliers":source.flatten()[::97].mul_(32)
        elif distribution=="tiny":source.mul_(1e-35)
        signs.random_(0,2).mul_(2).sub_(1);uniforms.uniform_()
        run(baseline,expected);graph.replay();torch.cuda.synchronize()
        for reference,value in zip(expected,actual):
            assert torch.isfinite(value.float()).all()
            torch.testing.assert_close(value.view(torch.uint8),reference.view(torch.uint8),atol=0,rtol=0)


@run_cuda
def test_lut_public_paired_forward_rng_current_buffers_memory_and_packed_gates():
    from benchmarks.compare_ms_eden_pipeline import schedule_pair,options
    from benchmarks.compare_mixed_nvfp4_backward import run_worker
    from benchmarks.study_ms_eden_backward import run_study
    from rtx.autotune.posthoc import PosthocPreparationSchedule
    shape=(512,512,128)
    schedules=schedule_pair({"normalization_math":"refine_warp_lut","correction_decode":"native"})
    record,bundle=run_worker(options(shape,schedules,pipeline="paired"),capture_only=True)
    assert record["guards"]["forward_bitwise_matched"]
    assert record["guards"]["fresh_rng_per_phase"]==[True,True]
    assert record["memory"]["ordinary_compiled_allocation_audit"]["live_growth_bytes"]<=0
    assert record["memory"]["replay_live_growth_bytes"]==0
    assert all(torch.isfinite(value).all() for value in bundle["grads"])
    study=run_study(shape,"dw","mma128",benchmark=False,
        schedule=PosthocPreparationSchedule(**json.loads(schedules["dw"])))
    assert study["metrics"]["packed"]["nrms"]<=.005
