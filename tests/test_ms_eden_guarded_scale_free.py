"""Exact equivalence to threshold_scale_free, NOT divide statistics.

Only packing changes: cached RN reciprocal proposals are certified by a W16
integer window or replaced by precise division. EDEN statistics continue to
consume the unchanged exact power-normalized inputs. Corrected scales may
differ from divide by one BF16 ULP; its gradient gates remain unchanged.
"""
from dataclasses import replace
import json
import os

import numpy as np
import pytest
import torch

from tests.test_ms_eden_threshold_scale_free import MIDPOINTS,quantize_codes,evaluate_cpu,bf16_bits


def guard(values,window=16):
    magnitude=values.view(np.uint32)&np.uint32(0x7fffffff)
    return (magnitude!=0)&(((magnitude+np.uint32(window))&np.uint32(0x1fffff))<=np.uint32(2*window))


def midpoint_cases():
    values,scales=[],[]
    for denominator in range(16,32):
        m=np.float32(denominator/16)
        words=np.multiply(m,MIDPOINTS,dtype=np.float32).view(np.uint32)
        neighbours=(words[:,None].astype(np.int64)+np.arange(-32,33)).astype(np.uint32).ravel().view(np.float32)
        values.extend(np.concatenate((neighbours,-neighbours)))
        scales.extend([m]*(2*len(neighbours)))
    return np.asarray(values,dtype=np.float32),np.asarray(scales,dtype=np.float32)


def test_cpu_w16_guard_midpoints_ties_and_all_reciprocal_mantissas():
    assert np.all((MIDPOINTS.view(np.uint32)&np.uint32(0x1fffff))==0)
    values,scales=midpoint_cases()
    reference=quantize_codes(np.divide(values,scales,dtype=np.float32))
    reciprocal=np.divide(np.float32(1),scales,dtype=np.float32)
    raw_errors=0
    for shift in (-1,0,1):
        r=(np.nextafter(reciprocal,np.float32(-np.inf if shift<0 else np.inf)) if shift else reciprocal)
        proposal=np.multiply(values,r,dtype=np.float32)
        proposed_code=quantize_codes(proposal)
        bad=proposed_code!=reference
        raw_errors+=int(bad.sum())
        assert not np.any(bad&~guard(proposal))
        accepted=np.where(guard(proposal),np.divide(values,scales,dtype=np.float32),proposal)
        np.testing.assert_array_equal(quantize_codes(accepted),reference)
    assert raw_errors>0  # This stresses the fallback rather than only easy cases.
    zero=np.array([0.,-0.],dtype=np.float32)
    assert not guard(zero).any()
    assert quantize_codes(zero).tolist()==[0,8]


def test_cpu_scale_free_contract_unchanged_and_divide_error_remains_bounded():
    rng=np.random.default_rng(955)
    values=rng.normal(0,.25,(4096,16)).astype(np.float32)
    values*=np.where(rng.random(values.shape)<.01,np.float32(32),np.float32(1))
    values=np.ldexp(values,rng.choice([-120,-80,-40,0,40,80,120],len(values))[:,None],dtype=np.float32)
    divide,threshold,code,threshold_code,safe,oracle,scale=evaluate_cpu(values)
    exponent=((scale.view(np.uint32)>>23)&255).astype(np.int32)-127
    u=np.ldexp(values,-exponent[:,None],dtype=np.float32)
    m=np.ldexp(scale,-exponent,dtype=np.float32)
    r=np.divide(np.float32(1),m,dtype=np.float32)
    proposal=np.multiply(u,r[:,None],dtype=np.float32)
    packing=np.where(guard(proposal),np.divide(values,scale[:,None],dtype=np.float32),proposal)
    np.testing.assert_array_equal(quantize_codes(packing)[safe],threshold_code[safe])
    # Statistics use u, not packing. Exact q-code equivalence leaves the
    # existing threshold_scale_free statistics/final BF16 cast unchanged.
    assert np.max(np.abs(bf16_bits(threshold).astype(np.int32)-bf16_bits(divide).astype(np.int32)))<=1


def test_opt_in_adapter_vpl_restrictions_and_distinct_identity():
    from rtx.autotune.posthoc import PosthocPreparationSchedule,make_posthoc_preparation_adapter
    from rtx.kernels.nvfp4_posthoc import HadamardPreparationKernel,PosthocScaleConfig
    from benchmarks.compare_ms_eden_pipeline import schedule_pair
    base=PosthocPreparationSchedule()
    candidate=replace(base,normalization_math="guarded_scale_free")
    adapter=make_posthoc_preparation_adapter((512,512,128),"dx","mma128",lambda _:None)
    assert base.normalization_math=="divide" and adapter.rejection(candidate) is None
    assert adapter.config_id(candidate)!=adapter.config_id(replace(base,normalization_math="threshold_scale_free"))
    assert adapter.deserialize(adapter.serialize(candidate))==candidate
    assert adapter.extra_features_fn(candidate)["a_normalization_guarded_scale_free"]==1.
    assert adapter.context.tags["schedule_features_revision"]==22
    assert all(json.loads(encoded)["normalization_math"]=="guarded_scale_free"
               for encoded in schedule_pair({"normalization_math":"guarded_scale_free"}).values())
    for transpose in (False,True):
        kwargs={"producer_transpose_values_per_lane":8,"load_mode":"coalesced_x8"} if transpose else {"producer_values_per_lane":8}
        with pytest.raises(ValueError):
            HadamardPreparationKernel(128,128,PosthocScaleConfig(),4,transpose,normalization_math="guarded_scale_free",**kwargs)


run_cuda=pytest.mark.skipif(os.environ.get("RTX_RUN_GUARDED_SCALE_FREE_CUDA")!="1",
                            reason="explicit assigned-GPU exact-contract qualification")


@run_cuda
def test_gpu_guarded_packing_vs_precise_division_midpoints_and_extremes():
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from cutlass import Float32,Int16,Int32
    from cutlass.experimental.primitives import nvvm_wrapper as nvvm
    from rtx.kernels.nvfp4_posthoc import HadamardPreparationKernel
    u,m=midpoint_cases()
    values,scales=[],[]
    with np.errstate(all="ignore"):
        for exponent in (-126,0,120,125):
            values.extend(np.ldexp(u,exponent,dtype=np.float32));scales.extend(np.ldexp(m,exponent,dtype=np.float32))
    edges=np.array([0,0x80000000,1,0x80000001,0x007fffff,0x00800000,0x7f7fffff,0xff7fffff,
        0x7f800000,0xff800000,0x7fc00000],dtype=np.uint32).view(np.float32)
    for scale in (0.,-0.,np.float32(2.**-126),1.,1.000001,np.float32(2.**127),np.inf,np.nan):
        values.extend(edges);scales.extend([np.float32(scale)]*len(edges))
    # Ineligible first-half lanes still own valid reciprocal table entries.
    values.extend(np.ones(32,dtype=np.float32))
    scales.extend(np.concatenate((np.full(16,np.nan,dtype=np.float32),
        (np.uint32(0x3f800000)|(np.arange(16,dtype=np.uint32)<<19)).view(np.float32))))
    padding=(-len(values))%128
    values=np.pad(np.asarray(values,dtype=np.float32),(0,padding))
    scales=np.pad(np.asarray(scales,dtype=np.float32),(0,padding),constant_values=1)
    class Probe(HadamardPreparationKernel):
        def __init__(self,count):self.count=count
        @cute.jit
        def __call__(self,v:cute.Tensor,s:cute.Tensor,out:cute.Tensor,ref:cute.Tensor,stream:cuda.CUstream):
            self.probe(v,s,out,ref).launch(grid=(self.count//128,1,1),block=(128,1,1),stream=stream)
        @cute.kernel
        def probe(self,v:cute.Tensor,s:cute.Tensor,out:cute.Tensor,ref:cute.Tensor):
            thread,_,_=cute.arch.thread_idx();block,_,_=cute.arch.block_idx();i=block*128+thread
            cached=self.warp_reciprocal_entry()
            r=self.warp_reciprocal_lookup(cached,s[i])
            eligible,m,power,unused=self.normalization_parameters(s[i],False)
            normalized=nvvm.inline_ptx_hl("mul.rn.f32 {$w0}, {$r0}, {$r1};",
                write_only_types=[Float32],read_only_args=[v[i],power])
            exact=nvvm.inline_ptx_hl("div.rn.f32 {$w0}, {$r0}, {$r1};",
                write_only_types=[Float32],read_only_args=[v[i],s[i]])
            packing=exact
            word=normalized.bitcast(Int32)&Int32(0x7fffffff)
            original=v[i].bitcast(Int32)&Int32(0x7fffffff)
            if eligible & (original<0x7f800000) & (word<=0x41800000) & ((word>=0x21800000)|((word==0)&(original==0))):
                packing=self.guarded_scale_free_proposal(normalized,v[i],s[i],r)
            actual=nvvm.inline_ptx_hl("{.reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, {$r0}, {$r0}; mov.b16 {$w0}, {b, 0};}",
                write_only_types=[Int16],read_only_args=[packing])
            expected=nvvm.inline_ptx_hl("{.reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, {$r0}, {$r0}; mov.b16 {$w0}, {b, 0};}",
                write_only_types=[Int16],read_only_args=[exact])
            out[i],ref[i]=Int32(actual)&15,Int32(expected)&15
    source,scale=(torch.tensor(value,device="cuda") for value in (values,scales))
    actual,expected=(torch.empty(len(values),device="cuda",dtype=torch.int32) for _ in range(2))
    fake=lambda dtype:cute.runtime.make_fake_tensor(dtype,(len(values),),stride=(1,),assumed_align=4)
    runner=cute.compile(Probe(len(values)),fake(Float32),fake(Float32),fake(Int32),fake(Int32),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options="--enable-tvm-ffi --opt-level 3")
    runner(source,scale,actual,expected)
    torch.testing.assert_close(actual,expected,atol=0,rtol=0)


@run_cuda
@pytest.mark.parametrize("shape,transpose,buffered,bits",[
    ((128,128),False,False,3),((4096,512),False,False,4),((4096,512),True,True,4)])
def test_guarded_all_five_buffers_exact_to_threshold_stats_and_quantify_divide_difference(shape,transpose,buffered,bits):
    from rtx.kernels.nvfp4_posthoc import compile_hadamard_preparation,compile_posthoc_scales,PosthocScaleConfig
    torch.manual_seed(956)
    rows,cols=shape[::-1] if transpose else shape
    source=torch.empty(shape,device="cuda",dtype=torch.bfloat16)
    signs=torch.ones(128,device="cuda");uniforms=torch.empty(source.numel()//16,device="cuda")
    kwargs=dict(mantissa_bits=bits,transpose=transpose,load_mode="coalesced_x8",signs_mode="register",
        stages=2 if buffered else 1,async_copy=buffered,correction_decode="native",partial_max_scope="cta",
        producer_unroll_rows=False,producer_row_tile_batch=2 if not transpose else 1)
    runners=[];count=None
    for mode in ("divide","threshold_scale_free","guarded_scale_free"):
        producer,partial_count=compile_hadamard_preparation(*shape,PosthocScaleConfig(128,256),normalization_math=mode,**kwargs)
        assert count is None or count==partial_count
        count=partial_count;runners.append(producer)
    finalizer=compile_posthoc_scales(rows,cols//16,PosthocScaleConfig(128,224,"mma128",probability_math="power2"),count)
    def buffers():return (torch.empty(source.numel()//2,device="cuda",dtype=torch.uint8),
        torch.empty(source.numel()//16,device="cuda",dtype=torch.bfloat16),torch.empty(count,device="cuda"),
        torch.empty(source.numel()//16,device="cuda",dtype=torch.float8_e4m3fn),torch.empty(1,device="cuda"))
    divide,threshold,guarded=buffers(),buffers(),buffers()
    def run(producer,outputs):
        packed,corrected,partials,scales,factor=outputs
        producer(source.flatten(),signs,packed,corrected,partials)
        finalizer(corrected,partials,uniforms,scales,factor)
    source.zero_();uniforms.fill_(.5)
    stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):run(runners[2],guarded)
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):run(runners[2],guarded)
    for distribution in ("zero","gaussian","sparse","outliers","tiny","huge"):
        source.normal_(0,.25)
        if distribution=="zero":source.zero_()
        elif distribution=="sparse":source.mul_((torch.rand(shape,device="cuda")<.05).to(source.dtype))
        elif distribution=="outliers":source.flatten()[::97].mul_(32)
        elif distribution=="tiny":source.mul_(1e-35)
        elif distribution=="huge":source.mul_(1e30)
        signs.random_(0,2).mul_(2).sub_(1);uniforms.uniform_()
        run(runners[0],divide);run(runners[1],threshold);graph.replay();torch.cuda.synchronize()
        for reference,actual in zip(threshold,guarded):
            assert torch.isfinite(actual.float()).all()
            torch.testing.assert_close(actual.view(torch.uint8),reference.view(torch.uint8),atol=0,rtol=0)
        torch.testing.assert_close(guarded[0],divide[0],atol=0,rtol=0)
        ulp=(guarded[1].view(torch.int16).int()-divide[1].view(torch.int16).int()).abs()
        assert ulp.max()<=1
        relative=(guarded[1].float()-divide[1].float()).abs()/divide[1].float().abs().clamp_min(torch.finfo(torch.float32).tiny)
        print(dict(distribution=distribution,changed_corrected_bf16=int((ulp!=0).sum()),
            max_bf16_ulp=int(ulp.max()),max_relative_with_fp32_tiny_floor=float(relative.max())))


@run_cuda
def test_guarded_public_current_inputs_rng_memory_and_unchanged_gradient_gates():
    from benchmarks.compare_ms_eden_pipeline import schedule_pair,options
    from benchmarks.compare_mixed_nvfp4_backward import run_worker
    from benchmarks.study_ms_eden_backward import run_study
    from rtx.autotune.posthoc import PosthocPreparationSchedule
    shape=(512,512,128)
    pair=schedule_pair({"normalization_math":"guarded_scale_free","correction_decode":"native"})
    record,bundle=run_worker(options(shape,pair,pipeline="paired"),capture_only=True)
    assert record["guards"]["forward_bitwise_matched"] and record["guards"]["fresh_rng_per_phase"]==[True,True]
    assert record["memory"]["ordinary_compiled_allocation_audit"]["live_growth_bytes"]<=0
    assert record["memory"]["replay_live_growth_bytes"]==0
    assert all(torch.isfinite(value).all() for value in bundle["grads"])
    study=run_study(shape,"dw","mma128",benchmark=False,
        schedule=PosthocPreparationSchedule(**json.loads(pair["dw"])))
    assert study["metrics"]["packed"]["nrms"]<=.005
