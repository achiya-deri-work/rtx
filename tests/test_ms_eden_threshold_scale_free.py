"""Exact FP4 packing, numerically reassociated FP32 EDEN correction.

This option is NOT byte-equivalent to divide: corrected BF16 scales can move
one ULP. Production Gaussian/packed-gradient gates are not changed or relaxed.
"""
from dataclasses import asdict, replace
import json
import os

import numpy as np
import pytest
import torch


LEVELS=np.array([0,.5,1,1.5,2,3,4,6],dtype=np.float32)
MIDPOINTS=np.array([.25,.75,1.25,1.75,2.5,3.5,5],dtype=np.float32)
UPPER_EVEN=np.array([False,True,False,True,False,True,False])


def quantize_codes(values, thresholds=MIDPOINTS):
    absolute=np.abs(values)
    thresholds=np.asarray(thresholds)
    if thresholds.ndim==2 and values.ndim==2: thresholds=thresholds[:,None,:]
    code=(np.sum(absolute[...,None]>thresholds,axis=-1)
          +np.sum((absolute[...,None]==thresholds)&UPPER_EVEN,axis=-1)).astype(np.uint8)
    return code|((values.view(np.uint32)>>np.uint32(28)).astype(np.uint8)&8)


def balanced_dot(a,b):
    value=np.multiply(a,b,dtype=np.float32)
    while value.shape[1]>1:value=np.add(value[:,::2],value[:,1::2],dtype=np.float32)
    return value[:,0]


def bf16_bits(values):
    words=np.asarray(values,dtype=np.float32).view(np.uint32)
    return ((words+np.uint32(0x7fff)+((words>>16)&1))>>16).astype(np.uint16)


def oracle_bf16_bits(values):
    """Direct positive FP64 -> BF16 nearest/even, without an FP32 double round."""
    representable=(np.arange(0x7f80,dtype=np.uint32)<<16).view(np.float32).astype(np.float64)
    boundaries=(representable[:-1]+representable[1:])*.5
    values=np.asarray(values,dtype=np.float64)
    code=np.searchsorted(boundaries,values,side="left").astype(np.uint16)
    at=np.minimum(code.astype(np.int64),len(boundaries)-1)
    tie=(values==boundaries[at])&((code&1)!=0)
    return code+tie.astype(np.uint16)


def evaluate_cpu(values, mantissa_bits=4):
    """Explicit FP32 mul/add model, not a claim about GPU FMA contraction."""
    values=np.asarray(values,dtype=np.float32)
    maximum=np.abs(values).max(axis=1)
    mask=np.uint32(0xffffffff^((1<<(23-mantissa_bits))-1))
    with np.errstate(all="ignore"):
        scale=(np.divide(maximum,np.float32(6),dtype=np.float32).view(np.uint32)&mask).view(np.float32)
        scale=np.where(scale<np.finfo(np.float32).tiny,np.float32(0),scale)
        normalized=np.zeros_like(values)
        np.divide(values,scale[:,None],out=normalized,where=scale[:,None]>0)
        code=quantize_codes(normalized)
        q=np.copysign(LEVELS[code&7],normalized)
        denominator=balanced_dot(normalized,q)
        correction=np.ones(len(values),dtype=np.float32)
        np.divide(balanced_dot(normalized,normalized),denominator,out=correction,where=denominator>0)
        baseline=np.multiply(scale,correction,dtype=np.float32)
        exponent=((scale.view(np.uint32)>>23)&255).astype(np.int32)-127
        u=np.ldexp(values,-exponent[:,None],dtype=np.float32)
        m=np.ldexp(scale,-exponent,dtype=np.float32)
        safe=(scale>0)&(exponent>=-126)&(exponent<=126)&np.isfinite(values).all(axis=1)
        safe&=np.all(((u==0)&(values==0))|(np.abs(u)>=np.float32(2.**-60)),axis=1)
        safe&=np.all(np.abs(u)<=16,axis=1)
        maximum_u=np.ldexp(maximum,-exponent,dtype=np.float32)
        safe&=(maximum_u>=m*6)&(maximum_u<=m*np.float32(6*(1+2.**-mantissa_bits)))
        threshold_code=quantize_codes(u,np.multiply(m[:,None],MIDPOINTS,dtype=np.float32))
        threshold_q=np.copysign(LEVELS[threshold_code&7],u)
        new_denominator=balanced_dot(u,threshold_q)
        ratio=np.ones(len(values),dtype=np.float32)
        np.divide(balanced_dot(u,u),new_denominator,out=ratio,where=new_denominator>0)
        candidate=np.where(safe,np.ldexp(ratio,exponent,dtype=np.float32),baseline)
        # Independent real-valued formula; no production normalization/reduction.
        oracle_denominator=np.sum(values.astype(np.float64)*q.astype(np.float64),axis=1)
        oracle=scale.astype(np.float64).copy()
        np.divide(np.sum(values.astype(np.float64)**2,axis=1),oracle_denominator,
                  out=oracle,where=oracle_denominator>0)
    return baseline,candidate,code,threshold_code,safe,oracle,scale


def threshold_cases(exponents):
    values,scales=[],[]
    with np.errstate(all="ignore"):
        for exponent in exponents:
            for mantissa in range(16,32):
                scale=np.ldexp(np.float32(mantissa/16),exponent,dtype=np.float32)
                boundaries=np.multiply(scale,MIDPOINTS,dtype=np.float32)
                if not np.isfinite(boundaries).all():continue
                assert np.array_equal(boundaries.astype(np.float64),np.float64(scale)*MIDPOINTS.astype(np.float64))
                for boundary in boundaries:
                    points=[boundary]
                    lo,hi=boundary,boundary
                    for _ in range(4):
                        lo=np.nextafter(lo,np.float32(-np.inf));hi=np.nextafter(hi,np.float32(np.inf))
                        points.extend((lo,hi))
                    points=np.array(points,dtype=np.float32)
                    points=np.concatenate((points,-points))
                    values.extend(points);scales.extend([scale]*len(points))
    return np.asarray(values,dtype=np.float32),np.asarray(scales,dtype=np.float32)


def test_all_scale_mantissas_threshold_neighbours_double_rounding_and_signed_zero():
    values,scales=threshold_cases((-126,-125,-90,-40,-1,0,1,40,90,120,124,125))
    baseline=quantize_codes(np.divide(values,scales,dtype=np.float32))
    direct=quantize_codes(values,np.multiply(scales[:,None],MIDPOINTS,dtype=np.float32))
    np.testing.assert_array_equal(baseline,direct)
    assert quantize_codes(np.array([0.,-0.],dtype=np.float32)).tolist()==[0,8]


def test_bounded_scale_free_cpu_study_against_fp64_oracle_without_relaxing_packing():
    rng=np.random.default_rng(20261027)
    for distribution in ("gaussian","outliers","sparse","zero","unsafe_dynamic"):
        values=rng.normal(0,.25,(8192,16)).astype(np.float32)
        if distribution=="outliers":values*=np.where(rng.random(values.shape)<.01,np.float32(32),np.float32(1))
        if distribution=="sparse":values*=rng.random(values.shape)<.05
        if distribution=="zero":values.fill(-0.)
        if distribution=="unsafe_dynamic":values[:,0]*=np.float32(1e20)
        exponents=rng.choice([-120,-80,-40,0,40] if distribution=="unsafe_dynamic" else [-120,-80,-40,0,40,80,120],len(values))
        values=np.ldexp(values,exponents[:,None],dtype=np.float32)
        baseline,candidate,code,new_code,safe,oracle,scale=evaluate_cpu(values)
        assert np.isfinite(candidate).all()
        np.testing.assert_array_equal(code[safe],new_code[safe])
        np.testing.assert_array_equal(candidate[~safe],baseline[~safe])
        # This explicitly permits ONE BF16 ULP for this new numerical option;
        # it does not modify any divide/refine exactness or public-gradient gate.
        distance=np.abs(bf16_bits(candidate).astype(np.int32)-oracle_bf16_bits(oracle).astype(np.int32))
        assert np.max(distance[safe],initial=0)<=1
        assert np.max(np.abs(bf16_bits(candidate).astype(np.int32)-bf16_bits(baseline).astype(np.int32)))<=1
        if distribution=="unsafe_dynamic":assert np.count_nonzero(~safe)>8000


def test_known_rounding_counterexample_is_not_mislabeled_exact():
    words=np.array([0x280fb96,0x82385013,0x8262a00e,0x819e0b2e,0x82618496,0x828cc94f,0x2714206,0x295a795,
        0x2fd80ef,0x1e62323,0x815f7b94,0x81495985,0x818dc9e5,0x282ae99,0x26fd59a,0x2616e7],dtype=np.uint32)
    baseline,candidate,_,_,safe,_,_=evaluate_cpu(words.view(np.float32)[None,:])
    assert safe[0] and abs(int(bf16_bits(candidate)[0])-int(bf16_bits(baseline)[0]))==1


def test_scale_free_opt_in_identity_features_and_json_forwarding():
    from rtx.autotune.posthoc import PosthocPreparationSchedule,make_posthoc_preparation_adapter
    from benchmarks.compare_ms_eden_pipeline import schedule_pair
    base=PosthocPreparationSchedule()
    candidate=replace(base,normalization_math="threshold_scale_free")
    assert base.normalization_math=="divide"
    adapter=make_posthoc_preparation_adapter((512,512,128),"dw","mma128",lambda _:None)
    assert "threshold_scale_free" in adapter.axes["normalization_math"]
    assert adapter.config_id(base)!=adapter.config_id(candidate)
    assert adapter.deserialize(adapter.serialize(candidate))==candidate
    for operand in ("a","b"):
        assert candidate.operand_schedule(operand).normalization_math=="threshold_scale_free"
        assert adapter.extra_features_fn(candidate)[operand+"_normalization_threshold_scale_free"]==1.
    assert all(json.loads(value)["normalization_math"]=="threshold_scale_free"
               for value in schedule_pair({"normalization_math":"threshold_scale_free"}).values())


run_cuda=pytest.mark.skipif(os.environ.get("RTX_RUN_SCALE_FREE_CUDA")!="1",reason="explicit assigned-GPU qualification")


@run_cuda
def test_gpu_threshold_bisection_matches_explicit_division_and_native_conversion():
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from cutlass import Float32,Int16,Int32
    from cutlass.experimental.primitives import nvvm_wrapper as nvvm
    from rtx.kernels.nvfp4_posthoc import HadamardPreparationKernel
    values,scales=threshold_cases((-126,-125,-90,-1,0,1,90,124,125))
    special=np.array([0,0x80000000,1,0x80000001,0x7f800000,0xff800000,0x7fc00000],dtype=np.uint32).view(np.float32)
    for scale in (0.,-0.,1.,1.000001,np.inf,np.nan):
        values=np.concatenate((values,special));scales=np.concatenate((scales,np.full(len(special),scale,dtype=np.float32)))
    class Probe(HadamardPreparationKernel):
        def __init__(self,count):self.count=count
        @cute.jit
        def __call__(self,v:cute.Tensor,s:cute.Tensor,out:cute.Tensor,ref:cute.Tensor,stream:cuda.CUstream):
            self.probe(v,s,out,ref).launch(grid=((self.count+127)//128,1,1),block=(128,1,1),stream=stream)
        @cute.kernel
        def probe(self,v:cute.Tensor,s:cute.Tensor,out:cute.Tensor,ref:cute.Tensor):
            thread,_,_=cute.arch.thread_idx();block,_,_=cute.arch.block_idx();i=block*128+thread
            if i<self.count:
                value,scale=v[i],s[i]
                eligible,m,power,r=self.normalization_parameters(scale,False)
                u=nvvm.inline_ptx_hl("mul.rn.f32 {$w0}, {$r0}, {$r1};",write_only_types=[Float32],read_only_args=[value,power])
                quotient=nvvm.inline_ptx_hl("div.rn.f32 {$w0}, {$r0}, {$r1};",write_only_types=[Float32],read_only_args=[value,scale])
                packed=nvvm.inline_ptx_hl("{.reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, {$r0}, {$r0}; mov.b16 {$w0}, {b, 0};}",
                    write_only_types=[Int16],read_only_args=[quotient])
                expected=Int32(packed)&15;actual=expected
                ab=u.bitcast(Int32)&0x7fffffff
                if eligible & ((value.bitcast(Int32)&0x7fffffff)<0x7f800000) & (ab<=0x41800000):
                    actual=self.threshold_code(u,*self.threshold_parameters(m))
                out[i],ref[i]=actual,expected
    source,scale=(torch.tensor(value,device="cuda") for value in (values,scales))
    out,reference=(torch.empty(len(values),device="cuda",dtype=torch.int32) for _ in range(2))
    fake=lambda dtype:cute.runtime.make_fake_tensor(dtype,(len(values),),stride=(1,),assumed_align=4)
    runner=cute.compile(Probe(len(values)),fake(Float32),fake(Float32),fake(Int32),fake(Int32),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options="--enable-tvm-ffi --opt-level 3")
    runner(source,scale,out,reference)
    torch.testing.assert_close(out,reference,atol=0,rtol=0)


@run_cuda
def test_gpu_scale_free_correction_within_one_bf16_ulp_of_independent_fp64_oracle():
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from cutlass import BFloat16,Float32,Int32
    from cutlass.experimental.primitives import nvvm_wrapper as nvvm
    from rtx.kernels.nvfp4_posthoc import HadamardPreparationKernel
    rng=np.random.default_rng(952)
    values=rng.normal(0,.25,(4096,16)).astype(np.float32)
    values*=np.where(rng.random(values.shape)<.01,np.float32(32),np.float32(1))
    values=np.ldexp(values,rng.choice([-120,-80,-40,0,40,80,120],len(values))[:,None],dtype=np.float32)
    baseline,candidate,code,new_code,safe,oracle,scales=evaluate_cpu(values)
    values,scales,oracle=values[safe],scales[safe],oracle[safe]
    assert len(values)>4000
    class Probe(HadamardPreparationKernel):
        def __init__(self,count):self.count=count
        @cute.jit
        def __call__(self,v:cute.Tensor,s:cute.Tensor,out:cute.Tensor,stream:cuda.CUstream):
            self.probe(v,s,out).launch(grid=((self.count+31)//32,1,1),block=(128,1,1),stream=stream)
        @cute.kernel
        def probe(self,v:cute.Tensor,s:cute.Tensor,out:cute.Tensor):
            thread,_,_=cute.arch.thread_idx();block,_,_=cute.arch.block_idx()
            group=block*32+thread//4
            if group<self.count:
                scale=s[group]
                eligible,m,power,unused=self.normalization_parameters(scale,False)
                thresholds=self.threshold_parameters(m)
                xx,xy=Float32(0.),Float32(0.)
                for pair in cutlass.range_constexpr(2):
                    index=group*16+(thread%4)*4+pair*2
                    a=nvvm.inline_ptx_hl("mul.rn.f32 {$w0}, {$r0}, {$r1};",write_only_types=[Float32],read_only_args=[v[index],power])
                    b=nvvm.inline_ptx_hl("mul.rn.f32 {$w0}, {$r0}, {$r1};",write_only_types=[Float32],read_only_args=[v[index+1],power])
                    qa,qb=self.decode(self.threshold_code(a,*thresholds)),self.decode(self.threshold_code(b,*thresholds))
                    xx+=a*a+b*b;xy+=a*qa+b*qb
                for shift in cutlass.range_constexpr(2):
                    xx+=cute.arch.shuffle_sync_bfly(xx,1<<shift)
                    xy+=cute.arch.shuffle_sync_bfly(xy,1<<shift)
                if thread%4==0:
                    restore=(scale.bitcast(Int32)&Int32(0x7f800000)).bitcast(Float32)
                    out[group]=(restore*(xx/xy)).to(BFloat16)
    source=torch.tensor(values.flatten(),device="cuda")
    scale=torch.tensor(scales,device="cuda")
    output=torch.empty(len(values),device="cuda",dtype=torch.bfloat16)
    fake=lambda dtype,size:cute.runtime.make_fake_tensor(dtype,(size,),stride=(1,),assumed_align=4)
    runner=cute.compile(Probe(len(values)),fake(Float32,values.size),fake(Float32,len(values)),
        fake(BFloat16,len(values)),cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options="--enable-tvm-ffi --opt-level 3")
    runner(source,scale,output)
    assert torch.isfinite(output.float()).all()
    actual=output.view(torch.int16).cpu().numpy().view(np.uint16).astype(np.int32)
    expected=oracle_bf16_bits(oracle).astype(np.int32)
    assert np.max(np.abs(actual-expected))<=1


@run_cuda
@pytest.mark.parametrize("transpose,buffered",[(False,False),(True,True)])
def test_scale_free_producer_exact_codes_bounded_scales_and_changed_graph_buffers(transpose,buffered):
    from rtx.kernels.nvfp4_posthoc import compile_hadamard_preparation,PosthocScaleConfig
    shape=(512,256)
    source=torch.empty(shape,device="cuda",dtype=torch.bfloat16)
    signs=torch.ones(128,device="cuda")
    config=PosthocScaleConfig(128,128)
    kwargs=dict(transpose=transpose,load_mode="coalesced_x8",signs_mode="register",
        stages=2 if buffered else 1,async_copy=buffered,correction_decode="native",partial_max_scope="cta")
    baseline,count=compile_hadamard_preparation(*shape,config,**kwargs)
    candidate,other=compile_hadamard_preparation(*shape,config,normalization_math="threshold_scale_free",**kwargs)
    assert count==other
    def buffers():return (torch.empty(source.numel()//2,device="cuda",dtype=torch.uint8),
        torch.empty(source.numel()//16,device="cuda",dtype=torch.bfloat16),torch.empty(count,device="cuda"))
    expected,actual=buffers(),buffers()
    source.zero_();stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):candidate(source.flatten(),signs,*actual)
    stream.synchronize();graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):candidate(source.flatten(),signs,*actual)
    for distribution in ("zero","gaussian","sparse","outliers","tiny"):
        source.normal_(0,.25)
        if distribution=="zero":source.zero_()
        elif distribution=="sparse":source.mul_((torch.rand(shape,device="cuda")<.05).to(source.dtype))
        elif distribution=="outliers":source.flatten()[::97].mul_(32)
        elif distribution=="tiny":source.mul_(1e-35)
        signs.random_(0,2).mul_(2).sub_(1)
        baseline(source.flatten(),signs,*expected);graph.replay();torch.cuda.synchronize()
        torch.testing.assert_close(actual[0],expected[0],atol=0,rtol=0)
        assert torch.isfinite(actual[1].float()).all() and torch.isfinite(actual[2]).all()
        assert (actual[1].view(torch.int16).int()-expected[1].view(torch.int16).int()).abs().max()<=1


@run_cuda
def test_scale_free_public_fullgraph_rng_current_buffers_memory_and_packed_gates():
    from benchmarks.compare_ms_eden_pipeline import schedule_pair,options
    from benchmarks.compare_mixed_nvfp4_backward import run_worker
    from benchmarks.study_ms_eden_backward import run_study
    from rtx.autotune.posthoc import PosthocPreparationSchedule
    shape=(512,512,128)
    schedules=schedule_pair({"normalization_math":"threshold_scale_free","correction_decode":"native"})
    record,bundle=run_worker(options(shape,schedules,pipeline="paired"),capture_only=True)
    assert record["guards"]["forward_bitwise_matched"]
    assert record["guards"]["fresh_rng_per_phase"]==[True,True]
    assert record["memory"]["ordinary_compiled_allocation_audit"]["live_growth_bytes"]<=0
    assert record["memory"]["replay_live_growth_bytes"]==0
    assert all(torch.isfinite(value).all() for value in bundle["grads"])
    study=run_study(shape,"dw","mma128",benchmark=False,
        schedule=PosthocPreparationSchedule(**json.loads(schedules["dw"])))
    assert study["metrics"]["packed"]["nrms"]<=.005
