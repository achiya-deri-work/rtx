"""Exact power-of-two probability division, with independent format oracles."""
from bisect import bisect_left
from contextlib import nullcontext
from dataclasses import asdict, replace
import json
import math
import os
import struct

import pytest
import torch

from rtx.autotune.posthoc import PosthocPreparationSchedule,make_posthoc_preparation_adapter


def f32(value):
    return struct.unpack("f",struct.pack("f",value))[0]


def bits(value):
    return struct.unpack("I",struct.pack("f",value))[0]


def e4m3_positive(code):
    """Independent real-valued E4M3FN definition, including subnormals."""
    exponent,mantissa=code//8,code%8
    return mantissa*2.**-9 if exponent==0 else (1+mantissa/8)*2.**(exponent-7)


LEVELS=[e4m3_positive(code) for code in range(127)]


def probability_cases():
    values,lows,highs=[],[],[]
    for low,high in zip(LEVELS,LEVELS[1:]):
        gap=high-low
        midpoint=torch.tensor(low+gap/2,dtype=torch.float32)
        points=[low,low+gap/4,float(torch.nextafter(midpoint,torch.tensor(-math.inf))),
                float(midpoint),float(torch.nextafter(midpoint,torch.tensor(math.inf))),
                low+3*gap/4,high]
        values.extend(points); lows.extend([low]*len(points)); highs.extend([high]*len(points))
    # Signed zero, subnormal results/underflow, and explicit division fallbacks.
    for value,low,high in [(-0.,0.,2.**-9),(2.**-149,0.,2.**-9),
        (-2.**-149,0.,32.),(2.**-149,0.,32.),(.125,0.,.75),
        (2.**-149,0.,2.**-148),(1.,1.,1.),(0.,math.nan,1.),
        (0.,0.,math.inf),(math.inf,0.,math.inf)]:
        values.append(value); lows.append(low); highs.append(high)
    return values,lows,highs


def test_all_adjacent_e4m3_gaps_have_exact_normal_power2_reciprocal():
    assert LEVELS[0]==0 and LEVELS[-1]==448
    for low,high in zip(LEVELS,LEVELS[1:]):
        gap=high-low
        mantissa,exponent=math.frexp(gap)
        assert mantissa==.5 and 2.**-9<=gap<=32
        reciprocal=struct.unpack("f",struct.pack("I",0x7f000000-bits(gap)))[0]
        assert reciprocal==2.**(1-exponent)==1/gap
    values,lows,highs=probability_cases()
    for value,low,high in zip(values,lows,highs):
        if not high>low: continue
        gap=f32(high-low)
        if math.isfinite(gap) and math.frexp(gap)[0]==.5 and 2.**-9<=gap<=32:
            numerator=f32(value-low)
            # Every FP32 value is a dyadic number. A power-of-two scale shift
            # is exact in Python's double domain before the single FP32 RN.
            assert bits(f32(numerator/gap))==bits(f32(numerator*f32(1/gap)))


def test_probability_option_identity_features_validation_and_json():
    from rtx.kernels.nvfp4_posthoc import PosthocScaleConfig
    base=PosthocPreparationSchedule()
    candidate=replace(base,probability_math="power2")
    assert PosthocScaleConfig().probability_math==base.probability_math=="divide"
    assert PosthocPreparationSchedule(**json.loads(json.dumps(asdict(candidate))))==candidate
    adapter=make_posthoc_preparation_adapter((512,512,512),"dx","mma128",lambda _:None)
    assert adapter.axes["probability_math"]==("divide","power2")
    assert adapter.config_id(candidate)!=adapter.config_id(base)
    assert adapter.rejection(candidate) is None
    for operand in ("a","b"):
        assert adapter.extra_features_fn(candidate)[f"{operand}_probability_power2"]==1
        assert candidate.operand_schedule(operand).probability_math=="power2"
    for constructor in (PosthocScaleConfig,PosthocPreparationSchedule):
        with pytest.raises(ValueError,match="probability math"):
            constructor(probability_math="approximate")


@pytest.mark.parametrize("seeded",[False,True])
def test_runtime_study_and_philox_receive_probability_config(monkeypatch,seeded):
    import rtx.fp4 as fp4
    import rtx.experimental_ms_eden as eden
    import rtx.kernels.nvfp4_posthoc as kernels
    import rtx.kernels.nvfp4_rng as rng
    from benchmarks.study_ms_eden_backward import prepare
    seen=[]
    def finalizer(*args):
        seen.append(args[2].probability_math)
        return lambda *args:None
    monkeypatch.setattr(kernels,"compile_hadamard_preparation",lambda *args,**kw:((lambda *args:None),4))
    monkeypatch.setattr(kernels,"compile_posthoc_scales",finalizer)
    monkeypatch.setattr(rng,"compile_philox_posthoc_scales",finalizer)
    monkeypatch.setattr(kernels,"compile_posthoc_scale_product",lambda:lambda *args:None)
    monkeypatch.setattr(fp4,"compile_nvfp4_gemm",lambda *args:lambda *args:None)
    monkeypatch.setattr(fp4,"_packed_fp4_view",lambda value:value)
    monkeypatch.setattr(fp4,"_check_sm12x",lambda device:None)
    monkeypatch.setattr(torch.cuda,"device",lambda device:nullcontext())
    source=torch.empty(128,128,dtype=torch.bfloat16)
    signs,uniforms=torch.ones(128),torch.zeros(1024)
    seed=torch.zeros(2,dtype=torch.int64) if seeded else None
    config=PosthocPreparationSchedule(probability_math="power2",rng_mode="philox" if seeded else "explicit")
    eden._build_gradient(source,source,signs,uniforms,uniforms,"dx",config,(128,128,128),seed=seed)
    prepare(source,True,"mma128",config)
    assert seen==["power2"]*3


run_cuda=pytest.mark.skipif(os.environ.get("RTX_RUN_PROBABILITY_MATH_CUDA")!="1",
    reason="explicit assigned idle GPU qualification opt-in")


@run_cuda
def test_gpu_probability_bits_all_boundaries_and_exception_fallback():
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from cutlass import Float32
    from rtx.kernels.nvfp4_posthoc import PosthocScaleKernel,PosthocScaleConfig
    values,lows,highs=probability_cases()
    class ProbabilityProbe(PosthocScaleKernel):
        def __init__(self,mode):
            super().__init__(1,len(values),PosthocScaleConfig(probability_math=mode))
        @cute.jit
        def __call__(self,v:cute.Tensor,lo:cute.Tensor,hi:cute.Tensor,out:cute.Tensor,stream:cuda.CUstream):
            self.probe(v,lo,hi,out).launch(grid=((self.groups+255)//256,1,1),block=(256,1,1),stream=stream)
        @cute.kernel
        def probe(self,v:cute.Tensor,lo:cute.Tensor,hi:cute.Tensor,out:cute.Tensor):
            thread,_,_=cute.arch.thread_idx()
            block,_,_=cute.arch.block_idx()
            i=block*256+thread
            if i<self.groups: out[i]=self.rounding_probability(v[i],lo[i],hi[i])
    inputs=[torch.tensor(x,device="cuda",dtype=torch.float32) for x in (values,lows,highs)]
    fake=lambda:cute.runtime.make_fake_tensor(Float32,(len(values),),stride=(1,),assumed_align=4)
    outputs=[]
    for mode in ("divide","power2"):
        runner=cute.compile(ProbabilityProbe(mode),fake(),fake(),fake(),fake(),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options="--enable-tvm-ffi --opt-level 3")
        output=torch.empty_like(inputs[0]); runner(*inputs,output); outputs.append(output)
    assert torch.equal(outputs[0].view(torch.int32),outputs[1].view(torch.int32))
    expected=torch.tensor([f32(f32(v-lo)/f32(hi-lo)) if hi>lo else 0.
        for v,lo,hi in zip(values,lows,highs)],dtype=torch.float32,device="cuda")
    finite=torch.isfinite(expected)
    assert torch.equal(outputs[1][finite].view(torch.int32),expected[finite].view(torch.int32))
    assert torch.equal(torch.isnan(outputs[1]),torch.isnan(expected))


def finalizer_inputs():
    source=[]
    for low,high in zip(LEVELS,LEVELS[1:]):
        gap=high-low; midpoint=torch.tensor(low+gap/2,dtype=torch.bfloat16)
        source.extend([low,low+gap/4,float(torch.nextafter(midpoint,torch.tensor(-math.inf,dtype=torch.bfloat16))),
            float(midpoint),float(torch.nextafter(midpoint,torch.tensor(math.inf,dtype=torch.bfloat16))),
            low+3*gap/4,high])
    source.extend([0.,-0.,math.inf,-math.inf,math.nan,-1.,448.,512.])
    source.extend([0.]*(1024-len(source)))
    value=torch.tensor(source,dtype=torch.bfloat16)
    probabilities=[]; pairs=[]
    for v in value.float().tolist():
        if not math.isfinite(v) or v<0 or bits(v)==0x80000000:
            probabilities.append(0.); pairs.append(None); continue
        v=min(v,448.)
        index=bisect_left(LEVELS,v)
        if LEVELS[index]==v:
            lo=hi=index; p=0.
        else:
            lo,hi=index-1,index
            p=f32(f32(v-LEVELS[lo])/f32(LEVELS[hi]-LEVELS[lo]))
        probabilities.append(p); pairs.append((lo,hi))
    return value,torch.tensor(probabilities),pairs


def test_independent_finalizer_oracle_covers_every_positive_code_and_tie():
    source,probabilities,pairs=finalizer_inputs()
    assert source.numel()==probabilities.numel()==len(pairs)==1024
    assert set(LEVELS)<=set(source.float().tolist())
    assert ((probabilities>=0)&(probabilities<=1)).all()
    assert .25 in probabilities and .5 in probabilities and .75 in probabilities
    assert {pair[0] for pair in pairs if pair is not None}==set(range(127))


@run_cuda
@pytest.mark.parametrize("layout",["row_major","mma128"])
@pytest.mark.parametrize("partial_count",[1,1024])
def test_finalizer_exact_all_boundaries_and_uniform_probability_neighbors(layout,partial_count):
    from rtx.kernels.nvfp4_posthoc import compile_posthoc_scales,PosthocScaleConfig
    source,probabilities,pairs=finalizer_inputs()
    source=source.cuda(); probabilities=probabilities.cuda()
    maximum=torch.zeros(partial_count,device="cuda"); maximum[0]=448.
    uniforms=torch.empty(1024,device="cuda")
    outputs=[torch.empty(1024,device="cuda",dtype=torch.float8_e4m3fn) for _ in range(2)]
    factors=[torch.empty(1,device="cuda") for _ in range(2)]
    runners=[compile_posthoc_scales(128,8,PosthocScaleConfig(128,128,layout,probability_math=mode),partial_count)
        for mode in ("divide","power2")]
    graph=torch.cuda.CUDAGraph()
    runners[1](source,maximum,uniforms,outputs[1],factors[1])
    with torch.cuda.graph(graph): runners[1](source,maximum,uniforms,outputs[1],factors[1])
    for direction in (-math.inf,0.,math.inf):
        target=probabilities if direction==0 else torch.nextafter(probabilities,torch.full_like(probabilities,direction))
        uniforms.copy_(target.clamp(0.,1.-2.**-24))
        runners[0](source,maximum,uniforms,outputs[0],factors[0]); graph.replay()
        assert torch.equal(outputs[0].view(torch.uint8),outputs[1].view(torch.uint8))
        assert torch.equal(factors[0],factors[1]) and factors[0].item()==1.
        logical=outputs[1].view(torch.uint8)
        if layout=="mma128":
            logical=logical.reshape(1,1,2,32,4,4).permute(0,4,3,1,2,5).contiguous().reshape(-1)
        draws=uniforms.cpu().tolist()
        expected=torch.tensor([lohi[int(draw<probability)] if lohi is not None else 0
            for lohi,draw,probability in zip(pairs,draws,probabilities.cpu().tolist())],dtype=torch.uint8,device="cuda")
        eligible=torch.tensor([p is not None for p in pairs],device="cuda")
        assert torch.equal(logical[eligible],expected[eligible])


@run_cuda
@pytest.mark.parametrize("packet_size",[1,4])
def test_philox_inherited_probability_math_exact(packet_size):
    from rtx.kernels.nvfp4_posthoc import PosthocScaleConfig
    from rtx.kernels.nvfp4_rng import compile_philox_posthoc_scales
    source,_,_=finalizer_inputs(); source=source.cuda()
    maximum=torch.tensor([448.],device="cuda")
    seed=torch.tensor([952,23456789],device="cuda",dtype=torch.int64)
    outputs=[]; factors=[]
    for mode in ("divide","power2"):
        runner=compile_philox_posthoc_scales(128,8,
            PosthocScaleConfig(128,128,"mma128",probability_math=mode),1,0,packet_size)
        output=torch.empty(1024,device="cuda",dtype=torch.float8_e4m3fn)
        factor=torch.empty(1,device="cuda");runner(source,maximum,seed,output,factor)
        outputs.append(output);factors.append(factor)
    assert torch.equal(outputs[0].view(torch.uint8),outputs[1].view(torch.uint8))
    assert torch.equal(factors[0],factors[1])
