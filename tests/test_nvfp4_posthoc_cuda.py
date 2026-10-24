import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),reason="CUDA required")


def test_ms_eden_timing_records_bounded_time_based_warmup():
    from benchmarks.study_ms_eden_backward import run_study
    result=run_study((128,128,128),"dw","mma128")
    assert result["measurement_contract"] == "graph8x10_samples15_warm200ms_revision2"
    for timing in result["timings"].values():
        assert 10 <= timing["graph_warmup"]["warmup_replays"] <= 4096
        assert timing["graph_warmup"]["pilot_graph_ms"] > 0
        assert len(timing["samples_ms"]) == 15


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("layout", ["row_major", "mma128"])
@pytest.mark.parametrize("order", ["row_major", "reduction_major"])
@pytest.mark.parametrize("reduction", ["inline", "staged"])
@pytest.mark.parametrize("load", ["direct", "coalesced"])
def test_preparation_grid_changes_preserve_packed_values_and_scales(transpose, layout, order, reduction, load):
    from benchmarks.study_ms_eden_backward import prepare
    from rtx.autotune.posthoc import PosthocPreparationSchedule
    torch.manual_seed(809)
    source = torch.randn(2048, 768, device="cuda", dtype=torch.bfloat16)
    signs = (torch.randint(2, (128,), device="cuda") * 2 - 1).float()
    baseline = prepare(source, transpose, layout, PosthocPreparationSchedule())
    for blocks in (192, 1376):
        candidate = prepare(source, transpose, layout, PosthocPreparationSchedule(
            producer_blocks=blocks, finalizer_blocks=blocks, producer_order=order,
            amax_reduction=reduction, producer_load=load))
        candidate["uniforms"].copy_(baseline["uniforms"])
        for zero in (False, True):
            if zero:
                source.zero_()
            baseline["run"](signs)
            candidate["run"](signs)
            for name in ("packed", "scales", "factor"):
                torch.testing.assert_close(candidate[name].view(torch.uint8),
                                           baseline[name].view(torch.uint8), rtol=0, atol=0)
        source.normal_()


@pytest.mark.parametrize("count", [1, 3, 512, 2816, 8192])
def test_partial_maximum_independent_reference_and_replay(count):
    from rtx.kernels.nvfp4_posthoc import compile_partial_maximum
    torch.manual_seed(819)
    values=torch.rand(count,device="cuda")
    maximum=torch.empty(1,device="cuda")
    runner=compile_partial_maximum(count)
    runner(values,maximum)
    torch.testing.assert_close(maximum,values.amax().reshape(1),rtol=0,atol=0)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        runner(values,maximum)
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        runner(values,maximum)
    for zero in (False,True):
        values.uniform_(0,100)
        if zero:
            values.zero_()
        graph.replay()
        torch.testing.assert_close(maximum,values.amax().reshape(1),rtol=0,atol=0)


@pytest.mark.parametrize("threads", [64,128,256,512])
@pytest.mark.parametrize("packed_source", [False,True])
@pytest.mark.parametrize("order", ["row_major","reduction_major"])
def test_coalesced_producer_thread_geometry(threads,packed_source,order):
    from benchmarks.study_ms_eden_backward import prepare
    from rtx.autotune.posthoc import PosthocPreparationSchedule
    from benchmarks.compare_nvfp4_packed_gemm import decode
    from types import SimpleNamespace
    torch.manual_seed(820)
    source=torch.randn(256,384,device="cuda",dtype=torch.bfloat16)*.2
    packed=None
    if packed_source:
        # Exercise every legal nibble directly, independent of the forward
        # quantizer and its toolkit/driver capability discovery.
        packed=SimpleNamespace(
            qdata=torch.randint(256,(256,192),device="cuda",dtype=torch.uint8),
            scale=(torch.rand(256,24,device="cuda")*.3).to(torch.float8_e4m3fn),
            per_tensor_scale=torch.tensor([.17],device="cuda"))
        source=decode(packed.qdata,packed.scale,packed.per_tensor_scale).bfloat16()
    reference=prepare(source,True,"mma128",PosthocPreparationSchedule())
    candidate=prepare(source,True,"mma128",PosthocPreparationSchedule(
        producer_threads=threads,producer_blocks=3,producer_order=order,producer_load="coalesced"),packed)
    signs=(torch.randint(2,(128,),device="cuda")*2-1).float()
    candidate["uniforms"].copy_(reference["uniforms"])
    reference["run"](signs);candidate["run"](signs)
    for name in ("packed","scales","factor"):
        torch.testing.assert_close(candidate[name].view(torch.uint8),reference[name].view(torch.uint8),rtol=0,atol=0)


@pytest.mark.parametrize("phase",["dx","dw"])
@pytest.mark.parametrize("layout",["row_major","mma128"])
def test_posthoc_gradient_gemm_packed_reference(phase,layout):
    from benchmarks.study_ms_eden_backward import run_study
    result=run_study((128,256,128),phase,layout,benchmark=False)
    assert result["metrics"]["packed"]["nrms"] < .005
    # Broad smoke bound only; this is not the MXFP8 accuracy contract.
    assert result["metrics"]["original_bf16"]["nrms"] < .3


def assert_corrected_scales(actual,reference):
    expected=reference.corrected_scales.flatten()
    different=actual != expected
    if different.any():
        actual_bits=actual.view(torch.int16).to(torch.int32)
        expected_bits=expected.view(torch.int16).to(torch.int32)
        assert ((actual_bits-expected_bits).abs()[different] == 1).all()
        midpoint=(actual.float()+expected.float())*.5
        fp32=reference.fp32_corrected_scales.flatten()
        assert ((midpoint-fp32).abs()[different] <=
                (8*torch.finfo(torch.float32).eps*fp32.abs())[different]).all()


@pytest.mark.parametrize("transpose",[False,True])
@pytest.mark.parametrize("bits",[3,4])
@pytest.mark.parametrize("rows,cols",[(128,256),(384,512)])
def test_fused_hadamard_preparation(transpose,bits,rows,cols):
    from benchmarks.reference_ms_eden import prepare_extended
    from benchmarks.compare_nvfp4_packed_gemm import decode
    from rtx.kernels.nvfp4_posthoc import compile_hadamard_preparation,PosthocScaleConfig
    torch.manual_seed(806)
    x=(torch.randn(rows,cols,device="cuda")*.25).bfloat16()
    signs=(torch.randint(2,(128,),device="cuda")*2-1).float()
    h=torch.ones(1,1,device="cuda",dtype=torch.float64)
    for _ in range(7):
        h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
    logical=x.T if transpose else x
    normalization=torch.tensor(0.08838834764831845,device="cuda",dtype=torch.float32).double()
    rotated=((logical.double().reshape(-1,128)*signs.double())@h*normalization).float().reshape(logical.shape)
    reference=prepare_extended(rotated,mantissa_bits=bits)
    runner,count=compile_hadamard_preparation(rows,cols,PosthocScaleConfig(128,3),bits,transpose)
    packed=torch.empty(x.numel()//2,device="cuda",dtype=torch.uint8)
    scales=torch.empty(x.numel()//16,device="cuda",dtype=torch.bfloat16)
    partials=torch.empty(count,device="cuda")
    runner(x.flatten(),signs,packed,scales,partials)
    actual=decode(packed.reshape(logical.shape[0],logical.shape[1]//2),
                  torch.ones_like(scales).reshape(logical.shape[0],-1),1.)
    torch.testing.assert_close(actual.reshape(reference.values.shape),reference.values,rtol=0,atol=0)
    assert_corrected_scales(scales,reference)
    torch.testing.assert_close(partials.amax(),scales.float().amax(),rtol=0,atol=0)


@pytest.mark.parametrize("bits",[3,4])
@pytest.mark.parametrize("rows,groups",[(3,7),(128,8),(512,96)])
def test_two_pass_extended_preparation(rows,groups,bits):
    from benchmarks.reference_ms_eden import prepare_extended,finalize_scales
    from benchmarks.compare_nvfp4_packed_gemm import decode
    from rtx.kernels.nvfp4_posthoc import compile_extended_preparation,compile_posthoc_scales,PosthocScaleConfig
    torch.manual_seed(805)
    x=torch.randn(rows,groups*16,device="cuda")*.25
    config=PosthocScaleConfig(128,3)
    producer,count=compile_extended_preparation(rows,groups,config,bits)
    finalizer=compile_posthoc_scales(rows,groups,config,count)
    packed=torch.empty(rows*groups*8,device="cuda",dtype=torch.uint8)
    temporary=torch.empty(rows*groups,device="cuda",dtype=torch.bfloat16)
    partials=torch.empty(count,device="cuda")
    scales=torch.empty(rows*groups,device="cuda",dtype=torch.float8_e4m3fn)
    outer=torch.empty(1,device="cuda")
    u=torch.rand(rows,groups,device="cuda")
    def run():
        producer(x.flatten(),packed,temporary,partials)
        finalizer(temporary,partials,u.flatten(),scales,outer)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        run()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        run()
    for zero in (False,True):
        if zero:
            x.zero_()
        graph.replay()
        reference=prepare_extended(x,mantissa_bits=bits)
        actual_values=decode(packed.reshape(rows,groups*8),torch.ones_like(u),1.).reshape(rows,groups,16)
        torch.testing.assert_close(actual_values,reference.values,rtol=0,atol=0)
        # Independent FP32 association may cross a BF16 midpoint only.
        assert_corrected_scales(temporary,reference)
        torch.testing.assert_close(partials.amax(),temporary.float().amax(),rtol=0,atol=0)
        # Finalizer remains bit-exact against its actual BF16 input contract.
        expected,factor=finalize_scales(temporary.reshape(rows,groups),u)
        torch.testing.assert_close(scales.view(torch.uint8),expected.flatten().view(torch.uint8),rtol=0,atol=0)
        torch.testing.assert_close(outer,factor.reshape(1),rtol=0,atol=0)


@pytest.mark.parametrize("rows,groups,layout", [(3,7,"row_major"),(128,8,"row_major"),(256,16,"mma128")])
@pytest.mark.parametrize("threads", [64,128,256,512])
def test_posthoc_scale_finalization_matches_independent_oracle(rows,groups,layout,threads):
    from benchmarks.reference_ms_eden import finalize_scales
    from benchmarks.compare_nvfp4_packed_gemm import native_scales
    from rtx.kernels.nvfp4_posthoc import compile_posthoc_scales, PosthocScaleConfig
    torch.manual_seed(801)
    src = (torch.rand(rows,groups,device="cuda")*.03).to(torch.bfloat16)
    u = torch.rand(rows,groups,device="cuda")
    out = torch.empty(rows*groups,device="cuda",dtype=torch.float8_e4m3fn)
    outer = torch.empty(1,device="cuda")
    runner = compile_posthoc_scales(rows,groups,PosthocScaleConfig(threads,3,layout))
    for zero in (False, True):
        if zero:
            src.zero_()
        maximum = src.float().amax().reshape(1)
        expected,expected_outer = finalize_scales(src,u)
        runner(src.flatten(),maximum,u.flatten(),out,outer)
        if layout == "mma128":
            expected=native_scales(expected)
        torch.testing.assert_close(out.view(torch.uint8),expected.flatten().view(torch.uint8),rtol=0,atol=0)
        torch.testing.assert_close(outer,expected_outer.reshape(1),rtol=0,atol=0)


def test_posthoc_fp8_subnormal_scales():
    from benchmarks.reference_ms_eden import finalize_scales
    from rtx.kernels.nvfp4_posthoc import compile_posthoc_scales
    src=torch.tensor([[0.,2**-20,2**-10,2**-9,.003,1.03,448.]],device="cuda",dtype=torch.bfloat16)
    u=torch.tensor([[.0,.0,.1,.9,.3,.8,.9]],device="cuda")
    out=torch.empty(7,device="cuda",dtype=torch.float8_e4m3fn)
    outer=torch.empty(1,device="cuda")
    compile_posthoc_scales(1,7)(src.flatten(),src.float().amax().reshape(1),u.flatten(),out,outer)
    expected,factor=finalize_scales(src,u)
    torch.testing.assert_close(out.view(torch.uint8),expected.flatten().view(torch.uint8),rtol=0,atol=0)
    torch.testing.assert_close(outer,factor.reshape(1),rtol=0,atol=0)


@pytest.mark.parametrize("magnitude", [1e-20,1.,1e20])
def test_posthoc_capture_replays_changed_rng_and_scale_inputs(magnitude):
    from benchmarks.reference_ms_eden import finalize_scales
    from rtx.kernels.nvfp4_posthoc import compile_posthoc_scales, PosthocScaleConfig
    torch.manual_seed(802)
    src = (torch.rand(129,17,device="cuda")*magnitude).to(torch.bfloat16)
    uniforms = torch.rand_like(src,dtype=torch.float32)
    maximum = src.float().amax().reshape(1)
    out = torch.empty(src.numel(),device="cuda",dtype=torch.float8_e4m3fn)
    outer = torch.empty(1,device="cuda")
    runner = compile_posthoc_scales(129,17,PosthocScaleConfig(128,7))
    def run():
        runner(src.flatten(),maximum,uniforms.flatten(),out,outer)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        run()
    for seed in (803,804):
        torch.manual_seed(seed)
        src.copy_((torch.rand_like(src,dtype=torch.float32)*magnitude).to(torch.bfloat16))
        uniforms.uniform_()
        maximum.copy_(src.float().amax().reshape(1))
        graph.replay()
        expected,factor=finalize_scales(src,uniforms)
        torch.testing.assert_close(out.view(torch.uint8),expected.flatten().view(torch.uint8),rtol=0,atol=0)
        torch.testing.assert_close(outer,factor.reshape(1),rtol=0,atol=0)
@pytest.mark.parametrize("phase", ["dx", "dw"])
@pytest.mark.parametrize("distribution", ["gaussian", "student3", "sparse", "outliers"])
def test_stochastic_gradient_study_reproducible(phase, distribution):
    from benchmarks.study_ms_eden_backward import run_study
    first = run_study((128,256,128), phase, "mma128", benchmark=False, stochastic_samples=4, distribution=distribution)
    second = run_study((128,256,128), phase, "mma128", benchmark=False, stochastic_samples=4, distribution=distribution)
    assert first["stochastic"] == second["stochastic"]
    for mode in ("fixed", "resampled"):
        checkpoints = first["stochastic"][mode]["checkpoints"]
        assert [point["samples"] for point in checkpoints] == [1,2,4]
        assert checkpoints[0]["mean_noise_nrms"] == 0
        assert checkpoints[-1]["mean_noise_nrms"] > 0
        assert checkpoints[-1]["reference_rms"] > 0


def test_stochastic_decoded_samples_localize_kernel_error():
    from benchmarks.study_ms_eden_backward import run_study
    result = run_study((128,256,128), "dw", "mma128", benchmark=False,
                       stochastic_samples=4, distribution="sparse", decoded_samples=True, oracle_samples=True)
    for mode in ("fixed", "resampled"):
        for point in result["stochastic"][mode]["checkpoints"]:
            assert point["worst_sample_packed_nrms"] < .005
            assert abs(point["mean_nrms"] - point["decoded_mean_nrms"]) <= point["kernel_vs_decoded_mean_nrms"] + 1e-12
            assert point["decoded_vs_oracle_mean_nrms"] < .005
            assert abs(point["decoded_mean_nrms"] - point["oracle_mean_nrms"]) <= point["decoded_vs_oracle_mean_nrms"] + 1e-12


@pytest.mark.parametrize("phase", ["dx", "dw"])
def test_forward_roundtrip_is_a_distinct_backward_numerical_policy(phase):
    from benchmarks.study_ms_eden_backward import run_study
    from rtx.autotune.posthoc import PosthocPreparationSchedule
    original = run_study((128,256,128),phase,"mma128",benchmark=False)
    reused = run_study((128,256,128),phase,"mma128",benchmark=False,
                       source_policy="forward_current_roundtrip_bf16")
    direct = run_study((128,256,128),phase,"mma128",benchmark=False,
                       source_policy="forward_current_roundtrip_bf16",packed_reader=True)
    staged = run_study((128,256,128),phase,"mma128",benchmark=False,
                       source_policy="forward_current_roundtrip_bf16",packed_reader=True,
                       schedule=PosthocPreparationSchedule(reader_mode="staged",decoder_threads=256,decoder_blocks=512))
    assert original["source_reference_nrms"] == 0
    assert reused["source_reference_nrms"] > .01
    assert reused["policy"] != original["policy"]
    assert reused["metrics"]["packed"]["nrms"] < .005
    assert reused["metrics"]["original_bf16"] != reused["metrics"]["backward_source"]
    assert direct["metrics"] == reused["metrics"]
    assert staged["metrics"] == direct["metrics"]
    assert direct["source_policy"] == reused["source_policy"]


@pytest.mark.parametrize("transpose", [False, True])
@pytest.mark.parametrize("bits", [3,4])
@pytest.mark.parametrize("order", ["row_major", "reduction_major"])
def test_packed_hadamard_matches_explicit_bf16_roundtrip(transpose,bits,order):
    from benchmarks.compare_nvfp4_packed_gemm import decode
    from rtx.kernels.nvfp4_posthoc import (compile_hadamard_preparation,
        compile_packed_hadamard_preparation,PosthocScaleConfig)
    torch.manual_seed(811)
    rows,cols=128,256
    packed=torch.randint(256,(rows,cols//2),device="cuda",dtype=torch.uint8)
    scales=(torch.rand(rows,cols//16,device="cuda")*.3).to(torch.float8_e4m3fn)
    outer=torch.tensor([.17],device="cuda")
    source=decode(packed,scales,outer).bfloat16()
    signs=(torch.randint(2,(128,),device="cuda")*2-1).float()
    config=PosthocScaleConfig(128,3)
    fused,count=compile_packed_hadamard_preparation(rows,cols,config,bits,transpose,order)
    explicit,count2=compile_hadamard_preparation(rows,cols,config,bits,transpose,order)
    assert count == count2
    def outputs():
        return (torch.empty(rows*cols//2,device="cuda",dtype=torch.uint8),
                torch.empty(rows*cols//16,device="cuda",dtype=torch.bfloat16),
                torch.empty(count,device="cuda"))
    actual,expected=outputs(),outputs()
    fused(packed.flatten(),scales.flatten(),outer,signs,*actual)
    explicit(source.flatten(),signs,*expected)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    packed_flat,scales_flat=packed.flatten(),scales.flatten()
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        fused(packed_flat,scales_flat,outer,signs,*actual)
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        fused(packed_flat,scales_flat,outer,signs,*actual)
    packed.zero_()
    outer.fill_(.03125)
    signs.neg_()
    graph.replay()
    source.copy_(decode(packed,scales,outer).bfloat16())
    explicit(source.flatten(),signs,*expected)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
        assert torch.count_nonzero(a) == 0


@pytest.mark.parametrize("shape",[(3,16),(128,256)])
def test_staged_decode_independent_reference(shape):
    from benchmarks.compare_nvfp4_packed_gemm import decode
    from rtx.kernels.nvfp4_posthoc import compile_packed_bf16_decode,PosthocScaleConfig
    rows,cols=shape
    data=torch.randint(256,(rows,cols//2),device="cuda",dtype=torch.uint8)
    scales=torch.rand(rows,cols//16,device="cuda").to(torch.float8_e4m3fn)
    outer=torch.tensor([.137],device="cuda")
    out=torch.empty(rows,cols,device="cuda",dtype=torch.bfloat16)
    run=compile_packed_bf16_decode(rows,cols,PosthocScaleConfig(64,3))
    run(data.flatten(),scales.flatten(),outer,out.flatten())
    torch.testing.assert_close(out,decode(data,scales,outer).bfloat16(),rtol=0,atol=0)
