"""CPU policy contracts plus explicit opt-in native E2M1 decode qualification.

No approximate tolerance: FP32 decode bits (including -0), packed bytes,
corrected BF16 scales, partial maxima, finalized FP8 scales and factor match.
"""
import inspect
import os
import struct
from contextlib import nullcontext
from dataclasses import asdict, replace
import json
from types import SimpleNamespace

import pytest
import torch


def fp4_bits(code):
    """Independent E2M1 format oracle, not the production integer LUT."""
    exponent, mantissa = (code >> 1) & 3, code & 1
    magnitude = mantissa * .5 if exponent == 0 else (1 + mantissa * .5) * 2 ** (exponent - 1)
    value = -magnitude if code & 8 else magnitude
    return struct.unpack("i", struct.pack("f", value))[0]


def test_native_decode_is_explicit_opt_in_and_invalid_modes_fail_before_compile():
    from rtx.kernels.nvfp4_posthoc import (
        HadamardPreparationKernel, PosthocScaleConfig, compile_hadamard_preparation,
        compile_packed_hadamard_preparation)
    for compile in (compile_hadamard_preparation, compile_packed_hadamard_preparation):
        assert inspect.signature(compile).parameters["correction_decode"].default == "lut"
        with pytest.raises(ValueError, match="decode mode"):
            compile(128, 128, correction_decode="approximate")
    for mode in ("lut", "native"):
        kernel = HadamardPreparationKernel(128, 128, PosthocScaleConfig(), 4, False,
                                           correction_decode=mode)
        assert kernel.correction_decode == mode


def test_independent_e2m1_oracle_includes_both_signed_zeros():
    assert fp4_bits(0) == 0
    assert fp4_bits(8) == -(1 << 31)
    assert struct.unpack("f", struct.pack("i", fp4_bits(7)))[0] == 6.
    assert struct.unpack("f", struct.pack("i", fp4_bits(15)))[0] == -6.


def test_native_schedule_adapter_identity_features_and_comparator_json():
    from rtx.autotune.posthoc import PosthocPreparationSchedule, make_posthoc_preparation_adapter
    from benchmarks.compare_ms_eden_pipeline import schedule_pair
    baseline = PosthocPreparationSchedule(a_geometry={"producer_blocks": 32})
    native = replace(baseline, correction_decode="native")
    for operand in ("a", "b"):
        assert native.operand_schedule(operand).correction_decode == "native"
    assert native.operand_schedule("a").producer_blocks == 32
    assert PosthocPreparationSchedule(**json.loads(json.dumps(asdict(native)))) == native
    adapter = make_posthoc_preparation_adapter((512, 512, 512), "dx", "mma128", lambda _: None)
    assert adapter.axes["correction_decode"] == ("lut", "native")
    assert adapter.rejection(native) is None
    assert adapter.config_id(native) != adapter.config_id(baseline)
    assert adapter.deserialize(adapter.serialize(native)) == native
    for operand in ("a", "b"):
        assert adapter.extra_features_fn(native)[f"{operand}_correction_decode_native"] == 1.
        assert adapter.extra_features_fn(baseline)[f"{operand}_correction_decode_native"] == 0.
    assert adapter.context.tags["schedule_features_revision"] == 22
    assert all(json.loads(encoded)["correction_decode"] == "native"
               for encoded in schedule_pair({"correction_decode": "native"}).values())
    with pytest.raises(ValueError, match="correction decode"):
        PosthocPreparationSchedule(correction_decode="approximate")


@pytest.mark.parametrize("packed", [False, True])
def test_native_runtime_and_study_forward_mode_without_losing_operand_geometry(monkeypatch, packed):
    """Wrapper routing only: CPU stand-ins never launch or prove GPU kernels."""
    import rtx.fp4 as fp4
    import rtx.kernels.nvfp4_posthoc as kernels
    import rtx.experimental_ms_eden as eden
    from benchmarks.study_ms_eden_backward import prepare
    from rtx.autotune.posthoc import PosthocPreparationSchedule
    calls = []
    def producer(kind):
        def compile(*args, **kwargs):
            assert kwargs["normalization_math"] == "refine"
            calls.append((kind, args[2], kwargs["correction_decode"]))
            return (lambda *args: None), 4
        return compile
    monkeypatch.setattr(kernels, "compile_hadamard_preparation", producer("bf16"))
    monkeypatch.setattr(kernels, "compile_packed_hadamard_preparation", producer("packed"))
    monkeypatch.setattr(kernels, "compile_posthoc_scales", lambda *args: lambda *args: None)
    monkeypatch.setattr(kernels, "compile_posthoc_scale_product", lambda: lambda *args: None)
    monkeypatch.setattr(fp4, "compile_nvfp4_gemm", lambda *args: lambda *args: None)
    monkeypatch.setattr(fp4, "_packed_fp4_view", lambda value: value)
    monkeypatch.setattr(fp4, "_check_sm12x", lambda device: None)
    monkeypatch.setattr(torch.cuda, "device", lambda device: nullcontext())
    source = torch.empty(128, 128, dtype=torch.bfloat16)
    codes = torch.empty(128, 64, dtype=torch.uint8)
    scales = torch.empty(128 * 8, dtype=torch.float8_e4m3fn)
    outer = torch.ones(1)
    signs, uniforms = torch.ones(128), torch.zeros(128 * 8)
    schedule = PosthocPreparationSchedule(correction_decode="native", normalization_math="refine",
        a_geometry={"producer_blocks": 32}, b_geometry={"producer_blocks": 64})
    eden._build_gradient(source, codes if packed else source, signs, uniforms, uniforms,
        "dx", schedule, (128, 128, 128), scales if packed else None, outer if packed else None)
    assert [(kind, config.blocks, mode) for kind, config, mode in calls] == [
        ("bf16", 32, "native"), ("packed" if packed else "bf16", 64, "native")]
    calls.clear()
    saved = SimpleNamespace(qdata=codes, scale=scales, per_tensor_scale=outer) if packed else None
    prepare(source, True, "mma128", schedule, packed_source=saved, operand="b")
    assert [(kind, config.blocks, mode) for kind, config, mode in calls] == [
        ("packed" if packed else "bf16", 64, "native")]


run_cuda = pytest.mark.skipif(os.environ.get("RTX_RUN_NATIVE_DECODE_CUDA") != "1",
                              reason="explicit GPU qualification opt-in")


@run_cuda
def test_native_decode_all_256_pairs_exact_fp32_bits():
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from cutlass import Float32, Int32, Uint8
    from rtx.kernels.nvfp4_posthoc import ExtendedPreparationKernel

    class DecodeProbe(ExtendedPreparationKernel):
        def __init__(self, mode):
            self.mode = mode

        @cute.jit
        def __call__(self, source: cute.Tensor, output: cute.Tensor, stream: cuda.CUstream):
            self.probe(source, output).launch(grid=(1, 1, 1), block=(256, 1, 1), stream=stream)

        @cute.kernel
        def probe(self, source: cute.Tensor, output: cute.Tensor):
            index, _, _ = cute.arch.thread_idx()
            a, b = self.decode_pair(Int32(source[index]), self.mode)
            output[index * 2], output[index * 2 + 1] = a, b

    source = torch.arange(256, device="cuda").to(torch.uint8)
    expected = torch.tensor([fp4_bits(code) for byte in range(256)
                             for code in (byte & 15, byte >> 4)], dtype=torch.int32, device="cuda")
    for mode in ("lut", "native"):
        output = torch.empty(512, device="cuda")
        fake = lambda dtype, size: cute.runtime.make_fake_tensor(dtype, (size,), stride=(1,), assumed_align=4)
        runner = cute.compile(DecodeProbe(mode), fake(Uint8, 256), fake(Float32, 512),
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi --opt-level 3")
        runner(source, output)
        torch.testing.assert_close(output.view(torch.int32), expected, atol=0, rtol=0)


@run_cuda
@pytest.mark.parametrize("shape,transpose,buffered", [
    ((128, 128), False, False), ((128, 128), True, False),
    ((4096, 512), False, False), ((4096, 512), True, True)])
def test_native_producer_and_finalizer_byte_exact_current_buffer_replay(shape, transpose, buffered):
    from rtx.kernels.nvfp4_posthoc import (
        PosthocScaleConfig, compile_hadamard_preparation, compile_posthoc_scales)
    torch.manual_seed(941)
    rows, cols = shape[::-1] if transpose else shape
    source = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    signs = torch.ones(128, device="cuda")
    uniforms = torch.empty(source.numel() // 16, device="cuda")
    config = PosthocScaleConfig(128, 256)
    kwargs = dict(transpose=transpose, tile_order="reduction_major" if transpose else "row_major",
        load_mode="coalesced_wide", signs_mode="register", stages=2 if buffered else 1,
        async_copy=buffered)
    baseline, count = compile_hadamard_preparation(*shape, config, **kwargs)
    native, native_count = compile_hadamard_preparation(*shape, config, correction_decode="native", **kwargs)
    assert native_count == count
    # MMA layout exercises the production finalizer's physical scale mapping.
    finalizer = compile_posthoc_scales(rows, cols // 16, PosthocScaleConfig(128, 224, "mma128"), count)
    def buffers():
        return (torch.empty(source.numel() // 2, device="cuda", dtype=torch.uint8),
                torch.empty(source.numel() // 16, device="cuda", dtype=torch.bfloat16),
                torch.empty(count, device="cuda"),
                torch.empty(source.numel() // 16, device="cuda", dtype=torch.float8_e4m3fn),
                torch.empty(1, device="cuda"))
    expected, actual = buffers(), buffers()
    def run(producer, outputs):
        packed, corrected, partials, scales, factor = outputs
        producer(source.flatten(), signs, packed, corrected, partials)
        finalizer(corrected, partials, uniforms, scales, factor)
    source.zero_()
    uniforms.fill_(.5)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            run(native, actual)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        run(native, actual)
    for distribution in ("zero", "negative_zero", "gaussian", "sparse", "outliers"):
        source.normal_(0, .25)
        if distribution in ("zero", "negative_zero"):
            source.fill_(-0. if distribution == "negative_zero" else 0.)
        elif distribution == "sparse":
            source.mul_((torch.rand(shape, device="cuda") < .05).to(source.dtype))
        elif distribution == "outliers":
            source.flatten()[::97].mul_(32)
        signs.random_(0, 2).mul_(2).sub_(1)
        uniforms.uniform_()
        run(baseline, expected)
        graph.replay()
        torch.cuda.synchronize()
        for reference, result in zip(expected, actual):
            assert torch.isfinite(result.float()).all()
            torch.testing.assert_close(result.view(torch.uint8), reference.view(torch.uint8), atol=0, rtol=0)
