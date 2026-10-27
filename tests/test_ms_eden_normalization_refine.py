"""Guarded exact shared-reciprocal refinement; GPU checks require opt-in.

The CPU model forms the small first residual exactly in binary64. Final
binary64 multiply/add rounding error is far below the proven 1/62 FP32-ULP
distance to a midpoint, so it cannot change the final FP32 result here.
This is not a generic exact-FMA simulator; explicit GPU div.rn remains a gate.
"""
from dataclasses import replace
import inspect
import json
import os

import numpy as np
import pytest
import torch


def refine_cpu_normalized(x, denominator):
    m = np.float32(denominator / 16)
    reciprocal = np.divide(np.float32(1), m, dtype=np.float32)
    estimate = np.multiply(x, reciprocal, dtype=np.float32)
    exact_residual = x.astype(np.float64) - estimate.astype(np.float64) * np.float64(m)
    residual = exact_residual.astype(np.float32)
    assert np.array_equal(residual.astype(np.float64), exact_residual)
    return (residual.astype(np.float64) * np.float64(reciprocal) + estimate.astype(np.float64)).astype(np.float32)


def check_significands(count, chunk=1 << 18):
    checked = 0
    for exponent in (-90, 0, 3):
        for sign in (0, 1):
            for base in range(0, count, chunk):
                # Full count enumerates all significands; small count spreads
                # samples across the same complete interval, not only near 1.
                index = np.arange(base, min(base + chunk, count), dtype=np.uint32)
                mantissa = index if count == 1 << 23 else index * np.uint32((1 << 23) // count)
                words = mantissa | np.uint32((exponent + 127) << 23) | np.uint32(sign << 31)
                values = words.view(np.float32)
                for denominator in range(16, 32):
                    actual = refine_cpu_normalized(values, denominator)
                    expected = np.divide(values, np.float32(denominator / 16), dtype=np.float32)
                    np.testing.assert_array_equal(actual.view(np.uint32), expected.view(np.uint32))
                    checked += len(values)
    return checked


def test_cpu_representative_significands_both_signs_all_denominators():
    assert check_significands(4096) == 393216


@pytest.mark.skipif(os.environ.get("RTX_RUN_REFINE_EXHAUSTIVE_CPU") != "1",
                    reason="explicit exhaustive CPU evidence run")
def test_cpu_exhaustive_significands():
    assert check_significands(1 << 23) == 805306368


def test_refine_is_opt_in_roundtrips_and_has_distinct_cache_features():
    from rtx.autotune.posthoc import PosthocPreparationSchedule, make_posthoc_preparation_adapter
    from rtx.kernels.nvfp4_posthoc import compile_hadamard_preparation, compile_packed_hadamard_preparation
    from benchmarks.compare_ms_eden_pipeline import schedule_pair
    base = PosthocPreparationSchedule(a_geometry={"producer_blocks": 32})
    assert base.normalization_math == "divide"
    adapter = make_posthoc_preparation_adapter((512, 512, 512), "dx", "mma128", lambda _: None)
    candidate = replace(base, normalization_math="refine")
    assert adapter.axes["normalization_math"] == ("divide", "refine", "threshold_scale_free", "refine_warp_lut", "guarded_scale_free")
    assert adapter.rejection(candidate) is None
    assert adapter.config_id(candidate) != adapter.config_id(base)
    assert adapter.deserialize(adapter.serialize(candidate)) == candidate
    for operand in ("a", "b"):
        assert candidate.operand_schedule(operand).normalization_math == "refine"
        assert adapter.extra_features_fn(candidate)[f"{operand}_normalization_refine"] == 1.
        assert adapter.extra_features_fn(base)[f"{operand}_normalization_refine"] == 0.
    assert adapter.context.tags["schedule_features_revision"] == 22
    assert all(json.loads(value)["normalization_math"] == "refine"
               for value in schedule_pair({"normalization_math": "refine"}).values())
    with pytest.raises(ValueError, match="normalization math"):
        PosthocPreparationSchedule(normalization_math="approximate")
    for compile in (compile_hadamard_preparation, compile_packed_hadamard_preparation):
        assert inspect.signature(compile).parameters["normalization_math"].default == "divide"
        with pytest.raises(ValueError, match="normalization math"):
            compile(128, 128, normalization_math="approximate")


run_cuda = pytest.mark.skipif(os.environ.get("RTX_RUN_REFINE_CUDA") != "1",
                              reason="explicit assigned-GPU qualification opt-in")


@run_cuda
def test_gpu_refinement_matches_independent_explicit_division_all_denominators_and_fallbacks(
        scale_exponents=(-126,-90,0,90,126,127), normalized_extra=()):
    import cutlass
    import cutlass.cute as cute
    import cuda.bindings.driver as cuda
    from cutlass import Float32
    from cutlass.experimental.primitives import nvvm_wrapper as nvvm
    from rtx.kernels.nvfp4_posthoc import HadamardPreparationKernel
    inputs, scales = [], []
    generator = np.random.default_rng(948)
    normalized = (generator.integers(0, 1 << 23, size=1024, dtype=np.uint32)
                  | np.uint32(127 << 23)).view(np.float32)
    normalized = np.concatenate((normalized, -normalized,
        np.array([0., -0., 2.**-90, -2.**-90, 16., -16., 2.**-91, 32., np.inf, -np.inf, np.nan], dtype=np.float32)))
    normalized=np.concatenate((normalized,np.asarray(normalized_extra,dtype=np.float32)))
    original_edges = np.array([0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000,
        0x7f7fffff, 0xff7fffff, 0x7f800000, 0xff800000, 0x7f800001, 0x7fc00000], dtype=np.uint32).view(np.float32)
    with np.errstate(all="ignore"):
        for exponent in scale_exponents:
            for denominator in range(16, 32):
                values = np.concatenate((np.ldexp(normalized, exponent, dtype=np.float32), original_edges))
                values = np.pad(values, (0, (-len(values)) % 4))
                inputs.extend(values)
                scales.extend([np.ldexp(np.float32(denominator / 16), exponent, dtype=np.float32)] * (len(values) // 4))
    # Unsupported denominators, scales, and special values must use div.rn.
    for scale_bits in (0, 0x80000000, 1, 0x007fffff, 0xbf800000, 0x3f812345,
                       0x7f800000, 0xff800000, 0x7f800001, 0x7fc00000):
        inputs.extend(original_edges)
        scales.extend([np.array([scale_bits], dtype=np.uint32).view(np.float32)[0]] * 3)

    class DivisionProbe(HadamardPreparationKernel):
        def __init__(self, groups):
            self.groups = groups

        @cute.jit
        def __call__(self, source: cute.Tensor, scale: cute.Tensor, actual: cute.Tensor,
                     expected: cute.Tensor, stream: cuda.CUstream):
            self.probe(source, scale, actual, expected).launch(
                grid=((self.groups + 127) // 128, 1, 1), block=(128, 1, 1), stream=stream)

        @cute.kernel
        def probe(self, source: cute.Tensor, scale: cute.Tensor, actual: cute.Tensor, expected: cute.Tensor):
            thread, _, _ = cute.arch.thread_idx()
            block, _, _ = cute.arch.block_idx()
            group = block * 128 + thread
            if group < self.groups:
                divisor = scale[group]
                eligible, m, power, reciprocal = self.normalization_parameters(divisor)
                for j in cutlass.range_constexpr(4):
                    value = source[group * 4 + j]
                    actual[group * 4 + j] = self.normalize_refined(value, divisor, eligible, m, power, reciprocal)
                    expected[group * 4 + j] = nvvm.inline_ptx_hl("div.rn.f32 {$w0}, {$r0}, {$r1};",
                        write_only_types=[Float32], read_only_args=[value, divisor])
    source = torch.tensor(np.asarray(inputs), device="cuda")
    divisors = torch.tensor(np.asarray(scales), device="cuda")
    actual, expected = torch.empty_like(source), torch.empty_like(source)
    fake = lambda size: cute.runtime.make_fake_tensor(Float32, (size,), stride=(1,), assumed_align=4)
    runner = cute.compile(DivisionProbe(len(scales)), fake(len(inputs)), fake(len(scales)),
        fake(len(inputs)), fake(len(inputs)), cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 3")
    runner(source, divisors, actual, expected)
    torch.testing.assert_close(actual.view(torch.int32), expected.view(torch.int32), atol=0, rtol=0)


@run_cuda
@pytest.mark.parametrize("shape,transpose,buffered,mantissa_bits", [
    ((128, 128), False, False, 3), ((128, 128), True, False, 4),
    ((4096, 512), False, False, 4), ((4096, 512), True, True, 4)])
def test_refined_producer_finalizer_byte_exact_and_changed_graph_buffers(shape, transpose, buffered, mantissa_bits):
    from rtx.kernels.nvfp4_posthoc import compile_hadamard_preparation, compile_posthoc_scales, PosthocScaleConfig
    torch.manual_seed(949)
    rows, cols = shape[::-1] if transpose else shape
    source = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
    signs = torch.ones(128, device="cuda")
    uniforms = torch.empty(source.numel() // 16, device="cuda")
    config = PosthocScaleConfig(128, 256)
    kwargs = dict(mantissa_bits=mantissa_bits, transpose=transpose,
        tile_order="reduction_major" if transpose else "row_major", load_mode="coalesced_wide",
        signs_mode="register", stages=2 if buffered else 1, async_copy=buffered,
        correction_decode="native", partial_max_scope="cta")
    baseline, count = compile_hadamard_preparation(*shape, config, **kwargs)
    candidate, candidate_count = compile_hadamard_preparation(*shape, config, normalization_math="refine", **kwargs)
    assert count == candidate_count
    finalizer = compile_posthoc_scales(rows, cols // 16,
        PosthocScaleConfig(128, 224, "mma128", probability_math="power2"), count)
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
    source.zero_(); uniforms.fill_(.5)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): run(candidate, actual)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream): run(candidate, actual)
    for distribution in ("zero", "negative_zero", "gaussian", "sparse", "outliers", "tiny"):
        source.normal_(0, .25)
        if distribution in ("zero", "negative_zero"):
            source.fill_(-0. if distribution == "negative_zero" else 0.)
        elif distribution == "sparse": source.mul_((torch.rand(shape, device="cuda") < .05).to(source.dtype))
        elif distribution == "outliers": source.flatten()[::97].mul_(32)
        elif distribution == "tiny": source.mul_(1e-35)
        signs.random_(0, 2).mul_(2).sub_(1); uniforms.uniform_()
        run(baseline, expected); graph.replay(); torch.cuda.synchronize()
        for reference, result in zip(expected, actual):
            assert torch.isfinite(result.float()).all()
            torch.testing.assert_close(result.view(torch.uint8), reference.view(torch.uint8), atol=0, rtol=0)
