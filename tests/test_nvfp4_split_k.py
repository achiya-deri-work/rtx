"""Research NVFP4 split-K: FP32 partials, one outer factor, one BF16 cast.

CPU tests prove configuration/ABI contracts only. GPU numerical/capture tests
are explicitly enabled on the assigned idle device; no timing is claimed.
"""
from dataclasses import replace
import inspect
import os
from types import SimpleNamespace

import pytest
import torch

from rtx.configs.nvfp4 import NVFP4GemmConfig, NVFP4Problem
from rtx.kernels import nvfp4_gemm as kernels


DIRECT = NVFP4GemmConfig(epilogue="direct", store_vec=1)


def test_split_grid_workspace_and_raw_partial_contract():
    problem = NVFP4Problem(512, 512, 4096)
    baseline = kernels.NVFP4GemmKernel(problem, DIRECT)
    split = kernels.NVFP4SplitGemmKernel(problem, DIRECT, split_reduction=2, reduction_tile=2048)
    assert baseline.total_work_tiles == 16
    assert split.total_work_tiles == 32 and split.grid_ctas == 32
    assert split.workspace_elements == 2 * 512 * 512
    assert split.workspace_bytes == 2 * 512 * 512 * 4
    assert not split.apply_output_scale and not split.atomic_output and not split.cluster_output
    assert baseline.apply_output_scale and baseline.split_reduction == 1
    # New research entry points do not change the original compiler's ABI.
    signature=inspect.signature(kernels.compile_nvfp4_gemm)
    assert tuple(signature.parameters) == ("problem", "config", "outer_factor_mode")
    assert signature.parameters["outer_factor_mode"].kind==inspect.Parameter.KEYWORD_ONLY
    assert signature.parameters["outer_factor_mode"].default=="product_kernel"
    assert NVFP4GemmConfig().epilogue == "tma"


@pytest.mark.parametrize("split,tile", [(1, 4096), (3, 1536), (64, 128), (True, 4096),
    (2, True), (2, 0), (2, 2000), (4, 2048), (4, 512)])
def test_invalid_split_geometry_fails_before_compilation(split, tile):
    with pytest.raises(ValueError):
        kernels.NVFP4SplitGemmKernel(NVFP4Problem(512, 512, 4096), DIRECT,
                                     split_reduction=split, reduction_tile=tile)


@pytest.mark.parametrize("changes", [dict(epilogue="tma"), dict(tiles_per_cta=2),
                                     dict(persistent_waves=1)])
def test_split_rejects_incompatible_schedule_instead_of_rewriting(changes):
    with pytest.raises(ValueError):
        kernels.NVFP4SplitGemmKernel(NVFP4Problem(512, 512, 4096), replace(DIRECT, **changes),
                                     split_reduction=2, reduction_tile=2048)


def test_unequal_final_partition_is_legal_and_nonempty():
    split = kernels.NVFP4SplitGemmKernel(NVFP4Problem(128, 256, 896), DIRECT,
                                         split_reduction=4, reduction_tile=256)
    assert [(part * split.reduction_tile, min((part + 1) * split.reduction_tile, split.problem.k))
            for part in range(split.split_reduction)] == [(0, 256), (256, 512), (512, 768), (768, 896)]


def test_compilers_use_fp32_workspace_and_separate_final_factor(monkeypatch):
    from cutlass import BFloat16, Float32
    records = []
    def fake(dtype, shape, **kwargs):
        return SimpleNamespace(dtype=dtype, shape=shape, kwargs=kwargs)
    def compile(kernel, *args, **kwargs):
        record = dict(kernel=kernel, tensors=args[:-1], launched=[])
        records.append(record)
        return lambda *values: record["launched"].append(values)
    monkeypatch.setattr(kernels.cute.runtime, "make_fake_tensor", fake)
    monkeypatch.setattr(kernels.cute, "compile", compile)
    kernels.compile_nvfp4_split_gemm.cache_clear()
    kernels.compile_nvfp4_split_workspace_reduce.cache_clear()
    try:
        partial = kernels.compile_nvfp4_split_gemm(NVFP4Problem(128, 256, 512), DIRECT,
            split_reduction=2, reduction_tile=256)
        record = records[-1]
        assert record["tensors"][-2].dtype == Float32
        assert record["tensors"][-2].shape == (2 * 128 * 256,)
        qx, qw, sx, sw, workspace = (object() for _ in range(5))
        partial(qx, qw, sx, sw, workspace)
        assert record["launched"] == [(qx, qw, sx, sw, workspace, workspace)]
        final = kernels.compile_nvfp4_split_workspace_reduce(128, 256, 2)
        record = records[-1]
        assert [tensor.dtype for tensor in record["tensors"]] == [Float32, BFloat16, Float32]
        assert record["tensors"][-1].shape == (1,)
        out, alpha = object(), object()
        final(workspace, out, alpha)
        assert record["launched"] == [(workspace, out, alpha)]
    finally:
        kernels.compile_nvfp4_split_gemm.cache_clear()
        kernels.compile_nvfp4_split_workspace_reduce.cache_clear()


def decode_e2m1(packed, row_scales):
    """Independent format oracle, not a quantization or production kernel."""
    code = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(-2).long()
    exponent, mantissa = (code >> 1) & 3, code & 1
    magnitude = torch.where(exponent == 0, mantissa.double() * .5,
        (1 + mantissa.double() * .5) * torch.pow(2., exponent.double() - 1))
    value = torch.where(code & 8 != 0, -magnitude, magnitude)
    return value * row_scales.double().repeat_interleave(16, dim=1)


def test_independent_decoder_and_pre_bf16_reduction_adversary():
    codes = torch.tensor([[0x10, 0x32, 0x54, 0x76, 0x98, 0xba, 0xdc, 0xfe]], dtype=torch.uint8)
    values = decode_e2m1(codes, torch.ones(1, 1))
    assert values.tolist() == [[0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.]]
    assert values.signbit()[0, 8]
    partials = torch.tensor([1.00390625, -1.], dtype=torch.float32)
    correct = (partials.sum() * .125).bfloat16()
    premature = (partials.bfloat16().float().sum() * .125).bfloat16()
    assert correct != premature and correct == 2.**-11


run_cuda = pytest.mark.skipif(os.environ.get("RTX_RUN_NVFP4_SPLIT_CUDA") != "1",
                              reason="explicit assigned-GPU research qualification")


def gpu_operands(problem, layout, exact=False):
    from rtx.fp4 import _packed_fp4_view
    from benchmarks.compare_nvfp4_packed_gemm import native_scales
    def operand(rows):
        packed = (torch.full((rows, problem.k // 2), 0x22, dtype=torch.uint8, device="cuda") if exact
                  else torch.randint(0, 256, (rows, problem.k // 2), dtype=torch.uint8, device="cuda"))
        scales = (torch.ones(rows, problem.k // 16, device="cuda") if exact
                  else torch.tensor([.5, .75, 1., 1.5, 2.], device="cuda")[
                      torch.randint(0, 5, (rows, problem.k // 16), device="cuda")])
        scales = scales.to(torch.float8_e4m3fn)
        kernel_scales = native_scales(scales) if layout == "mma128" else scales
        return packed, _packed_fp4_view(packed), scales, kernel_scales
    return operand(problem.m), operand(problem.n)


def assert_fp32_oracle(actual, reference, gate=1e-4):
    assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
    error = actual.double() - reference.double()
    nrms = float(error.norm() / reference.double().norm().clamp_min(1e-12))
    assert nrms <= gate, dict(nrms=nrms, max_abs=float(error.abs().max()), gate=gate)


@run_cuda
@pytest.mark.parametrize("shape,split,tile,layout", [
    ((128, 128, 512), 2, 256, "row_major"),
    ((256, 128, 896), 4, 256, "row_major"),
    ((512, 512, 4096), 2, 2048, "mma128"),
    ((128, 256, 4096), 4, 1024, "mma128"),
    ((128, 128, 8192), 8, 1024, "mma128")])
def test_split_fp32_partials_and_final_output_against_unpacked_oracle(shape, split, tile, layout):
    torch.manual_seed(951)
    problem = NVFP4Problem(*shape)
    config = replace(DIRECT, scale_layout=layout, scale_role="tma" if layout == "mma128" else "consumers")
    a, b = gpu_operands(problem, layout)
    workspace = torch.full((split, problem.m, problem.n), torch.nan, device="cuda")
    out = torch.empty(problem.m, problem.n, dtype=torch.bfloat16, device="cuda")
    alpha = torch.tensor([.15625], device="cuda")
    partial = kernels.compile_nvfp4_split_gemm(problem, config, split_reduction=split, reduction_tile=tile)
    final = kernels.compile_nvfp4_split_workspace_reduce(problem.m, problem.n, split)
    def run():
        partial(a[1], b[1], a[3], b[3], workspace.flatten())
        final(workspace.flatten(), out, alpha)
    run()
    # FP64 unpacked-operand GEMMs isolate arithmetic from quantization error.
    decoded_a, decoded_b = decode_e2m1(a[0], a[2]), decode_e2m1(b[0], b[2])
    for part in range(split):
        start, end = part * tile, min((part + 1) * tile, problem.k)
        reference = decoded_a[:, start:end] @ decoded_b[:, start:end].T
        assert_fp32_oracle(workspace[part], reference)
    reference = (decoded_a @ decoded_b.T) * alpha.double()
    # Existing packed GEMM's Gaussian NRMS gate, not a relaxed FP4 policy gate.
    assert_fp32_oracle(out, reference, gate=.005)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): run()
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream): run()
    # Current packed operands and alpha must be read on every replay.
    a[0].bitwise_xor_(0x88)
    alpha.fill_(.3125)
    graph.replay(); torch.cuda.synchronize()
    decoded_a = decode_e2m1(a[0], a[2])
    assert_fp32_oracle(out, (decoded_a @ decoded_b.T) * alpha.double(), gate=.005)
    before = torch.cuda.memory_allocated()
    for _ in range(20): graph.replay()
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == before


@run_cuda
def test_split_exact_dyadic_payload_and_single_outer_factor():
    problem = NVFP4Problem(128, 128, 896)
    a, b = gpu_operands(problem, "row_major", exact=True)
    workspace = torch.empty(4, 128, 128, device="cuda")
    out = torch.empty(128, 128, dtype=torch.bfloat16, device="cuda")
    partial = kernels.compile_nvfp4_split_gemm(problem, DIRECT, split_reduction=4, reduction_tile=256)
    final = kernels.compile_nvfp4_split_workspace_reduce(128, 128, 4)
    partial(a[1], b[1], a[3], b[3], workspace.flatten())
    for part, length in enumerate((256, 256, 256, 128)):
        torch.testing.assert_close(workspace[part], torch.full_like(workspace[part], length), atol=0, rtol=0)
    for factor in (0., -0., .15625, -.3125, 3.1415927):
        alpha = torch.tensor([factor], device="cuda")
        final(workspace.flatten(), out, alpha)
        reference = (torch.full_like(out, 896, dtype=torch.float32) * alpha).bfloat16()
        torch.testing.assert_close(out.view(torch.int16), reference.view(torch.int16), atol=0, rtol=0)


@run_cuda
@pytest.mark.parametrize("algorithm", ["serial", "tree", "persistent_tree"])
def test_reducer_preserves_fp32_partial_bits_until_final_scaled_cast(algorithm):
    workspace = torch.zeros(4, 5, 7, device="cuda")
    workspace[0].fill_(1.00390625); workspace[1].fill_(-1.)
    alpha = torch.tensor([.125], device="cuda")
    out = torch.empty(5, 7, dtype=torch.bfloat16, device="cuda")
    reduce = kernels.compile_nvfp4_split_workspace_reduce(5, 7, 4, algorithm=algorithm)
    reduce(workspace.flatten(), out, alpha)
    reference = (workspace.double().sum(0).float() * alpha).bfloat16()
    torch.testing.assert_close(out.view(torch.int16), reference.view(torch.int16), atol=0, rtol=0)
    assert torch.all(out != 0)
