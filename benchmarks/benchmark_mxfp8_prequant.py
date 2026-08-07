"""Benchmark dynamic BF16 quantization plus prequantized SM120 MXFP8 GEMM."""

from __future__ import annotations

import argparse
import atexit
import ctypes
from dataclasses import asdict
import json
from pathlib import Path
import statistics

import torch

from rtx.kernels.mxfp8 import MXFP8Problem
from rtx.kernels.mxfp8_gemm import MXFP8GemmConfig, compile_mxfp8_gemm
from rtx.kernels.mxfp8_quant import (
    MXFP8QuantConfig,
    compile_mxfp8_dual_quant,
    compile_mxfp8_quant,
)


def _reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    def quantize(src: torch.Tensor):
        rows, k = src.shape
        blocks = src.reshape(rows, k // 32, 32)
        amax = blocks.abs().amax(dim=-1, keepdim=True).float()
        exponent = ((amax.view(torch.int32) >> 23) & 0xFF) - 127
        code = ((exponent - 8).clamp(-127, 128) + 127).to(torch.uint8)
        code = torch.where(torch.isnan(amax), torch.full_like(code, 255), code)
        scale = (code.int() << 23).view(torch.float32).clamp_min(2**-126)
        q = (blocks.float() / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
        return q.reshape(rows, k), code.view(torch.float8_e8m0fnu).squeeze(-1)

    qx, sx = quantize(x)
    qw, sw = quantize(weight)
    dx = qx.float() * sx.float().repeat_interleave(32, dim=-1)
    dw = qw.float() * sw.float().repeat_interleave(32, dim=-1)
    return (dx @ dw.T).bfloat16()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=1536)
    parser.add_argument("--k", type=int, default=1536)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--samples", type=int, default=51)
    parser.add_argument("--calls-per-sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quant-launches",
        choices=("dual", "separate"),
        default="dual",
    )
    parser.add_argument(
        "--scale-layout",
        choices=("row_major", "mma128", "mma64x128"),
        default="row_major",
    )
    parser.add_argument(
        "--component",
        choices=("e2e", "quant", "gemm"),
        default="e2e",
    )
    parser.add_argument(
        "--l2-fetch-granularity",
        type=int,
        choices=(0, 32, 64, 128),
        help="temporarily set cudaLimitMaxL2FetchGranularity for this process",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        parser.error("benchmark requires an SM120/SM121 CUDA GPU")
    previous_fetch_granularity = None
    if args.l2_fetch_granularity is not None:
        site_packages = Path(torch.__file__).resolve().parent.parent
        runtime_path = next(
            (site_packages / "nvidia").glob("cu*/lib/libcudart.so.*")
        )
        runtime = ctypes.CDLL(str(runtime_path))
        runtime.cudaDeviceGetLimit.argtypes = [
            ctypes.POINTER(ctypes.c_size_t),
            ctypes.c_int,
        ]
        runtime.cudaDeviceSetLimit.argtypes = [ctypes.c_int, ctypes.c_size_t]
        value = ctypes.c_size_t()
        if runtime.cudaDeviceGetLimit(ctypes.byref(value), 5) != 0:
            raise RuntimeError("cudaDeviceGetLimit(MaxL2FetchGranularity) failed")
        previous_fetch_granularity = value.value
        if runtime.cudaDeviceSetLimit(5, args.l2_fetch_granularity) != 0:
            raise RuntimeError("cudaDeviceSetLimit(MaxL2FetchGranularity) failed")
        atexit.register(runtime.cudaDeviceSetLimit, 5, previous_fetch_granularity)
    torch.manual_seed(args.seed)
    problem = MXFP8Problem(args.m, args.n, args.k)
    x = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(args.n, args.k, device="cuda", dtype=torch.bfloat16)
    qx = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    qw = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    out = torch.empty(args.m, args.n, device="cuda", dtype=torch.bfloat16)

    weight_scale_layout = None
    if args.scale_layout == "row_major":
        quant_config = MXFP8QuantConfig()
        gemm_config = MXFP8GemmConfig()
        sx = torch.empty(
            args.m, args.k // 32, device="cuda", dtype=torch.float8_e8m0fnu
        )
        sw = torch.empty(
            args.n, args.k // 32, device="cuda", dtype=torch.float8_e8m0fnu
        )
    elif args.scale_layout == "mma128":
        quant_config = MXFP8QuantConfig(
            load_bits=32,
            maxrregcount=96,
            persistent_waves=6,
            quant_amax="fp32",
            quant_math="fp32",
            scale_layout="mma128",
        )
        gemm_config = MXFP8GemmConfig(
            atom_layout_m=4,
            b_swizzle="128b",
            consumer_registers=232,
            producer_registers=64,
            scale_role="tma",
            scale_layout="mma128",
        )
        sx = torch.empty(
            args.m // 128,
            args.k // 128,
            512,
            device="cuda",
            dtype=torch.float8_e8m0fnu,
        )
        sw = torch.empty(
            args.n // 128,
            args.k // 128,
            512,
            device="cuda",
            dtype=torch.float8_e8m0fnu,
        )
    else:
        quant_config = MXFP8QuantConfig(scale_layout="mma64")
        weight_scale_layout = "mma128"
        gemm_config = MXFP8GemmConfig(
            tile_m=64,
            atom_layout_m=2,
            stages=1,
            scale_role="tma",
            scale_layout="mma64x128",
        )
        sx = torch.empty(
            args.m // 64,
            args.k // 128,
            512,
            device="cuda",
            dtype=torch.float8_e8m0fnu,
        )
        sw = torch.empty(
            args.n // 128,
            args.k // 128,
            512,
            device="cuda",
            dtype=torch.float8_e8m0fnu,
        )
    quant_x = compile_mxfp8_quant(args.m, args.k, quant_config)
    weight_quant_config = MXFP8QuantConfig(
        **{
            **asdict(quant_config),
            "scale_layout": weight_scale_layout or quant_config.scale_layout,
        }
    )
    quant_w = compile_mxfp8_quant(args.n, args.k, weight_quant_config)
    quant_dual = compile_mxfp8_dual_quant(
        args.m,
        args.n,
        args.k,
        quant_config,
        weight_scale_layout=weight_scale_layout,
    )
    gemm = compile_mxfp8_gemm(problem, gemm_config)

    def quantize() -> None:
        if args.quant_launches == "dual":
            quant_dual(x, weight, qx, qw, sx, sw)
        else:
            quant_x(x, qx, sx)
            quant_w(weight, qw, sw)

    def run() -> None:
        quantize()
        gemm(qx, qw, sx, sw, out)

    run()
    torch.cuda.synchronize()
    expected = _reference(x, weight)
    torch.cuda.synchronize()
    # The tensor-core K-tile accumulation order is not identical to PyTorch's
    # dequantized FP32 GEMM reduction order. Quantized values/scales are checked
    # bit-for-bit in tests; allow only final BF16 rounding-level differences.
    torch.testing.assert_close(out, expected, rtol=0.05, atol=0.5)

    measured = run
    if args.component == "quant":
        measured = quantize
    elif args.component == "gemm":
        measured = lambda: gemm(qx, qw, sx, sw, out)

    for _ in range(args.warmup):
        measured()
    torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(args.samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _call in range(args.calls_per_sample):
            measured()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)) / args.calls_per_sample)

    median_ms = statistics.median(timings)
    print(
        json.dumps(
            {
                "shape": {"m": args.m, "n": args.n, "k": args.k},
                "median_ms": median_ms,
                "min_ms": min(timings),
                "max_ms": max(timings),
                "tflops": 2 * args.m * args.n * args.k / (median_ms * 1e9),
                "quant_config": asdict(quant_config),
                "weight_quant_config": asdict(weight_quant_config),
                "quant_launches": args.quant_launches,
                "scale_layout": args.scale_layout,
                "component": args.component,
                "l2_fetch_granularity": args.l2_fetch_granularity,
                "previous_l2_fetch_granularity": previous_fetch_granularity,
                "gemm_config": asdict(gemm_config),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
