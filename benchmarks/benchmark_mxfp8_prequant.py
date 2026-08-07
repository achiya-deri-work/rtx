"""Benchmark dynamic BF16 quantization plus prequantized SM120 MXFP8 GEMM."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import statistics

import torch

from rtx.kernels.mxfp8 import MXFP8Problem
from rtx.kernels.mxfp8_gemm import MXFP8GemmConfig, compile_mxfp8_gemm
from rtx.kernels.mxfp8_quant import MXFP8QuantConfig, compile_mxfp8_quant


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
    args = parser.parse_args()

    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        parser.error("benchmark requires an SM120/SM121 CUDA GPU")
    torch.manual_seed(args.seed)
    problem = MXFP8Problem(args.m, args.n, args.k)
    x = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(args.n, args.k, device="cuda", dtype=torch.bfloat16)
    qx = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    qw = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    sx = torch.empty(
        args.m, args.k // 32, device="cuda", dtype=torch.float8_e8m0fnu
    )
    sw = torch.empty(
        args.n, args.k // 32, device="cuda", dtype=torch.float8_e8m0fnu
    )
    out = torch.empty(args.m, args.n, device="cuda", dtype=torch.bfloat16)

    quant_config = MXFP8QuantConfig()
    gemm_config = MXFP8GemmConfig()
    quant_x = compile_mxfp8_quant(args.m, args.k, quant_config)
    quant_w = compile_mxfp8_quant(args.n, args.k, quant_config)
    gemm = compile_mxfp8_gemm(problem, gemm_config)

    def run() -> None:
        quant_x(x, qx, sx)
        quant_w(weight, qw, sw)
        gemm(qx, qw, sx, sw, out)

    run()
    torch.cuda.synchronize()
    expected = _reference(x, weight)
    torch.cuda.synchronize()
    # The tensor-core K-tile accumulation order is not identical to PyTorch's
    # dequantized FP32 GEMM reduction order. Quantized values/scales are checked
    # bit-for-bit in tests; allow only final BF16 rounding-level differences.
    torch.testing.assert_close(out, expected, rtol=0.05, atol=0.5)

    for _ in range(args.warmup):
        run()
    torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(args.samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _call in range(args.calls_per_sample):
            run()
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
                "gemm_config": asdict(gemm_config),
            },
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
