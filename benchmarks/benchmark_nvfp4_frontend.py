"""Paired compiled NVFP4 versus MXFP8 dynamic-forward benchmark."""

from __future__ import annotations

import argparse
import json
import statistics

import torch

import rtx


def _time(function, x, weight, calls: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(calls):
        function(x, weight)
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / calls


def _error_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_f32 = actual.float()
    error = actual_f32 - reference
    reference_rms = reference.square().mean().sqrt()
    error_rms = error.square().mean().sqrt()
    cosine = torch.nn.functional.cosine_similarity(
        actual_f32.flatten(), reference.flatten(), dim=0
    )
    return {
        "mean_abs_error": float(error.abs().mean()),
        "rmse": float(error_rms),
        "normalized_rmse": float(error_rms / reference_rms),
        "cosine_similarity": float(cosine),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=1536)
    parser.add_argument("--k", type=int, default=1536)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument(
        "--scaling", choices=("block", "current", "regional"), default="block"
    )
    args = parser.parse_args()
    torch.manual_seed(20260810)
    x = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(args.n, args.k, device="cuda", dtype=torch.bfloat16)

    def mx(a, b):
        return rtx.mxfp8_linear(a, b, backend="auto", autotune="cache")

    def nv(a, b):
        return rtx.nvfp4_linear(a, b, scaling=args.scaling, backend="auto")

    mx_compiled = torch.compile(mx, fullgraph=True, dynamic=False)
    nv_compiled = torch.compile(nv, fullgraph=True, dynamic=False)
    with torch.inference_mode():
        mx_out = mx_compiled(x, weight)
        nv_out = nv_compiled(x, weight)
        for _ in range(args.warmup):
            mx_compiled(x, weight)
            nv_compiled(x, weight)
        torch.cuda.synchronize()
        mx_times = []
        nv_times = []
        for index in range(args.samples):
            if index % 2:
                nv_times.append(_time(nv_compiled, x, weight, args.calls))
                mx_times.append(_time(mx_compiled, x, weight, args.calls))
            else:
                mx_times.append(_time(mx_compiled, x, weight, args.calls))
                nv_times.append(_time(nv_compiled, x, weight, args.calls))
    reference = x.float() @ weight.float().T
    mx_ms = statistics.median(mx_times)
    nv_ms = statistics.median(nv_times)
    mx_error = _error_metrics(mx_out, reference)
    nv_error = _error_metrics(nv_out, reference)
    print(json.dumps({
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "scaling": args.scaling,
        "mxfp8_ms": mx_ms,
        "nvfp4_ms": nv_ms,
        "nvfp4_speedup": mx_ms / nv_ms,
        "mxfp8_error": mx_error,
        "nvfp4_error": nv_error,
        "mxfp8_timings_ms": mx_times,
        "nvfp4_timings_ms": nv_times,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
