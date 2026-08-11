"""Paired NVFP4-versus-MXFP8 forward/backward training benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import torch

import rtx


def _parse_shape(value: str) -> tuple[int, int, int]:
    shape = tuple(int(item) for item in value.lower().replace("x", ",").split(","))
    if len(shape) != 3 or min(shape) <= 0:
        raise argparse.ArgumentTypeError("shape must be M,N,K")
    return shape  # type: ignore[return-value]


def _time_batch(function, calls: int) -> float:
    # Backward may execute dX and dW on private CUDA streams.  An event on the
    # caller stream can therefore complete before all requested work, severely
    # under-reporting latency.  Device fences make this an honest end-to-end
    # measurement (and amortize the host overhead across ``calls``).
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(calls):
        function()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1_000.0 / calls


def _benchmark_shape(
    shape: tuple[int, int, int],
    *,
    warmup: int,
    rounds: int,
    calls: int,
    seed: int,
    compile_modules: bool,
    profile: bool,
    nv_scaling: str,
    nv_backend: str,
    mx_backend: str,
) -> dict[str, object]:
    m, n, k = shape
    torch.manual_seed(seed)
    mx = rtx.MXFP8Linear(
        k,
        n,
        device="cuda",
        backend=mx_backend,
        autotune="cache",
    )
    nv = rtx.NVFP4Linear(
        k,
        n,
        device="cuda",
        scaling=nv_scaling,
        backend=nv_backend,
    )
    with torch.no_grad():
        nv.weight.copy_(mx.weight)
    x_mx = torch.randn(m, k, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    x_nv = x_mx.detach().clone().requires_grad_(True)
    grad = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    if compile_modules:
        # Delayed state is intentionally shape-specific. Bootstrap once before
        # capturing the steady-state full graph, matching production usage.
        if nv_scaling == "delayed":
            nv(x_nv)
            torch.cuda.synchronize()
        mx = torch.compile(mx, fullgraph=True, dynamic=False)
        nv = torch.compile(nv, fullgraph=True, dynamic=False)

    def mx_step() -> None:
        out = mx(x_mx)
        torch.autograd.grad(out, (x_mx, mx.weight), grad, retain_graph=False)

    def nv_step() -> None:
        out = nv(x_nv)
        torch.autograd.grad(out, (x_nv, nv.weight), grad, retain_graph=False)

    def mx_forward() -> None:
        mx(x_mx)

    def nv_forward() -> None:
        nv(x_nv)

    # Both frontends intentionally use the same MXFP8 backward.  Verify that
    # the benchmark cannot report a speedup by silently dropping a gradient or
    # selecting a different numerical path under torch.compile.
    check_mx = torch.autograd.grad(
        mx(x_mx), (x_mx, mx.weight), grad, retain_graph=False
    )
    check_nv = torch.autograd.grad(
        nv(x_nv), (x_nv, nv.weight), grad, retain_graph=False
    )
    gradient_checks = []
    for mx_grad, nv_grad in zip(check_mx, check_nv, strict=True):
        difference = (mx_grad.float() - nv_grad.float()).abs()
        gradient_checks.append(
            {
                "mxfp8_norm": float(mx_grad.float().norm()),
                "nvfp4_norm": float(nv_grad.float().norm()),
                "max_abs_difference": float(difference.max()),
                "mean_abs_difference": float(difference.mean()),
            }
        )

    profiles: dict[str, list[dict[str, object]]] = {}
    if profile:
        for name, step in (("mxfp8", mx_step), ("nvfp4", nv_step)):
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as execution:
                step()
                torch.cuda.synchronize()
            events = []
            for event in execution.key_averages():
                cuda_us = float(getattr(event, "self_device_time_total", 0.0))
                cpu_us = float(getattr(event, "self_cpu_time_total", 0.0))
                if cuda_us > 0 or cpu_us >= 10.0:
                    events.append(
                        {
                            "name": event.key,
                            "count": int(event.count),
                            "self_cuda_us": cuda_us,
                            "self_cpu_us": cpu_us,
                        }
                    )
            profiles[name] = sorted(
                events, key=lambda item: float(item["self_cuda_us"]), reverse=True
            )

    for _ in range(warmup):
        mx_step()
        nv_step()
    torch.cuda.synchronize()
    mx_times: list[float] = []
    nv_times: list[float] = []
    mx_forward_times: list[float] = []
    nv_forward_times: list[float] = []
    for index in range(rounds):
        if index % 2:
            nv_times.append(_time_batch(nv_step, calls))
            mx_times.append(_time_batch(mx_step, calls))
            nv_forward_times.append(_time_batch(nv_forward, calls))
            mx_forward_times.append(_time_batch(mx_forward, calls))
        else:
            mx_times.append(_time_batch(mx_step, calls))
            nv_times.append(_time_batch(nv_step, calls))
            mx_forward_times.append(_time_batch(mx_forward, calls))
            nv_forward_times.append(_time_batch(nv_forward, calls))
    mx_median = statistics.median(mx_times)
    nv_median = statistics.median(nv_times)
    mx_forward_median = statistics.median(mx_forward_times)
    nv_forward_median = statistics.median(nv_forward_times)
    return {
        "shape": {"m": m, "n": n, "k": k},
        "compiled": compile_modules,
        "mxfp8_backend": mx_backend,
        "nvfp4_backend": nv_backend,
        "nvfp4_scaling": nv_scaling,
        "mxfp8_training_ms": mx_median,
        "nvfp4_training_ms": nv_median,
        "end_to_end_speedup": mx_median / nv_median,
        "mxfp8_forward_ms": mx_forward_median,
        "nvfp4_forward_ms": nv_forward_median,
        "forward_speedup": mx_forward_median / nv_forward_median,
        "gradient_checks": gradient_checks,
        "profiles": profiles,
        "mxfp8_backward_delta_ms": mx_median - mx_forward_median,
        "nvfp4_backward_delta_ms": nv_median - nv_forward_median,
        "mxfp8_timings_ms": mx_times,
        "nvfp4_timings_ms": nv_times,
        "mxfp8_forward_timings_ms": mx_forward_times,
        "nvfp4_forward_timings_ms": nv_forward_times,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", action="append", type=_parse_shape)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=11)
    parser.add_argument("--calls", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument(
        "--mx-backend",
        choices=("auto", "fused", "materialized"),
        default="auto",
    )
    parser.add_argument(
        "--nv-backend", choices=("auto", "fused", "materialized"), default="auto"
    )
    parser.add_argument(
        "--nv-scaling",
        choices=("delayed", "current", "regional", "block"),
        default="block",
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        parser.error("benchmark requires an SM120/SM121 CUDA GPU")
    shapes = args.shape or [(256, 1536, 1536), (1024, 1536, 1536)]
    results = [
        _benchmark_shape(
            shape,
            warmup=args.warmup,
            rounds=args.rounds,
            calls=args.calls,
            seed=args.seed + index,
            compile_modules=args.compile,
            profile=args.profile,
            nv_scaling=args.nv_scaling,
            nv_backend=args.nv_backend,
            mx_backend=args.mx_backend,
        )
        for index, shape in enumerate(shapes)
    ]
    document = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "results": results,
    }
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")


if __name__ == "__main__":
    main()
