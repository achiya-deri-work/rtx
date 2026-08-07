"""Benchmark the fullgraph dynamic-weight MXFP8 torch frontend."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from rtx import mxfp8_linear


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=1536)
    parser.add_argument("--k", type=int, default=1536)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--samples", type=int, default=51)
    parser.add_argument("--calls-per-sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        parser.error("benchmark requires an SM120/SM121 CUDA GPU")
    torch.manual_seed(args.seed)
    x = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(args.n, args.k, device="cuda", dtype=torch.bfloat16)

    def forward_dynamic_weight(
        input_bf16: torch.Tensor, weight_bf16: torch.Tensor
    ) -> torch.Tensor:
        return mxfp8_linear(
            input_bf16, weight_bf16, backend="prequant", autotune=False
        )

    compiled = torch.compile(
        forward_dynamic_weight,
        fullgraph=True,
        dynamic=False,
        options={"triton.cudagraphs": False, "freezing": False},
    )
    with torch.inference_mode():
        compile_started = time.perf_counter()
        actual = compiled(x, weight)
        torch.cuda.synchronize()
        first_call_s = time.perf_counter() - compile_started
        eager = forward_dynamic_weight(x, weight)
        changed = compiled(x, -weight)
        torch.cuda.synchronize()
        torch.testing.assert_close(actual, eager, rtol=0, atol=0)
        weight_is_runtime_input = not torch.equal(actual, changed)
        profile_events: list[dict[str, object]] = []
        if args.profile:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as profiler:
                compiled(x, weight)
                torch.cuda.synchronize()
            for event in profiler.key_averages():
                device_us = float(getattr(event, "device_time_total", 0.0))
                if device_us:
                    profile_events.append({
                        "name": event.key,
                        "calls": event.count,
                        "device_time_total_us": device_us,
                    })
            profile_events.sort(
                key=lambda event: float(event["device_time_total_us"]),
                reverse=True,
            )
        for _ in range(args.warmup):
            compiled(x, weight)
        torch.cuda.synchronize()
        timings: list[float] = []
        for _ in range(args.samples):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _call in range(args.calls_per_sample):
                compiled(x, weight)
            end.record()
            end.synchronize()
            timings.append(float(start.elapsed_time(end)) / args.calls_per_sample)
    median_ms = statistics.median(timings)
    print(json.dumps({
        "implementation": "rtx.mxfp8.prequant_native",
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "fullgraph": True,
        "dynamic": False,
        "dynamic_weight": weight_is_runtime_input,
        "first_call_s": first_call_s,
        "median_ms": median_ms,
        "min_ms": min(timings),
        "max_ms": max(timings),
        "tflops": 2 * args.m * args.n * args.k / (median_ms * 1e9),
        "warmup": args.warmup,
        "samples": args.samples,
        "calls_per_sample": args.calls_per_sample,
        "profile_events": profile_events,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
