"""Benchmark TorchAO dynamic FP8 rowwise linear with runtime BF16 X and W."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch
from torch import nn
from torchao.float8 import (
    Float8LinearConfig,
    Float8LinearRecipeName,
    convert_to_float8_training,
)


COMPILE_OPTIONS = {
    "epilogue_fusion": True,
    "prologue_fusion": True,
    "max_autotune": True,
    "triton.cudagraphs": False,
    "max_autotune_gemm": True,
    "aggressive_fusion": True,
    "max_autotune_pointwise": True,
    "online_softmax": True,
    "layout_optimization": True,
    "b2b_gemm_pass": True,
    "joint_graph_constant_folding": True,
    "benchmark_epilogue_fusion": True,
    "coordinate_descent_tuning": False,
    "triton.autotune_cublasLt": False,
    "memory_planning": True,
    "inplace_buffers": True,
    "allow_buffer_reuse": True,
    "reorder_for_locality": True,
    "auto_chunker.enable": True,
    "reorder_for_compute_comm_overlap": True,
    "group_fusion": True,
    "triton.persistent_reductions": True,
    "use_fast_math": True,
    # X and W are both runtime BF16 tensors and are quantized on every call.
    "freezing": False,
}


def _time_cuda(
    fn,
    args: tuple[torch.Tensor, ...],
    *,
    warmup: int,
    samples: int,
    calls_per_sample: int,
) -> list[float]:
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()

    timings_ms: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(calls_per_sample):
            fn(*args)
        end.record()
        end.synchronize()
        timings_ms.append(float(start.elapsed_time(end)) / calls_per_sample)
    return timings_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=1536)
    parser.add_argument("--k", type=int, default=1536)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--samples", type=int, default=51)
    parser.add_argument("--calls-per-sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    x = torch.randn(args.m, args.k, device=device, dtype=torch.bfloat16)
    weight = torch.randn(args.n, args.k, device=device, dtype=torch.bfloat16)

    linear = nn.Linear(
        args.k,
        args.n,
        bias=False,
        device=device,
        dtype=torch.bfloat16,
    ).eval()
    recipe = Float8LinearConfig.from_recipe_name(Float8LinearRecipeName.ROWWISE)
    rowwise = convert_to_float8_training(linear, config=recipe).eval()

    # Make W an explicit graph input.  This preserves dynamic BF16->FP8 weight
    # quantization even when the requested Inductor `freezing` option is enabled.
    def forward_dynamic_weight(
        input_bf16: torch.Tensor, weight_bf16: torch.Tensor
    ) -> torch.Tensor:
        return torch.func.functional_call(
            rowwise,
            {"weight": weight_bf16},
            (input_bf16,),
        )

    compiled = torch.compile(
        forward_dynamic_weight,
        fullgraph=True,
        dynamic=False,
        options=COMPILE_OPTIONS,
    )

    with torch.inference_mode():
        compile_started = time.perf_counter()
        actual = compiled(x, weight)
        torch.cuda.synchronize()
        first_call_s = time.perf_counter() - compile_started

        bf16_reference = torch.nn.functional.linear(x, weight)
        difference = (actual.float() - bf16_reference.float()).abs()
        compiled_vs_bf16 = {
            "max_abs_error": float(difference.max()),
            "mean_abs_error": float(difference.mean()),
            "finite": bool(torch.isfinite(actual).all()),
        }

        # Prove W remains a live runtime input instead of a frozen/prequantized
        # module constant.  A complete replacement makes the check unambiguous.
        changed_weight = -weight
        changed = compiled(x, changed_weight)
        torch.cuda.synchronize()
        weight_is_runtime_input = not torch.equal(actual, changed)

        timings_ms = _time_cuda(
            compiled,
            (x, weight),
            warmup=args.warmup,
            samples=args.samples,
            calls_per_sample=args.calls_per_sample,
        )

        profile_events = []
        if args.profile:
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ]
            ) as prof:
                compiled(x, weight)
                torch.cuda.synchronize()
            for event in prof.key_averages():
                device_us = float(getattr(event, "device_time_total", 0.0))
                if device_us > 0:
                    profile_events.append(
                        {
                            "name": event.key,
                            "calls": event.count,
                            "device_time_total_us": device_us,
                        }
                    )
            profile_events.sort(
                key=lambda event: event["device_time_total_us"], reverse=True
            )

    median_ms = float(statistics.median(timings_ms))
    flop = 2 * args.m * args.n * args.k
    result = {
        "implementation": "torchao.float8.rowwise",
        "torch_version": torch.__version__,
        "torchao_version": __import__("torchao").__version__,
        "cuda_version": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "input_dtype": str(x.dtype),
        "weight_dtype": str(weight.dtype),
        "output_dtype": str(actual.dtype),
        "bias": False,
        "recipe": str(Float8LinearRecipeName.ROWWISE.value),
        "dynamic_weight_quantization_in_timed_region": True,
        "weight_is_runtime_input": weight_is_runtime_input,
        "compile": {
            "fullgraph": True,
            "dynamic": False,
            "options": COMPILE_OPTIONS,
            "first_call_s": first_call_s,
        },
        "correctness": {"compiled_rowwise_vs_bf16": compiled_vs_bf16},
        "timing": {
            "warmup": args.warmup,
            "samples": args.samples,
            "calls_per_sample": args.calls_per_sample,
            "median_ms": median_ms,
            "min_ms": min(timings_ms),
            "max_ms": max(timings_ms),
            "tflops": flop / (median_ms * 1e9),
            "samples_ms": timings_ms,
        },
        "profile_events": profile_events,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
