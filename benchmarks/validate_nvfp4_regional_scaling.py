"""Matched numerical study for current-JIT and regional-delayed NVFP4.

This benchmark deliberately initializes delayed scaling from a separate
calibration distribution.  Initializing it from the tensor being evaluated
would erase the temporal-staleness question the study is intended to measure.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import statistics
from typing import Callable

import torch

from rtx.configs.nvfp4 import NVFP4Problem
from rtx.fp4 import (
    _make_jit_region_dynamic_runner,
    _make_region_delayed_dynamic_runner,
)
from rtx.nvfp4_inference_autotune import (
    preferred_jit_row_region_config,
    preferred_region_delayed_config,
)


def _metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float]:
    actual_f32 = actual.float()
    error = actual_f32 - reference
    reference_rms = reference.square().mean().sqrt().clamp_min(1.0e-30)
    error_rms = error.square().mean().sqrt()
    return {
        "mae": float(error.abs().mean()),
        "rmse": float(error_rms),
        "nrmse": float(error_rms / reference_rms),
        "cosine": float(
            torch.nn.functional.cosine_similarity(
                actual_f32.reshape(-1), reference.reshape(-1), dim=0
            )
        ),
        "max_abs": float(error.abs().max()),
        "finite_fraction": float(torch.isfinite(actual_f32).float().mean()),
    }


def _gradient_metrics(
    actual: torch.Tensor,
    reference: torch.Tensor,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> dict[str, float]:
    # Compare gradients induced by forward error under a fixed smooth loss.
    # This isolates the consequence of forward scaling; both public modes use
    # the same MXFP8 backward implementation after dY has been formed.
    target = reference * 0.9
    divisor = float(reference.numel())
    reference_dy = (reference - target) * (2.0 / divisor)
    actual_dy = (actual.float() - target) * (2.0 / divisor)
    reference_dx = reference_dy @ weight.float()
    actual_dx = actual_dy @ weight.float()
    reference_dw = reference_dy.transpose(0, 1) @ x.float()
    actual_dw = actual_dy.transpose(0, 1) @ x.float()
    return {
        "dx_nrmse": _metrics(actual_dx, reference_dx)["nrmse"],
        "dw_nrmse": _metrics(actual_dw, reference_dw)["nrmse"],
    }


def _scale_metrics(current: torch.Tensor, stale: torch.Tensor) -> dict[str, float]:
    current_f32 = current.float().clamp_min(torch.finfo(torch.float32).tiny)
    stale_f32 = stale.float().clamp_min(torch.finfo(torch.float32).tiny)
    log2_error = (torch.log2(stale_f32) - torch.log2(current_f32)).abs()
    return {
        "equal_fraction": float((current == stale).float().mean()),
        "mean_abs_log2_error": float(log2_error.mean()),
        "max_abs_log2_error": float(log2_error.max()),
    }


def _time(call: Callable[[], None], warmup: int, samples: int, calls: int) -> dict:
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(calls):
            call()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)) / calls)
    return {
        "median_ms": statistics.median(timings),
        "minimum_ms": min(timings),
        "maximum_ms": max(timings),
        "timings_ms": timings,
    }


def _scenario_tensors(
    name: str,
    x: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    calibration_x = x
    calibration_weight = weight
    target_x = x
    target_weight = weight
    if name == "activation_up_64x":
        target_x = x * 64.0
    elif name == "activation_down_64x":
        calibration_x = x * 64.0
    elif name == "weight_up_8x":
        target_weight = weight * 8.0
    elif name == "heterogeneous_rows":
        x_range = torch.exp2(
            torch.linspace(-8, 8, x.shape[0], device=x.device)
        ).to(torch.bfloat16)
        weight_range = torch.exp2(
            torch.linspace(-4, 4, weight.shape[0], device=weight.device)
        ).to(torch.bfloat16)
        target_x = x * x_range[:, None]
        target_weight = weight * weight_range[:, None]
    elif name == "outlier_recovery":
        calibration_x = x.clone()
        calibration_weight = weight.clone()
        calibration_x[:16].mul_(256.0)
        calibration_weight[:8].mul_(256.0)
    elif name == "alternating_extremes":
        calibration_x = x * 64.0
        calibration_weight = weight * 16.0
        target_x = x / 64.0
        target_weight = weight / 16.0
    elif name == "within_region_outliers":
        x_exponent = torch.where(
            torch.arange(x.shape[0], device=x.device) % 16 == 0,
            30.0,
            -30.0,
        )
        weight_exponent = torch.where(
            torch.arange(weight.shape[0], device=weight.device) % 8 == 0,
            24.0,
            -24.0,
        )
        target_x = x * torch.exp2(x_exponent).to(torch.bfloat16)[:, None]
        target_weight = weight * torch.exp2(weight_exponent).to(
            torch.bfloat16
        )[:, None]
    elif name != "stationary":
        raise ValueError(f"unknown scenario {name!r}")
    return calibration_x, calibration_weight, target_x, target_weight


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=1024)
    parser.add_argument("--n", type=int, default=768)
    parser.add_argument("--k", type=int, default=768)
    parser.add_argument("--x-region-rows", type=int, default=16)
    parser.add_argument("--weight-region-rows", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=21)
    parser.add_argument("--calls", type=int, default=50)
    parser.add_argument("--pdl", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        parser.error("benchmark requires an SM120/SM121 CUDA GPU")

    problem = NVFP4Problem(args.m, args.n, args.k)
    torch.manual_seed(20260814)
    x = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(args.n, args.k, device="cuda", dtype=torch.bfloat16)
    delayed_config = replace(
        preferred_region_delayed_config(problem),
        x_scale_region_rows=args.x_region_rows,
        weight_scale_region_rows=args.weight_region_rows,
    )
    jit_config = replace(
        preferred_jit_row_region_config(problem),
        gemm=delayed_config.gemm,
        x_scale_region_rows=args.x_region_rows,
        weight_scale_region_rows=args.weight_region_rows,
        region_waves=delayed_config.region_waves,
        region_ownership=delayed_config.region_ownership,
        programmatic_dependent_launch=args.pdl,
    )
    for label, config in (("jit", jit_config), ("delayed", delayed_config)):
        rejection = config.rejection(problem)
        if rejection is not None:
            parser.error(f"{label} configuration is illegal: {rejection}")

    scenarios = {}
    for name in (
        "stationary",
        "activation_up_64x",
        "activation_down_64x",
        "weight_up_8x",
        "heterogeneous_rows",
        "outlier_recovery",
        "alternating_extremes",
        "within_region_outliers",
    ):
        calibration_x, calibration_weight, target_x, target_weight = (
            _scenario_tensors(name, x, weight)
        )
        jit = _make_jit_region_dynamic_runner(problem, jit_config, x.device)
        delayed = _make_region_delayed_dynamic_runner(
            problem, delayed_config, x.device
        )
        delayed.initialize(calibration_x, calibration_weight)
        delayed_used_scales = delayed.region_scales.clone()
        jit_out = torch.empty(
            (problem.m, problem.n), device=x.device, dtype=torch.bfloat16
        )
        delayed_out = torch.empty_like(jit_out)
        recovered_out = torch.empty_like(jit_out)
        jit(target_x, target_weight, jit_out)
        jit_scales = jit.region_scales.clone()
        delayed(target_x, target_weight, delayed_out)
        delayed(target_x, target_weight, recovered_out)
        torch.cuda.synchronize()
        reference = target_x.float() @ target_weight.float().transpose(0, 1)
        jit_metrics = _metrics(jit_out, reference)
        stale_metrics = _metrics(delayed_out, reference)
        recovered_metrics = _metrics(recovered_out, reference)
        if name != "within_region_outliers":
            jit_metrics.update(
                _gradient_metrics(jit_out, reference, target_x, target_weight)
            )
            stale_metrics.update(
                _gradient_metrics(
                    delayed_out, reference, target_x, target_weight
                )
            )
            recovered_metrics.update(
                _gradient_metrics(
                    recovered_out, reference, target_x, target_weight
                )
            )
        scenario = {
            "jit": jit_metrics,
            "delayed_stale": stale_metrics,
            "delayed_recovered": recovered_metrics,
            "stale_scale_error_vs_current": _scale_metrics(
                jit_scales, delayed_used_scales
            ),
        }
        if name == "within_region_outliers":
            low_x = torch.arange(problem.m, device=x.device) % 16 != 0
            low_weight = torch.arange(problem.n, device=x.device) % 8 != 0
            low_low = low_x[:, None] & low_weight[None, :]
            scenario["low_magnitude_output"] = {
                "jit": _metrics(jit_out[low_low], reference[low_low]),
                "delayed_stale": _metrics(
                    delayed_out[low_low], reference[low_low]
                ),
                "delayed_recovered": _metrics(
                    recovered_out[low_low], reference[low_low]
                ),
            }
        scenarios[name] = scenario

    jit = _make_jit_region_dynamic_runner(problem, jit_config, x.device)
    delayed = _make_region_delayed_dynamic_runner(problem, delayed_config, x.device)
    delayed.initialize(x, weight)
    jit_out = torch.empty((problem.m, problem.n), device=x.device, dtype=torch.bfloat16)
    delayed_out = torch.empty_like(jit_out)
    performance = {
        "jit": _time(
            lambda: jit(x, weight, jit_out), args.warmup, args.samples, args.calls
        ),
        "regional_delayed": _time(
            lambda: delayed(x, weight, delayed_out),
            args.warmup,
            args.samples,
            args.calls,
        ),
    }
    performance["jit_over_delayed"] = (
        performance["jit"]["median_ms"]
        / performance["regional_delayed"]["median_ms"]
    )
    report = {
        "type": "nvfp4_matched_regional_scaling_validation",
        "device": torch.cuda.get_device_name(),
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "region_rows": {
            "x": args.x_region_rows,
            "weight": args.weight_region_rows,
        },
        "pdl": args.pdl,
        "jit_config": str(jit_config),
        "delayed_config": str(delayed_config),
        "scenarios": scenarios,
        "performance": performance,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
