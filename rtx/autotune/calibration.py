"""Portable empirical calibration used by cross-device cost-model features."""

from __future__ import annotations

from datetime import datetime, timezone
import math
import statistics
from typing import Callable

import torch

from .hardware import device_properties, static_device_profile


def _time_cuda(function: Callable[[], None], *, calls: int, samples: int) -> list[float]:
    for _ in range(3):
        function()
    torch.cuda.synchronize()
    timings: list[float] = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _call in range(calls):
            function()
        end.record()
        end.synchronize()
        timings.append(float(start.elapsed_time(end)) / calls)
    return timings


def _copy_bandwidth(
    device: torch.device,
    *,
    bytes_per_buffer: int,
    target_ms: float,
    samples: int,
) -> tuple[float, list[float]]:
    elements = max(1024, bytes_per_buffer // 4)
    source = torch.empty(elements, dtype=torch.float32, device=device).normal_()
    destination = torch.empty_like(source)
    pilot = _time_cuda(lambda: destination.copy_(source), calls=2, samples=1)[0]
    calls = max(1, min(4096, math.ceil(target_ms / max(pilot, 1e-6))))
    timings = _time_cuda(lambda: destination.copy_(source), calls=calls, samples=samples)
    # A device copy reads and writes the complete buffer.
    bandwidths = [2.0 * source.numel() * source.element_size() / (ms * 1.0e6) for ms in timings]
    return float(statistics.median(bandwidths)), timings


def _bf16_matmul(device: torch.device, *, size: int, samples: int) -> tuple[float, list[float]]:
    a = torch.randn(size, size, dtype=torch.bfloat16, device=device)
    b = torch.randn(size, size, dtype=torch.bfloat16, device=device)
    out = torch.empty_like(a)
    timings = _time_cuda(lambda: torch.mm(a, b, out=out), calls=1, samples=samples)
    tflops = [2.0 * size**3 / (ms * 1.0e9) for ms in timings]
    return float(statistics.median(tflops)), timings


def _native_mxfp8(device: torch.device, *, samples: int) -> dict[str, object]:
    from ..fp8 import DEFAULT_MXFP8_PREQUANT_CONFIG
    from ..prequant_experiments import (
        BenchmarkProtocol,
        PrequantBenchmarkHarness,
        ShapeSpec,
    )

    shape = ShapeSpec(512, 1536, 1536, name="hardware_calibration")
    protocol = BenchmarkProtocol(
        warmup_calls=3,
        samples=max(3, samples),
        confirm_samples=3,
        race_rounds=3,
        target_batch_ms=20.0,
        bootstrap_resamples=200,
        max_rotation_buffers=2,
        telemetry=False,
    )
    harness = PrequantBenchmarkHarness(shape, "hot", protocol, device=device, seed=0)
    outcome = harness.measure(
        DEFAULT_MXFP8_PREQUANT_CONFIG,
        samples=max(3, samples),
        seed=0,
        components=True,
    )
    result: dict[str, object] = {"native_mxfp8_outcome": outcome}
    if outcome.get("status") != "ok":
        return result
    components = outcome.get("components")
    if not isinstance(components, dict):
        return result
    gemm = components.get("gemm_hot_materialized")
    if isinstance(gemm, dict) and isinstance(gemm.get("summary_ms"), dict):
        median_ms = float(gemm["summary_ms"]["median"])
        result["measured_native_mxfp8_gemm_tflops"] = (
            2.0 * shape.m * shape.n * shape.k / (median_ms * 1.0e9)
        )
    quant = components.get("dual_quant")
    if isinstance(quant, dict) and isinstance(quant.get("summary_ms"), dict):
        median_ms = float(quant["summary_ms"]["median"])
        # Read BF16 X/W and write E4M3 X/W plus E8M0 block scales.
        values = (shape.m + shape.n) * shape.k
        traffic = values * 3 + (shape.m + shape.n) * (shape.k // 32)
        result["measured_mxfp8_quant_bandwidth_gbps"] = traffic / (median_ms * 1.0e6)
    return result


def calibrate_device(
    device: torch.device | str = "cuda",
    *,
    samples: int = 5,
    target_ms: float = 30.0,
    include_native_mxfp8: bool = True,
) -> dict[str, object]:
    """Measure portable rooflines and optionally the project's native pipeline."""

    resolved = torch.device(device)
    properties = device_properties(resolved)
    l2_bytes = int(properties.get("l2_cache_size", 0) or 0)
    free_bytes, _ = torch.cuda.mem_get_info(resolved)
    hot_bytes = min(64 << 20, max(1 << 20, l2_bytes // 2 if l2_bytes else 8 << 20))
    dram_bytes = min(512 << 20, max(64 << 20, l2_bytes * 4), int(free_bytes * 0.1))
    l2_bandwidth, l2_timings = _copy_bandwidth(
        resolved, bytes_per_buffer=hot_bytes, target_ms=target_ms, samples=samples
    )
    dram_bandwidth, dram_timings = _copy_bandwidth(
        resolved, bytes_per_buffer=dram_bytes, target_ms=target_ms, samples=samples
    )
    matmul_size = 4096 if free_bytes >= 1 << 30 else 2048
    bf16_tflops, bf16_timings = _bf16_matmul(
        resolved, size=matmul_size, samples=samples
    )
    calibration: dict[str, object] = {
        "schema_version": 1,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "measured_l2_copy_bandwidth_gbps": l2_bandwidth,
        "measured_dram_bandwidth_gbps": dram_bandwidth,
        "measured_bf16_matmul_tflops": bf16_tflops,
        "l2_copy_buffer_bytes": hot_bytes,
        "dram_copy_buffer_bytes": dram_bytes,
        "bf16_matmul_size": matmul_size,
        "raw_l2_copy_timings_ms": l2_timings,
        "raw_dram_copy_timings_ms": dram_timings,
        "raw_bf16_matmul_timings_ms": bf16_timings,
    }
    if include_native_mxfp8:
        try:
            calibration.update(_native_mxfp8(resolved, samples=samples))
        except Exception as exc:
            # Partially configured SM120/SM121 machines can still contribute
            # memory/BF16 rooflines when native-kernel calibration is unavailable.
            calibration["native_mxfp8_outcome"] = {
                "status": "unavailable",
                "error": f"{type(exc).__name__}: {exc}"[:4000],
            }
    calibration["hardware_profile"] = static_device_profile(
        resolved, calibration=calibration
    )
    return calibration


__all__ = ["calibrate_device"]
