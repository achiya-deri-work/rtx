"""Persistent correctness-gated coordinate sweep for native-scale quantization."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from rtx.kernels.mxfp8_quant import MXFP8QuantConfig, compile_mxfp8_dual_quant


COORDINATES: tuple[tuple[str, tuple[dict[str, object], ...]], ...] = (
    (
        "vector_load",
        tuple(
            {"quant_vec": vector, "load_bits": bits}
            for vector in (1, 2, 4, 8)
            for bits in (16, 32, 64, 128)
            if bits <= vector * 16 and (vector * 16) % bits == 0
        ),
    ),
    (
        "quant_arithmetic",
        tuple(
            {"quant_math": math, "quant_amax": amax, "reduction": reduction}
            for math in ("fp32", "bf16x2")
            for amax in ("fp32", "bf16_bits")
            for reduction in ("shuffle", "redux")
        ),
    ),
    (
        "launch_shape",
        tuple(
            {"num_warps": warps, "persistent_waves": waves}
            for warps in (4, 8, 16)
            for waves in (1, 2, 3, 4, 6, 8)
        ),
    ),
    (
        "register_budget",
        tuple({"maxrregcount": registers} for registers in (64, 96, 128, 160, 192)),
    ),
    (
        "scale_store",
        ({"native_scale_store": "scalar"}, {"native_scale_store": "packed"}),
    ),
)


def _config_id(config: MXFP8QuantConfig) -> str:
    encoded = json.dumps(asdict(config), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _save(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _logical_scales(native: torch.Tensor, rows: int, k: int) -> torch.Tensor:
    row = torch.arange(rows, device="cuda")[:, None]
    block = torch.arange(k // 32, device="cuda")[None, :]
    physical = (row % 32) * 16 + ((row // 32) % 4) * 4 + block % 4
    return native[row // 128, block // 4, physical]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=1536)
    parser.add_argument("--k", type=int, default=1536)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--calls-per-sample", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1706)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        parser.error("tuning requires an SM120/SM121 CUDA GPU")
    output = args.output or Path("autotune_results") / (
        f"mxfp8_native_quant_m{args.m}_n{args.n}_k{args.k}.json"
    )
    torch.manual_seed(args.seed)
    x = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(args.n, args.k, device="cuda", dtype=torch.bfloat16)
    qx = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    qw = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
    sx = torch.empty(
        args.m // 128, args.k // 128, 512,
        device="cuda", dtype=torch.float8_e8m0fnu,
    )
    sw = torch.empty(
        args.n // 128, args.k // 128, 512,
        device="cuda", dtype=torch.float8_e8m0fnu,
    )
    initial = MXFP8QuantConfig(scale_layout="mma128")
    compile_mxfp8_dual_quant(args.m, args.n, args.k, initial)(
        x, weight, qx, qw, sx, sw
    )
    torch.cuda.synchronize()
    reference_qx = qx.clone()
    reference_qw = qw.clone()
    reference_sx = _logical_scales(sx, args.m, args.k).clone()
    reference_sw = _logical_scales(sw, args.n, args.k).clone()

    if output.exists():
        state = json.loads(output.read_text())
        trials: dict[str, dict[str, object]] = state.get("trials", {})
        incumbent = MXFP8QuantConfig(**state.get("best_config", asdict(initial)))
        best_us = float(state.get("best_us", "inf"))
    else:
        state = {}
        trials = {}
        incumbent = initial
        best_us = float("inf")

    def evaluate(config: MXFP8QuantConfig, coordinate: str) -> float | None:
        nonlocal state
        config_id = _config_id(config)
        previous = trials.get(config_id)
        if previous is not None:
            value = previous.get("median_us")
            return float(value) if value is not None else None
        rejection_m = config.rejection(args.m, args.k)
        rejection_n = config.rejection(args.n, args.k)
        if rejection_m or rejection_n:
            reason = rejection_m or rejection_n
            trials[config_id] = {
                "config": asdict(config), "coordinate": coordinate,
                "status": "rejected", "reason": reason,
            }
            _save(output, {**state, "trials": trials})
            print(f"REJECT {coordinate:20s} {config_id} {reason}", flush=True)
            return None
        started = time.time()
        try:
            quant = compile_mxfp8_dual_quant(args.m, args.n, args.k, config)
            quant(x, weight, qx, qw, sx, sw)
            torch.cuda.synchronize()
            torch.testing.assert_close(qx, reference_qx, rtol=0, atol=0)
            torch.testing.assert_close(qw, reference_qw, rtol=0, atol=0)
            torch.testing.assert_close(
                _logical_scales(sx, args.m, args.k), reference_sx, rtol=0, atol=0
            )
            torch.testing.assert_close(
                _logical_scales(sw, args.n, args.k), reference_sw, rtol=0, atol=0
            )
            for _ in range(args.warmup):
                quant(x, weight, qx, qw, sx, sw)
            torch.cuda.synchronize()
            samples: list[float] = []
            for _ in range(args.samples):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _call in range(args.calls_per_sample):
                    quant(x, weight, qx, qw, sx, sw)
                end.record()
                end.synchronize()
                samples.append(
                    float(start.elapsed_time(end)) * 1000 / args.calls_per_sample
                )
            median_us = statistics.median(samples)
            trials[config_id] = {
                "config": asdict(config), "coordinate": coordinate, "status": "ok",
                "median_us": median_us, "min_us": min(samples),
                "max_us": max(samples),
                "compile_and_test_seconds": time.time() - started,
            }
            print(
                f"RUN    {coordinate:20s} {config_id} median={median_us:.4f} us "
                f"min={min(samples):.4f} us", flush=True,
            )
        except Exception as exc:
            torch.cuda.synchronize()
            trials[config_id] = {
                "config": asdict(config), "coordinate": coordinate, "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
                "compile_and_test_seconds": time.time() - started,
            }
            print(
                f"ERROR  {coordinate:20s} {config_id} "
                f"{type(exc).__name__}: {exc}", flush=True,
            )
            median_us = None
        state = {
            "shape": {"m": args.m, "n": args.n, "k": args.k},
            "best_config": asdict(incumbent), "best_us": best_us,
            "trials": trials, "updated_at": time.time(),
        }
        _save(output, state)
        return median_us

    initial_us = evaluate(incumbent, "initial")
    if initial_us is not None and initial_us < best_us:
        best_us = initial_us
    for pass_index in range(args.passes):
        changed = False
        print(f"PASS {pass_index + 1}/{args.passes} incumbent={best_us:.4f} us", flush=True)
        for coordinate, variants in COORDINATES:
            axis_best, axis_best_us = incumbent, best_us
            for updates in variants:
                candidate = replace(incumbent, **updates)
                value = evaluate(candidate, coordinate)
                if value is not None and value < axis_best_us:
                    axis_best, axis_best_us = candidate, value
            if axis_best != incumbent:
                incumbent, best_us = axis_best, axis_best_us
                changed = True
                print(
                    f"BEST   {coordinate:20s} median={best_us:.4f} us "
                    f"config={_config_id(incumbent)}", flush=True,
                )
                state.update(
                    best_config=asdict(incumbent), best_us=best_us,
                    updated_at=time.time(),
                )
                _save(output, state)
        if not changed:
            break
    print(json.dumps({
        "output": str(output), "best_us": best_us,
        "best_config": asdict(incumbent), "trial_count": len(trials),
    }, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
