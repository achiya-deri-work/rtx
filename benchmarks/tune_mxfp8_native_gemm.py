"""Persistent correctness-gated coordinate sweep for native-scale MXFP8 GEMM."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from rtx.kernels.mxfp8 import MXFP8Problem
from rtx.kernels.mxfp8_gemm import MXFP8GemmConfig, compile_mxfp8_gemm
from rtx.kernels.mxfp8_quant import (
    MXFP8QuantConfig,
    compile_mxfp8_dual_quant,
)


COORDINATES: tuple[tuple[str, tuple[dict[str, object], ...]], ...] = (
    (
        "pipeline_geometry",
        tuple(
            {"stages": stages, "atom_layout_m": atom_m}
            for stages in (1, 2, 3, 4)
            for atom_m in (2, 4, 8)
        ),
    ),
    (
        "ldmatrix",
        tuple(
            {"a_ldmatrix_matrices": a, "b_ldmatrix_matrices": b}
            for a in (1, 2, 4)
            for b in (1, 2, 4)
        ),
    ),
    (
        "smem_swizzle",
        tuple(
            {"a_swizzle": a, "b_swizzle": b}
            for a in ("32b", "64b", "128b")
            for b in ("32b", "64b", "128b")
        ),
    ),
    (
        "scale_s2r",
        tuple(
            {"sfa_s2r_bits": a, "sfb_s2r_bits": b}
            for a in (0, 8)
            for b in (0, 8)
        ),
    ),
    (
        "register_budget",
        tuple(
            {
                "producer_registers": producer,
                "consumer_registers": consumer,
                "maxrregcount": maximum,
            }
            for producer in (32, 48, 64)
            for consumer, maximum in (
                (128, 192),
                (160, 224),
                (192, 255),
                (224, 255),
                (232, 255),
            )
        ),
    ),
    (
        "epilogue",
        (
            {"epilogue": "tma", "store_vec": 1},
            {"epilogue": "tma", "store_vec": 2},
            {"epilogue": "tma", "store_vec": 4},
            {"epilogue": "direct", "epilogue_stages": 1, "store_vec": 1},
        ),
    ),
    (
        "raster_group",
        tuple(
            {"raster": raster, "grid_swizzle": group}
            for raster in ("m", "n")
            for group in (1, 2, 4, 8)
        ),
    ),
)


def _config_id(config: MXFP8GemmConfig) -> str:
    encoded = json.dumps(asdict(config), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def _save(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=1536)
    parser.add_argument("--k", type=int, default=1536)
    parser.add_argument("--passes", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--calls-per-sample", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1705)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        parser.error("tuning requires an SM120/SM121 CUDA GPU")

    problem = MXFP8Problem(args.m, args.n, args.k)
    output = args.output or Path("autotune_results") / (
        f"mxfp8_native_gemm_m{args.m}_n{args.n}_k{args.k}.json"
    )
    torch.manual_seed(args.seed)
    x = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(args.n, args.k, device="cuda", dtype=torch.bfloat16)
    qx = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    qw = torch.empty_like(weight, dtype=torch.float8_e4m3fn)
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
    out = torch.empty(args.m, args.n, device="cuda", dtype=torch.bfloat16)
    compile_mxfp8_dual_quant(
        args.m,
        args.n,
        args.k,
        MXFP8QuantConfig(scale_layout="mma128"),
    )(x, weight, qx, qw, sx, sw)

    initial = MXFP8GemmConfig(scale_role="tma", scale_layout="mma128")
    reference = torch.empty_like(out)
    compile_mxfp8_gemm(problem, initial)(qx, qw, sx, sw, reference)
    torch.cuda.synchronize()

    if output.exists():
        state = json.loads(output.read_text())
        trials: dict[str, dict[str, object]] = state.get("trials", {})
        incumbent = MXFP8GemmConfig(**state.get("best_config", asdict(initial)))
        best_us = float(state.get("best_us", "inf"))
    else:
        trials = {}
        incumbent = initial
        best_us = float("inf")
        state = {}

    def evaluate(config: MXFP8GemmConfig, coordinate: str) -> float | None:
        nonlocal state
        config_id = _config_id(config)
        previous = trials.get(config_id)
        if previous is not None:
            value = previous.get("median_us")
            return float(value) if value is not None else None
        rejection = config.rejection(problem)
        if rejection is not None:
            trials[config_id] = {
                "config": asdict(config),
                "coordinate": coordinate,
                "status": "rejected",
                "reason": rejection,
            }
            _save(output, {**state, "trials": trials})
            print(f"REJECT {coordinate:20s} {config_id} {rejection}", flush=True)
            return None
        started = time.time()
        try:
            gemm = compile_mxfp8_gemm(problem, config)
            gemm(qx, qw, sx, sw, out)
            torch.cuda.synchronize()
            torch.testing.assert_close(out, reference, rtol=0.05, atol=0.5)
            for _ in range(args.warmup):
                gemm(qx, qw, sx, sw, out)
            torch.cuda.synchronize()
            samples: list[float] = []
            for _ in range(args.samples):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _call in range(args.calls_per_sample):
                    gemm(qx, qw, sx, sw, out)
                end.record()
                end.synchronize()
                samples.append(
                    float(start.elapsed_time(end)) * 1000 / args.calls_per_sample
                )
            median_us = statistics.median(samples)
            trials[config_id] = {
                "config": asdict(config),
                "coordinate": coordinate,
                "status": "ok",
                "median_us": median_us,
                "min_us": min(samples),
                "max_us": max(samples),
                "compile_and_test_seconds": time.time() - started,
            }
            print(
                f"RUN    {coordinate:20s} {config_id} "
                f"median={median_us:.4f} us min={min(samples):.4f} us",
                flush=True,
            )
        except Exception as exc:
            torch.cuda.synchronize()
            trials[config_id] = {
                "config": asdict(config),
                "coordinate": coordinate,
                "status": "error",
                "reason": f"{type(exc).__name__}: {exc}",
                "compile_and_test_seconds": time.time() - started,
            }
            print(
                f"ERROR  {coordinate:20s} {config_id} "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
            median_us = None
        state = {
            "shape": asdict(problem),
            "best_config": asdict(incumbent),
            "best_us": best_us,
            "trials": trials,
            "updated_at": time.time(),
        }
        _save(output, state)
        return median_us

    initial_us = evaluate(incumbent, "initial")
    if initial_us is not None and initial_us < best_us:
        best_us = initial_us
    for pass_index in range(args.passes):
        changed = False
        print(
            f"PASS {pass_index + 1}/{args.passes} incumbent={best_us:.4f} us",
            flush=True,
        )
        for coordinate, variants in COORDINATES:
            axis_best = incumbent
            axis_best_us = best_us
            for updates in variants:
                candidate = replace(incumbent, **updates)
                value = evaluate(candidate, coordinate)
                if value is not None and value < axis_best_us:
                    axis_best = candidate
                    axis_best_us = value
            if axis_best != incumbent:
                incumbent = axis_best
                best_us = axis_best_us
                changed = True
                print(
                    f"BEST   {coordinate:20s} median={best_us:.4f} us "
                    f"config={_config_id(incumbent)}",
                    flush=True,
                )
                state.update(
                    best_config=asdict(incumbent),
                    best_us=best_us,
                    updated_at=time.time(),
                )
                _save(output, state)
        if not changed:
            break
    print(
        json.dumps(
            {
                "output": str(output),
                "best_us": best_us,
                "best_config": asdict(incumbent),
                "trial_count": len(trials),
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
