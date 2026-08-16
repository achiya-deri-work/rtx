"""Compare fused MMA-thread and warp-specialized NVFP4 regional epilogues."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import statistics

import torch

from rtx.configs.nvfp4 import NVFP4Problem
from rtx.fp4 import _make_jit_region_dynamic_runner
from rtx.nvfp4_inference_autotune import preferred_jit_row_region_config


def _measure(runner, x, weight, out, warmup: int, samples: int, calls: int):
    for _ in range(warmup):
        runner(x, weight, out)
    torch.cuda.synchronize()
    values = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(calls):
            runner(x, weight, out)
        end.record()
        end.synchronize()
        values.append(float(start.elapsed_time(end)) / calls)
    return {
        "median_ms": statistics.median(values),
        "minimum_ms": min(values),
        "samples_ms": values,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=32768)
    parser.add_argument("--n", type=int, default=2304)
    parser.add_argument("--k", type=int, default=768)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--calls", type=int, default=10)
    args = parser.parse_args()

    torch.manual_seed(20260816)
    problem = NVFP4Problem(args.m, args.n, args.k)
    base = preferred_jit_row_region_config(problem)
    x = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(args.n, args.k, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(args.m, args.n, device="cuda", dtype=torch.bfloat16)

    configs = {
        "mma": replace(
            base,
            gemm=replace(
                base.gemm,
                stages=3,
                regional_epilogue_schedule="mma",
                tiles_per_cta=2,
            ),
        )
    }
    for warps in (1, 2, 4, 8):
        configs[f"epilogue_warps_{warps}"] = replace(
            base,
            gemm=replace(
                base.gemm,
                stages=1,
                epilogue="direct",
                epilogue_stages=1,
                store_vec=1,
                regional_scale_epilogue="factorized",
                regional_epilogue_schedule="warp_specialized",
                regional_epilogue_warps=warps,
                regional_epilogue_registers=(32 if warps == 1 else 48),
                tiles_per_cta=max(2, base.gemm.tiles_per_cta),
            ),
            programmatic_dependent_launch=False,
        )
    warp8 = configs["epilogue_warps_8"]
    for tiles in (2, 4, 8):
        configs[f"epilogue_warps_8_tiles_{tiles}"] = replace(
            warp8, gemm=replace(warp8.gemm, tiles_per_cta=tiles)
        )
    for locality in ("same_a", "same_b", "serpentine_a", "serpentine_b"):
        configs[f"epilogue_warps_8_{locality}"] = replace(
            warp8, gemm=replace(warp8.gemm, tile_locality=locality)
        )
    for registers in (24, 80):
        configs[f"epilogue_warps_8_regs_{registers}"] = replace(
            warp8,
            gemm=replace(
                warp8.gemm, regional_epilogue_registers=registers
            ),
        )

    report = {"shape": {"m": args.m, "n": args.n, "k": args.k}, "results": {}}
    with torch.inference_mode():
        for name, config in configs.items():
            rejection = config.rejection(problem)
            if rejection is not None:
                report["results"][name] = {"rejection": rejection}
                continue
            runner = _make_jit_region_dynamic_runner(problem, config, x.device)
            result = _measure(
                runner, x, weight, out, args.warmup, args.samples, args.calls
            )
            result["epilogue_warps"] = config.gemm.num_epilogue_warps
            result["operand_stages"] = config.gemm.stages
            report["results"][name] = result
            print(f"{name:20s} {result['median_ms'] * 1000.0:9.3f} us", flush=True)
    baseline = report["results"]["mma"].get("median_ms")
    if baseline is not None:
        for result in report["results"].values():
            if "median_ms" in result:
                result["speedup_over_mma"] = baseline / result["median_ms"]
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
