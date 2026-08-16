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
    parser.add_argument("--x-region-rows", type=int)
    parser.add_argument("--weight-region-rows", type=int)
    args = parser.parse_args()

    torch.manual_seed(20260816)
    problem = NVFP4Problem(args.m, args.n, args.k)
    base = preferred_jit_row_region_config(problem)
    if args.x_region_rows is not None or args.weight_region_rows is not None:
        base = replace(
            base,
            x_scale_region_rows=(
                args.x_region_rows
                if args.x_region_rows is not None
                else base.x_scale_region_rows
            ),
            weight_scale_region_rows=(
                args.weight_region_rows
                if args.weight_region_rows is not None
                else base.weight_scale_region_rows
            ),
        )
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
    same_b = configs["epilogue_warps_8_same_b"]
    for strategy in ("factorized", "product"):
        configs[f"epilogue_warps_8_same_b_{strategy}"] = replace(
            same_b,
            gemm=replace(
                same_b.gemm,
                regional_scale_epilogue=strategy,
            ),
        )
    product = configs["epilogue_warps_8_same_b_product"]
    for values in (2, 4, 8):
        configs[f"epilogue_warps_8_same_b_product_v{values}"] = replace(
            product,
            gemm=replace(
                product.gemm,
                regional_epilogue_values=values,
            ),
        )
    packet4 = configs["epilogue_warps_8_same_b_product_v4"]
    for warps in (2, 4, 8):
        for tiles in (2, 4, 8):
            configs[f"packet4_w{warps}_t{tiles}"] = replace(
                packet4,
                gemm=replace(
                    packet4.gemm,
                    regional_epilogue_warps=warps,
                    tiles_per_cta=tiles,
                ),
            )
    if "epilogue_warps_8_same_b_product_v8" in configs:
        packet8 = configs["epilogue_warps_8_same_b_product_v8"]
        for warps in (2, 4, 8):
            for tiles in (2, 4, 8):
                configs[f"packet8_w{warps}_t{tiles}"] = replace(
                    packet8,
                    gemm=replace(
                        packet8.gemm,
                        regional_epilogue_warps=warps,
                        tiles_per_cta=tiles,
                    ),
                )
    for warps in (4, 8):
        for tiles in (2, 4, 8):
            configs[f"tile64_packet4_w{warps}_t{tiles}"] = replace(
                packet4,
                quant=replace(packet4.quant, scale_layout="row_major"),
                gemm=replace(
                    packet4.gemm,
                    tile_m=64,
                    atom_layout_m=2,
                    scale_layout="row_major",
                    scale_role="consumers",
                    regional_epilogue_warps=warps,
                    tiles_per_cta=tiles,
                ),
            )
    for atom_m in (4, 8):
        for warps in (4, 8):
            for tiles in (2, 4, 8):
                configs[
                    f"packet4_atom{atom_m}_w{warps}_t{tiles}"
                ] = replace(
                    packet4,
                    gemm=replace(
                        packet4.gemm,
                        atom_layout_m=atom_m,
                        regional_epilogue_warps=warps,
                        tiles_per_cta=tiles,
                    ),
                )
    atom4 = configs["packet4_atom4_w8_t4"]
    for registers in (24, 32, 40, 48, 64, 80):
        configs[f"atom4_epi_regs_{registers}"] = replace(
            atom4,
            gemm=replace(
                atom4.gemm,
                regional_epilogue_registers=registers,
            ),
        )
    for consumer_registers in (96, 112, 128, 160, 192):
        configs[f"atom4_mma_regs_{consumer_registers}"] = replace(
            atom4,
            gemm=replace(
                atom4.gemm,
                consumer_registers=consumer_registers,
            ),
        )
    for strategy in ("direct", "expanded_factors", "factorized", "product"):
        configs[f"atom4_scale_{strategy}"] = replace(
            atom4,
            gemm=replace(atom4.gemm, regional_scale_epilogue=strategy),
        )
    for waves in (0, 1, 2, 3, 4):
        configs[f"atom4_waves_{waves}"] = replace(
            atom4,
            gemm=replace(atom4.gemm, persistent_waves=waves),
        )
    for raster in ("m", "n"):
        for swizzle in (1, 2, 4, 8):
            configs[f"atom4_{raster}_swizzle_{swizzle}"] = replace(
                atom4,
                gemm=replace(
                    atom4.gemm,
                    raster=raster,
                    grid_swizzle=swizzle,
                ),
            )
    atom4_t8 = configs["packet4_atom4_w8_t8"]
    for locality in (
        "raster",
        "same_a",
        "same_b",
        "serpentine_a",
        "serpentine_b",
    ):
        configs[f"atom4_t4_{locality}"] = replace(
            atom4,
            gemm=replace(atom4.gemm, tile_locality=locality),
        )
    for locality in (
        "raster",
        "same_a",
        "same_b",
        "serpentine_a",
        "serpentine_b",
    ):
        configs[f"atom4_t8_{locality}"] = replace(
            atom4_t8,
            gemm=replace(atom4_t8.gemm, tile_locality=locality),
        )
    for raster in ("m", "n"):
        for swizzle in (1, 2, 4, 8):
            configs[f"atom4_t8_{raster}_swizzle_{swizzle}"] = replace(
                atom4_t8,
                gemm=replace(
                    atom4_t8.gemm,
                    raster=raster,
                    grid_swizzle=swizzle,
                ),
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
