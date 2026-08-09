"""Benchmark fused MXFP8 backward families against the decomposed baseline."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import time

import torch

from rtx.bwd_autotune import bwd_config_id, bwd_config_to_dict, update_bwd_config
from rtx.bwd_experiments import BwdBenchmarkHarness
from rtx.kernels.mxfp8 import MXFP8Problem, normalize_fwd_config
from rtx.kernels.mxfp8_bwd import (
    DEFAULT_DECOMPOSED_MXFP8_BWD_CONFIG,
    DEFAULT_DUAL_DECOMPOSED_MXFP8_BWD_CONFIG,
    DEFAULT_FUSED_MXFP8_BWD_CONFIG,
    DEFAULT_SEPARATE_DECOMPOSED_MXFP8_BWD_CONFIG,
)
from rtx.prequant_experiments import BenchmarkProtocol, ShapeSpec


def _split_for(k: int) -> tuple[int, int] | None:
    choices = [
        (parts, tile)
        for parts in (2, 4, 8, 16, 32)
        for tile in (128, 256, 512, 1024, 2048, 4096)
        if (parts - 1) * tile < k <= parts * tile
    ]
    if not choices:
        return None
    # Prefer fewer partials, then the least overcoverage.
    return min(choices, key=lambda item: (item[0], item[0] * item[1] - k))


def _configs(shape: ShapeSpec, *, reuse_sweep: bool = False) -> dict[str, object]:
    tma_three_role = normalize_fwd_config(
        load_engine="tma",
        schedule="three_role",
        bf16_tile_k=32,
        bf16_swizzle="none",
        bf16_stages=2,
        mxfp8_stages=2,
        quantizer_warps=4,
        quant_vec=8,
        quant_math="bf16x2",
        quant_amax="bf16_bits",
        # One transposing x4 ldmatrix loads an 8-row x 32-K logical tile from
        # MN-major TMA staging and delivers eight BF16 values per lane.
        quant_load_bits=128,
    )
    fused_tma = update_bwd_config(
        DEFAULT_FUSED_MXFP8_BWD_CONFIG,
        {
            "dx": {"fused": asdict(tma_three_role)},
            "dw": {"fused": asdict(tma_three_role)},
        },
    )
    tma_m64 = normalize_fwd_config(tma_three_role, tile_m=64)
    fused_tma_m64 = update_bwd_config(
        DEFAULT_FUSED_MXFP8_BWD_CONFIG,
        {
            "dx": {"fused": asdict(tma_m64)},
            "dw": {"fused": asdict(tma_m64)},
        },
    )
    tma_n256_reuse_a = normalize_fwd_config(
        tma_three_role,
        tile_n=256,
        mxfp8_stages=1,
        consumer_registers=96,
        maxrregcount=160,
    )
    tma_m256_reuse_b = normalize_fwd_config(
        tma_three_role,
        tile_m=256,
        mxfp8_stages=1,
        consumer_registers=96,
        maxrregcount=160,
    )
    fused_tma_dw_n256 = update_bwd_config(
        fused_tma,
        {"dw": {"fused": asdict(tma_n256_reuse_a)}},
    )
    fused_tma_dw_m256 = update_bwd_config(
        fused_tma,
        {"dw": {"fused": asdict(tma_m256_reuse_b)}},
    )
    configs = {
        "decomposed_separate": DEFAULT_SEPARATE_DECOMPOSED_MXFP8_BWD_CONFIG,
        "decomposed_dual": DEFAULT_DUAL_DECOMPOSED_MXFP8_BWD_CONFIG,
        "decomposed_dual_interleaved": update_bwd_config(
            DEFAULT_DUAL_DECOMPOSED_MXFP8_BWD_CONFIG,
            {"execution_order": "interleaved"},
        ),
        "decomposed_quad_quant": update_bwd_config(
            DEFAULT_DUAL_DECOMPOSED_MXFP8_BWD_CONFIG,
            {"quant_schedule": "quad"},
        ),
        "decomposed_quad_quant_dual_stream": update_bwd_config(
            DEFAULT_DUAL_DECOMPOSED_MXFP8_BWD_CONFIG,
            {"quant_schedule": "quad", "stream_schedule": "dual_stream"},
        ),
        "fused_scalar": DEFAULT_FUSED_MXFP8_BWD_CONFIG,
        "fused_tma_three_role": fused_tma,
        "fused_tma_m64": fused_tma_m64,
        "fused_tma_m64_dual_stream": update_bwd_config(
            fused_tma_m64, {"stream_schedule": "dual_stream"}
        ),
        "fused_tma_dw_n256_reuse_a": fused_tma_dw_n256,
        "fused_tma_dw_n256_reuse_a_dual_stream": update_bwd_config(
            fused_tma_dw_n256, {"stream_schedule": "dual_stream"}
        ),
        "fused_tma_dw_m256_reuse_b": fused_tma_dw_m256,
        "fused_tma_dw_m256_reuse_b_dual_stream": update_bwd_config(
            fused_tma_dw_m256, {"stream_schedule": "dual_stream"}
        ),
        "fused_tma_dual_stream": update_bwd_config(
            fused_tma, {"stream_schedule": "dual_stream"}
        ),
    }
    for store_bits in (8, 16, 32):
        configs[f"decomposed_quad_cpasync_store{store_bits}"] = update_bwd_config(
            DEFAULT_DECOMPOSED_MXFP8_BWD_CONFIG,
            {
                "dx": {
                    "quant_b": {
                        "transposed_load_engine": "cp_async",
                        "transposed_smem_padding": 0,
                        "transposed_tile_rows": 128,
                        "quant_store_bits": store_bits,
                        "quant_vec": 4 if store_bits == 32 else 2,
                    }
                },
                "dw": {
                    "quant_a": {
                        "transposed_load_engine": "cp_async",
                        "transposed_smem_padding": 0,
                        "transposed_tile_rows": 128,
                        "quant_store_bits": store_bits,
                        "quant_vec": 4 if store_bits == 32 else 2,
                    },
                    "quant_b": {
                        "transposed_load_engine": "cp_async",
                        "transposed_smem_padding": 0,
                        "transposed_tile_rows": 128,
                        "quant_store_bits": store_bits,
                        "quant_vec": 4 if store_bits == 32 else 2,
                    },
                },
            },
        )
    for store_bits in (8, 32):
        configs[f"decomposed_quad_register_tile128_store{store_bits}"] = update_bwd_config(
            DEFAULT_DECOMPOSED_MXFP8_BWD_CONFIG,
            {
                "dx": {
                    "quant_b": {
                        "transposed_load_engine": "register",
                        "transposed_smem_padding": 1,
                        "transposed_tile_rows": 128,
                        "quant_store_bits": store_bits,
                        "quant_vec": 4 if store_bits == 32 else 2,
                    }
                },
                "dw": {
                    "quant_a": {
                        "transposed_load_engine": "register",
                        "transposed_smem_padding": 1,
                        "transposed_tile_rows": 128,
                        "quant_store_bits": store_bits,
                        "quant_vec": 4 if store_bits == 32 else 2,
                    },
                    "quant_b": {
                        "transposed_load_engine": "register",
                        "transposed_smem_padding": 1,
                        "transposed_tile_rows": 128,
                        "quant_store_bits": store_bits,
                        "quant_vec": 4 if store_bits == 32 else 2,
                    },
                },
            },
        )
    for engine, scale_store in (
        ("register", "scalar"),
        ("register", "packed"),
        ("cp_async", "packed"),
    ):
        transport = {
            "transposed_load_engine": engine,
            "transposed_smem_padding": 0 if engine == "cp_async" else 1,
            "transposed_tile_k": 128,
            "native_scale_store": scale_store,
        }
        configs[
            f"decomposed_quad_k128_{engine}_{scale_store}"
        ] = update_bwd_config(
            DEFAULT_DECOMPOSED_MXFP8_BWD_CONFIG,
            {
                "dx": {"quant_b": transport},
                "dw": {"quant_a": transport, "quant_b": transport},
            },
        )
    for engine in ("register", "cp_async"):
        transport = {
            "transposed_load_engine": engine,
            "transposed_smem_padding": 0 if engine == "cp_async" else 1,
            "transposed_tile_k": 64,
            "native_scale_store": "scalar",
        }
        configs[f"decomposed_quad_k64_{engine}_scalar"] = update_bwd_config(
            DEFAULT_DECOMPOSED_MXFP8_BWD_CONFIG,
            {
                "dx": {"quant_b": transport},
                "dw": {"quant_a": transport, "quant_b": transport},
            },
        )
    problem = MXFP8Problem(shape.m, shape.n, shape.k)
    for operand in ("a", "b"):
        for cluster_size in (2, 4):
            clustered = normalize_fwd_config(
                tma_three_role,
                cluster_reuse_tile=(operand, cluster_size),
            )
            candidate = update_bwd_config(
                DEFAULT_FUSED_MXFP8_BWD_CONFIG,
                {
                    "dx": {"fused": asdict(clustered)},
                    "dw": {"fused": asdict(clustered)},
                    "stream_schedule": "dual_stream",
                },
            )
            if candidate.implementation_rejection(problem) is None:
                configs[
                    f"fused_tma_cluster_reuse_{operand}{cluster_size}_dual_stream"
                ] = candidate
    split = _split_for(shape.m)
    if split is not None:
        parts, tile = split
        configs["fused_tma_workspace"] = update_bwd_config(
            fused_tma,
            {
                "dw": {
                    "reduction": "split_fp32_workspace",
                    "split_reduction": parts,
                    "reduction_tile": tile,
                    "workspace_epilogue": "tree",
                }
            },
        )
        configs["fused_tma_atomic"] = update_bwd_config(
            fused_tma,
            {
                "dw": {
                    "reduction": "split_fp32_atomic",
                    "split_reduction": parts,
                    "reduction_tile": tile,
                    "workspace_epilogue": "none",
                }
            },
        )
    if reuse_sweep:
        resource_points = (
            # quantizer warps, consumer registers, quantizer registers
            (4, 96, 64),
            (4, 96, 80),
            (4, 96, 96),
            (4, 112, 64),
            (4, 112, 80),
            (4, 112, 96),
            (4, 128, 64),
        )
        for tile_m, tile_n, reuse in (
            (128, 256, "a"),
            (256, 128, "b"),
        ):
            for bf16_stages in (1, 2):
                for (
                    quantizer_warps,
                    consumer_registers,
                    quantizer_registers,
                ) in resource_points:
                    wide = normalize_fwd_config(
                        tma_three_role,
                        tile_m=tile_m,
                        tile_n=tile_n,
                        bf16_stages=bf16_stages,
                        mxfp8_stages=1,
                        quantizer_warps=quantizer_warps,
                        consumer_registers=consumer_registers,
                        quantizer_registers=quantizer_registers,
                        maxrregcount=160,
                    )
                    candidate = update_bwd_config(
                        fused_tma,
                        {"dw": {"fused": asdict(wide)}},
                    )
                    name = (
                        f"fused_tma_dw_reuse_{reuse}_bf16s{bf16_stages}_"
                        f"q{quantizer_warps}_c{consumer_registers}_"
                        f"qr{quantizer_registers}_dual_stream"
                    )
                    configs[name] = update_bwd_config(
                        candidate, {"stream_schedule": "dual_stream"}
                    )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=1536)
    parser.add_argument("--k", type=int, default=1536)
    parser.add_argument("--regime", choices=("hot", "rotate"), default="hot")
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--samples", type=int, default=11)
    parser.add_argument("--race-rounds", type=int, default=11)
    parser.add_argument("--target-batch-ms", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=20260809)
    parser.add_argument(
        "--only",
        help="comma-separated configuration names to benchmark",
    )
    parser.add_argument(
        "--reuse-sweep",
        action="store_true",
        help="sweep legal wide-CTA dW register/stage reuse basins",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        parser.error("benchmark requires an SM120/SM121 CUDA GPU")

    shape = ShapeSpec(args.m, args.n, args.k)
    protocol = BenchmarkProtocol(
        warmup_calls=args.warmups,
        samples=args.samples,
        confirm_samples=args.samples,
        race_rounds=args.race_rounds,
        target_batch_ms=args.target_batch_ms,
        correctness_rtol=0.07,
        correctness_atol=1.0,
        telemetry=True,
    )
    harness = BwdBenchmarkHarness(
        shape, protocol, regime=args.regime, seed=args.seed
    )
    configs = _configs(shape, reuse_sweep=args.reuse_sweep)
    if args.only:
        requested = tuple(part.strip() for part in args.only.split(",") if part.strip())
        missing = tuple(name for name in requested if name not in configs)
        if missing:
            parser.error(f"unknown --only configurations: {', '.join(missing)}")
        configs = {name: configs[name] for name in requested}
    measurements: dict[str, object] = {}
    for index, (name, config) in enumerate(configs.items()):
        print(f"START {name} {bwd_config_id(config)}", flush=True)
        started = time.monotonic()
        result = harness.measure(
            config,
            samples=args.samples,
            seed=args.seed + index,
            components=True,
        )
        result["wall_s"] = time.monotonic() - started
        result["config_id"] = bwd_config_id(config)
        result["config"] = bwd_config_to_dict(config)
        measurements[name] = result
        summary = result.get("summary_ms", {})
        print(
            f"SAVE  {name} status={result['status']} "
            f"median_ms={summary.get('median')}",
            flush=True,
        )

    races: dict[str, object] = {}
    reference_name = next(
        (name for name in ("decomposed_quad_quant_dual_stream", "decomposed_dual") if name in configs),
        next(iter(configs)),
    )
    reference = configs[reference_name]
    for index, (name, config) in enumerate(configs.items()):
        if name == reference_name or measurements[name].get("status") != "ok":
            continue
        print(f"RACE  {reference_name} vs {name}", flush=True)
        races[name] = harness.race(
            reference, config, seed=args.seed + 100 + index
        )

    fused_races: dict[str, object] = {}
    fused_reference_name = "fused_tma_dual_stream"
    if fused_reference_name in configs:
        fused_reference = configs[fused_reference_name]
        for index, (name, config) in enumerate(configs.items()):
            if (
                name == fused_reference_name
                or "reuse_" not in name
                or "dual_stream" not in name
                or measurements[name].get("status") != "ok"
            ):
                continue
            print(f"RACE  {fused_reference_name} vs {name}", flush=True)
            fused_races[name] = harness.race(
                fused_reference, config, seed=args.seed + 1000 + index
            )

    payload = {
        "schema_version": 1,
        "type": "mxfp8_backward_family_benchmark",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "shape": asdict(shape),
        "regime": args.regime,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "protocol": {
            "warmups": args.warmups,
            "samples": args.samples,
            "race_rounds": args.race_rounds,
            "target_batch_ms": args.target_batch_ms,
            "seed": args.seed,
        },
        "measurements": measurements,
        "race_reference": reference_name,
        "races_vs_reference": races,
        "fused_race_reference": fused_reference_name,
        "fused_races_vs_reference": fused_races,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"WROTE {args.output}", flush=True)
    if any(value.get("status") != "ok" for value in measurements.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
