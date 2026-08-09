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
from rtx.kernels.mxfp8 import normalize_fwd_config
from rtx.kernels.mxfp8_bwd import (
    DEFAULT_DECOMPOSED_MXFP8_BWD_CONFIG,
    DEFAULT_FUSED_MXFP8_BWD_CONFIG,
)
from rtx.prequant_experiments import BenchmarkProtocol, ShapeSpec


def _split_for(k: int) -> tuple[int, int]:
    choices = [
        (parts, tile)
        for parts in (2, 4, 8, 16, 32)
        for tile in (128, 256, 512, 1024, 2048, 4096)
        if (parts - 1) * tile < k <= parts * tile
    ]
    if not choices:
        raise ValueError(f"no represented split covers reduction length {k}")
    # Prefer fewer partials, then the least overcoverage.
    return min(choices, key=lambda item: (item[0], item[0] * item[1] - k))


def _configs(shape: ShapeSpec) -> dict[str, object]:
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
        # Logical-transpose inputs are contiguous along MN, not K.  TMA stages
        # them in an MN-major CuTe layout and the quantizer uses layout-aware
        # scalar loads before emitting K-major FP8 MMA tiles.
        quant_load_bits=16,
    )
    fused_tma = update_bwd_config(
        DEFAULT_FUSED_MXFP8_BWD_CONFIG,
        {
            "dx": {"fused": asdict(tma_three_role)},
            "dw": {"fused": asdict(tma_three_role)},
        },
    )
    parts, tile = _split_for(shape.m)
    workspace = update_bwd_config(
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
    atomic = update_bwd_config(
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
    return {
        "decomposed": DEFAULT_DECOMPOSED_MXFP8_BWD_CONFIG,
        "fused_scalar": DEFAULT_FUSED_MXFP8_BWD_CONFIG,
        "fused_tma_three_role": fused_tma,
        "fused_tma_workspace": workspace,
        "fused_tma_atomic": atomic,
        "fused_tma_dual_stream": update_bwd_config(
            fused_tma, {"stream_schedule": "dual_stream"}
        ),
    }


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
    configs = _configs(shape)
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
    reference = configs["decomposed"]
    for index, (name, config) in enumerate(configs.items()):
        if name == "decomposed" or measurements[name].get("status") != "ok":
            continue
        print(f"RACE  decomposed vs {name}", flush=True)
        races[name] = harness.race(
            reference, config, seed=args.seed + 100 + index
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
        "races_vs_decomposed": races,
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
