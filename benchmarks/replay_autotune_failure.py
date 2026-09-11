"""Replay one saved candidate with fresh seeded inputs in an isolated process.

This is a correctness diagnostic, not a benchmark or exact tensor replay.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

from rtx.autotune.dataset import FusedFwdBenchmarkHarness, _source_snapshot
from rtx.bwd_autotune import bwd_config_from_dict
from rtx.bwd_experiments import BwdBenchmarkHarness
from rtx.kernels.mxfp8 import MXFP8FwdConfig
from rtx.prequant_experiments import BenchmarkProtocol, ShapeSpec


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--config-id", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--shape", type=int, nargs=3, metavar=("M", "N", "K"))
    parser.add_argument("--quantizer-warps", type=int,
                        help="Override dX three-role quantizer warps for fault localization")
    parser.add_argument("--mma-m-atoms", type=int,
                        help="Override the NVFP4 GEMM M-atom layout for fault localization")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    row = next((json.loads(line)
                for path in sorted(args.dataset.rglob("observations.jsonl"))
                for line in path.open()
                if json.loads(line).get("config_id") == args.config_id), None)
    if row is None:
        parser.error("configuration not found")
    dims = args.shape or [int(row["features"][f"context.workload.{d}"]) for d in "mnk"]
    shape = ShapeSpec(*dims)
    protocol = BenchmarkProtocol(
        correctness_rtol=0.07 if row["family"] == "mxfp8_bwd" else 0.05,
        correctness_atol=0.5,
    )
    result = dict(config_id=args.config_id, family=row["family"], shape=dims,
                  seed=args.seed, config=row["config"], source=_source_snapshot(),
                  protocol=asdict(protocol))
    try:
        if row["family"] == "mxfp8_fused_fwd":
            config = MXFP8FwdConfig(**row["config"])
            harness = FusedFwdBenchmarkHarness(shape, "hot", protocol, seed=args.seed)
        elif row["family"] == "mxfp8_bwd":
            config = bwd_config_from_dict(row["config"])
            if args.quantizer_warps is not None:
                from rtx.kernels.mxfp8 import normalize_fwd_config
                config = replace(config, dx=replace(config.dx, fused=normalize_fwd_config(
                    config.dx.fused, quantizer_warps=args.quantizer_warps)))
                result["config"] = asdict(config)
            harness = BwdBenchmarkHarness(shape, protocol, seed=args.seed)
        else:
            from rtx.nvfp4_inference_autotune import (
                dynamic_config_from_dict, weight_prequant_config_from_dict,
                fully_prequant_config_from_dict,
            )
            from rtx.nvfp4_inference_experiments import (
                NVFP4DelayedBenchmarkHarness, NVFP4WeightPrequantBenchmarkHarness,
                NVFP4FullyPrequantBenchmarkHarness,
            )
            factories = {
                "nvfp4_delayed_fwd": (dynamic_config_from_dict, NVFP4DelayedBenchmarkHarness),
                "nvfp4_weight_prequant_fwd": (weight_prequant_config_from_dict, NVFP4WeightPrequantBenchmarkHarness),
                "nvfp4_fully_prequant_fwd": (fully_prequant_config_from_dict, NVFP4FullyPrequantBenchmarkHarness),
            }
            if row["family"] not in factories:
                parser.error("unsupported replay family")
            deserialize, factory = factories[row["family"]]
            config = deserialize(row["config"])
            if args.mma_m_atoms is not None:
                config = replace(config, gemm=replace(config.gemm, atom_layout_m=args.mma_m_atoms))
                result["config"] = asdict(config)
            harness = factory(shape, "hot", protocol, seed=args.seed)
        prepared = harness.prepare(config)
        result.update(status="ok", max_abs_error=prepared.max_abs_error)
    except Exception as exc:
        result.update(status="failed", error=f"{type(exc).__name__}: {exc}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("config", "source")}))
    raise SystemExit(result["status"] != "ok")


if __name__ == "__main__":
    main()
