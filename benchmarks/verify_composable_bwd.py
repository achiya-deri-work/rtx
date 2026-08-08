"""Independently measure and race saved MXFP8 backward configurations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rtx.bwd_autotune import bwd_config_from_dict
from rtx.bwd_experiments import BwdBenchmarkHarness
from rtx.prequant_experiments import BenchmarkProtocol, ShapeSpec


def _saved_config(path: Path):
    document = json.loads(path.read_text(encoding="utf-8"))
    return bwd_config_from_dict(document["config"])


def _legacy_best(path: Path):
    document = json.loads(path.read_text(encoding="utf-8"))
    return bwd_config_from_dict(document["best"]["config"])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, default=512)
    parser.add_argument("--n", type=int, default=1536)
    parser.add_argument("--k", type=int, default=1536)
    parser.add_argument("--legacy-db", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    parser.add_argument("--challenger", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=31)
    parser.add_argument("--rounds", type=int, default=75)
    parser.add_argument("--target-batch-ms", type=float, default=40.0)
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=20260827)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    protocol = BenchmarkProtocol(
        warmup_calls=25,
        samples=args.samples,
        confirm_samples=args.samples,
        race_rounds=args.rounds,
        target_batch_ms=args.target_batch_ms,
        correctness_rtol=0.07,
        correctness_atol=1.0,
        practical_threshold=0.0,
        bootstrap_resamples=args.bootstrap,
        telemetry=False,
    )
    harness = BwdBenchmarkHarness(
        ShapeSpec(args.m, args.n, args.k), protocol, seed=args.seed
    )
    configs = {
        "legacy": _legacy_best(args.legacy_db),
        "incumbent": _saved_config(args.incumbent),
        "challenger": _saved_config(args.challenger),
    }
    measurements: dict[str, object] = {}
    for index, (name, config) in enumerate(configs.items()):
        result = harness.measure(
            config,
            samples=args.samples,
            seed=args.seed + index,
            components=True,
        )
        measurements[name] = result
    payload = {
        "shape": {"m": args.m, "n": args.n, "k": args.k},
        "protocol": {
            "samples": args.samples,
            "rounds": args.rounds,
            "target_batch_ms": args.target_batch_ms,
            "bootstrap": args.bootstrap,
            "seed": args.seed,
        },
        "paths": {
            "legacy": str(args.legacy_db),
            "incumbent": str(args.incumbent),
            "challenger": str(args.challenger),
        },
        "measurements": measurements,
        "incumbent_vs_challenger": harness.race(
            configs["incumbent"], configs["challenger"], seed=args.seed + 3
        ),
        "legacy_vs_challenger": harness.race(
            configs["legacy"], configs["challenger"], seed=args.seed + 4
        ),
    }
    encoded = json.dumps(payload, indent=2, sort_keys=True)
    print(encoded)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
