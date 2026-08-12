"""Check compiled decoder throughput ratios from append-only training logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


def _steady_rate(path: Path, *, tail: int) -> float:
    records = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    rates = [
        float(record["tokens_per_second"])
        for record in records
        if record.get("record_type") == "training_step"
        and int(record.get("step", 0)) > 1
    ]
    if not rates:
        raise RuntimeError(f"no steady-state throughput records in {path}")
    return float(statistics.median(rates[-tail:]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--tail", type=int, default=3)
    parser.add_argument("--minimum-mxfp8", type=float, default=1.30)
    parser.add_argument("--minimum-nvfp4", type=float, default=1.35)
    args = parser.parse_args()
    if args.tail <= 0:
        parser.error("--tail must be positive")

    rates = {
        precision: _steady_rate(args.run / precision / "metrics.jsonl", tail=args.tail)
        for precision in ("bf16", "mxfp8", "nvfp4")
    }
    speedups = {
        precision: rates[precision] / rates["bf16"]
        for precision in ("mxfp8", "nvfp4")
    }
    passed = (
        speedups["mxfp8"] >= args.minimum_mxfp8
        and speedups["nvfp4"] >= args.minimum_nvfp4
    )
    print(
        json.dumps(
            {
                "run": str(args.run),
                "rates_tokens_per_second": rates,
                "speedup_over_bf16": speedups,
                "minimum_speedup": {
                    "mxfp8": args.minimum_mxfp8,
                    "nvfp4": args.minimum_nvfp4,
                },
                "passed": passed,
            },
            indent=2,
            sort_keys=True,
        )
    )
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
