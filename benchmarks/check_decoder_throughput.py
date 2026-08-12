"""Summarize four-mode decoder runtime and convergence release evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics


MODES = ("bf16", "mxfp8", "nvfp4_delayed", "nvfp4_block")


def _records(path: Path, *, mode: str) -> list[dict[str, object]]:
    journal = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    records: list[dict[str, object]] = []
    for record in journal:
        record_type = record.get("record_type")
        if record_type == "resume_boundary":
            if record.get("precision") != mode:
                raise RuntimeError(f"foreign resume boundary in {path}")
            checkpoint_step = int(record["checkpoint_step"])
            records = [
                previous
                for previous in records
                if int(previous["step"]) <= checkpoint_step
            ]
        elif record_type == "training_step":
            records.append(record)
    if not records:
        raise RuntimeError(f"no training records in {path}")
    wrong = sorted(
        {
            str(record.get("precision"))
            for record in records
            if record.get("precision") != mode
        }
    )
    if wrong:
        raise RuntimeError(
            f"{path} contains modes {wrong}, expected only {mode!r}"
        )
    steps = [int(record["step"]) for record in records]
    if steps != sorted(set(steps)):
        raise RuntimeError(f"{path} contains duplicate or non-monotonic steps")
    return records


def _mode_summary(
    path: Path,
    *,
    mode: str,
    tail: int,
) -> dict[str, object]:
    records = _records(path, mode=mode)
    steady = [
        float(record["tokens_per_second"])
        for record in records
        if int(record["step"]) > 1
    ]
    if not steady:
        raise RuntimeError(f"no steady-state throughput records in {path}")
    selected = steady[-tail:]
    final = records[-1]
    validation = final.get("validation_loss")
    if validation is None:
        raise RuntimeError(f"final record has no validation loss in {path}")
    median = float(statistics.median(selected))
    return {
        "final_step": int(final["step"]),
        "final_train_loss": float(final["train_loss"]),
        "final_validation_loss": float(validation),
        "steady_tokens_per_second": median,
        "throughput_samples": len(selected),
        "throughput_min": min(selected),
        "throughput_max": max(selected),
        "throughput_coefficient_of_variation": (
            float(statistics.pstdev(selected) / median)
            if len(selected) > 1 and median
            else 0.0
        ),
        "measured_training_seconds": float(
            sum(float(record["elapsed_seconds"]) for record in records)
        ),
        "validation_curve": [
            {
                "step": int(record["step"]),
                "loss": float(record["validation_loss"]),
            }
            for record in records
            if record.get("validation_loss") is not None
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--tail", type=int, default=9)
    parser.add_argument("--minimum-mxfp8", type=float, default=1.30)
    parser.add_argument("--minimum-nvfp4-block", type=float, default=1.35)
    parser.add_argument("--minimum-nvfp4-delayed", type=float, default=0.0)
    args = parser.parse_args()
    if args.tail <= 0:
        parser.error("--tail must be positive")

    manifest_path = args.run / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_modes = tuple(manifest.get("precisions", ()))
    if manifest_modes != MODES:
        raise RuntimeError(
            f"{manifest_path} modes are {manifest_modes}, expected {MODES}"
        )
    policy = manifest.get("training_policy")
    if not isinstance(policy, dict):
        raise RuntimeError(f"{manifest_path} has no explicit training policy")

    modes = {
        mode: _mode_summary(
            args.run / mode / "metrics.jsonl",
            mode=mode,
            tail=args.tail,
        )
        for mode in MODES
    }
    final_steps = {int(value["final_step"]) for value in modes.values()}
    if len(final_steps) != 1:
        raise RuntimeError(f"training modes ended at different steps: {final_steps}")

    baseline = float(modes["bf16"]["steady_tokens_per_second"])
    speedups = {
        mode: float(modes[mode]["steady_tokens_per_second"]) / baseline
        for mode in MODES[1:]
    }
    minimums = {
        "mxfp8": args.minimum_mxfp8,
        "nvfp4_delayed": args.minimum_nvfp4_delayed,
        "nvfp4_block": args.minimum_nvfp4_block,
    }
    passed = all(speedups[mode] >= minimum for mode, minimum in minimums.items())
    print(
        json.dumps(
            {
                "run": str(args.run),
                "git_commit": manifest.get("git_commit"),
                "device": manifest.get("device"),
                "torch_version": manifest.get("torch_version"),
                "cuda_version": manifest.get("cuda_version"),
                "training_policy": policy,
                "modes": modes,
                "speedup_over_bf16": speedups,
                "minimum_speedup": minimums,
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
