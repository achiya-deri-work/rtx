"""Small controlled convergence study for NVFP4 tensor-scale policies."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import statistics

import torch
from torch import nn

import rtx


def _median_window(values: list[float], width: int = 5) -> float:
    return statistics.median(values[-min(width, len(values)):])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        parser.error("validation requires an SM120/SM121 CUDA GPU")
    if min(args.steps, args.batch, args.features) <= 0:
        parser.error("steps, batch, and features must be positive")
    if args.features % 64:
        parser.error("features must be divisible by 64")

    torch.manual_seed(args.seed)
    features = args.features
    target_weight = torch.randn(
        features, features, device="cuda", dtype=torch.bfloat16
    ) / features**0.5
    initial_weight = torch.randn_like(target_weight) / features**0.5
    exact_config = rtx.NVFP4ScaleConfig(tensor_scale_mode="exact")
    models: dict[str, nn.Module] = {
        "bf16": nn.Linear(
            features,
            features,
            bias=False,
            device="cuda",
            dtype=torch.bfloat16,
        ),
        "delayed_power2": rtx.NVFP4Linear(
            features, features, device="cuda", scaling="delayed"
        ),
        "delayed_exact": rtx.NVFP4Linear(
            features,
            features,
            device="cuda",
            scaling="delayed",
            scale_config=exact_config,
        ),
        "current_power2": rtx.NVFP4Linear(
            features, features, device="cuda", scaling="current"
        ),
        "jit_row_region": rtx.NVFP4Linear(
            features, features, device="cuda", scaling="jit_row_region"
        ),
        "block_only": rtx.NVFP4Linear(
            features, features, device="cuda", scaling="block"
        ),
    }
    for model in models.values():
        with torch.no_grad():
            model.weight.copy_(initial_weight)
    optimizers = {
        name: torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        for name, model in models.items()
    }
    losses: dict[str, list[float]] = {name: [] for name in models}
    for step in range(args.steps):
        generator = torch.Generator(device="cuda")
        generator.manual_seed(args.seed + step + 1)
        x = torch.randn(
            args.batch,
            features,
            device="cuda",
            dtype=torch.bfloat16,
            generator=generator,
        )
        target = x.float() @ target_weight.float().T
        for name, model in models.items():
            optimizer = optimizers[name]
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x).float()
            loss = (prediction - target).square().mean()
            loss.backward()
            optimizer.step()
            losses[name].append(float(loss.detach()))
    torch.cuda.synchronize()
    summaries = {}
    for name, values in losses.items():
        final = _median_window(values)
        initial = statistics.median(values[: min(5, len(values))])
        summaries[name] = {
            "initial_median": initial,
            "final_median": final,
            "final_over_initial": final / initial,
            "finite": all(torch.isfinite(torch.tensor(values))),
            "losses": values,
        }
    bf16_final = float(summaries["bf16"]["final_median"])
    passed = bool(summaries["bf16"]["finite"])
    for name in (
        "delayed_power2",
        "delayed_exact",
        "current_power2",
        "jit_row_region",
        "block_only",
    ):
        result = summaries[name]
        result["final_over_bf16"] = float(result["final_median"]) / bf16_final
        passed = passed and bool(result["finite"])
        passed = passed and float(result["final_over_initial"]) < 0.9
        passed = passed and float(result["final_over_bf16"]) < 1.15
    document = {
        "device": torch.cuda.get_device_name(),
        "steps": args.steps,
        "batch": args.batch,
        "features": features,
        "learning_rate": args.learning_rate,
        "passed": passed,
        "policies": summaries,
    }
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
