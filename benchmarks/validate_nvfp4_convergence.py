"""Synthetic linear-regression training check, not transformer qualification."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
import math
from pathlib import Path
import statistics
import platform

import torch
from torch import nn

import rtx


def _median_window(values: list[float], width: int = 5) -> float:
    return statistics.median(values[-min(width, len(values)):])


def _adaptive_configs(batch: int, features: int) -> dict[str, rtx.NVFP4DynamicConfig]:
    from rtx.nvfp4_inference_autotune import preferred_jit_row_region_config

    base = preferred_jit_row_region_config(rtx.NVFP4Problem(batch, features, features))
    base = replace(base, x_scale_region_rows=8, weight_scale_region_rows=8,
                   tensor_scale_mode="exact")
    configs = {
        selection: replace(base, quant=replace(base.quant, values_per_lane=16,
                           load_bits=128, block_scale_selection=selection))
        for selection in ("amax", "lsq1", "mse5", "four_over_six")
    }
    # Keep the existing exact control: block-grid choice and outer headroom are
    # separate interventions, never an implicit change to four_over_six.
    configs["four_over_six_headroom"] = replace(
        configs["four_over_six"], tensor_scale_mode="exact_four_over_six"
    )
    return configs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--features", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260810)
    parser.add_argument("--seeds", type=int, nargs="+", help="Optional seed sweep")
    parser.add_argument("--feature-sizes", type=int, nargs="+", help="Optional square shape sweep")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--adaptive", action="store_true",
                        help="Compare BF16/MXFP8, exact block-scale policies, and explicit headroom")
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        parser.error("validation requires an SM120/SM121 CUDA GPU")
    sizes = args.feature_sizes or [args.features]
    if min(args.steps, args.batch, *sizes) <= 0:
        parser.error("steps, batch, and features must be positive")
    if any(size % 64 for size in sizes):
        parser.error("features must be divisible by 64")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("learning-rate must be finite and positive")
    cases = [run_case(args, seed, features)
             for seed in (args.seeds or [args.seed]) for features in sizes]
    document = cases[0] if len(cases) == 1 else {
        "evidence_scope": "synthetic_linear_regression_not_transformer_qualification",
        "passed": all(case["passed"] for case in cases), "cases": cases,
    }
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    if not document["passed"]:
        raise SystemExit(1)


def run_case(args: argparse.Namespace, seed: int, features: int) -> dict:
    torch.manual_seed(seed)
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
    configs = {}
    if args.adaptive:
        models = {
            "bf16": models["bf16"],
            "mxfp8": rtx.MXFP8Linear(features,features,device="cuda",autotune="off",
                                      backward_config=rtx.DEFAULT_MXFP8_BWD_CONFIG),
        }
        configs = _adaptive_configs(args.batch, features)
        for name, config in configs.items():
            models[name] = rtx.NVFP4Linear(features,features,device="cuda",
                scaling="jit_row_region",scale_region_rows=8,
                scale_config=rtx.NVFP4ScaleConfig(tensor_scale_mode=config.tensor_scale_mode),
                dynamic_config=config,autotune="off",
                backward_config=rtx.DEFAULT_MXFP8_BWD_CONFIG)
    for model in models.values():
        with torch.no_grad():
            model.weight.copy_(initial_weight)
    losses: dict[str, list[float]] = {name: [] for name in models}
    finite_gradients = {}
    finite_parameters = {}
    for name, original_model in models.items():
        # Independent fixed specializations must not compete for Dynamo's cache.
        if args.compile:
            torch._dynamo.reset()
        model = torch.compile(original_model, fullgraph=True, dynamic=False) if args.compile else original_model
        optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
        finite_gradients[name] = True
        for step in range(args.steps):
            generator = torch.Generator(device="cuda").manual_seed(seed + step + 1)
            x = torch.randn(args.batch, features, device="cuda", dtype=torch.bfloat16,
                            generator=generator)
            target = x.float() @ target_weight.float().T
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x).float()
            loss = (prediction - target).square().mean()
            loss.backward()
            finite_gradients[name] = finite_gradients[name] and bool(
                original_model.weight.grad is not None
                and torch.isfinite(original_model.weight.grad).all()
            )
            optimizer.step()
            losses[name].append(float(loss.detach()))
        finite_parameters[name] = bool(torch.isfinite(original_model.weight).all())
    torch.cuda.synchronize()
    summaries = {}
    for name, values in losses.items():
        final = _median_window(values)
        initial = statistics.median(values[: min(5, len(values))])
        summaries[name] = {
            "initial_median": initial,
            "final_median": final,
            "final_over_initial": final / initial,
            "finite": all(math.isfinite(value) for value in values),
            "finite_weight_gradients": finite_gradients[name],
            "finite_final_parameters": finite_parameters[name],
            "losses": values,
        }
    bf16_final = float(summaries["bf16"]["final_median"])
    passed = (bool(summaries["bf16"]["finite"])
              and finite_gradients["bf16"] and finite_parameters["bf16"])
    for name in (name for name in models if name != "bf16"):
        result = summaries[name]
        result["final_over_bf16"] = float(result["final_median"]) / bf16_final
        passed = passed and bool(result["finite"]) and bool(result["finite_weight_gradients"])
        passed = passed and bool(result["finite_final_parameters"])
        passed = passed and float(result["final_over_initial"]) < 0.9
        passed = passed and float(result["final_over_bf16"]) < 1.15
    document = {
        "device": torch.cuda.get_device_name(),
        "evidence_scope": "synthetic_linear_regression_not_transformer_qualification",
        "seed": seed,
        "input_seed_rule": "seed + step + 1",
        "parameter_dtype": "torch.bfloat16",
        "backward": "MXFP8 for quantized models; weight gradient only (inputs do not require grad)",
        "optimizer": "SGD",
        "dynamic_configs": {name: asdict(config) for name, config in configs.items()},
        "backward_config": asdict(rtx.DEFAULT_MXFP8_BWD_CONFIG) if args.adaptive else None,
        "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                        "cuda": torch.version.cuda, "compute_capability": torch.cuda.get_device_capability()},
        "gates": {"final_over_initial_lt": 0.9, "final_over_bf16_lt": 1.15},
        "steps": args.steps,
        "batch": args.batch,
        "features": features,
        "learning_rate": args.learning_rate,
        "compiled": args.compile,
        "adaptive": args.adaptive,
        "passed": passed,
        "policies": summaries,
    }
    return document


if __name__ == "__main__":
    main()
