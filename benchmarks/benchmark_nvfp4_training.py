"""Paired NVFP4 delayed-training versus MXFP8 fused-forward benchmark."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
import statistics

import torch

from rtx.autotune.winners import load_runtime_winner, runtime_winner_key
from rtx.configs.nvfp4 import NVFP4Problem, normalize_nvfp4_fwd_config
from rtx.fp4 import _delayed_amax_state, _fallback_fused_config
from rtx.kernels.mxfp8 import (
    DEFAULT_MXFP8_FWD_CONFIG,
    MXFP8Problem,
    fwd_config_from_dict,
)
from rtx.kernels.mxfp8_fwd import compile_mxfp8_fwd
from rtx.kernels.nvfp4_fwd import (
    compile_nvfp4_fwd,
    nvfp4_telemetry_values,
)


def _parse_shape(value: str) -> tuple[int, int, int]:
    try:
        shape = tuple(int(item) for item in value.lower().replace("x", ",").split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("shape must be M,N,K") from exc
    if len(shape) != 3 or min(shape) <= 0:
        raise argparse.ArgumentTypeError("shape must contain three positive integers")
    return shape  # type: ignore[return-value]


def _time_batch(launch, calls: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(calls):
        launch()
    end.record()
    end.synchronize()
    return float(start.elapsed_time(end)) / calls


def _benchmark_shape(
    shape: tuple[int, int, int],
    *,
    winner_root: Path | None,
    warmup: int,
    rounds: int,
    calls: int,
    seed: int,
) -> dict[str, object]:
    m, n, k = shape
    torch.manual_seed(seed)
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    mx_problem = MXFP8Problem(m, n, k)
    nv_problem = NVFP4Problem(m, n, k)

    mx_config = load_runtime_winner(
        runtime_winner_key("mxfp8_fused_fwd", mx_problem, device=x.device),
        lambda value: fwd_config_from_dict(dict(value)),
        root=winner_root,
        rejection=lambda value: value.implementation_rejection(mx_problem),
    )
    if mx_config is None:
        mx_config = DEFAULT_MXFP8_FWD_CONFIG
    nv_config = load_runtime_winner(
        runtime_winner_key("nvfp4_fused_fwd", mx_problem, device=x.device),
        lambda value: normalize_nvfp4_fwd_config(**dict(value)),
        root=winner_root,
        rejection=lambda value: value.implementation_rejection(nv_problem),
    )
    if nv_config is None:
        nv_config = _fallback_fused_config(nv_problem)
    nv_config = replace(nv_config, collect_amax=True)

    mx_launcher = compile_mxfp8_fwd(mx_problem, mx_config)
    nv_launcher = compile_nvfp4_fwd(nv_problem, nv_config)
    mx_out = torch.empty(m, n, device="cuda", dtype=torch.bfloat16)
    nv_out = torch.empty_like(mx_out)
    state_values = nvfp4_telemetry_values(nv_problem, nv_config)
    x_amax_state = _delayed_amax_state(x, state_values)
    weight_amax_state = _delayed_amax_state(weight, state_values)
    next_x_amax_state = torch.empty_like(x_amax_state)
    next_weight_amax_state = torch.empty_like(weight_amax_state)

    def launch_mx() -> None:
        mx_launcher(x, weight, mx_out)

    def launch_nv() -> None:
        nonlocal x_amax_state, weight_amax_state
        nonlocal next_x_amax_state, next_weight_amax_state
        if nv_config.telemetry_layout == "scalar_atomic":
            from rtx.fp8_bwd import _zero_tensor_async

            _zero_tensor_async(next_x_amax_state)
            _zero_tensor_async(next_weight_amax_state)
        nv_launcher(
            x,
            weight,
            nv_out,
            x_amax_state,
            weight_amax_state,
            next_x_amax_state,
            next_weight_amax_state,
        )
        x_amax_state, next_x_amax_state = (
            next_x_amax_state,
            x_amax_state,
        )
        weight_amax_state, next_weight_amax_state = (
            next_weight_amax_state,
            weight_amax_state,
        )

    for _ in range(warmup):
        launch_mx()
        launch_nv()
    torch.cuda.synchronize()
    mx_times: list[float] = []
    nv_times: list[float] = []
    for round_index in range(rounds):
        if round_index % 2:
            nv_times.append(_time_batch(launch_nv, calls))
            mx_times.append(_time_batch(launch_mx, calls))
        else:
            mx_times.append(_time_batch(launch_mx, calls))
            nv_times.append(_time_batch(launch_nv, calls))

    reference = x.float() @ weight.float().T
    mx_error = (mx_out.float() - reference).abs()
    nv_error = (nv_out.float() - reference).abs()
    mx_median = statistics.median(mx_times)
    nv_median = statistics.median(nv_times)
    return {
        "shape": {"m": m, "n": n, "k": k},
        "mxfp8_ms": mx_median,
        "nvfp4_training_ms": nv_median,
        "speedup": mx_median / nv_median,
        "mxfp8_timings_ms": mx_times,
        "nvfp4_timings_ms": nv_times,
        "mxfp8_error": {
            "mean_abs": float(mx_error.mean()),
            "max_abs": float(mx_error.max()),
        },
        "nvfp4_error": {
            "mean_abs": float(nv_error.mean()),
            "max_abs": float(nv_error.max()),
        },
        "mxfp8_config": asdict(mx_config),
        "nvfp4_config": asdict(nv_config),
        "mxfp8_runtime_winner": mx_config != DEFAULT_MXFP8_FWD_CONFIG,
        "nvfp4_runtime_winner": nv_config
        != replace(_fallback_fused_config(nv_problem), collect_amax=True),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shape",
        action="append",
        type=_parse_shape,
        default=None,
        help="repeatable M,N,K shape (defaults: 256x1536x1536 and 1024x1536x1536)",
    )
    parser.add_argument("--winner-root", type=Path)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--rounds", type=int, default=21)
    parser.add_argument("--calls", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--minimum-speedup", type=float, default=1.5)
    parser.add_argument(
        "--allow-default-mxfp8",
        action="store_true",
        help="permit an untuned MXFP8 fallback (disabled for the release gate)",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        parser.error("benchmark requires an SM120/SM121 CUDA GPU")
    if min(args.warmup, args.rounds, args.calls) <= 0:
        parser.error("warmup, rounds, and calls must be positive")
    shapes = args.shape or [(256, 1536, 1536), (1024, 1536, 1536)]
    results = [
        _benchmark_shape(
            shape,
            winner_root=args.winner_root,
            warmup=args.warmup,
            rounds=args.rounds,
            calls=args.calls,
            seed=args.seed + index,
        )
        for index, shape in enumerate(shapes)
    ]
    document = {
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "minimum_speedup": args.minimum_speedup,
        "passed": all(
            float(result["speedup"]) >= args.minimum_speedup
            and (
                bool(result["mxfp8_runtime_winner"])
                or args.allow_default_mxfp8
            )
            for result in results
        ),
        "requires_tuned_mxfp8": not args.allow_default_mxfp8,
        "results": results,
    }
    payload = json.dumps(document, indent=2, sort_keys=True) + "\n"
    print(payload, end="")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload, encoding="utf-8")
    if not document["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
