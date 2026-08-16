"""Run the unified MXFP8Linear/NVFP4Linear production-readiness matrix."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import torch

import rtx
import validate_mxfp8_production as mx


def _nv_dynamic_case(
    *, compiled: bool, training: bool, scaling: str, backend: str
) -> dict[str, object]:
    torch.manual_seed(601 + int(compiled) + 2 * int(training))
    layer = rtx.NVFP4Linear(
        128,
        128,
        device="cuda",
        scaling=scaling,
        backend=backend,
        autotune="off",
    )
    layer.train(training)
    if scaling == "delayed":
        with torch.no_grad():
            layer(torch.randn(128, 128, device="cuda", dtype=torch.bfloat16))
    x = torch.randn(
        128,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=training,
    )
    reference_weight = layer.weight.detach().float().clone()
    function = (
        torch.compile(
            layer,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
        if compiled
        else layer
    )
    context = nullcontext() if training else torch.inference_mode()
    with context:
        actual = function(x)
        expected = x.detach().float() @ reference_weight.T
        result: dict[str, object] = {
            "scaling": scaling,
            "backend": backend,
            "forward_relative_l2": mx._relative_error(actual, expected),
        }
        if training:
            grad = torch.randn_like(actual)
            actual.backward(grad)
            result.update(
                dx_relative_l2=mx._relative_error(
                    x.grad, grad.float() @ reference_weight
                ),
                dw_relative_l2=mx._relative_error(
                    layer.weight.grad, grad.float().T @ x.detach().float()
                ),
            )
    torch.cuda.synchronize()
    if float(result["forward_relative_l2"]) >= 0.18:
        raise AssertionError(f"NVFP4 forward error exceeds 18%: {result}")
    gradient_errors = [
        float(result[name])
        for name in ("dx_relative_l2", "dw_relative_l2")
        if name in result
    ]
    if gradient_errors and max(gradient_errors) >= 0.07:
        raise AssertionError(f"MXFP8 backward error exceeds 7%: {result}")
    return result


def _nv_partial_training_case(*, gradient: str, compiled: bool) -> dict[str, object]:
    torch.manual_seed(641 + int(compiled))
    layer = rtx.NVFP4Linear(
        128,
        128,
        device="cuda",
        scaling="block",
        backend="materialized",
        autotune="off",
    )
    if gradient == "dx":
        layer.weight.requires_grad_(False)
    elif gradient != "dw":
        raise ValueError(f"unknown gradient {gradient!r}")
    x = torch.randn(
        128,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=gradient == "dx",
    )
    reference_weight = layer.weight.detach().float().clone()
    function = torch.compile(layer, fullgraph=True, dynamic=False) if compiled else layer
    actual = function(x)
    grad = torch.randn_like(actual)
    actual.backward(grad)
    if gradient == "dx":
        if x.grad is None or layer.weight.grad is not None:
            raise AssertionError("NVFP4 dX-only autograd materialized the wrong gradients")
        error = mx._relative_error(x.grad, grad.float() @ reference_weight)
    else:
        if x.grad is not None or layer.weight.grad is None:
            raise AssertionError("NVFP4 dW-only autograd materialized the wrong gradients")
        error = mx._relative_error(
            layer.weight.grad, grad.float().T @ x.detach().float()
        )
    torch.cuda.synchronize()
    if error >= 0.07:
        raise AssertionError(f"NVFP4 {gradient}-only error exceeds 7%: {error}")
    return {"gradient": gradient, "relative_l2": error}


def _nv_packed_case(*, fully_prequantized: bool, compiled: bool) -> dict[str, object]:
    torch.manual_seed(683 + int(compiled) + 2 * int(fully_prequantized))
    dynamic = rtx.NVFP4Linear(
        128,
        128,
        device="cuda",
        scaling="block",
        backend="materialized",
        autotune="off",
    ).eval()
    packed = dynamic.to_quantized_weight()
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    operand = rtx.quantize_nvfp4(x) if fully_prequantized else x
    function = torch.compile(packed, fullgraph=True, dynamic=False) if compiled else packed
    with torch.inference_mode():
        actual = function(operand)
        expected = x.float() @ dynamic.weight.detach().float().T
    torch.cuda.synchronize()
    error = mx._relative_error(actual, expected)
    if error >= 0.18:
        raise AssertionError(f"packed NVFP4 relative error exceeds 18%: {error}")
    return {
        "activation": "packed" if fully_prequantized else "dynamic_bf16",
        "weight": "packed",
        "compiled": compiled,
        "forward_relative_l2": error,
    }


def _nv_delayed_state_case() -> dict[str, object]:
    torch.manual_seed(719)
    layer = rtx.NVFP4Linear(128, 128, device="cuda", autotune="off")
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    streams = (torch.cuda.Stream(), torch.cuda.Stream())
    stream_ids = []
    outputs = []
    for stream in streams:
        with torch.cuda.stream(stream):
            outputs.append(layer(x))
            stream_ids.append(layer._delayed_problem[2])
    torch.cuda.synchronize()
    if stream_ids[0] == stream_ids[1] or not all(
        bool(torch.isfinite(value).all()) for value in outputs
    ):
        raise AssertionError("delayed state was not isolated by CUDA stream")
    layer.load_state_dict(layer.state_dict())
    if layer._delayed_initialized or layer._x_amax_state.numel():
        raise AssertionError("state_dict load did not reset delayed telemetry")
    return {"streams": 2, "state_reset": True}


def _nv_variable_shape_cache_case() -> dict[str, object]:
    rtx.clear_runtime_caches()
    layer = rtx.NVFP4Linear(
        128,
        128,
        device="cuda",
        scaling="block",
        backend="materialized",
        autotune="off",
    ).eval()
    with torch.inference_mode():
        for rows in range(128, 128 * 11, 128):
            layer(torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16))
    torch.cuda.synchronize()
    stats = rtx.clear_runtime_caches()["fp4"]["block_dynamic"]
    if int(stats["entries"]) > int(stats["max_entries"]):
        raise AssertionError(f"NVFP4 runner cache exceeded its bound: {stats}")
    return stats


def _mxfp8_cases(args) -> dict[str, object]:
    long_m = 1024 if args.quick else args.long_m
    return {
        "eager_dynamic_training": lambda: mx._dynamic_case(compiled=False, training=True),
        "compiled_dynamic_training": lambda: mx._dynamic_case(compiled=True, training=True),
        "eager_dx_only_training": lambda: mx._partial_training_case(gradient="dx", compiled=False),
        "compiled_dx_only_training": lambda: mx._partial_training_case(gradient="dx", compiled=True),
        "eager_dw_only_training": lambda: mx._partial_training_case(gradient="dw", compiled=False),
        "compiled_dw_only_training": lambda: mx._partial_training_case(gradient="dw", compiled=True),
        "eager_dynamic_inference": lambda: mx._dynamic_case(compiled=False, training=False),
        "compiled_dynamic_inference": lambda: mx._dynamic_case(compiled=True, training=False),
        "eager_dynamic_x_prequantized_weight": lambda: mx._packed_inference_case(
            fully_prequantized=False, compiled=False
        ),
        "compiled_dynamic_x_prequantized_weight": lambda: mx._packed_inference_case(
            fully_prequantized=False, compiled=True
        ),
        "eager_fully_prequantized": lambda: mx._packed_inference_case(
            fully_prequantized=True, compiled=False
        ),
        "compiled_fully_prequantized": lambda: mx._packed_inference_case(
            fully_prequantized=True, compiled=True
        ),
        "oriented_cpasync_cluster_backward": mx._cpasync_cluster_backward_case,
        "long_sequence_dw_full": lambda: mx._long_reduction_case(long_m, reduction="full_fp32"),
        "long_sequence_dw_workspace": lambda: mx._long_reduction_case(long_m, reduction="split_fp32_workspace"),
        "long_sequence_dw_atomic": lambda: mx._long_reduction_case(long_m, reduction="split_fp32_atomic"),
        "long_sequence_dw_cluster": lambda: mx._long_reduction_case(long_m, reduction="cluster_fp32"),
        "multiple_streams": mx._multiple_stream_case,
        "variable_shapes_bounded_cache": mx._variable_shape_cache_case,
    }


def _nvfp4_cases() -> dict[str, object]:
    cases: dict[str, object] = {}
    for compiled in (False, True):
        prefix = "compiled" if compiled else "eager"
        cases[f"{prefix}_delayed_training"] = lambda compiled=compiled: _nv_dynamic_case(
            compiled=compiled, training=True, scaling="delayed", backend="materialized"
        )
        cases[f"{prefix}_materialized_training"] = lambda compiled=compiled: _nv_dynamic_case(
            compiled=compiled, training=True, scaling="block", backend="materialized"
        )
        cases[f"{prefix}_current_scale_training"] = lambda compiled=compiled: _nv_dynamic_case(
            compiled=compiled, training=True, scaling="current", backend="materialized"
        )
        cases[f"{prefix}_materialized_inference"] = lambda compiled=compiled: _nv_dynamic_case(
            compiled=compiled, training=False, scaling="block", backend="materialized"
        )
        cases[f"{prefix}_current_scale_inference"] = lambda compiled=compiled: _nv_dynamic_case(
            compiled=compiled, training=False, scaling="current", backend="materialized"
        )
        cases[f"{prefix}_dx_only_training"] = lambda compiled=compiled: _nv_partial_training_case(
            gradient="dx", compiled=compiled
        )
        cases[f"{prefix}_dw_only_training"] = lambda compiled=compiled: _nv_partial_training_case(
            gradient="dw", compiled=compiled
        )
        cases[f"{prefix}_dynamic_x_prequantized_weight"] = lambda compiled=compiled: _nv_packed_case(
            fully_prequantized=False, compiled=compiled
        )
        cases[f"{prefix}_fully_prequantized"] = lambda compiled=compiled: _nv_packed_case(
            fully_prequantized=True, compiled=compiled
        )
    cases["delayed_state_stream_and_load_reset"] = _nv_delayed_state_case
    cases["variable_shapes_bounded_cache"] = _nv_variable_shape_cache_case
    return cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--long-m", type=int, default=8192)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--frontend", choices=("both", "mxfp8", "nvfp4"), default="both"
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        parser.error("validation requires an SM120/SM121 CUDA GPU")

    groups = {}
    if args.frontend in ("both", "mxfp8"):
        groups["mxfp8"] = _mxfp8_cases(args)
    if args.frontend in ("both", "nvfp4"):
        groups["nvfp4"] = _nvfp4_cases()
    results: dict[str, object] = {}
    for frontend, cases in groups.items():
        for name, case in cases.items():
            qualified = f"{frontend}.{name}"
            started = torch.cuda.Event(enable_timing=True)
            finished = torch.cuda.Event(enable_timing=True)
            try:
                # Many cases intentionally compile the same module forward
                # code object for distinct operand subclasses and policies.
                # Keep them independent so Dynamo's per-frame recompile limit
                # does not turn matrix breadth into a false product failure.
                torch._dynamo.reset()
                started.record()
                detail = case()
                finished.record()
                finished.synchronize()
                results[qualified] = {
                    "status": "ok",
                    "elapsed_ms": float(started.elapsed_time(finished)),
                    "detail": detail,
                }
                print(f"PASS {qualified}", flush=True)
            except Exception as exc:
                torch.cuda.synchronize()
                results[qualified] = {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
                print(f"FAIL {qualified}: {type(exc).__name__}: {exc}", flush=True)
    report = {
        "schema_version": 1,
        "type": "rtx_production_matrix",
        "frontends": list(groups),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "passed": sum(value["status"] == "ok" for value in results.values()),
        "failed": sum(value["status"] != "ok" for value in results.values()),
        "results": results,
    }
    encoded = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    if report["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
