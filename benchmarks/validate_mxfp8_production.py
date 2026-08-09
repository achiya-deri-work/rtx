"""Run the MXFP8Linear production-readiness matrix on one SM120/SM121 GPU."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import traceback

import torch

import rtx
from rtx.bwd_autotune import update_bwd_config


def _relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    numerator = (actual.detach().float() - expected.detach().float()).norm()
    denominator = expected.detach().float().norm().clamp_min(1.0e-12)
    return float(numerator / denominator)


def _dynamic_case(*, compiled: bool, training: bool) -> dict[str, object]:
    torch.manual_seed(101 + int(compiled) + 2 * int(training))
    layer = rtx.MXFP8Linear(
        128,
        128,
        device="cuda",
        backend="fused",
        autotune="off",
    )
    layer.train(training)
    x = torch.randn(
        128,
        128,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=training,
    )
    weight_reference = layer.weight.detach().float().clone()
    function = layer
    if compiled:
        function = torch.compile(layer, fullgraph=True, dynamic=False)
    context = nullcontext() if training else torch.inference_mode()
    with context:
        actual = function(x)
        expected = x.detach().float() @ weight_reference.T
        forward_error = _relative_error(actual, expected)
        result: dict[str, object] = {"forward_relative_l2": forward_error}
        if training:
            grad = torch.randn_like(actual)
            actual.backward(grad)
            expected_dx = grad.float() @ weight_reference
            expected_dw = grad.float().T @ x.detach().float()
            result.update(
                dx_relative_l2=_relative_error(x.grad, expected_dx),
                dw_relative_l2=_relative_error(layer.weight.grad, expected_dw),
            )
    torch.cuda.synchronize()
    thresholds = [float(result["forward_relative_l2"])]
    thresholds.extend(
        float(result[name])
        for name in ("dx_relative_l2", "dw_relative_l2")
        if name in result
    )
    if max(thresholds) >= 0.07:
        raise AssertionError(f"relative error exceeds 7%: {result}")
    return result


def _packed_inference_case(*, fully_prequantized: bool) -> dict[str, object]:
    torch.manual_seed(211 + int(fully_prequantized))
    dynamic = rtx.MXFP8Linear(
        128, 128, device="cuda", backend="prequant", autotune="off"
    ).eval()
    packed = dynamic.to_quantized_weight()
    x = torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
    operand = rtx.quantize_mxfp8(x) if fully_prequantized else x
    with torch.inference_mode():
        actual = packed(operand)
        expected = x.float() @ dynamic.weight.detach().float().T
    torch.cuda.synchronize()
    error = _relative_error(actual, expected)
    if error >= 0.07:
        raise AssertionError(f"packed inference relative error exceeds 7%: {error}")
    return {
        "activation": "packed" if fully_prequantized else "dynamic_bf16",
        "weight": "packed",
        "forward_relative_l2": error,
    }


def _long_reduction_case(m: int, *, reduction: str) -> dict[str, object]:
    torch.manual_seed(307)
    n = k = 512
    x = torch.randn(m, k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(n, k, device="cuda", dtype=torch.bfloat16)
    grad_output = torch.randn(m, n, device="cuda", dtype=torch.bfloat16)
    config = rtx.DEFAULT_MXFP8_BWD_CONFIG
    if reduction != "full_fp32":
        choices = tuple(
            (parts, reduction_tile)
            for parts in (2, 4, 8, 16, 32)
            for reduction_tile in (128, 256, 512, 1024, 2048, 4096)
        )
        split_count, tile = next(
            (parts, reduction_tile)
            for parts, reduction_tile in choices
            if (parts - 1) * reduction_tile < m <= parts * reduction_tile
        )
        config = update_bwd_config(
            config,
            {
                "dw": {
                    "reduction": reduction,
                    "split_reduction": split_count,
                    "reduction_tile": tile,
                    "workspace_epilogue": (
                        "tree" if reduction == "split_fp32_workspace" else "none"
                    ),
                }
            },
        )
    _dx, actual_dw = rtx.mxfp8_linear_backward(
        grad_output, x, weight, config=config, autotune="off"
    )
    expected_dw = grad_output.float().T @ x.float()
    torch.cuda.synchronize()
    error = _relative_error(actual_dw, expected_dw)
    if error >= 0.07:
        raise AssertionError(f"long-reduction dW relative error exceeds 7%: {error}")
    return {
        "m": m,
        "n": n,
        "k": k,
        "reduction": reduction,
        "dw_relative_l2": error,
    }


def _multiple_stream_case() -> dict[str, object]:
    torch.manual_seed(401)
    layer = rtx.MXFP8Linear(
        128, 128, device="cuda", backend="prequant", autotune="off"
    ).eval()
    streams = [torch.cuda.Stream() for _ in range(3)]
    inputs = [
        torch.randn(128, 128, device="cuda", dtype=torch.bfloat16)
        for _ in streams
    ]
    outputs = []
    for stream, value in zip(streams, inputs):
        with torch.cuda.stream(stream), torch.inference_mode():
            outputs.append(layer(value))
    for stream in streams:
        stream.synchronize()
    errors = [
        _relative_error(output, value.float() @ layer.weight.detach().float().T)
        for output, value in zip(outputs, inputs)
    ]
    if max(errors) >= 0.07:
        raise AssertionError(f"multi-stream relative error exceeds 7%: {errors}")
    return {"streams": len(streams), "maximum_forward_relative_l2": max(errors)}


def _variable_shape_cache_case() -> dict[str, object]:
    rtx.clear_runtime_caches()
    layer = rtx.MXFP8Linear(
        128, 128, device="cuda", backend="prequant", autotune="off"
    ).eval()
    with torch.inference_mode():
        for rows in range(128, 128 * 11, 128):
            layer(torch.randn(rows, 128, device="cuda", dtype=torch.bfloat16))
    torch.cuda.synchronize()
    before_clear = rtx.clear_runtime_caches()
    stats = before_clear["fp8"]["prequant"]
    if int(stats["entries"]) > int(stats["max_entries"]):
        raise AssertionError(f"runner cache exceeded its bound: {stats}")
    if int(stats["evictions"]) == 0 and int(stats["max_entries"]) < 11:
        raise AssertionError(f"variable-shape run did not exercise eviction: {stats}")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--long-m", type=int, default=8192)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="use M=1024 for the long-reduction check",
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
        parser.error("validation requires an SM120/SM121 CUDA GPU")

    cases = {
        "eager_dynamic_training": lambda: _dynamic_case(compiled=False, training=True),
        "compiled_dynamic_training": lambda: _dynamic_case(compiled=True, training=True),
        "eager_dynamic_inference": lambda: _dynamic_case(compiled=False, training=False),
        "compiled_dynamic_inference": lambda: _dynamic_case(compiled=True, training=False),
        "dynamic_x_prequantized_weight": lambda: _packed_inference_case(
            fully_prequantized=False
        ),
        "fully_prequantized": lambda: _packed_inference_case(
            fully_prequantized=True
        ),
        "long_sequence_dw_full": lambda: _long_reduction_case(
            1024 if args.quick else args.long_m, reduction="full_fp32"
        ),
        "long_sequence_dw_workspace": lambda: _long_reduction_case(
            1024 if args.quick else args.long_m,
            reduction="split_fp32_workspace",
        ),
        "long_sequence_dw_atomic": lambda: _long_reduction_case(
            1024 if args.quick else args.long_m,
            reduction="split_fp32_atomic",
        ),
        "multiple_streams": _multiple_stream_case,
        "variable_shapes_bounded_cache": _variable_shape_cache_case,
    }
    results: dict[str, object] = {}
    for name, case in cases.items():
        started = torch.cuda.Event(enable_timing=True)
        finished = torch.cuda.Event(enable_timing=True)
        try:
            started.record()
            detail = case()
            finished.record()
            finished.synchronize()
            results[name] = {
                "status": "ok",
                "elapsed_ms": float(started.elapsed_time(finished)),
                "detail": detail,
            }
            print(f"PASS {name}", flush=True)
        except Exception as exc:  # keep the matrix actionable after one failure
            torch.cuda.synchronize()
            results[name] = {
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
            print(f"FAIL {name}: {type(exc).__name__}: {exc}", flush=True)
    report = {
        "schema_version": 1,
        "type": "rtx_mxfp8_production_matrix",
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
