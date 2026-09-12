"""Capture MXFP8 rounding disagreements; this is not a release pass/fail gate.

Run on the failing device first, then replay its inputs.pt on another device.
The independent quantization oracle comes from the repository's kernel tests.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import sys

import torch


def round_fp64_to_bf16(value: torch.Tensor) -> torch.Tensor:
    """Nearest-even rounding for this finite, ordinary-range diagnostic.

    A plain .bfloat16() conversion may narrow through FP32. Compare both
    neighboring BF16 values in FP64 to avoid double rounding at a midpoint.
    """
    if value.dtype != torch.float64 or not bool(torch.isfinite(value).all()):
        raise ValueError("expected finite FP64 values")
    initial = value.to(torch.bfloat16)
    if not bool(torch.isfinite(initial).all()):
        raise ValueError("diagnostic does not handle BF16 overflow")
    direction = torch.where(value >= initial.double(), torch.inf, -torch.inf)
    neighbor = torch.nextafter(initial, direction.to(torch.bfloat16))
    initial_error = (value - initial.double()).abs()
    neighbor_error = (value - neighbor.double()).abs()
    choose_neighbor = (neighbor_error < initial_error) | (
        (neighbor_error == initial_error) & ((initial.view(torch.int16) & 1) != 0)
    )
    return torch.where(choose_neighbor, neighbor, initial)


def _digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.cpu().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--inputs", type=Path, help="Replay captured BF16 operands")
    parser.add_argument("--seed", type=int, default=321)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    # Never replace an existing capture or contaminate it with another run.
    args.output_dir.mkdir(parents=True, exist_ok=False)
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tests"))
    from test_mxfp8_fwd import _reference_to_mx
    from rtx.autotune.dataset import _source_snapshot
    from rtx.kernels.mxfp8 import MXFP8Problem, normalize_fwd_config
    from rtx.kernels.mxfp8_fwd import compile_mxfp8_fwd
    import rtx

    if args.inputs is None:
        torch.manual_seed(args.seed)
        x = (torch.randn(257, 256, device="cuda") * 0.5).bfloat16()
        weight = (torch.randn(385, 256, device="cuda") * 0.5).bfloat16()
    else:
        saved = torch.load(args.inputs, map_location="cpu", weights_only=True)
        x, weight = saved["x"].cuda(), saved["weight"].cuda()
    for value in (x, weight):
        if value.ndim != 2 or value.dtype != torch.bfloat16 or not bool(torch.isfinite(value).all()):
            raise ValueError("capture must contain finite 2D BF16 operands")
    if x.shape[1] != weight.shape[1]:
        raise ValueError("operand K dimensions differ")
    x, weight = x.contiguous(), weight.contiguous()
    torch.save({"x": x.cpu(), "weight": weight.cpu()}, args.output_dir / "inputs.pt")
    problem = MXFP8Problem(x.shape[0], weight.shape[0], x.shape[1])
    qx, sx = _reference_to_mx(x)
    qw, sw = _reference_to_mx(weight)
    a = qx.float() * sx.float().repeat_interleave(32, -1)
    b = qw.float() * sw.float().repeat_interleave(32, -1)
    fp32 = (a @ b.T).cpu()
    ac, bc = a.cpu().double(), b.cpu().double()
    fp64 = ac @ bc.T
    rounded = round_fp64_to_bf16(fp64)
    reference = fp32.bfloat16()
    disagreements = (reference != rounded).nonzero().tolist()
    cells = {tuple(index) for index in disagreements[:8]}
    if problem.m > 91 and problem.n > 247:
        cells.add((91, 247))
    results, outputs = {}, {}
    configs = {
        "scalar": normalize_fwd_config(),
        "scalar_m64": normalize_fwd_config(tile_m=64),
        "scalar_k256": normalize_fwd_config(tile_k=256, k_unroll=2),
        "tma": normalize_fwd_config(load_engine="tma"),
        "three_role": normalize_fwd_config(load_engine="tma", schedule="three_role",
                                            bf16_tile_k=32, bf16_stages=2,
                                            mxfp8_stages=2, quantizer_warps=4),
    }
    for name, config in configs.items():
        reason = config.implementation_rejection(problem)
        if reason is not None:
            results[name] = {"status": "rejected", "reason": reason}
            continue
        out = torch.empty(problem.m, problem.n, device="cuda", dtype=torch.bfloat16)
        compile_mxfp8_fwd(problem, config)(x, weight, out)
        torch.cuda.synchronize()
        out = out.cpu()
        outputs[name] = out
        mismatch = (out != rounded).nonzero().tolist()
        cells.update(tuple(index) for index in mismatch[:8])
        error = out.double() - fp64
        results[name] = {
            "status": "measured", "config": asdict(config),
            "finite": bool(torch.isfinite(out).all()),
            "mismatches_vs_fp32_reference": int((out != reference).sum()),
            "mismatches_vs_fp64_rounded": len(mismatch),
            "max_abs_error_vs_fp64": error.abs().max().item(),
            "rms_error_vs_fp64": error.square().mean().sqrt().item(),
            "reference_rms": fp64.square().mean().sqrt().item(),
            "relative_l2_vs_fp64": (error.norm() / fp64.norm().clamp_min(1e-30)).item(),
        }
    details = []
    for i, j in sorted(cells):
        exact = sum((Fraction(float(u)) * Fraction(float(v))
                     for u, v in zip(ac[i], bc[j])), Fraction())
        details.append({
            "index": [i, j], "exact_numerator": str(exact.numerator),
            "exact_denominator": str(exact.denominator), "exact_as_float64": float(exact),
            "fp64_sum": fp64[i, j].item(), "fp32_sum": fp32[i, j].item(),
            "fp32_reference_bf16": reference[i, j].item(),
            "fp64_rounded_bf16": rounded[i, j].item(),
            "kernel_bf16": {name: out[i, j].item() for name, out in outputs.items()},
        })
    report = {
        "type": "mxfp8_rounding_diagnostic", "not_a_release_gate": True,
        "source": _source_snapshot(), "runtime": rtx.validate_runtime_environment(),
        "diagnostic_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "device": torch.cuda.get_device_name(),
        "sm_count": torch.cuda.get_device_properties(0).multi_processor_count,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "input_origin": str(args.inputs) if args.inputs else "generated_on_device",
        "seed": args.seed if args.inputs is None else None,
        "shape": asdict(problem), "x_sha256": _digest(x), "weight_sha256": _digest(weight),
        "qx_sha256": _digest(qx), "qw_sha256": _digest(qw),
        "fp32_reference_mismatches_vs_fp64": len(disagreements),
        "cases": results, "cells": details,
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
