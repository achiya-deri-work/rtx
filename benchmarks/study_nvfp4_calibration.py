"""CPU-only offline NVFP4 weight-scale calibration study (not a kernel benchmark).

Implements the block-diagonal second-moment objective described at
https://nvidia.github.io/Model-Optimizer/announcements/20260909-local-hessian.html
independently, without importing RTX or ModelOpt. Each row's 16-value K block
uses E2M1 RTNE codes and one positive finite E4M3FN scale, multiplied by a fixed
FP32 *tensor* outer scale. By default searches match RTX's min-normal scale
domain. The explicit full-e4m3 option searches all 126 positive scales, including
subnormals; that is a research domain, not RTX's current quantizer contract.
There is no arbitrary per-channel factor to fold into a GEMM epilogue.

Only weights are quantized here. Activation quantization, CUDA packing/layout,
GEMM accumulation, production kernel equivalence, timings, and model quality are
not validated. Local-Hessian ignores cross-block error interactions: its local
objective need not improve full calibration or held-out layer output error.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def _check_matrix(x: torch.Tensor, name: str) -> None:
    if x.device.type != "cpu" or x.ndim != 2 or min(x.shape) <= 0:
        raise ValueError(f"{name} must be a nonempty CPU matrix")
    if not x.is_floating_point() or not bool(torch.isfinite(x).all()):
        raise ValueError(f"{name} must contain finite floating-point values")


def e4m3_candidates(scale_domain: str = "full-e4m3") -> torch.Tensor:
    """Decode unsigned E4M3FN codes 1..126 exactly, including subnormals."""
    if scale_domain not in ("full-e4m3", "production-normal"):
        raise ValueError(f"unknown scale domain: {scale_domain}")
    code = torch.arange(1 if scale_domain == "full-e4m3" else 8, 127)
    exponent, mantissa = code // 8, code % 8
    return torch.where(exponent == 0, mantissa.float() * 2.0 ** -9,
                       (1 + mantissa.float() / 8) * torch.exp2(exponent.float() - 7))


def _round_e4m3_positive(x: torch.Tensor, scale_domain: str) -> torch.Tensor:
    # Positive nonzero scale contract; clamp underflow to the smallest subnormal.
    # A midpoint rounds to the even encoded mantissa, including exponent edges.
    levels = e4m3_candidates(scale_domain)
    midpoints = (levels[:-1] + levels[1:]) / 2
    index = torch.bucketize(x.contiguous(), midpoints)
    lower_code = index + (1 if scale_domain == "full-e4m3" else 8)
    tie = (index < levels.numel()-1) & (x == midpoints[index.clamp_max(levels.numel()-2)])
    index = index + (tie & (lower_code % 2 == 1)).long()
    return levels[index]


def fp4_round(x: torch.Tensor) -> torch.Tensor:
    """Independent saturating E2M1 RTNE, with decoded rather than packed codes."""
    levels = x.new_tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    edges = (levels[1:] + levels[:-1]) / 2
    a = x.abs()
    index = torch.bucketize(a.contiguous(), edges)
    tie = (index < 7) & (a == edges[index.clamp_max(6)])
    index = index + (tie & (index % 2 == 1)).long()
    return levels[index] * x.sign()


def local_second_moments(activations: torch.Tensor) -> torch.Tensor:
    """Return E[x_b x_b.T], not centered covariance, shape [ceil(K/16),16,16].

    Activations have shape [tokens,K]. Ragged K is padded with zero coordinates,
    not extra observations. The 1/N normalization does not affect the argmin.
    """
    _check_matrix(activations, "activations")
    x = F.pad(activations.float(), (0, (-activations.shape[1]) % 16))
    blocks = x.reshape(x.shape[0], -1, 16)
    return torch.einsum("tbi,tbj->bij", blocks, blocks) / x.shape[0]


@dataclass
class CalibratedWeight:
    codes: torch.Tensor  # decoded E2M1 grid, [out,ceil(K/16),16]
    block_scales: torch.Tensor  # decoded, exactly representable positive E4M3FN
    outer_scale: torch.Tensor  # scalar FP32, fixed across every policy
    input_features: int

    def dequantize(self) -> torch.Tensor:
        return (self.codes * (self.block_scales * self.outer_scale)[..., None]
                ).reshape(self.codes.shape[0], -1)[:, :self.input_features]


def calibrate_weight(weight: torch.Tensor, policy: str, *,
                     second_moments: torch.Tensor | None = None,
                     scale_domain: str = "production-normal",
                     outer_scale: torch.Tensor | None = None) -> CalibratedWeight:
    """Offline weight calibration, minimizing each block independently.

    Policies: max (amax/6), four_over_six (encoded amax/6 vs amax/4, weight SSE
    selector), mse (exhaustive weight SSE), local_hessian (exhaustive delta H
    delta). Exhaustive means all finite positive scales in the chosen domain.
    This two-candidate baseline is not an implementation of every
    Four-Over-Six paper variant. Ties choose the first candidate deterministically.
    """
    _check_matrix(weight, "weight")
    if policy not in ("max", "four_over_six", "mse", "local_hessian"):
        raise ValueError(f"unknown policy: {policy}")
    w = weight.float()
    k = w.shape[1]
    blocks = F.pad(w, (0, (-k) % 16)).reshape(w.shape[0], -1, 16)
    amax = w.abs().max()
    outer = torch.where(amax == 0, torch.ones_like(amax),
                        (amax / 2688).clamp_min(torch.finfo(torch.float32).tiny))
    if outer_scale is not None:
        if (outer_scale.device.type != "cpu" or outer_scale.numel() != 1 or
                not bool(torch.isfinite(outer_scale).all()) or float(outer_scale) <= 0):
            raise ValueError("outer_scale must be a finite positive CPU scalar")
        outer = outer_scale.float().reshape(())
    if second_moments is not None:
        if (second_moments.device.type != "cpu" or
                second_moments.shape != (blocks.shape[1], 16, 16) or
                not bool(torch.isfinite(second_moments).all())):
            raise ValueError("second_moments must be finite CPU [ceil(K/16),16,16]")
    if policy == "local_hessian" and second_moments is None:
        raise ValueError("local_hessian requires calibration second_moments")
    raw = (blocks.abs().amax(dim=-1) / outer) / 6
    max_scale = _round_e4m3_positive(raw, scale_domain)
    if policy in ("max", "four_over_six"):
        candidates = [max_scale]
        if policy == "four_over_six":
            candidates.append(_round_e4m3_positive(raw * 1.5, scale_domain))
    else:
        candidates = [s.expand_as(max_scale) for s in e4m3_candidates(scale_domain)]
    best_error = torch.full_like(max_scale, float("inf"))
    best_scale = max_scale.clone()
    best_codes = torch.zeros_like(blocks)
    for scale in candidates:
        effective_scale = scale * outer
        codes = fp4_round(blocks / effective_scale[..., None])
        delta = codes * effective_scale[..., None] - blocks
        if policy == "local_hessian":
            error = torch.einsum("obi,bij,obj->ob", delta,
                                 second_moments.float(), delta)
        else:
            error = delta.square().sum(dim=-1)
        improve = error < best_error
        best_codes = torch.where(improve[..., None], codes, best_codes)
        best_scale = torch.where(improve, scale, best_scale)
        best_error = torch.minimum(error, best_error)
    return CalibratedWeight(best_codes, best_scale, outer, k)


def metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    delta = candidate.float() - reference.float()
    rms_error = delta.square().mean().sqrt()
    reference_rms = reference.float().square().mean().sqrt()
    return dict(finite=bool(candidate.isfinite().all()),
                nonfinite_count=int((~candidate.isfinite()).sum()),
                max_abs=float(delta.abs().max()), rms_error=float(rms_error),
                reference_rms=float(reference_rms),
                normalized_rms=float(rms_error / reference_rms.clamp_min(1e-20)),
                normalized_max=float(delta.abs().max() /
                                     reference.float().abs().max().clamp_min(1e-20)))


def make_problem(out_features: int, in_features: int, tokens: int, seed: int,
                 distribution: str) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fixed weights/distribution; independently seeded calibration and test draws."""
    wg = torch.Generator().manual_seed(seed)
    weight = torch.randn(out_features, in_features, generator=wg) / math.sqrt(in_features)

    def draw(split: int):
        g = torch.Generator().manual_seed(seed + 100_003 * split)
        x = torch.randn(tokens, in_features, generator=g)
        if distribution == "correlated":
            shared = torch.randn(tokens, (in_features + 15) // 16, 1, generator=g)
            x = .15 * x + shared.expand(-1, -1, 16).reshape(tokens, -1)[:, :in_features]
        elif distribution == "anisotropic":
            x *= torch.exp2(torch.linspace(-3, 3, in_features))
        elif distribution != "normal":
            raise ValueError(distribution)
        return x.bfloat16().float()

    return weight.bfloat16().float(), draw(1), draw(2)


def load_tensor_problem(path: Path) -> tuple[dict[str, torch.Tensor], dict]:
    """Load only tensor-safe data, with caller-owned statistical independence.

    Required fields are weight [N,K], calibration_activations [Tcal,K], and
    heldout_activations [Ttest,K]. Optional provenance must be JSON-compatible.
    Content hashes document identity, not independent collection or no leakage.
    """
    payload = torch.load(path, map_location="cpu", weights_only=True)
    names = ("weight", "calibration_activations", "heldout_activations")
    if not isinstance(payload, dict) or any(name not in payload for name in names):
        raise ValueError(f"input must be a dictionary with distinct fields {names}")
    tensors = {}
    descriptions = {}
    for name in names:
        tensor = payload[name]
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"{name} must be a tensor")
        _check_matrix(tensor, name)
        if tensor.layout != torch.strided:
            raise ValueError(f"{name} must be strided")
        tensors[name] = tensor
        raw = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
        descriptions[name] = dict(shape=list(tensor.shape), dtype=str(tensor.dtype),
                                  sha256=hashlib.sha256(raw).hexdigest())
    if any(tensors[name].shape[1] != tensors["weight"].shape[1] for name in names[1:]):
        raise ValueError("weight and both activation matrices must have the same K")
    provenance = payload.get("provenance", {})
    # Fail before opening the result file if provenance cannot be serialized.
    json.dumps(provenance, allow_nan=False)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return tensors, dict(path=str(path.resolve()), file_sha256=digest.hexdigest(),
                         tensors=descriptions, provenance=provenance,
                         calibration_heldout_identical=(
                             descriptions[names[1]]["shape"] == descriptions[names[2]]["shape"] and
                             torch.equal(tensors[names[1]], tensors[names[2]])),
                         independence="caller responsibility; hashes do not establish split independence")


def compare_policies(w: torch.Tensor, calibration: torch.Tensor, heldout: torch.Tensor,
                     scale_domain: str, *, outer_scale: torch.Tensor | None = None):
    """Compare all policies without using heldout activations for calibration."""
    w, calibration, heldout = w.float(), calibration.float(), heldout.float()
    h = local_second_moments(calibration)
    calibration_reference, heldout_reference = calibration @ w.T, heldout @ w.T
    for policy in ("max", "four_over_six", "mse", "local_hessian"):
        quantized = calibrate_weight(w, policy, second_moments=h,
                                     scale_domain=scale_domain, outer_scale=outer_scale)
        qw = quantized.dequantize()
        delta = F.pad(qw-w, (0, (-w.shape[1]) % 16)).reshape(w.shape[0], -1, 16)
        yield dict(kind="result", policy=policy, scale_domain=scale_domain,
                   weight=metrics(qw, w),
                   calibration_output=metrics(calibration @ qw.T, calibration_reference),
                   heldout_output=metrics(heldout @ qw.T, heldout_reference),
                   local_objective=float(torch.einsum("obi,bij,obj->", delta, h, delta)),
                   outer_scale=float(quantized.outer_scale))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--out-features", type=int, default=16)
    parser.add_argument("--in-features", type=int, default=65)
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--input-tensors", type=Path,
                        help="tensor-safe .pt dictionary; replaces synthetic mode")
    parser.add_argument("--weight-rows", "--output-rows", type=int,
                        help="first N supplied output rows; preserves full input tensor outer scale")
    parser.add_argument("--scale-domain", choices=("production-normal", "full-e4m3"),
                        default="production-normal")
    args = parser.parse_args()
    if min(args.out_features, args.in_features, args.tokens, args.seeds) <= 0:
        parser.error("all sizes and seeds must be positive")
    if args.weight_rows is not None and (args.weight_rows <= 0 or args.input_tensors is None):
        parser.error("--weight-rows must be positive and requires --input-tensors")
    torch.set_num_threads(1)
    supplied = load_tensor_problem(args.input_tensors) if args.input_tensors else None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        def save(record):
            out.write(json.dumps(record, allow_nan=False) + "\n")
            out.flush()
        save(dict(kind="environment", torch=torch.__version__, device="cpu",
                  scope="offline weight-only numerical study, no CUDA or model qualification",
                  oracle=("FP32 GEMM on supplied tensor values converted to FP32" if supplied else
                          "FP32 GEMM on BF16-rounded inputs/weights"), floor=1e-20,
                  scale_contract="tensor FP32 outer maxabs/2688; positive finite E4M3 block16",
                  scale_domain=args.scale_domain,
                  scale_candidates=int(e4m3_candidates(args.scale_domain).numel()),
                  headroom="amax block has scale 448 at amax/6; amax/4 saturates to same scale",
                  calibration_test_split=("caller-supplied splits, independence not established by this script"
                                          if supplied else "independent seeds, same generating distribution"),
                  arguments={k: str(v) if isinstance(v, Path) else v
                             for k, v in vars(args).items()}))
        if supplied:
            tensors, description = supplied
            full_w = tensors["weight"].float()
            amax = full_w.abs().max()
            outer = torch.where(amax == 0, torch.ones_like(amax),
                                (amax / 2688).clamp_min(torch.finfo(torch.float32).tiny))
            w = full_w[:args.weight_rows] if args.weight_rows else full_w
            save(dict(kind="input", **description, evaluated_weight_shape=list(w.shape),
                      weight_row_selection="all" if args.weight_rows is None else f"first {w.shape[0]}",
                      outer_scale_source="full supplied weight tensor"))
            for record in compare_policies(w, tensors["calibration_activations"],
                                           tensors["heldout_activations"], args.scale_domain,
                                           outer_scale=outer):
                save(dict(distribution="supplied", **record))
            print(f"DONE supplied weight={list(w.shape)}", flush=True)
            return
        for distribution in ("normal", "correlated", "anisotropic"):
            for seed in range(args.seeds):
                w, calibration, heldout = make_problem(args.out_features, args.in_features,
                                                       args.tokens, seed, distribution)
                for record in compare_policies(w, calibration, heldout, args.scale_domain):
                    save(dict(seed=seed, distribution=distribution, **record))
                print(f"DONE seed={seed} distribution={distribution}", flush=True)


if __name__ == "__main__":
    main()
