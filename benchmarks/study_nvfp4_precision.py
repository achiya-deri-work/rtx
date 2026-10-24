"""Independent NVFP4 scale-policy study; oracle costs are NOT kernel timings.

Runs without importing RTX so the numerical experiments can use a newer Torch
without bypassing the production runtime contract. JSONL records survive an
interrupted sweep. All policies see identical BF16 inputs and an FP32 GEMM oracle.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F


def fp4_round(x: torch.Tensor) -> torch.Tensor:
    """Saturating E2M1 round-to-nearest, ties-to-even, independently decoded."""
    a = x.abs()
    levels = x.new_tensor([0, .5, 1, 1.5, 2, 3, 4, 6])
    midpoints = x.new_tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5])
    index = torch.bucketize(a.contiguous(), midpoints)
    # bucketize chooses the lower code on ties; odd lower codes must round up.
    for i in (1, 3, 5):
        index = torch.where(a == midpoints[i], i + 1, index)
    return levels[index] * x.sign()


def quantize(x: torch.Tensor, rows: int, mode: str, *, search: bool = False,
             ideal_block: bool = False, scale_factor: float = 1.,
             refine_steps: int = 0) -> torch.Tensor:
    """Reference two-level quantization, with optional block-MSE candidate search.

    Search includes the unchanged amax candidate, so its per-block squared
    reconstruction error cannot increase. This does NOT imply lower GEMM error.
    """
    m, k = x.shape
    padded = F.pad(x.float(), (0, (-k) % 16, 0, (-m) % rows))
    regions = padded.reshape(-1, rows, padded.shape[1])
    amax = regions.abs().amax(dim=(1, 2), keepdim=True)
    outer = (amax / 2688).clamp_min(2.0 ** -126)
    if mode == "power2":
        outer = torch.exp2(torch.ceil(torch.log2(outer)))
    elif mode == "block":
        outer = torch.ones_like(outer)
    elif mode != "exact":
        raise ValueError(mode)
    outer = torch.where(amax == 0, 1., outer).expand_as(regions).reshape(-1, 16)[..., :1]
    blocks = padded.reshape(-1, 16)
    raw = blocks.abs().amax(dim=1, keepdim=True) / (6 * outer)
    best = None
    error = None
    for factor in ((1., .75, .875, 1.125, 1.25) if search else (1.,)):
        scale = raw * factor * scale_factor
        if not ideal_block:
            scale = scale.clamp(2.0 ** -6, 448).to(torch.float8_e4m3fn).float()
        scale = scale * outer
        scale = torch.where(scale == 0, 1., scale)
        decoded = fp4_round(blocks / scale) * scale
        candidate_error = (blocks - decoded).square().sum(dim=1, keepdim=True)
        if best is None:
            best, error = decoded, candidate_error
            best_scale = scale
        else:
            replace = candidate_error < error
            best = torch.where(replace, decoded, best)
            best_scale = torch.where(replace, scale, best_scale)
            error = torch.minimum(error, candidate_error)
    # Alternating least-squares scale refinement with discrete E4M3 rounding.
    # Accept only improvements measured AFTER E2M1 requantization. The FP32
    # outer factor remains unchanged and the stored block scale stays E4M3.
    for _ in range(refine_steps):
        codes = fp4_round(blocks / best_scale)
        fitted = (blocks * codes).sum(dim=1, keepdim=True) / codes.square().sum(
            dim=1, keepdim=True).clamp_min(1e-20)
        scale = (fitted / outer).clamp(2.0 ** -6, 448).to(torch.float8_e4m3fn).float() * outer
        decoded = fp4_round(blocks / scale) * scale
        candidate_error = (blocks-decoded).square().sum(dim=1,keepdim=True)
        replace = candidate_error < error
        best = torch.where(replace, decoded, best)
        best_scale = torch.where(replace, scale, best_scale)
        error = torch.minimum(error, candidate_error)
    return best.reshape_as(padded)[:m, :k]


def metrics(candidate: torch.Tensor, reference: torch.Tensor) -> dict:
    c, r = candidate.float(), reference.float()
    delta = c - r
    rms = r.square().mean().sqrt()
    return dict(finite=bool(c.isfinite().all()), max_abs=float(delta.abs().max()),
                rms_error=float(delta.square().mean().sqrt()), reference_rms=float(rms),
                normalized_rms=float(delta.square().mean().sqrt() / rms.clamp_min(1e-20)))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--shape', action='append', default=[])
    p.add_argument('--seeds', type=int, default=3)
    args = p.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    shapes = [tuple(map(int, s.split(','))) for s in args.shape] or [
        (128, 256, 256), (513, 768, 768), (2048, 768, 1536)]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as out:
        def save(record):
            out.write(json.dumps(record, allow_nan=False) + '\n')
            out.flush()
        save(dict(kind='environment', torch=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), oracle='FP32 GEMM, TF32 disabled',
                  scope='numerical simulation only; not production timings'))
        for m, n, k in shapes:
            for seed in range(args.seeds):
                for distribution in ('normal', 'uniform', 'row_dynamic', 'outliers'):
                    gen = torch.Generator(device='cuda').manual_seed(seed)
                    x = torch.randn(m, k, device='cuda', generator=gen)
                    w = torch.randn(n, k, device='cuda', generator=gen) / math.sqrt(k)
                    if distribution == 'uniform':
                        x = (torch.rand(m, k, device='cuda', generator=gen) - .5)
                    elif distribution == 'row_dynamic':
                        x *= torch.exp2(torch.linspace(-6, 6, m, device='cuda'))[:, None]
                        w *= torch.exp2(torch.linspace(-3, 3, n, device='cuda'))[:, None]
                    elif distribution == 'outliers':
                        x[:, ::97] *= 20
                        w[:, ::113] *= 10
                    x, w = x.bfloat16(), w.bfloat16()
                    reference = x.float() @ w.float().T
                    for xr, wr in ((1, 1), (5, 4), (8, 4), (8, 8), (32, 32)):
                        for mode, search, ideal, factor, refine in (
                            ('power2', False, False, 1., 0), ('exact', False, False, 1., 0),
                            ('exact', True, False, 1., 0), ('exact', False, True, 1., 0),
                            ('exact', False, False, .875, 0), ('exact', False, False, .9375, 0),
                            ('exact', False, False, 1.0625, 0), ('exact', False, False, 1.125, 0),
                            ('exact', False, False, 1., 1), ('exact', False, False, 1., 2)):
                            qx = quantize(x, xr, mode, search=search, ideal_block=ideal, scale_factor=factor, refine_steps=refine)
                            qw = quantize(w, wr, mode, search=search, ideal_block=ideal, scale_factor=factor, refine_steps=refine)
                            record = dict(kind='result', shape=[m,n,k], seed=seed,
                                distribution=distribution, x_rows=xr, w_rows=wr,
                                mode=mode, block_mse_search=search, ideal_fp32_block=ideal,
                                scale_factor=factor,
                                refine_steps=refine,
                                x=metrics(qx,x), w=metrics(qw,w),
                                output=metrics(qx @ qw.T, reference))
                            save(record)
                    print(f'DONE shape={m,n,k} seed={seed} distribution={distribution}', flush=True)


if __name__ == '__main__':
    main()
