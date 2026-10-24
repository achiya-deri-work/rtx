"""Opt-in, offline weight-only NVFP4 post-training calibration.

The search runs on CPU and returns native row-major packed ``NVFP4Tensor``
storage on the weight's original device. No activation quantizer, dynamic
training policy, runtime kernel, or model-quality guarantee is implied.
Local-Hessian uses uncentered block-diagonal second moments: it omits
cross-block interactions and need not improve full-layer or held-out error.
"""

from __future__ import annotations

import math
from numbers import Real

import torch
import torch.nn.functional as F

from .configs.nvfp4 import nvfp4_storage_k
from .formats.nvfp4 import NVFP4Tensor, make_nvfp4_tensor


def _cpu_float(tensor: torch.Tensor, name: str) -> torch.Tensor:
    if (not isinstance(tensor, torch.Tensor) or tensor.layout != torch.strided
            or not tensor.is_floating_point()):
        raise ValueError(f"{name} must be a strided floating-point tensor")
    result = tensor.detach().to(device="cpu", dtype=torch.float32)
    if not bool(result.isfinite().all()):
        raise ValueError(f"{name} must be finite and representable in FP32")
    return result


class NVFP4SecondMomentCollector:
    """Streaming CPU accumulator of E[x_b x_b.T], not centered covariance.

    ``update`` accepts dense floating activations ``[..., in_features]``;
    tokens are accumulated in FP64, with ragged coordinates zero-padded.
    Collection is explicit and detached; no hooks or model mutations occur.
    """

    def __init__(self, in_features: int):
        if not isinstance(in_features, int) or isinstance(in_features, bool) or in_features <= 0:
            raise ValueError("in_features must be a positive integer")
        self.in_features = in_features
        self.num_samples = 0
        self._sum = torch.zeros(((in_features + 15) // 16, 16, 16), dtype=torch.float64)

    @torch.no_grad()
    def update(self, activations: torch.Tensor) -> None:
        x = _cpu_float(activations, "activations")
        if x.ndim < 2 or x.shape[-1] != self.in_features or x.numel() == 0:
            raise ValueError("activations must be nonempty [..., in_features]")
        x = x.reshape(-1, self.in_features).double()
        x = F.pad(x, (0, (-self.in_features) % 16)).reshape(x.shape[0], -1, 16)
        contribution = torch.einsum("tbi,tbj->bij", x, x)
        if not bool(contribution.isfinite().all()):
            raise ValueError("activation second moments overflowed FP64")
        self._sum.add_(contribution)
        self.num_samples += x.shape[0]

    def second_moments(self) -> torch.Tensor:
        if self.num_samples == 0:
            raise ValueError("collect at least one activation sample first")
        result = (self._sum / self.num_samples).float()
        if not bool(result.isfinite().all()):
            raise ValueError("activation second moments are not representable in FP32")
        return result


def local_second_moments(activations: torch.Tensor) -> torch.Tensor:
    """Compute CPU FP32 uncentered [ceil(K/16),16,16] second moments."""
    if not isinstance(activations, torch.Tensor) or activations.ndim < 2:
        raise ValueError("activations must be nonempty [..., K]")
    collector = NVFP4SecondMomentCollector(activations.shape[-1])
    collector.update(activations)
    return collector.second_moments()


def _normal_scales() -> torch.Tensor:
    # All 119 strictly positive normal finite E4M3FN values (codes 8..126).
    code = torch.arange(8, 127)
    return (1 + (code % 8).float() / 8) * torch.exp2((code // 8).float() - 7)


def _round_scale(raw: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    edges = (scales[:-1] + scales[1:]) / 2
    index = torch.bucketize(raw.contiguous(), edges)
    tie = (index < scales.numel() - 1) & (raw == edges[index.clamp_max(scales.numel() - 2)])
    # First positive normal has encoded code 8, so index parity is code parity.
    return scales[index + (tie & (index % 2 == 1)).long()]


def _fp4_indices(values: torch.Tensor) -> torch.Tensor:
    edges = values.new_tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.])
    absolute = values.abs()
    index = torch.bucketize(absolute.contiguous(), edges)
    tie = (index < 7) & (absolute == edges[index.clamp_max(6)])
    return index + (tie & (index % 2 == 1)).long()


@torch.no_grad()
def calibrate_nvfp4_weight(
    weight: torch.Tensor,
    policy: str = "local_hessian",
    *,
    second_moments: torch.Tensor | None = None,
    outer_scale: torch.Tensor | None = None,
    rows_per_chunk: int = 64,
    hessian_shrinkage: float = 0.0,
) -> NVFP4Tensor:
    """Pack a calibrated [out_features,K] weight for explicit offline PTQ.

    Policies are ``max`` (RTNE E4M3 amax/6), ``four_over_six`` (weight-SSE
    selector between encoded amax/6 and amax/4), ``mse`` (exhaustive weight
    SSE), and ``local_hessian`` (exhaustive delta H delta). The two-candidate
    control is not every Four-Over-Six variant. Searches use only production
    normal E4M3 scales [2**-6,448], never research-only subnormals. Ascending
    exhaustive candidates break ties deterministically toward smaller scales.

    A single FP32 outer decode scale is fixed across blocks and policies:
    max(abs(weight))/2688 (one for zero weights), or explicit ``outer_scale``.
    No per-channel factor or epilogue correction is introduced. Input tensors
    are detached and not mutated. Row chunking bounds search scratch storage;
    calibration activations and their split independence remain caller-owned.

    Opt-in ``hessian_shrinkage`` in [0,1] replaces H by
    (1-alpha)*H + alpha*mean(diag(H))*I. Zero preserves the original local
    objective, including its deterministic ties. One selects exactly the MSE
    result. With positive shrinkage, zero-energy H blocks fall back to MSE
    rather than an uninformative all-zero objective. Nonzero shrinkage is
    accepted only for ``local_hessian``. This regularizer is not evidence of
    improved held-out accuracy and never changes the outer decode scale.
    """
    w = _cpu_float(weight, "weight")
    if w.ndim != 2 or min(w.shape) <= 0:
        raise ValueError("weight must be a nonempty [out_features,K] matrix")
    if policy not in ("max", "four_over_six", "mse", "local_hessian"):
        raise ValueError(f"unknown NVFP4 calibration policy: {policy!r}")
    if (not isinstance(hessian_shrinkage, Real) or isinstance(hessian_shrinkage, bool)
            or not math.isfinite(hessian_shrinkage) or not 0 <= hessian_shrinkage <= 1):
        raise ValueError("hessian_shrinkage must be a finite real number in [0,1]")
    if hessian_shrinkage != 0 and policy != "local_hessian":
        raise ValueError("nonzero hessian_shrinkage requires policy='local_hessian'")
    if not isinstance(rows_per_chunk, int) or isinstance(rows_per_chunk, bool) or rows_per_chunk <= 0:
        raise ValueError("rows_per_chunk must be a positive integer")
    rows, k = w.shape
    blocks = (k + 15) // 16
    h = None
    if second_moments is not None:
        h = _cpu_float(second_moments, "second_moments")
        if h.shape != (blocks, 16, 16):
            raise ValueError("second_moments must have shape [ceil(K/16),16,16]")
        if not torch.allclose(h, h.transpose(-1, -2), rtol=1e-5, atol=1e-7):
            raise ValueError("second_moments must be symmetric")
        h = (h.double() + h.double().transpose(-1, -2)) / 2
        eigenvalues = torch.linalg.eigvalsh(h)
        tolerance = eigenvalues.abs().amax(dim=-1).clamp_min(1e-30) * 1e-5
        if bool((eigenvalues[:, 0] < -tolerance).any()):
            raise ValueError("second_moments must be positive semidefinite")
    if policy == "local_hessian" and h is None:
        raise ValueError("local_hessian requires calibration second_moments")
    mse_blocks = None
    if h is not None and policy == "local_hessian" and hessian_shrinkage > 0:
        energy = h.diagonal(dim1=-2, dim2=-1).mean(-1)
        mse_blocks = energy <= 0
        h = ((1 - hessian_shrinkage) * h
             + hessian_shrinkage * energy[:, None, None] * torch.eye(16, dtype=h.dtype))
    amax = w.abs().max()
    outer = torch.where(amax == 0, torch.ones_like(amax),
                        (amax / 2688).clamp_min(torch.finfo(torch.float32).tiny))
    if outer_scale is not None:
        outer = _cpu_float(outer_scale, "outer_scale")
        if outer.numel() != 1 or float(outer) <= 0:
            raise ValueError("outer_scale must be one finite positive FP32-representable value")
        outer = outer.reshape(())
    storage_k = nvfp4_storage_k(k)
    packed = torch.zeros((rows, storage_k // 2), dtype=torch.uint8)
    block_scales = torch.full((rows, storage_k // 16), 2**-6)
    scales = _normal_scales()
    levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    for start in range(0, rows, rows_per_chunk):
        end = min(start + rows_per_chunk, rows)
        values = F.pad(w[start:end], (0, blocks * 16 - k)).reshape(end - start, blocks, 16)
        max_scale = _round_scale((values.abs().amax(-1) / outer) / 6, scales)
        if policy in ("max", "four_over_six"):
            candidates = [max_scale]
            if policy == "four_over_six":
                candidates.append(_round_scale((values.abs().amax(-1) / outer) / 4, scales))
        else:
            candidates = scales
        best_error = torch.full(max_scale.shape, float("inf"), dtype=torch.float64)
        best_scale = max_scale.clone()
        best_indices = torch.zeros_like(values, dtype=torch.long)
        for scale in candidates:
            # FP64 division avoids under/overflow for caller-specified outer scales.
            effective = scale.double() * outer.double()
            indices = _fp4_indices(values.double() / effective[..., None])
            decoded = levels[indices].double() * values.sign().double()
            delta = decoded * effective[..., None] - values.double()
            if policy == "local_hessian" and hessian_shrinkage != 1:
                error = torch.einsum("obi,bij,obj->ob", delta, h, delta)
                if mse_blocks is not None:
                    error = torch.where(mse_blocks[None, :], delta.square().sum(-1), error)
            else:
                # alpha=1 is proportional to SSE for nonzero blocks; use SSE
                # directly to preserve exact MSE tie-breaking and zero fallback.
                error = delta.square().sum(-1)
            improve = error < best_error
            best_error = torch.minimum(error, best_error)
            best_scale = torch.where(improve, scale, best_scale)
            best_indices = torch.where(improve[..., None], indices, best_indices)
        # First logical element occupies the low nibble, matching native PTX.
        codes = (best_indices | ((values < 0).long() << 3)).to(torch.uint8)
        codes = F.pad(codes.reshape(end - start, -1), (0, storage_k - blocks * 16))
        packed[start:end] = codes[:, 0::2] | (codes[:, 1::2] << 4)
        block_scales[start:end, :blocks] = best_scale
    return make_nvfp4_tensor(packed.to(weight.device),
                             block_scales.to(device=weight.device, dtype=torch.float8_e4m3fn),
                             outer.to(weight.device), tuple(weight.shape))


__all__ = ["NVFP4SecondMomentCollector", "local_second_moments", "calibrate_nvfp4_weight"]
