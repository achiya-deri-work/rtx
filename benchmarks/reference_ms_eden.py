"""Independent Torch oracle for post-hoc MS-EDEN scale preparation.

Research only: inputs are already rotated FP32 values. This is not a kernel,
not the complete gradient estimator, and not a claim of vector unbiasedness.
Motivation: Quartet II section 7, https://arxiv.org/html/2601.22813v2#S7.
The inspected upstream RHT implementation uses E8M4 truncation rather than
the paper's E8M3 description; mantissa_bits makes this distinction explicit.
"""
from dataclasses import dataclass
import torch


def fp4_rne(values):
    """E2M1 saturation and round-to-nearest-even, independently via a grid."""
    grid = values.new_tensor([0., .5, 1., 1.5, 2., 3., 4., 6.])
    magnitude = values.abs().clamp_max(6.)
    upper = torch.searchsorted(grid, magnitude.contiguous()).clamp_max(7)
    lower = (upper-1).clamp_min(0)
    dl, du = magnitude-grid[lower], grid[upper]-magnitude
    use_upper = (du < dl) | ((du == dl) & (upper.remainder(2) == 0))
    return grid[torch.where(use_upper, upper, lower)] * values.sign()


@dataclass
class ExtendedBlocks:
    values: torch.Tensor
    corrected_scales: torch.Tensor
    fp32_corrected_scales: torch.Tensor


def prepare_extended(rotated, *, mantissa_bits=4):
    if rotated.dtype != torch.float32 or rotated.shape[-1] % 16:
        raise ValueError("expected FP32 values with full 16-element blocks")
    if mantissa_bits not in (3, 4):
        raise ValueError("mantissa_bits must be 3 or 4")
    if not torch.isfinite(rotated).all():
        raise ValueError("nonfinite values are outside this oracle's contract")
    blocks = rotated.reshape(*rotated.shape[:-1], -1, 16)
    scale = (blocks.abs().amax(-1) / 6).contiguous()
    # Explicit RTZ of the FP32 significand; FTZ matches the inspected producer.
    bits = scale.view(torch.int32)
    truncated = (bits & ~((1 << (23-mantissa_bits))-1)).view(torch.float32)
    truncated = torch.where(truncated >= torch.finfo(torch.float32).tiny, truncated, 0.)
    normalized = blocks / torch.where(truncated > 0, truncated, 1.).unsqueeze(-1)
    normalized = torch.where((truncated > 0).unsqueeze(-1), normalized, 0.)
    quantized = fp4_rne(normalized)
    xx = normalized.square().sum(-1)
    xy = (normalized*quantized).sum(-1)
    correction = torch.where(xy > 0, xx/xy.clamp_min(torch.finfo(torch.float32).tiny), 1.)
    corrected = truncated*correction
    stored = corrected.to(torch.bfloat16)
    if not torch.isfinite(stored).all():
        raise ValueError("corrected scales overflow BF16 temporary storage")
    return ExtendedBlocks(quantized, stored, corrected)


def scale_brackets(values):
    """Adjacent finite nonnegative E4M3 values; no nearest-rounding shortcut."""
    grid = torch.arange(127, device=values.device, dtype=torch.uint8).view(torch.float8_e4m3fn).float()
    if not torch.isfinite(values).all() or (values < 0).any() or (values > 448).any():
        raise ValueError("FP8 scale outside finite nonnegative range")
    hi = torch.searchsorted(grid, values.contiguous()).clamp_max(126)
    lo = (hi-1).clamp_min(0)
    low, high = grid[lo], grid[hi]
    probability = torch.where(high > low, (values-low)/(high-low).clamp_min(2**-9), 0.)
    return low, high, probability


def finalize_scales(corrected_scales, uniforms):
    """Scale-only SR with caller-owned randomness and an FP32 outer factor.

    Zero tensors use outer=1 and zero FP8 scales. The maximum is taken from
    stored BF16 scales, guaranteeing no clipping in this reference policy.
    """
    if corrected_scales.dtype != torch.bfloat16 or uniforms.shape != corrected_scales.shape:
        raise ValueError("expected BF16 scales and identically shaped uniforms")
    if not torch.isfinite(uniforms).all() or (uniforms < 0).any() or (uniforms >= 1).any():
        raise ValueError("uniforms must lie in [0,1)")
    scales = corrected_scales.float()
    maximum = scales.amax()
    outer = torch.where(maximum > 0, maximum*(1./448.), 1.)
    low, high, probability = scale_brackets((scales/outer).clamp_max(448.))
    rounded = torch.where(uniforms < probability, high, low)
    return rounded.to(torch.float8_e4m3fn), outer
