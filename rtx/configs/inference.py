"""State-specific MXFP8 inference schedules.

These contracts intentionally omit work which happens outside the timed
invocation. An AOT-weight candidate cannot tune the one-time W quantizer, and
a fully packed candidate cannot tune either quantizer.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .mxfp8 import MXFP8GemmConfig, MXFP8QuantConfig
from ..kernels.mxfp8 import MXFP8Problem


def gemm_operand_scale_layouts(gemm_layout: str) -> tuple[str, str]:
    try:
        return {
            "row_major": ("row_major", "row_major"),
            "mma128": ("mma128", "mma128"),
            "mma64x128": ("mma64", "mma128"),
        }[gemm_layout]
    except KeyError as exc:
        raise ValueError(f"unknown GEMM scale layout {gemm_layout!r}") from exc


def canonical_packing_quant(scale_layout: str) -> MXFP8QuantConfig:
    """Stable untimed producer used to materialize an inference operand."""

    return replace(MXFP8QuantConfig(), scale_layout=scale_layout)


def _l2_rejection(value: int | None) -> str | None:
    if value not in (None, 0, 32, 64, 128):
        return "L2 fetch granularity must be None, 0, 32, 64, or 128"
    return None


@dataclass(frozen=True, slots=True)
class MXFP8WeightPrequantConfig:
    """Per-call schedule for BF16 X and an AOT-packed W."""

    quant_x: MXFP8QuantConfig = MXFP8QuantConfig()
    gemm: MXFP8GemmConfig = MXFP8GemmConfig(epilogue="direct", store_vec=1)
    l2_fetch_granularity: int | None = None

    @property
    def operand_scale_layouts(self) -> tuple[str, str]:
        return gemm_operand_scale_layouts(self.gemm.scale_layout)

    def weight_packing_quant(self) -> MXFP8QuantConfig:
        return canonical_packing_quant(self.operand_scale_layouts[1])

    def rejection(self, problem: MXFP8Problem) -> str | None:
        try:
            x_layout, _weight_layout = self.operand_scale_layouts
        except ValueError as exc:
            return str(exc)
        if self.quant_x.scale_layout != x_layout:
            return (
                f"GEMM layout {self.gemm.scale_layout} requires X scales "
                f"in {x_layout}, got {self.quant_x.scale_layout}"
            )
        reason = self.quant_x.rejection(problem.m, problem.k)
        if reason is not None:
            return f"activation quantizer: {reason}"
        reason = self.weight_packing_quant().rejection(problem.n, problem.k)
        if reason is not None:
            return f"AOT weight packing: {reason}"
        return _l2_rejection(self.l2_fetch_granularity) or self.gemm.rejection(problem)


@dataclass(frozen=True, slots=True)
class MXFP8FullyPrequantConfig:
    """Per-call schedule when both operands are already packed."""

    gemm: MXFP8GemmConfig = MXFP8GemmConfig(epilogue="direct", store_vec=1)
    l2_fetch_granularity: int | None = None

    @property
    def operand_scale_layouts(self) -> tuple[str, str]:
        return gemm_operand_scale_layouts(self.gemm.scale_layout)

    def activation_packing_quant(self) -> MXFP8QuantConfig:
        return canonical_packing_quant(self.operand_scale_layouts[0])

    def weight_packing_quant(self) -> MXFP8QuantConfig:
        return canonical_packing_quant(self.operand_scale_layouts[1])

    def rejection(self, problem: MXFP8Problem) -> str | None:
        try:
            quant_x = self.activation_packing_quant()
            quant_w = self.weight_packing_quant()
        except ValueError as exc:
            return str(exc)
        reason = quant_x.rejection(problem.m, problem.k)
        if reason is not None:
            return f"AOT activation packing: {reason}"
        reason = quant_w.rejection(problem.n, problem.k)
        if reason is not None:
            return f"AOT weight packing: {reason}"
        return _l2_rejection(self.l2_fetch_granularity) or self.gemm.rejection(problem)


__all__ = [
    "MXFP8FullyPrequantConfig",
    "MXFP8WeightPrequantConfig",
    "canonical_packing_quant",
    "gemm_operand_scale_layouts",
]
