"""Legality and schedule model for MXFP8 linear backward on SM120.

Source orientation is represented exclusively by CuTe tensor layouts. A
logical transpose has shape ``[rows, K]`` and stride ``(1, rows)`` over the
original row-major allocation. No BF16 transpose kernel, SMEM transpose, or
temporary BF16 orientation buffer exists in the backward implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .mxfp8 import MXFP8Problem
from ..configs import MXFP8GemmConfig, MXFP8QuantConfig


@dataclass(frozen=True, slots=True)
class MXFP8BwdMatmulConfig:
    """Logical source layouts, quantizers, GEMM, and reduction for one gradient."""

    a_orientation: str = "row"
    b_orientation: str = "transpose"
    quant_launches: str = "separate"
    quant_a: MXFP8QuantConfig = MXFP8QuantConfig(
        load_bits=32,
        maxrregcount=96,
        persistent_waves=6,
        quant_amax="fp32",
        quant_math="fp32",
        scale_layout="mma128",
    )
    quant_b: MXFP8QuantConfig | None = None
    gemm: MXFP8GemmConfig = MXFP8GemmConfig(
        atom_layout_m=4,
        b_swizzle="128b",
        consumer_registers=232,
        producer_registers=64,
        scale_role="tma",
        scale_layout="mma128",
    )
    reduction: str = "full_fp32"
    split_reduction: int = 1
    reduction_tile: int = 0
    workspace_epilogue: str = "none"
    tile_scheduler: str = "static"
    persistent_waves: int = 1
    tiles_per_cta: int = 1
    reuse_operand: str = "none"
    tile_locality: str = "raster"

    def resolved_quant_b(self) -> MXFP8QuantConfig:
        return self.quant_a if self.quant_b is None else self.quant_b

    def normalized(self) -> "MXFP8BwdMatmulConfig":
        quant_b = self.resolved_quant_b()
        return replace(self, quant_b=None if quant_b == self.quant_a else quant_b)

    def rejection(self, problem: MXFP8Problem) -> str | None:
        if self.a_orientation not in ("row", "transpose"):
            return "A orientation must be row or transpose"
        if self.b_orientation not in ("row", "transpose"):
            return "B orientation must be row or transpose"
        if self.quant_launches not in ("dual", "separate"):
            return "quant_launches must be dual or separate"
        quant_b = self.resolved_quant_b()
        reason = self.quant_a.rejection(problem.m, problem.k)
        if reason is not None:
            return f"A quantizer: {reason}"
        reason = quant_b.rejection(problem.n, problem.k)
        if reason is not None:
            return f"B quantizer: {reason}"
        if self.a_orientation == "transpose" and problem.m % 32:
            return "logical-transpose A rows must be divisible by 32"
        if self.b_orientation == "transpose" and problem.n % 32:
            return "logical-transpose B rows must be divisible by 32"
        if self.a_orientation == "transpose" and self.quant_a.native_scale_store == "packed":
            return "logical-transpose A currently requires scalar native scale stores"
        if self.b_orientation == "transpose" and quant_b.native_scale_store == "packed":
            return "logical-transpose B currently requires scalar native scale stores"
        expected_layouts = {
            "row_major": ("row_major", "row_major"),
            "mma128": ("mma128", "mma128"),
            "mma64x128": ("mma64", "mma128"),
        }
        expected = expected_layouts.get(self.gemm.scale_layout)
        actual = (self.quant_a.scale_layout, quant_b.scale_layout)
        if expected is None or actual != expected:
            return (
                f"GEMM scale layout {self.gemm.scale_layout!r} requires "
                f"quantizer layouts {expected}, got {actual}"
            )
        if self.quant_launches == "dual":
            if self.a_orientation == self.b_orientation and quant_b != self.quant_a:
                return "same-orientation dual quantization requires identical schedules"
            if self.a_orientation != self.b_orientation and (
                quant_b.num_warps != self.quant_a.num_warps
            ):
                return "mixed-orientation dual quantization requires one warp count"
        if self.reduction not in (
            "full_fp32",
            "split_fp32_workspace",
            "split_fp32_atomic",
            "cluster_fp32",
        ):
            return "unknown reduction strategy"
        if self.split_reduction not in (1, 2, 4, 8, 16, 32):
            return "split_reduction must be 1, 2, 4, 8, 16, or 32"
        if self.reduction_tile not in (0, 128, 256, 512, 1024, 2048, 4096):
            return "unsupported reduction tile"
        if self.workspace_epilogue not in (
            "none",
            "serial",
            "tree",
            "persistent_tree",
        ):
            return "unknown workspace epilogue"
        if self.reduction == "full_fp32" and (
            self.split_reduction != 1
            or self.reduction_tile != 0
            or self.workspace_epilogue != "none"
        ):
            return "full reduction has no split/workspace coordinates"
        if self.reduction != "full_fp32" and self.split_reduction == 1:
            return "split/cluster reductions require more than one partition"
        if self.tile_scheduler not in ("static", "persistent"):
            return "tile_scheduler must be static or persistent"
        if self.persistent_waves not in (1, 2, 3, 4, 6, 8):
            return "persistent_waves must be 1, 2, 3, 4, 6, or 8"
        if self.tiles_per_cta not in (1, 2, 3, 4, 8):
            return "tiles_per_cta must be 1, 2, 3, 4, or 8"
        if self.reuse_operand not in ("none", "a", "b", "both"):
            return "reuse_operand must be none, a, b, or both"
        if self.tile_locality not in (
            "raster",
            "same_a",
            "same_b",
            "serpentine",
            "l2_wave",
        ):
            return "unknown tile locality policy"
        if self.tile_scheduler == "static" and (
            self.persistent_waves != 1
            or self.tiles_per_cta != 1
            or self.reuse_operand != "none"
            or self.tile_locality != "raster"
        ):
            return "static tile scheduling has no persistence/reuse coordinates"
        return self.gemm.rejection(problem)

    def implementation_rejection(self, problem: MXFP8Problem) -> str | None:
        reason = self.rejection(problem)
        if reason is not None:
            return reason
        if self.reduction != "full_fp32":
            return f"reduction family {self.reduction!r} is not implemented yet"
        if self.tile_scheduler != "static":
            return "persistent/multi-output backward GEMM is not implemented yet"
        return None


def _default_dx_config() -> MXFP8BwdMatmulConfig:
    base = MXFP8BwdMatmulConfig(
        a_orientation="row", b_orientation="transpose"
    )
    transposed = replace(
        base.quant_a,
        quant_vec=2,
        load_bits=32,
        transposed_tile_rows=64,
    )
    return replace(base, quant_b=transposed)


def _default_dw_config() -> MXFP8BwdMatmulConfig:
    base = MXFP8BwdMatmulConfig(
        a_orientation="transpose", b_orientation="transpose"
    )
    transposed = replace(
        base.quant_a,
        quant_vec=2,
        load_bits=32,
        transposed_tile_rows=64,
    )
    return replace(base, quant_a=transposed)


@dataclass(frozen=True, slots=True)
class MXFP8BwdConfig:
    """Full backward for ``Y[M,N] = X[M,K] @ W[N,K].T``."""

    dx: MXFP8BwdMatmulConfig = field(default_factory=_default_dx_config)
    dw: MXFP8BwdMatmulConfig = field(default_factory=_default_dw_config)
    execution_order: str = "dx_first"
    stream_schedule: str = "single"

    def normalized(self) -> "MXFP8BwdConfig":
        return replace(self, dx=self.dx.normalized(), dw=self.dw.normalized())

    def rejection(self, forward: MXFP8Problem) -> str | None:
        try:
            forward.validate()
        except ValueError as exc:
            return str(exc)
        if forward.m % 32 or forward.n % 32:
            return "M and N become MXFP8 reduction axes and must be divisible by 32"
        if self.execution_order not in ("dx_first", "dw_first", "interleaved"):
            return "execution_order must be dx_first, dw_first, or interleaved"
        if self.stream_schedule not in ("single", "dual_stream", "graph"):
            return "stream_schedule must be single, dual_stream, or graph"
        if (self.dx.a_orientation, self.dx.b_orientation) != ("row", "transpose"):
            return "dX requires logical layouts A=row and B=transpose"
        if (self.dw.a_orientation, self.dw.b_orientation) != (
            "transpose",
            "transpose",
        ):
            return "dW requires logical layouts A=transpose and B=transpose"
        reason = self.dx.rejection(MXFP8Problem(forward.m, forward.k, forward.n))
        if reason is not None:
            return f"dX: {reason}"
        reason = self.dw.rejection(MXFP8Problem(forward.n, forward.k, forward.m))
        if reason is not None:
            return f"dW: {reason}"
        return None

    def implementation_rejection(self, forward: MXFP8Problem) -> str | None:
        reason = self.rejection(forward)
        if reason is not None:
            return reason
        if self.execution_order == "interleaved":
            return "interleaved dX/dW execution is not implemented yet"
        if self.stream_schedule != "single":
            return f"stream schedule {self.stream_schedule!r} is not implemented yet"
        reason = self.dx.implementation_rejection(
            MXFP8Problem(forward.m, forward.k, forward.n)
        )
        if reason is not None:
            return f"dX: {reason}"
        reason = self.dw.implementation_rejection(
            MXFP8Problem(forward.n, forward.k, forward.m)
        )
        return None if reason is None else f"dW: {reason}"


DEFAULT_MXFP8_BWD_CONFIG = MXFP8BwdConfig()


__all__ = [
    "DEFAULT_MXFP8_BWD_CONFIG",
    "MXFP8BwdConfig",
    "MXFP8BwdMatmulConfig",
]
