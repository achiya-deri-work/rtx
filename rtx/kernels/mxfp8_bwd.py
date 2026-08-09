"""Legality and schedule model for MXFP8 linear backward on SM120.

Source orientation is represented exclusively by CuTe tensor layouts. A
logical transpose has shape ``[rows, K]`` and stride ``(1, rows)`` over the
original row-major allocation. No BF16 transpose kernel, SMEM transpose, or
temporary BF16 orientation buffer exists in the backward implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

from .mxfp8 import DEFAULT_MXFP8_FWD_CONFIG, MXFP8FwdConfig, MXFP8Problem
from ..configs import MXFP8GemmConfig, MXFP8QuantConfig


@dataclass(frozen=True, slots=True)
class MXFP8BwdMatmulConfig:
    """Logical source layouts, quantizers, GEMM, and reduction for one gradient."""

    a_orientation: str = "row"
    b_orientation: str = "transpose"
    # ``fused`` runs the same BF16->E4M3/E8M0->MMA pipeline as dynamic
    # forward. ``decomposed`` remains as a measured reference and fallback.
    backend: str = "fused"
    fused: MXFP8FwdConfig = DEFAULT_MXFP8_FWD_CONFIG
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
    reduction_threads: int = 256
    reduction_vector: int = 4
    reduction_waves: int = 1
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
        if self.backend not in ("fused", "decomposed"):
            return "backend must be fused or decomposed"
        if self.backend == "fused":
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
            if self.reduction_threads not in (64, 128, 256, 512, 1024):
                return "reduction_threads must be 64, 128, 256, 512, or 1024"
            if self.reduction_vector not in (1, 2, 4, 8):
                return "reduction_vector must be 1, 2, 4, or 8"
            if self.reduction_waves not in (1, 2, 3, 4, 6, 8):
                return "reduction_waves must be 1, 2, 3, 4, 6, or 8"
            if self.reduction == "full_fp32" and (
                self.split_reduction != 1
                or self.reduction_tile != 0
                or self.workspace_epilogue != "none"
            ):
                return "full reduction has no split/workspace coordinates"
            if self.reduction in ("split_fp32_workspace", "split_fp32_atomic"):
                if (
                    self.reduction == "split_fp32_workspace"
                    and self.workspace_epilogue == "none"
                ):
                    return "split workspace reduction requires an epilogue"
                if (
                    self.reduction == "split_fp32_atomic"
                    and self.workspace_epilogue != "none"
                ):
                    return "atomic split reduction does not use a workspace epilogue"
                if not (
                    (self.split_reduction - 1) * self.reduction_tile < problem.k
                    <= self.split_reduction * self.reduction_tile
                ):
                    return "split count/tile must cover K without an empty partition"
                if self.reduction_tile % self.fused.bf16_tile_k:
                    return "reduction tile must be divisible by the BF16 transport tile"
                if self.fused.epilogue != "direct":
                    return "split partials require fused.epilogue='direct'"
                if self.fused.persistent:
                    return "split partial GEMM cannot also be persistent"
            reason = self.fused.oriented_implementation_rejection(
                problem, self.a_orientation, self.b_orientation
            )
            if reason is not None:
                return f"fused kernel: {reason}"
            if self.reduction == "cluster_fp32":
                return f"fused {self.reduction!r} reduction is not implemented yet"
            if self.tile_scheduler != "static":
                return (
                    "use fused.persistent/reuse for fused backward scheduling; "
                    "legacy decomposed tile_scheduler coordinates are inactive"
                )
            return None
        if self.quant_launches not in ("dual", "separate"):
            return "quant_launches must be dual or separate"
        quant_b = self.resolved_quant_b()
        reason = (
            self.quant_a.transposed_rejection(problem.m, problem.k)
            if self.a_orientation == "transpose"
            else self.quant_a.rejection(problem.m, problem.k)
        )
        if reason is not None:
            return f"A quantizer: {reason}"
        reason = (
            quant_b.transposed_rejection(problem.n, problem.k)
            if self.b_orientation == "transpose"
            else quant_b.rejection(problem.n, problem.k)
        )
        if reason is not None:
            return f"B quantizer: {reason}"
        if self.a_orientation == "transpose" and problem.m % 32:
            return "logical-transpose A rows must be divisible by 32"
        if self.b_orientation == "transpose" and problem.n % 32:
            return "logical-transpose B rows must be divisible by 32"
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
        if self.reduction_threads not in (64, 128, 256, 512, 1024):
            return "reduction_threads must be 64, 128, 256, 512, or 1024"
        if self.reduction_vector not in (1, 2, 4, 8):
            return "reduction_vector must be 1, 2, 4, or 8"
        if self.reduction_waves not in (1, 2, 3, 4, 6, 8):
            return "reduction_waves must be 1, 2, 3, 4, 6, or 8"
        if self.reduction == "full_fp32" and (
            self.split_reduction != 1
            or self.reduction_tile != 0
            or self.workspace_epilogue != "none"
        ):
            return "full reduction has no split/workspace coordinates"
        if self.reduction != "full_fp32" and self.split_reduction == 1:
            return "split/cluster reductions require more than one partition"
        if self.reduction in ("split_fp32_workspace", "split_fp32_atomic"):
            if (
                self.reduction == "split_fp32_workspace"
                and self.workspace_epilogue == "none"
            ):
                return "split workspace reduction requires an epilogue"
            if (
                self.reduction == "split_fp32_atomic"
                and self.workspace_epilogue != "none"
            ):
                return "atomic split reduction does not use a workspace epilogue"
            if not (
                (self.split_reduction - 1) * self.reduction_tile < problem.k
                <= self.split_reduction * self.reduction_tile
            ):
                return "split count/tile must cover K without an empty partition"
            if self.reduction_tile % self.gemm.tile_k:
                return "reduction tile must be divisible by GEMM tile_k"
            if self.gemm.epilogue != "direct" or self.gemm.store_vec != 1:
                return "decomposed split partials require gemm direct FP32 output"
            if self.gemm.tiles_per_cta != 1:
                return "decomposed split-K cannot also use multi-output persistence"
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
        if self.backend == "fused":
            return None
        if self.reduction == "cluster_fp32":
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
        quant_vec=4,
        load_bits=32,
        quant_store_bits=32,
        transposed_tile_rows=128,
    )
    return replace(base, quant_b=transposed)


def _default_dw_config() -> MXFP8BwdMatmulConfig:
    base = MXFP8BwdMatmulConfig(
        a_orientation="transpose", b_orientation="transpose"
    )
    transposed = replace(
        base.quant_a,
        quant_vec=4,
        load_bits=32,
        quant_store_bits=32,
        transposed_tile_rows=128,
    )
    return replace(base, quant_a=transposed)


@dataclass(frozen=True, slots=True)
class MXFP8BwdConfig:
    """Full backward for ``Y[M,N] = X[M,K] @ W[N,K].T``."""

    dx: MXFP8BwdMatmulConfig = field(default_factory=_default_dx_config)
    dw: MXFP8BwdMatmulConfig = field(default_factory=_default_dw_config)
    execution_order: str = "dx_first"
    stream_schedule: str = "single"
    quant_schedule: str = "per_matmul"

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
        if self.quant_schedule not in ("per_matmul", "quad"):
            return "quant_schedule must be per_matmul or quad"
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
            if self.stream_schedule != "single":
                return "interleaved execution requires the single-stream schedule"
            if self.dx.backend != "decomposed" or self.dw.backend != "decomposed":
                return "interleaved execution requires two decomposed matmuls"
        if self.stream_schedule == "graph":
            return f"stream schedule {self.stream_schedule!r} is not implemented yet"
        if self.quant_schedule == "quad":
            if self.dx.backend != "decomposed" or self.dw.backend != "decomposed":
                return "quad quantization requires two decomposed matmuls"
            if self.dx.quant_launches != "dual" or self.dw.quant_launches != "dual":
                return "quad quantization requires dual per-matmul quantizer configs"
            transposed = self.dx.resolved_quant_b()
            if self.dw.quant_a != transposed or self.dw.resolved_quant_b() != transposed:
                return "quad quantization requires one shared transposed schedule"
            if self.dx.quant_a.num_warps != transposed.num_warps:
                return "quad quantization requires one CTA warp count"
        reason = self.dx.implementation_rejection(
            MXFP8Problem(forward.m, forward.k, forward.n)
        )
        if reason is not None:
            return f"dX: {reason}"
        reason = self.dw.implementation_rejection(
            MXFP8Problem(forward.n, forward.k, forward.m)
        )
        return None if reason is None else f"dW: {reason}"


DEFAULT_FUSED_MXFP8_BWD_CONFIG = MXFP8BwdConfig()


DEFAULT_SEPARATE_DECOMPOSED_MXFP8_BWD_CONFIG = replace(
    DEFAULT_FUSED_MXFP8_BWD_CONFIG,
    dx=replace(DEFAULT_FUSED_MXFP8_BWD_CONFIG.dx, backend="decomposed"),
    dw=replace(DEFAULT_FUSED_MXFP8_BWD_CONFIG.dw, backend="decomposed"),
)

DEFAULT_DUAL_DECOMPOSED_MXFP8_BWD_CONFIG = replace(
    DEFAULT_SEPARATE_DECOMPOSED_MXFP8_BWD_CONFIG,
    dx=replace(
        DEFAULT_SEPARATE_DECOMPOSED_MXFP8_BWD_CONFIG.dx,
        quant_launches="dual",
    ),
    dw=replace(
        DEFAULT_SEPARATE_DECOMPOSED_MXFP8_BWD_CONFIG.dw,
        quant_launches="dual",
    ),
)

DEFAULT_DECOMPOSED_MXFP8_BWD_CONFIG = replace(
    DEFAULT_DUAL_DECOMPOSED_MXFP8_BWD_CONFIG,
    quant_schedule="quad",
    stream_schedule="dual_stream",
)

# The fused families are searchable, but until their CTAs share quantized
# operands the measured quantize-once implementation is the safe runtime seed.
DEFAULT_MXFP8_BWD_CONFIG = DEFAULT_DECOMPOSED_MXFP8_BWD_CONFIG


__all__ = [
    "DEFAULT_MXFP8_BWD_CONFIG",
    "DEFAULT_DECOMPOSED_MXFP8_BWD_CONFIG",
    "DEFAULT_DUAL_DECOMPOSED_MXFP8_BWD_CONFIG",
    "DEFAULT_FUSED_MXFP8_BWD_CONFIG",
    "DEFAULT_SEPARATE_DECOMPOSED_MXFP8_BWD_CONFIG",
    "MXFP8BwdConfig",
    "MXFP8BwdMatmulConfig",
]
