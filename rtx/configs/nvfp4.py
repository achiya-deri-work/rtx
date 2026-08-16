"""Architecture-independent configuration for RTX NVFP4 kernels."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .mxfp8 import MXFP8GemmConfig


NVFP4_SF_VEC_SIZE = 16


@dataclass(frozen=True, slots=True)
class NVFP4QuantConfig:
    """One independently compilable BF16-to-NVFP4 quantizer schedule."""

    values_per_lane: int = 2
    load_bits: int = 32
    reduction: Literal["shuffle", "redux"] = "redux"
    quant_math: Literal["fp32"] = "fp32"
    num_warps: int = 8
    persistent_waves: int = 4
    maxrregcount: int = 128
    scale_reciprocal: Literal["direct", "e4m3_lut", "rcp_approx"] = "direct"
    scale_compute: Literal["redundant", "leader_broadcast"] = "redundant"
    scale_layout: Literal["row_major", "mma128"] = "row_major"

    @property
    def threads_per_scale(self) -> int:
        return NVFP4_SF_VEC_SIZE // self.values_per_lane

    @property
    def blocks_per_warp(self) -> int:
        return 32 // self.threads_per_scale

    def rejection(self, rows: int, k: int) -> str | None:
        if rows <= 0 or k <= 0:
            return "rows and K must be positive"
        if self.values_per_lane not in (2, 4, 8, 16):
            return (
                "values_per_lane must be 2, 4, 8, or 16 so each lane owns "
                "whole packed FP4 bytes"
            )
        if self.load_bits not in (16, 32, 64, 128):
            return "load_bits must be one of 16, 32, 64, 128"
        if self.load_bits > self.values_per_lane * 16:
            return "load width exceeds the BF16 values owned by one lane"
        if (self.values_per_lane * 16) % self.load_bits:
            return "values_per_lane must contain whole vector loads"
        if self.reduction not in ("shuffle", "redux"):
            return "reduction must be shuffle or redux"
        if self.quant_math != "fp32":
            return "only the implemented FP32 NVFP4 quantization path is legal"
        if self.scale_reciprocal not in ("direct", "e4m3_lut", "rcp_approx"):
            return "scale_reciprocal must be direct, e4m3_lut, or rcp_approx"
        if self.scale_compute not in ("redundant", "leader_broadcast"):
            return "scale_compute must be redundant or leader_broadcast"
        if self.scale_layout not in ("row_major", "mma128"):
            return "scale_layout must be row_major or mma128"
        if self.scale_layout == "mma128" and (rows % 128 or k % 128):
            return "mma128 NVFP4 scales require rows and K divisible by 128"
        if self.num_warps not in (4, 8, 16):
            return "num_warps must be one of 4, 8, 16"
        if self.persistent_waves not in (1, 2, 3, 4, 6, 8):
            return "persistent_waves must be one of 1, 2, 3, 4, 6, 8"
        return None


@dataclass(frozen=True, slots=True)
class NVFP4Problem:
    """A flattened ``[M,K] @ [N,K].T -> [M,N]`` NVFP4 problem."""

    m: int
    n: int
    k: int

    def validate(self) -> None:
        if min(self.m, self.n, self.k) <= 0:
            raise ValueError(f"M, N and K must be positive, got {self}")

    @property
    def storage_k(self) -> int:
        """Smallest NVFP4 block-aligned packed reduction extent."""

        return ((self.k + NVFP4_SF_VEC_SIZE - 1) // NVFP4_SF_VEC_SIZE) * (
            NVFP4_SF_VEC_SIZE
        )


@dataclass(frozen=True, slots=True)
class NVFP4ScaleConfig:
    """Numerical policy for NVFP4 outer FP32 scaling.

    Kernel schedules live in :class:`NVFP4DynamicConfig`.  Keeping the scale
    policy separate makes every public field effective for every retained
    NVFP4 implementation and avoids exposing controls from the retired fused
    forward experiment.
    """

    tensor_scale_mode: Literal["power2", "exact"] = "power2"
    amax_history_len: Literal[1, 4, 16, 64] = 16
    amax_history_algo: Literal["most_recent", "window_max"] = "window_max"

    def __post_init__(self) -> None:
        if self.tensor_scale_mode not in ("power2", "exact"):
            raise ValueError("tensor_scale_mode must be power2 or exact")
        if self.amax_history_len not in (1, 4, 16, 64):
            raise ValueError("amax_history_len must be 1, 4, 16, or 64")
        if self.amax_history_algo not in ("most_recent", "window_max"):
            raise ValueError(
                "amax_history_algo must be most_recent or window_max"
            )


@dataclass(frozen=True, slots=True)
class NVFP4GemmConfig(MXFP8GemmConfig):
    """Prequantized SM120 NVFP4 GEMM schedule.

    The transport and epilogue controls intentionally match MXFP8 so the
    portable autotuner can compare identical schedule coordinates. NVFP4
    differs in its packed E2M1 operands and twice-as-dense E4M3 scale vectors.
    """

    stages: int = 2
    scale_load_vec: int = 4
    scale_layout: str = "row_major"
    regional_scale_epilogue: Literal[
        "direct", "expanded_factors", "factorized", "product"
    ] = "direct"

    @property
    def native_operand_bits(self) -> int:
        return 4

    @property
    def scale_vector_size(self) -> int:
        return NVFP4_SF_VEC_SIZE

    @property
    def native_mma_k(self) -> int:
        return 64

    def rejection(self, problem: NVFP4Problem) -> str | None:
        try:
            problem.validate()
        except ValueError as exc:
            return str(exc)
        if self.tile_k % 64:
            return "NVFP4 tile K must be divisible by 64"
        if self.tile_k == 64 and (
            self.a_swizzle not in ("none", "32b")
            or self.b_swizzle not in ("none", "32b")
        ):
            return (
                "NVFP4 K=64 GEMM requires none or 32-byte SMEM swizzles"
            )
        if self.scale_layout not in ("row_major", "mma128"):
            return "NVFP4 GEMM scales must use row_major or mma128 layout"
        if self.regional_scale_epilogue not in (
            "direct",
            "expanded_factors",
            "factorized",
            "product",
        ):
            return (
                "NVFP4 regional_scale_epilogue must be direct, "
                "expanded_factors, factorized, or product"
            )
        if (self.tile_k // NVFP4_SF_VEC_SIZE) % self.scale_load_vec:
            return "scale_load_vec must divide the NVFP4 K-tile scale count"
        # The common checker uses native_operand_bits and scale_vector_size,
        # so its capacity model covers packed E2M1 and twice-dense E4M3 scales.
        # Explicit super is required for the class object returned by
        # dataclass(slots=True) on Python 3.12/PyTorch 2.12 environments.
        return super(NVFP4GemmConfig, self).rejection(problem)  # type: ignore[arg-type]


def _inference_l2_rejection(value: int | None) -> str | None:
    if value not in (None, 0, 32, 64, 128):
        return "L2 fetch granularity must be None, 0, 32, 64, or 128"
    return None


@dataclass(frozen=True, slots=True)
class NVFP4WeightPrequantConfig:
    """Per-call schedule for BF16 X and a TorchAO-packed NVFP4 W."""

    quant_x: NVFP4QuantConfig = NVFP4QuantConfig()
    gemm: NVFP4GemmConfig = NVFP4GemmConfig(
        epilogue="direct", store_vec=1
    )
    l2_fetch_granularity: int | None = None

    def rejection(self, problem: NVFP4Problem) -> str | None:
        reason = self.quant_x.rejection(problem.m, problem.k)
        if reason is not None:
            return f"activation quantizer: {reason}"
        reason = NVFP4QuantConfig().rejection(problem.n, problem.k)
        if reason is not None:
            return f"AOT weight packing: {reason}"
        return (
            _inference_l2_rejection(self.l2_fetch_granularity)
            or self.gemm.rejection(problem)
        )


@dataclass(frozen=True, slots=True)
class NVFP4FullyPrequantConfig:
    """Per-call schedule when X and W are both TorchAO-packed NVFP4."""

    gemm: NVFP4GemmConfig = NVFP4GemmConfig(
        epilogue="direct", store_vec=1
    )
    l2_fetch_granularity: int | None = None

    def rejection(self, problem: NVFP4Problem) -> str | None:
        for label, rows in (("activation", problem.m), ("weight", problem.n)):
            reason = NVFP4QuantConfig().rejection(rows, problem.k)
            if reason is not None:
                return f"AOT {label} packing: {reason}"
        return (
            _inference_l2_rejection(self.l2_fetch_granularity)
            or self.gemm.rejection(problem)
        )


@dataclass(frozen=True, slots=True)
class NVFP4DynamicConfig:
    """Materialized per-call schedule for two dynamic BF16 operands."""

    quant: NVFP4QuantConfig = NVFP4QuantConfig()
    gemm: NVFP4GemmConfig = NVFP4GemmConfig(
        epilogue="direct", store_vec=1, a_swizzle="64b", b_swizzle="64b"
    )
    quant_launches: str = "dual"
    l2_fetch_granularity: int | None = None
    # A positive pair selects current, JIT-computed FP32 outer scales for
    # bounded row regions.  Zero on both operands selects the ordinary
    # tensor-scale/block-only materialized paths.  Keeping the policy in the
    # compound config makes region geometry a measured autotuning coordinate.
    x_scale_region_rows: int = 0
    weight_scale_region_rows: int = 0
    tensor_scale_mode: str = "power2"
    region_amax_load_bits: int = 128
    region_amax_unroll: int = 1
    region_waves: int = 4
    region_order: str = "x_first"
    region_ownership: str = "warp"
    # CUDA Programmatic Dependent Launch lets the native GEMM grid become
    # schedulable before the producer grid has completely retired.  The GEMM
    # still waits for all quantized operands to become visible, so this is a
    # schedule-only coordinate and never changes JIT row-region numerics.
    programmatic_dependent_launch: bool = False

    @property
    def jit_row_region(self) -> bool:
        return bool(self.x_scale_region_rows or self.weight_scale_region_rows)

    def rejection(self, problem: NVFP4Problem) -> str | None:
        if self.quant_launches not in ("dual", "independent", "concurrent"):
            return "quant_launches must be dual, independent, or concurrent"
        # The pointer-free block GEMM uses the packed E2M1 CTA value maps.
        # A 128-byte SMEM swizzle changes their top-level shape and CuTe rejects
        # the resulting TMA/CTA mapping before lowering.  This was confirmed by
        # every such candidate in the first revision-2 prospective run; keep it
        # out of the compiler rather than learning the same failure per shape.
        if self.gemm.a_swizzle == "128b" or self.gemm.b_swizzle == "128b":
            return "dynamic NVFP4 block GEMM does not support 128-byte swizzles"
        if bool(self.x_scale_region_rows) != bool(self.weight_scale_region_rows):
            return "JIT row-region scaling requires regions for both operands"
        if self.jit_row_region:
            if self.x_scale_region_rows < 1 or self.weight_scale_region_rows < 1:
                return "JIT row-region sizes must be positive"
            if self.tensor_scale_mode not in ("power2", "exact"):
                return "JIT row-region tensor_scale_mode must be power2 or exact"
            if self.region_amax_load_bits not in (16, 32, 64, 128):
                return "region_amax_load_bits must be 16, 32, 64, or 128"
            if self.region_amax_unroll not in (1, 2, 4, 8):
                return "region_amax_unroll must be 1, 2, 4, or 8"
            if self.region_waves not in (1, 2, 3, 4, 6, 8):
                return "region_waves must be 1, 2, 3, 4, 6, or 8"
            if self.region_order not in ("x_first", "weight_first"):
                return "region_order must be x_first or weight_first"
            if self.region_ownership not in ("warp", "cta"):
                return "region_ownership must be warp or cta"
            if self.quant_launches != "dual":
                return "the first JIT row-region implementation uses one dual launch"
            if (
                self.gemm.regional_scale_epilogue == "expanded_factors"
                and self.programmatic_dependent_launch
            ):
                return "expanded regional factors require ordinary launch ordering"
            if self.gemm.regional_scale_epilogue == "product":
                x_cache = (
                    self.gemm.tile_m + self.x_scale_region_rows - 2
                ) // self.x_scale_region_rows + 1
                w_cache = (
                    self.gemm.tile_n + self.weight_scale_region_rows - 2
                ) // self.weight_scale_region_rows + 1
                # Keep the product table bounded to 4 KiB. Larger tables erase
                # occupancy on SM120 and are better represented by the
                # independently tunable factorized cache.
                if x_cache * w_cache > 1024:
                    return "regional product cache exceeds its 4-KiB budget"
            if self.gemm.epilogue == "tma" and problem.k < 256:
                return (
                    "regional TMA epilogue requires K >= 256; small K uses "
                    "the direct epilogue to avoid disproportionate compilation"
                )
        elif self.programmatic_dependent_launch:
            return "programmatic dependent launch is implemented for JIT regions"
        native_scales = self.quant.scale_layout == "mma128"
        if native_scales != (
            self.gemm.scale_layout == "mma128" and self.gemm.scale_role == "tma"
        ):
            return (
                "dynamic NVFP4 quantizer and GEMM native scale transport "
                "must be selected together"
            )
        for label, rows in (("activation", problem.m), ("weight", problem.n)):
            reason = self.quant.rejection(rows, problem.k)
            if reason is not None:
                return f"{label} quantizer: {reason}"
        return (
            _inference_l2_rejection(self.l2_fetch_granularity)
            or self.gemm.rejection(problem)
        )


DEFAULT_NVFP4_GEMM_CONFIG = NVFP4GemmConfig()
DEFAULT_NVFP4_QUANT_CONFIG = NVFP4QuantConfig()
DEFAULT_NVFP4_SCALE_CONFIG = NVFP4ScaleConfig()
DEFAULT_NVFP4_DYNAMIC_CONFIG = NVFP4DynamicConfig()
NVFP4_KERNEL_REVISION = 7


__all__ = [
    "DEFAULT_NVFP4_DYNAMIC_CONFIG",
    "DEFAULT_NVFP4_GEMM_CONFIG",
    "DEFAULT_NVFP4_SCALE_CONFIG",
    "DEFAULT_NVFP4_QUANT_CONFIG",
    "NVFP4GemmConfig",
    "NVFP4DynamicConfig",
    "NVFP4FullyPrequantConfig",
    "NVFP4ScaleConfig",
    "NVFP4Problem",
    "NVFP4QuantConfig",
    "NVFP4WeightPrequantConfig",
    "NVFP4_KERNEL_REVISION",
    "NVFP4_SF_VEC_SIZE",
]
