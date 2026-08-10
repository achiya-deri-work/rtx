"""Architecture-independent configuration for RTX NVFP4 kernels."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from .mxfp8 import MXFP8GemmConfig
from ..kernels.mxfp8 import (
    FWD_SEARCH_SPACE,
    MXFP8FwdConfig,
    normalize_fwd_config,
)


NVFP4_SF_VEC_SIZE = 16


@dataclass(frozen=True, slots=True)
class NVFP4QuantConfig:
    """One independently compilable BF16-to-NVFP4 quantizer schedule."""

    values_per_lane: int = 2
    load_bits: int = 32
    reduction: str = "redux"
    quant_math: str = "fp32"
    num_warps: int = 8
    persistent_waves: int = 4
    maxrregcount: int = 128

    @property
    def threads_per_scale(self) -> int:
        return NVFP4_SF_VEC_SIZE // self.values_per_lane

    @property
    def blocks_per_warp(self) -> int:
        return 32 // self.threads_per_scale

    def rejection(self, rows: int, k: int) -> str | None:
        if rows <= 0 or k <= 0 or k % 64:
            return "rows must be positive and K must be divisible by 64"
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
        if (rows * (k // NVFP4_SF_VEC_SIZE)) % self.blocks_per_warp:
            return "the block grid must contain complete warp tasks"
        if self.reduction not in ("shuffle", "redux"):
            return "reduction must be shuffle or redux"
        if self.quant_math != "fp32":
            return "only the implemented FP32 NVFP4 quantization path is legal"
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
        if self.k % 64:
            raise ValueError(
                "SM120 NVFP4 MMA consumes K in groups of 64; "
                f"got K={self.k}"
            )


@dataclass(frozen=True, slots=True)
class NVFP4FwdConfig(MXFP8FwdConfig):
    """Fused BF16-to-NVFP4 forward schedule.

    The first implementation deliberately shares the mature SM120 scheduling
    coordinates with MXFP8. NVFP4 changes the scale vector, packed operand
    width, converter, and MMA atom; inherited coordinates still select real
    generated schedules rather than aliases.
    """

    tile_k: int = 256
    atom_layout_m: int = 8
    atom_layout_n: int = 2
    num_mma_warps: int = 16
    bf16_tile_k: int = 256
    mxfp8_stages: int = 1
    quant_vec: int = 16
    quant_load_bits: int = 128
    quant_math: str = "fp32"
    quant_amax: str = "fp32"
    quantizer_warps: int = 16
    k_unroll: int = 1
    num_threads: int = 512
    a_ldmatrix_matrices: int = 4
    b_ldmatrix_matrices: int = 4
    a_swizzle: str = "128b"
    b_swizzle: str = "128b"
    scale_reciprocal: str = "supplied_pow2_ptx_lut"
    tensor_scale_mode: str = "power2"
    # Zero selects one tensor-wide outer FP32 scale.  Positive values select
    # independently scaled contiguous row regions.  They are numerical-policy
    # coordinates, not hidden schedule aliases: the frontend emits one
    # three-value scale pack per region and the CTA selects its pack after
    # raster/persistent work assignment.
    x_scale_region_rows: int = 0
    weight_scale_region_rows: int = 0
    collect_amax: bool = False
    telemetry_layout: str = "scalar_atomic"
    telemetry_ownership: str = "operand_owner"
    amax_history_len: int = 16
    amax_history_algo: str = "window_max"

    @property
    def native_operand_bits(self) -> int:
        return 4

    @property
    def scale_vector_size(self) -> int:
        return NVFP4_SF_VEC_SIZE

    def implementation_rejection(self, problem: NVFP4Problem) -> str | None:
        reason = super().implementation_rejection(problem)  # type: ignore[arg-type]
        if reason is not None:
            return reason
        if self.quant_vec not in (2, 4, 8, 16):
            return "NVFP4 quant_vec must own whole packed pairs"
        if self.quant_math != "fp32":
            return "the fused NVFP4 converter currently consumes FP32 pairs"
        if self.quant_amax != "fp32":
            return "the fused NVFP4 scale path currently reduces FP32 amax"
        if self.scale_reciprocal not in (
            "direct",
            "supplied_exact",
            "supplied_pow2",
            "supplied_pow2_ptx_lut",
            "supplied_pow2_ptx_rcp",
        ):
            return (
                "NVFP4 scale_reciprocal must be direct, supplied_exact, or "
                "one of the supplied_pow2 variants"
            )
        if self.tensor_scale_mode not in ("power2", "exact"):
            return "NVFP4 tensor_scale_mode must be power2 or exact"
        for name, rows, problem_rows in (
            ("X", self.x_scale_region_rows, problem.m),
            ("weight", self.weight_scale_region_rows, problem.n),
        ):
            if rows < 0:
                return f"NVFP4 {name} scale-region rows cannot be negative"
            if rows:
                if self.collect_amax:
                    return (
                        "row-region JIT scaling and delayed amax are distinct "
                        "policies"
                    )
                if problem_rows % rows:
                    return (
                        f"NVFP4 {name} rows must be divisible by its scale region; "
                        f"got {problem_rows} and {rows}"
                    )
                if self.epilogue != "direct":
                    return (
                        "row-region JIT scaling currently requires the direct "
                        "per-element epilogue"
                    )
        if self.telemetry_layout not in ("per_cta", "scalar_atomic"):
            return "NVFP4 telemetry_layout must be per_cta or scalar_atomic"
        if self.telemetry_ownership not in ("all", "operand_owner"):
            return "NVFP4 telemetry_ownership must be all or operand_owner"
        if self.amax_history_len not in (1, 4, 16, 64):
            return "NVFP4 amax_history_len must be 1, 4, 16, or 64"
        if self.amax_history_algo not in ("most_recent", "window_max"):
            return "NVFP4 amax_history_algo must be most_recent or window_max"
        if self.bf16_tile_k < 64:
            return "NVFP4 staged transport tiles must cover a complete K=64 MMA"
        return None

    def oriented_implementation_rejection(
        self,
        problem: NVFP4Problem,
        a_orientation: str,
        b_orientation: str,
    ) -> str | None:
        reason = self.implementation_rejection(problem)
        if reason is not None:
            return reason
        if a_orientation != "row" or b_orientation != "row":
            return "the initial fused NVFP4 forward accepts row-major operands"
        return None


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

    @property
    def native_operand_bits(self) -> int:
        return 4

    @property
    def scale_vector_size(self) -> int:
        return NVFP4_SF_VEC_SIZE

    def rejection(self, problem: NVFP4Problem) -> str | None:
        try:
            problem.validate()
        except ValueError as exc:
            return str(exc)
        if self.tile_k % 64:
            return "NVFP4 tile K must be divisible by 64"
        if self.scale_layout != "row_major":
            return "the first NVFP4 GEMM revision accepts row-major scales"
        if (self.tile_k // NVFP4_SF_VEC_SIZE) % self.scale_load_vec:
            return "scale_load_vec must divide the NVFP4 K-tile scale count"
        # The common checker conservatively models operands as bytes. Passing
        # it therefore proves SMEM capacity for the half-width packed format.
        return super().rejection(problem)  # type: ignore[arg-type]


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


DEFAULT_NVFP4_GEMM_CONFIG = NVFP4GemmConfig()
DEFAULT_NVFP4_QUANT_CONFIG = NVFP4QuantConfig()
DEFAULT_NVFP4_FWD_CONFIG = NVFP4FwdConfig()
NVFP4_KERNEL_REVISION = 5


_NVFP4_EXCLUDED_COMPOUND_AXES = {
    "cta_reuse_tile",
    "cluster_reuse_tile",
    "cpasync_cluster_reuse_tile",
    "cpasync_ldmatrix_pipeline",
    "persistent_tma_pipeline",
}
NVFP4_FWD_SEARCH_SPACE: dict[str, tuple[object, ...]] = {
    name: tuple(values)
    for name, values in FWD_SEARCH_SPACE.items()
    if name not in _NVFP4_EXCLUDED_COMPOUND_AXES
}
NVFP4_FWD_SEARCH_SPACE.update(
    tile_k=(128, 256),
    bf16_tile_k=(64, 128, 256),
    quant_vec=(2, 4, 8, 16),
    quant_math=("fp32",),
    quant_amax=("fp32",),
    scale_reciprocal=(
        "supplied_pow2",
        "supplied_pow2_ptx_lut",
        "supplied_pow2_ptx_rcp",
    ),
    # Numerical scale policy is held fixed within one tuning context so exact
    # output comparison does not conflate schedules with quantization policy.
    tensor_scale_mode=("power2",),
    collect_amax=(True,),
    telemetry_layout=("per_cta", "scalar_atomic"),
    telemetry_ownership=("all", "operand_owner"),
    # History policy changes numerical behavior. Keep the production recipe
    # fixed in latency campaigns; numerical studies can explicitly override
    # these axes without conflating quality and schedule performance.
    amax_history_len=(16,),
    amax_history_algo=("window_max",),
)


def normalize_nvfp4_fwd_config(
    base: NVFP4FwdConfig | None = None,
    /,
    **updates: object,
) -> NVFP4FwdConfig:
    """Apply common compound coordinates while preserving NVFP4 fields."""

    selected = base or DEFAULT_NVFP4_FWD_CONFIG
    values = asdict(selected)
    scale_reciprocal = updates.pop(
        "scale_reciprocal", values.pop("scale_reciprocal")
    )
    tensor_scale_mode = updates.pop(
        "tensor_scale_mode", values.pop("tensor_scale_mode")
    )
    x_scale_region_rows = updates.pop(
        "x_scale_region_rows", values.pop("x_scale_region_rows")
    )
    weight_scale_region_rows = updates.pop(
        "weight_scale_region_rows", values.pop("weight_scale_region_rows")
    )
    collect_amax = updates.pop("collect_amax", values.pop("collect_amax"))
    telemetry_layout = updates.pop(
        "telemetry_layout", values.pop("telemetry_layout")
    )
    telemetry_ownership = updates.pop(
        "telemetry_ownership", values.pop("telemetry_ownership")
    )
    amax_history_len = updates.pop(
        "amax_history_len", values.pop("amax_history_len")
    )
    amax_history_algo = updates.pop(
        "amax_history_algo", values.pop("amax_history_algo")
    )
    common = MXFP8FwdConfig(**values)
    normalized = normalize_fwd_config(common, **updates)
    return NVFP4FwdConfig(
        **asdict(normalized),
        scale_reciprocal=str(scale_reciprocal),
        tensor_scale_mode=str(tensor_scale_mode),
        x_scale_region_rows=int(x_scale_region_rows),
        weight_scale_region_rows=int(weight_scale_region_rows),
        collect_amax=bool(collect_amax),
        telemetry_layout=str(telemetry_layout),
        telemetry_ownership=str(telemetry_ownership),
        amax_history_len=int(amax_history_len),
        amax_history_algo=str(amax_history_algo),
    )


__all__ = [
    "DEFAULT_NVFP4_GEMM_CONFIG",
    "DEFAULT_NVFP4_FWD_CONFIG",
    "DEFAULT_NVFP4_QUANT_CONFIG",
    "NVFP4GemmConfig",
    "NVFP4FullyPrequantConfig",
    "NVFP4FwdConfig",
    "NVFP4Problem",
    "NVFP4QuantConfig",
    "NVFP4WeightPrequantConfig",
    "NVFP4_KERNEL_REVISION",
    "NVFP4_FWD_SEARCH_SPACE",
    "normalize_nvfp4_fwd_config",
    "NVFP4_SF_VEC_SIZE",
]
