"""MXFP8 quantizer/GEMM configs which do not import a CuTe architecture."""

from __future__ import annotations

from dataclasses import dataclass

from ..kernels.mxfp8 import (
    MXFP8Problem,
    SM120_GEMM_RUNTIME_SMEM_RESERVE_BYTES,
    SM120_SMEM_CAPACITY_BYTES,
)


SF_VEC_SIZE = 32


@dataclass(frozen=True, slots=True)
class MXFP8QuantConfig:
    quant_vec: int = 4
    load_bits: int = 64
    quant_store_bits: int = 8
    quant_math: str = "bf16x2"
    quant_amax: str = "bf16_bits"
    # ``infinity`` selects the smallest E8M0 power of two that keeps E4M3's
    # finite 448 maximum in range.  ``floor`` retains the minimally required
    # OCP conversion for controlled experiments, but can clip training values.
    scale_rounding: str = "infinity"
    reduction: str = "shuffle"
    num_warps: int = 8
    persistent_waves: int = 4
    maxrregcount: int = 128
    scale_layout: str = "row_major"
    native_scale_store: str = "scalar"
    transposed_load_engine: str = "register"
    transposed_tile_rows: int = 128
    transposed_tile_k: int = 32
    transposed_smem_padding: int = 1

    def rejection(self, rows: int, k: int) -> str | None:
        if rows <= 0 or k <= 0:
            return "rows and K must be positive"
        if self.quant_vec not in (1, 2, 4, 8):
            return "quant_vec must be one of 1, 2, 4, 8"
        if self.load_bits not in (16, 32, 64, 128):
            return "load_bits must be one of 16, 32, 64, 128"
        if self.load_bits > self.quant_vec * 16:
            return "load width exceeds values owned by one lane"
        if (self.quant_vec * 16) % self.load_bits:
            return "quant_vec must contain an integer number of vector loads"
        if self.quant_store_bits not in (8, 16, 32):
            return "quant_store_bits must be one of 8, 16, 32"
        if self.quant_store_bits > self.quant_vec * 8:
            return "quantized store width exceeds values owned by one lane"
        if (self.quant_vec * 8) % self.quant_store_bits:
            return "quant_vec must contain an integer number of quantized stores"
        if self.quant_math not in ("fp32", "bf16x2"):
            return "quant_math must be fp32 or bf16x2"
        if self.quant_amax not in ("fp32", "bf16_bits"):
            return "quant_amax must be fp32 or bf16_bits"
        if self.scale_rounding not in ("floor", "infinity"):
            return "scale_rounding must be floor or infinity"
        if self.reduction not in ("shuffle", "redux"):
            return "reduction must be shuffle or redux"
        if self.num_warps not in (4, 8, 16):
            return "num_warps must be one of 4, 8, 16"
        if self.persistent_waves not in (1, 2, 3, 4, 6, 8):
            return "persistent_waves must be one of 1, 2, 3, 4, 6, 8"
        if self.scale_layout not in ("row_major", "mma64", "mma128"):
            return "scale_layout must be row_major, mma64, or mma128"
        if self.scale_layout == "mma128" and (rows % 128 or k % 128):
            return "mma128 scales require rows and K divisible by 128"
        if self.scale_layout == "mma64" and (rows % 64 or k % 128):
            return "mma64 scales require rows divisible by 64 and K by 128"
        if self.native_scale_store not in ("scalar", "packed"):
            return "native_scale_store must be scalar or packed"
        if self.native_scale_store == "packed" and (
            self.scale_layout != "mma128" or self.quant_vec != 4
        ):
            return "packed native scale stores require mma128 and quant_vec=4"
        if self.transposed_load_engine not in ("register", "cp_async"):
            return "transposed_load_engine must be register or cp_async"
        if self.transposed_tile_rows not in (32, 64, 128, 256):
            return "transposed_tile_rows must be 32, 64, 128, or 256"
        if self.transposed_tile_k not in (32, 64, 128):
            return "transposed_tile_k must be 32, 64, or 128"
        if self.transposed_smem_padding not in (0, 1, 2, 4, 8):
            return "transposed_smem_padding must be 0, 1, 2, 4, or 8"
        if (
            self.transposed_load_engine == "cp_async"
            and (self.transposed_tile_rows + self.transposed_smem_padding) * 2
            % 16
        ):
            return "cp_async transposed SMEM rows must match the copy alignment"
        return None

    def transposed_rejection(self, rows: int, k: int) -> str | None:
        reason = self.rejection(rows, k)
        if reason is not None:
            return reason
        if k % SF_VEC_SIZE:
            return "decomposed logical-transpose K must be divisible by 32"
        if k % self.transposed_tile_k:
            return "logical-transpose K must contain full SMEM tiles"
        if self.native_scale_store == "packed" and self.transposed_tile_k != 128:
            return "packed transposed native scales require tile_k=128"
        return None


@dataclass(frozen=True, slots=True)
class MXFP8GemmConfig:
    tile_m: int = 128
    tile_n: int = 128
    tile_k: int = 128
    atom_layout_m: int = 8
    atom_layout_n: int = 2
    stages: int = 2
    a_swizzle: str = "64b"
    b_swizzle: str = "64b"
    a_ldmatrix_matrices: int = 4
    b_ldmatrix_matrices: int = 4
    mma_schedule: str = "interleaved"
    sfa_s2r_bits: int = 8
    sfb_s2r_bits: int = 8
    scale_schedule: str = "before_wait"
    scale_recycle: str = "barrier"
    scale_load_vec: int = 4
    scale_smem_store: str = "scalar"
    scale_l2_prefetch: str = "none"
    scale_l1_evict: str = "default"
    scale_cache: str = "default"
    scale_role: str = "consumers"
    scale_layout: str = "row_major"
    epilogue: str = "tma"
    epilogue_stages: int = 1
    store_vec: int = 4
    maxrregcount: int = 255
    producer_registers: int = 48
    consumer_registers: int = 192
    raster: str = "n"
    grid_swizzle: int = 2
    tiles_per_cta: int = 1
    tile_locality: str = "raster"
    persistent_waves: int = 0

    @property
    def native_operand_bits(self) -> int:
        return 8

    @property
    def scale_vector_size(self) -> int:
        return SF_VEC_SIZE

    @property
    def native_mma_k(self) -> int:
        return 128

    @property
    def num_mma_warps(self) -> int:
        return self.atom_layout_m * self.atom_layout_n

    @property
    def num_threads(self) -> int:
        return (self.num_mma_warps + 1) * 32

    def rejection(self, problem: MXFP8Problem) -> str | None:
        try:
            problem.validate()
        except ValueError as exc:
            return str(exc)
        if (self.tile_m != 64 and self.tile_m % 128) or self.tile_n % 128:
            return "SM120 block-scale tiles require M=64 or M/N divisible by 128"
        if self.tile_k % self.native_mma_k:
            return f"tile K must be divisible by {self.native_mma_k}"
        if self.tile_n != 64 * self.atom_layout_n:
            return "tile_n must equal 64 * atom_layout_n"
        if self.tile_m == 64 and self.atom_layout_m != 2:
            return "64-row SFA fragments require atom_layout_m=2"
        if self.tile_m == 256 and self.atom_layout_m != 8:
            return "256-row SFA fragments require atom_layout_m=8"
        if self.num_threads > 1024:
            return "CUDA limits one CTA to 1024 threads"
        if self.stages not in (1, 2, 3, 4):
            return "stages must be one of 1, 2, 3, 4"
        if self.a_swizzle not in ("none", "32b", "64b", "128b"):
            return "invalid A swizzle"
        if self.b_swizzle not in ("none", "32b", "64b", "128b"):
            return "invalid B swizzle"
        if self.a_ldmatrix_matrices not in (1, 2, 4):
            return "A ldmatrix width must be x1, x2, or x4"
        if self.b_ldmatrix_matrices not in (1, 2, 4):
            return "B ldmatrix width must be x1, x2, or x4"
        if self.mma_schedule not in ("interleaved", "preload"):
            return "mma_schedule must be interleaved or preload"
        if self.sfa_s2r_bits not in (0, 8) or self.sfb_s2r_bits not in (0, 8):
            return "scale S2R widths must be auto or 8 bits"
        if self.scale_schedule not in ("after_wait", "before_wait"):
            return "scale_schedule must be after_wait or before_wait"
        if self.scale_recycle not in ("barrier", "staged"):
            return "scale_recycle must be barrier or staged"
        if self.scale_recycle == "staged" and (
            self.scale_role != "consumers" or self.stages < 2
        ):
            return "staged scale recycling requires consumer staging and 2+ stages"
        if self.scale_load_vec not in (1, 2, 4, 8):
            return "scale_load_vec must be x1, x2, x4, or x8"
        if self.scale_smem_store not in ("scalar", "packed"):
            return "scale_smem_store must be scalar or packed"
        if self.scale_smem_store == "packed" and (
            self.scale_role != "consumers" or self.scale_load_vec == 1
        ):
            return "packed scale SMEM stores require vectorized consumer staging"
        if (self.tile_k // self.scale_vector_size) % self.scale_load_vec:
            return "scale_load_vec must divide the K tile scale count"
        if self.scale_l2_prefetch not in ("none", "64b", "128b", "256b"):
            return "invalid scale L2 prefetch size"
        if self.scale_l1_evict not in (
            "default", "normal", "first", "last", "noallocate"
        ):
            return "invalid scale L1 eviction priority"
        if self.scale_cache not in ("default", "ca", "cg", "cs"):
            return "invalid scale cache modifier"
        if self.scale_l1_evict != "default" and self.scale_cache != "default":
            return "scale eviction and cache modifiers are mutually exclusive"
        if self.scale_role not in ("consumers", "producer", "tma"):
            return "scale_role must be consumers, producer, or tma"
        if self.scale_role == "tma" and (
            self.scale_schedule != "before_wait"
            or self.scale_load_vec != 4
            or self.scale_smem_store != "scalar"
            or self.scale_l2_prefetch != "none"
            or self.scale_l1_evict != "default"
            or self.scale_cache != "default"
        ):
            return "scalar scale-staging controls are inactive for TMA scales"
        if self.scale_load_vec == 1 and (
            self.scale_l2_prefetch != "none"
            or self.scale_l1_evict != "default"
            or self.scale_cache != "default"
        ):
            return "vector-load cache controls are inactive for scalar scale loads"
        if self.scale_layout not in ("row_major", "mma128", "mma64x128"):
            return "scale_layout must be row_major, mma128, or mma64x128"
        if self.scale_layout == "mma128" and (
            self.scale_role != "tma"
            or self.tile_m != 128
            or self.tile_n != 128
            or self.tile_k != 128
            or problem.m % 128
            or problem.n % 128
            or problem.k % 128
        ):
            return "mma128 scales require TMA and full 128-row operand tiles"
        if self.scale_role == "tma" and self.scale_layout != "mma128":
            if not (
                self.scale_layout == "mma64x128"
                and self.tile_m == 64
                and self.tile_n == 128
                and self.tile_k == 128
                and self.stages == 1
                and problem.m % 64 == 0
                and problem.n % 128 == 0
                and problem.k % 128 == 0
            ):
                return "TMA scale transport requires a compatible native layout"
        if self.epilogue not in ("direct", "tma"):
            return "epilogue must be direct or tma"
        if self.epilogue_stages not in (1, 2, 3, 4):
            return "epilogue_stages must be one of 1, 2, 3, 4"
        if self.epilogue == "direct" and self.epilogue_stages != 1:
            return "epilogue stages only apply to the TMA epilogue"
        if self.store_vec not in (1, 2, 4):
            return "store_vec must be x1, x2, or x4"
        if self.epilogue == "direct" and self.store_vec != 1:
            return "store_vec only applies to the TMA epilogue"
        if self.epilogue == "tma" and (
            problem.m % self.tile_m or problem.n % self.tile_n
        ):
            return "TMA epilogue requires full M/N tiles"
        for label, registers in (
            ("producer", self.producer_registers),
            ("consumer", self.consumer_registers),
        ):
            if not 24 <= registers <= 256 or registers % 8:
                return (
                    f"{label} setmaxregister value must be a multiple of 8 "
                    "between 24 and 256"
                )
        if not 1 <= self.maxrregcount <= 255:
            return "maxrregcount must be between 1 and 255"
        q_bytes = (
            self.stages
            * (self.tile_m + self.tile_n)
            * self.tile_k
            * self.native_operand_bits
            // 8
        )
        scale_bytes = self.stages * (
            ((self.tile_m + 127) // 128) * 128
            + ((self.tile_n + 127) // 128) * 128
        ) * (self.tile_k // self.scale_vector_size)
        out_bytes = (
            self.epilogue_stages * self.tile_m * self.tile_n * 2
            if self.epilogue == "tma"
            else 0
        )
        launch_bytes = (
            q_bytes
            + scale_bytes
            + out_bytes
            + SM120_GEMM_RUNTIME_SMEM_RESERVE_BYTES
        )
        if launch_bytes > SM120_SMEM_CAPACITY_BYTES:
            return (
                "prequantized GEMM exceeds SM120 shared-memory capacity "
                "including runtime reserve"
            )
        if self.raster not in ("m", "n") or self.grid_swizzle not in (1, 2, 4, 8):
            return "invalid raster/grid swizzle"
        if self.tiles_per_cta not in (1, 2, 4, 8):
            return "tiles_per_cta must be 1, 2, 4, or 8"
        if self.persistent_waves not in (0, 1, 2, 3, 4):
            return "persistent_waves must be zero or one to four waves"
        output_tiles = (
            (problem.m + self.tile_m - 1) // self.tile_m
        ) * ((problem.n + self.tile_n - 1) // self.tile_n)
        if self.tiles_per_cta > output_tiles:
            return "tiles_per_cta cannot exceed the logical output tile count"
        if self.tile_locality not in (
            "raster",
            "same_a",
            "same_b",
            "serpentine_a",
            "serpentine_b",
        ):
            return "invalid persistent tile locality"
        return None


__all__ = ["MXFP8GemmConfig", "MXFP8QuantConfig", "SF_VEC_SIZE"]
