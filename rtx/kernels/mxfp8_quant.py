"""Dynamic BF16 -> MXFP8/E8M0 row-block quantization for SM120.

This is the producer for the prequantized forward backend.  It intentionally
materializes dynamic operands in global memory once per invocation so the GEMM
grid does not repeat activation quantization across N tiles or weight
quantization across M tiles.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import lru_cache
import math
import os

os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")
os.environ.setdefault("QUACK_ARCH", "sm_120a")

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import (
    BFloat16,
    Float8E4M3FN,
    Float8E8M0FNU,
    Float32,
    Int16,
    Int32,
    Uint16,
    Uint8,
)
from cutlass.experimental.primitives import nvvm_wrapper as nvvm


SF_VEC_SIZE = 32
F32_MANTISSA_BITS = 23
F32_EXPONENT_BIAS = 127
F8_MAX_POW2 = 8
E8M0_MIN_UNBIASED = -127
E8M0_MAX_UNBIASED = 128


def _native_scale_tile_rows(scale_layout: str) -> int:
    return {"row_major": 0, "mma64": 64, "mma128": 128}[scale_layout]


@dataclass(frozen=True, slots=True)
class MXFP8QuantConfig:
    """One independently compilable dynamic quantizer schedule."""

    quant_vec: int = 4
    load_bits: int = 64
    quant_math: str = "bf16x2"
    quant_amax: str = "bf16_bits"
    reduction: str = "shuffle"
    num_warps: int = 8
    persistent_waves: int = 4
    maxrregcount: int = 128
    scale_layout: str = "row_major"
    native_scale_store: str = "scalar"

    def rejection(self, rows: int, k: int) -> str | None:
        if rows <= 0 or k <= 0 or k % SF_VEC_SIZE:
            return "rows must be positive and K must be divisible by 32"
        if self.quant_vec not in (1, 2, 4, 8):
            return "quant_vec must be one of 1, 2, 4, 8"
        if self.load_bits not in (16, 32, 64, 128):
            return "load_bits must be one of 16, 32, 64, 128"
        if self.load_bits > self.quant_vec * BFloat16.width:
            return "load width exceeds values owned by one lane"
        if (self.quant_vec * BFloat16.width) % self.load_bits:
            return "quant_vec must contain an integer number of vector loads"
        if (k // SF_VEC_SIZE) % self.quant_vec:
            return "K scale blocks must be divisible by quant_vec"
        if self.quant_math not in ("fp32", "bf16x2"):
            return "quant_math must be fp32 or bf16x2"
        if self.quant_amax not in ("fp32", "bf16_bits"):
            return "quant_amax must be fp32 or bf16_bits"
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
        return None


class MXFP8QuantKernel:
    """Warp-grouped, grid-stride dynamic MXFP8 quantization kernel."""

    def __init__(self, rows: int, k: int, config: MXFP8QuantConfig):
        rejection = config.rejection(rows, k)
        if rejection is not None:
            raise ValueError(f"illegal MXFP8 quantizer configuration: {rejection}")
        self.rows = rows
        self.k = k
        self.config = config
        self.scale_tile_rows = _native_scale_tile_rows(config.scale_layout)
        blocks_per_row = k // SF_VEC_SIZE
        self.task_groups = rows * blocks_per_row // config.quant_vec
        sm_count = utils.HardwareInfo().get_device_multiprocessor_count()
        max_ctas = sm_count * config.persistent_waves
        natural_ctas = cute.ceil_div(self.task_groups, config.num_warps)
        self.grid_ctas = min(natural_ctas, max_ctas)

    @cute.jit
    def __call__(
        self,
        src: cute.Tensor,
        quantized: cute.Tensor,
        scales: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(src, quantized, scales).launch(
            grid=(self.grid_ctas, 1, 1),
            block=(self.config.num_warps * 32, 1, 1),
            stream=stream,
        )

    @cute.jit
    def _scale_from_amax(self, amax: Float32):
        exponent = ((amax.bitcast(Int32) >> F32_MANTISSA_BITS) & 0xFF) - (
            F32_EXPONENT_BIAS + F8_MAX_POW2
        )
        exponent = cutlass.max(exponent, E8M0_MIN_UNBIASED)
        exponent = cutlass.min(exponent, E8M0_MAX_UNBIASED)
        biased = exponent + F32_EXPONENT_BIAS
        if cute.isnan(amax):
            biased = Int32(255)
        scale_e8m0 = Uint8(biased).bitcast(Float8E8M0FNU)

        fp32_exponent = Int32(1) if biased == 0 else biased
        reciprocal_bits = (Int32(254) - fp32_exponent) << F32_MANTISSA_BITS
        if fp32_exponent == 254:
            reciprocal_bits = Int32(1 << (F32_MANTISSA_BITS - 1))
        if fp32_exponent == 255:
            reciprocal_bits = Int32(0)
        return scale_e8m0, reciprocal_bits.bitcast(Float32)

    @cute.jit
    def _subwarp_amax(
        self,
        value: Float32,
        threads_in_group: cutlass.Constexpr,
        lane_idx: Int32,
    ):
        maximum_bits = value.bitcast(Int32) & Int32(0x7FFFFFFF)
        if cutlass.const_expr(self.config.reduction == "redux"):
            group_base = lane_idx & Int32(~(threads_in_group - 1))
            member_mask = Int32((1 << threads_in_group) - 1) << group_base
            return nvvm.redux_sync(
                maximum_bits,
                nvvm.ReductionKind.UMAX,
                member_mask,
            ).bitcast(Float32)

        maximum = maximum_bits.bitcast(Float32)
        mask_and_clamp = ((32 - threads_in_group) << 8) | (threads_in_group - 1)
        for level in cutlass.range_constexpr(int(math.log2(threads_in_group))):
            other = cute.arch.shuffle_sync_bfly(
                maximum,
                (threads_in_group // 2) >> level,
                mask_and_clamp=mask_and_clamp,
            )
            maximum = cute.arch.fmax(maximum, other, nan=True)
        return maximum

    @cute.jit
    def _quantize_task(
        self,
        src: cute.Tensor,
        quantized: cute.Tensor,
        scales: cute.Tensor,
        row: Int32,
        scale_block: Int32,
        lane_in_scale: Int32,
        k_base: Int32,
        scale_tile_rows: cutlass.Constexpr,
    ):
        """Quantize one lane's values inside a four/one/two/eight-block warp task."""

        cfg = self.config
        lane_idx = cute.arch.lane_idx()
        threads_per_scale = 32 // cfg.quant_vec
        scale_in_warp = lane_idx // threads_per_scale
        bf16_values = [BFloat16(0.0)] * cfg.quant_vec
        values_per_load = cfg.load_bits // BFloat16.width
        loads_per_lane = cfg.quant_vec // values_per_load
        src_row = src[row, None]
        for load_idx in cutlass.range_constexpr(loads_per_lane):
            vec_base = load_idx * values_per_load
            if cutlass.const_expr(values_per_load == 1):
                bf16_values[vec_base] = src_row[k_base + vec_base]
            else:
                loaded = nvvm.load_ext(
                    src_row.iterator + src_row.layout(k_base + vec_base),
                    dtype=Uint16,
                    count=values_per_load,
                ).bitcast(BFloat16)
                for load_vec in cutlass.range_constexpr(values_per_load):
                    bf16_values[vec_base + load_vec] = loaded[load_vec]

        values = [Float32(0.0)] * cfg.quant_vec
        local_maximum = Float32(0.0)
        local_maximum_bits = Int32(0)
        for vec in cutlass.range_constexpr(cfg.quant_vec):
            value = Float32(bf16_values[vec])
            values[vec] = value
            if cutlass.const_expr(cfg.quant_amax == "fp32"):
                magnitude = (
                    value.bitcast(Int32) & Int32(0x7FFFFFFF)
                ).bitcast(Float32)
                local_maximum = cute.arch.fmax(
                    local_maximum, magnitude, nan=True
                )
            else:
                magnitude_bits = Int32(
                    bf16_values[vec].bitcast(Uint16)
                ) & Int32(0x7FFF)
                local_maximum_bits = cutlass.max(
                    local_maximum_bits, magnitude_bits
                )
        if cutlass.const_expr(cfg.quant_amax == "bf16_bits"):
            local_maximum = Float32(
                Uint16(local_maximum_bits).bitcast(BFloat16)
            )

        amax = self._subwarp_amax(
            local_maximum, threads_per_scale, lane_idx
        )
        scale_e8m0, inv_scale_fp32 = self._scale_from_amax(amax)
        for pair in cutlass.range_constexpr((cfg.quant_vec + 1) // 2):
            vec0 = pair * 2
            vec1 = cutlass.min(vec0 + 1, cfg.quant_vec - 1)
            if cutlass.const_expr(cfg.quant_math == "bf16x2"):
                bits0 = bf16_values[vec0].bitcast(Uint16)
                bits1 = bf16_values[vec1].bitcast(Uint16)
                packed_values = Int32(bits0) | (Int32(bits1) << 16)
                inv_bits = BFloat16(inv_scale_fp32).bitcast(Uint16)
                packed_inv = Int32(inv_bits) | (Int32(inv_bits) << 16)
                scaled = nvvm.mul_bf16x2(packed_values, packed_inv)
                packed = nvvm.inline_ptx_hl(
                    "cvt.rn.satfinite.e4m3x2.bf16x2 {$w0}, {$r0};",
                    write_only_types=[Int16],
                    read_only_args=[scaled],
                )
                q0 = Uint8(packed & Int16(0xFF)).bitcast(Float8E4M3FN)
                q1 = Uint8(
                    (packed >> Int16(8)) & Int16(0xFF)
                ).bitcast(Float8E4M3FN)
            else:
                scaled0 = values[vec0] * inv_scale_fp32
                scaled1 = values[vec1] * inv_scale_fp32
                packed = nvvm.inline_ptx_hl(
                    "cvt.rn.satfinite.e4m3x2.f32 {$w0}, {$r0}, {$r1};",
                    write_only_types=[Int16],
                    read_only_args=[scaled0, scaled1],
                )
                q0 = Uint8(
                    (packed >> Int16(8)) & Int16(0xFF)
                ).bitcast(Float8E4M3FN)
                q1 = Uint8(packed & Int16(0xFF)).bitcast(Float8E4M3FN)
            quantized[row, k_base + vec0] = q0
            if vec1 != vec0:
                quantized[row, k_base + vec1] = q1

        if cutlass.const_expr(
            scale_tile_rows == 128
            and cfg.native_scale_store == "packed"
        ):
            scale_bits = Int32(scale_e8m0.bitcast(Uint8))
            packed_scales = cute.arch.shuffle_sync(scale_bits, 0)
            packed_scales |= cute.arch.shuffle_sync(scale_bits, 8) << 8
            packed_scales |= cute.arch.shuffle_sync(scale_bits, 16) << 16
            packed_scales |= cute.arch.shuffle_sync(scale_bits, 24) << 24
            if lane_idx == 0:
                physical = (
                    (row % 32) * 16
                    + ((row // 32) % 4) * 4
                    + (scale_block - scale_in_warp) % 4
                )
                nvvm.store_ext(
                    packed_scales,
                    scales.iterator
                    + scales.layout(
                        (
                            row // 128,
                            (scale_block - scale_in_warp) // 4,
                            physical,
                        )
                    ),
                )
        elif lane_in_scale == 0:
            if cutlass.const_expr(scale_tile_rows == 0):
                scales[row, scale_block] = scale_e8m0
            else:
                physical = (
                    (row % 32) * 16
                    + scale_block % 4
                )
                if cutlass.const_expr(scale_tile_rows == 64):
                    physical += ((row // 32) % 2) * 4
                    scales[row // 64, scale_block // 4, physical] = scale_e8m0
                else:
                    physical += ((row // 32) % 4) * 4
                    scales[row // 128, scale_block // 4, physical] = scale_e8m0

    @cute.kernel
    def kernel(
        self,
        src: cute.Tensor,
        quantized: cute.Tensor,
        scales: cute.Tensor,
    ):
        cfg = self.config
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block_idx, _, _ = cute.arch.block_idx()
        warp_linear = block_idx * cfg.num_warps + warp_idx
        warp_stride = self.grid_ctas * cfg.num_warps
        blocks_per_row = self.k // SF_VEC_SIZE
        threads_per_scale = 32 // cfg.quant_vec
        scale_in_warp = lane_idx // threads_per_scale
        lane_in_scale = lane_idx % threads_per_scale

        for task_group in cutlass.range(
            warp_linear, self.task_groups, warp_stride, unroll=1
        ):
            first_block = task_group * cfg.quant_vec
            row = first_block // blocks_per_row
            scale_block = first_block % blocks_per_row + scale_in_warp
            k_base = scale_block * SF_VEC_SIZE + lane_in_scale * cfg.quant_vec
            self._quantize_task(
                src,
                quantized,
                scales,
                row,
                scale_block,
                lane_in_scale,
                k_base,
                self.scale_tile_rows,
            )


class MXFP8DualQuantKernel(MXFP8QuantKernel):
    """Quantize activation and weight rows in one persistent launch."""

    def __init__(
        self,
        x_rows: int,
        weight_rows: int,
        k: int,
        config: MXFP8QuantConfig,
        weight_scale_layout: str | None = None,
    ):
        weight_config = replace(
            config,
            scale_layout=weight_scale_layout or config.scale_layout,
        )
        for rows, operand_config in (
            (x_rows, config),
            (weight_rows, weight_config),
        ):
            rejection = operand_config.rejection(rows, k)
            if rejection is not None:
                raise ValueError(
                    f"illegal dual MXFP8 quantizer configuration: {rejection}"
                )
        self.x_rows = x_rows
        self.weight_rows = weight_rows
        self.k = k
        self.config = config
        self.x_scale_tile_rows = _native_scale_tile_rows(config.scale_layout)
        self.weight_scale_tile_rows = _native_scale_tile_rows(
            weight_config.scale_layout
        )
        blocks_per_row = k // SF_VEC_SIZE
        self.x_task_groups = x_rows * blocks_per_row // config.quant_vec
        self.weight_task_groups = (
            weight_rows * blocks_per_row // config.quant_vec
        )
        self.task_groups = self.x_task_groups + self.weight_task_groups
        sm_count = utils.HardwareInfo().get_device_multiprocessor_count()
        max_ctas = sm_count * config.persistent_waves
        natural_ctas = cute.ceil_div(self.task_groups, config.num_warps)
        self.grid_ctas = min(natural_ctas, max_ctas)

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        qx: cute.Tensor,
        qw: cute.Tensor,
        sx: cute.Tensor,
        sw: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(x, weight, qx, qw, sx, sw).launch(
            grid=(self.grid_ctas, 1, 1),
            block=(self.config.num_warps * 32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        qx: cute.Tensor,
        qw: cute.Tensor,
        sx: cute.Tensor,
        sw: cute.Tensor,
    ):
        cfg = self.config
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block_idx, _, _ = cute.arch.block_idx()
        warp_linear = block_idx * cfg.num_warps + warp_idx
        warp_stride = self.grid_ctas * cfg.num_warps
        blocks_per_row = self.k // SF_VEC_SIZE
        threads_per_scale = 32 // cfg.quant_vec
        scale_in_warp = lane_idx // threads_per_scale
        lane_in_scale = lane_idx % threads_per_scale

        for task_group in cutlass.range(
            warp_linear, self.task_groups, warp_stride, unroll=1
        ):
            if task_group < self.x_task_groups:
                first_block = task_group * cfg.quant_vec
                row = first_block // blocks_per_row
                scale_block = first_block % blocks_per_row + scale_in_warp
                k_base = (
                    scale_block * SF_VEC_SIZE
                    + lane_in_scale * cfg.quant_vec
                )
                self._quantize_task(
                    x,
                    qx,
                    sx,
                    row,
                    scale_block,
                    lane_in_scale,
                    k_base,
                    self.x_scale_tile_rows,
                )
            else:
                local_task = task_group - self.x_task_groups
                first_block = local_task * cfg.quant_vec
                row = first_block // blocks_per_row
                scale_block = first_block % blocks_per_row + scale_in_warp
                k_base = (
                    scale_block * SF_VEC_SIZE
                    + lane_in_scale * cfg.quant_vec
                )
                self._quantize_task(
                    weight,
                    qw,
                    sw,
                    row,
                    scale_block,
                    lane_in_scale,
                    k_base,
                    self.weight_scale_tile_rows,
                )


@lru_cache(maxsize=None)
def compile_mxfp8_quant(
    rows: int,
    k: int,
    config: MXFP8QuantConfig = MXFP8QuantConfig(),
):
    """Compile and cache a shape/config-specialized TVM-FFI launcher."""

    kernel = MXFP8QuantKernel(rows, k, config)
    src = cute.runtime.make_fake_tensor(
        BFloat16,
        (rows, k),
        stride=(k, 1),
        assumed_align=16,
    )
    quantized = cute.runtime.make_fake_tensor(
        Float8E4M3FN,
        (rows, k),
        stride=(k, 1),
        assumed_align=16,
    )
    if config.scale_layout == "row_major":
        scales = cute.runtime.make_fake_tensor(
            Float8E8M0FNU,
            (rows, k // SF_VEC_SIZE),
            stride=(k // SF_VEC_SIZE, 1),
            assumed_align=16,
        )
    else:
        tile_rows = 64 if config.scale_layout == "mma64" else 128
        # The 64-row SM120 SFA layout has a physical cosize of 504 bytes.
        # Pad each global tile to 512 bytes so it can be moved with one
        # naturally aligned TMA transaction; offsets 504:512 are unused.
        tile_storage = 512
        scales = cute.runtime.make_fake_tensor(
            Float8E8M0FNU,
            (rows // tile_rows, k // 128, tile_storage),
            stride=(k // 128 * tile_storage, tile_storage, 1),
            assumed_align=16,
        )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        src,
        quantized,
        scales,
        stream,
        options=(
            "--enable-tvm-ffi --opt-level 3 "
            f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"
        ),
    )


@lru_cache(maxsize=None)
def compile_mxfp8_dual_quant(
    x_rows: int,
    weight_rows: int,
    k: int,
    config: MXFP8QuantConfig = MXFP8QuantConfig(),
    weight_scale_layout: str | None = None,
):
    """Compile one launcher that dynamically quantizes X and W together."""

    kernel = MXFP8DualQuantKernel(
        x_rows, weight_rows, k, config, weight_scale_layout
    )

    def fake(dtype, rows: int, columns: int):
        return cute.runtime.make_fake_tensor(
            dtype,
            (rows, columns),
            stride=(columns, 1),
            assumed_align=16,
        )

    x = fake(BFloat16, x_rows, k)
    weight = fake(BFloat16, weight_rows, k)
    qx = fake(Float8E4M3FN, x_rows, k)
    qw = fake(Float8E4M3FN, weight_rows, k)

    def fake_scales(rows: int, scale_layout: str):
        if scale_layout == "row_major":
            return fake(Float8E8M0FNU, rows, k // SF_VEC_SIZE)
        tile_rows = _native_scale_tile_rows(scale_layout)
        return cute.runtime.make_fake_tensor(
            Float8E8M0FNU,
            (rows // tile_rows, k // 128, 512),
            stride=(k // 128 * 512, 512, 1),
            assumed_align=16,
        )

    sx = fake_scales(x_rows, config.scale_layout)
    sw = fake_scales(
        weight_rows, weight_scale_layout or config.scale_layout
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        x,
        weight,
        qx,
        qw,
        sx,
        sw,
        stream,
        options=(
            "--enable-tvm-ffi --opt-level 3 "
            f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"
        ),
    )


__all__ = [
    "MXFP8DualQuantKernel",
    "MXFP8QuantConfig",
    "MXFP8QuantKernel",
    "compile_mxfp8_dual_quant",
    "compile_mxfp8_quant",
]
