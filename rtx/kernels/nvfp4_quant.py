"""Dynamic BF16 to packed NVFP4/E4M3 quantization for SM120/SM121."""

from __future__ import annotations

from functools import lru_cache
import math
import os

os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")
os.environ.setdefault("QUACK_ARCH", "sm_120a")

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import BFloat16, Float8E4M3FN, Float32, Int16, Int32, Uint16, Uint8
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

from ..configs.nvfp4 import NVFP4QuantConfig, NVFP4_SF_VEC_SIZE


F4_MAX = 6.0
F8_MAX = 448.0
E4M3_MIN_NORMAL = 0.015625


class NVFP4QuantKernel:
    """Persistent grid-stride quantizer with a supplied FP32 global scale."""

    def __init__(self, rows: int, k: int, config: NVFP4QuantConfig):
        rejection = config.rejection(rows, k)
        if rejection is not None:
            raise ValueError(f"illegal NVFP4 quantizer configuration: {rejection}")
        self.rows = rows
        self.k = k
        self.config = config
        blocks = rows * (k // NVFP4_SF_VEC_SIZE)
        self.task_groups = blocks // config.blocks_per_warp
        sm_count = utils.HardwareInfo().get_device_multiprocessor_count()
        natural_ctas = cute.ceil_div(self.task_groups, config.num_warps)
        self.grid_ctas = min(
            natural_ctas,
            sm_count * config.persistent_waves,
        )

    @cute.jit
    def __call__(
        self,
        src: cute.Tensor,
        quantized_packed: cute.Tensor,
        scales: cute.Tensor,
        tensor_scale: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.kernel(src, quantized_packed, scales, tensor_scale).launch(
            grid=(self.grid_ctas, 1, 1),
            block=(self.config.num_warps * 32, 1, 1),
            stream=stream,
        )

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
        mask_and_clamp = ((32 - threads_in_group) << 8) | (
            threads_in_group - 1
        )
        for level in cutlass.range_constexpr(int(math.log2(threads_in_group))):
            other = cute.arch.shuffle_sync_bfly(
                maximum,
                (threads_in_group // 2) >> level,
                mask_and_clamp=mask_and_clamp,
            )
            maximum = cute.arch.fmax(maximum, other, nan=True)
        return maximum

    @cute.jit
    def _e4m3_scale(self, raw_scale: Float32):
        raw_scale = cute.arch.fmax(raw_scale, E4M3_MIN_NORMAL, nan=True)
        raw_scale = cute.arch.fmin(raw_scale, F8_MAX, nan=True)
        packed = nvvm.inline_ptx_hl(
            "cvt.rn.satfinite.e4m3x2.f32 {$w0}, {$r0}, {$r0};",
            write_only_types=[Int16],
            read_only_args=[raw_scale],
        )
        return Uint8(packed & Int16(0xFF)).bitcast(Float8E4M3FN)

    @cute.jit
    def _quantize_pair(self, value0: Float32, value1: Float32):
        return nvvm.inline_ptx_hl(
            "{.reg .b8 b; "
            "cvt.rn.satfinite.e2m1x2.f32 b, {$r1}, {$r0}; "
            "mov.b16 {$w0}, {b, 0};}",
            write_only_types=[Int16],
            read_only_args=[value0, value1],
        )

    @cute.jit
    def _quantize_task(
        self,
        src: cute.Tensor,
        quantized_packed: cute.Tensor,
        scales: cute.Tensor,
        tensor_scale: Float32,
        row: Int32,
        scale_block: Int32,
        lane_in_scale: Int32,
    ):
        cfg = self.config
        k_base = (
            scale_block * NVFP4_SF_VEC_SIZE
            + lane_in_scale * cfg.values_per_lane
        )
        values_bf16 = [BFloat16(0.0)] * cfg.values_per_lane
        values_per_load = cfg.load_bits // BFloat16.width
        loads_per_lane = cfg.values_per_lane // values_per_load
        src_row = src[row, None]
        for load_idx in cutlass.range_constexpr(loads_per_lane):
            value_base = load_idx * values_per_load
            if cutlass.const_expr(values_per_load == 1):
                values_bf16[value_base] = src_row[k_base + value_base]
            else:
                loaded = nvvm.load_ext(
                    src_row.iterator + src_row.layout(k_base + value_base),
                    dtype=Uint16,
                    count=values_per_load,
                ).bitcast(BFloat16)
                for vec in cutlass.range_constexpr(values_per_load):
                    values_bf16[value_base + vec] = loaded[vec]

        values = [Float32(0.0)] * cfg.values_per_lane
        local_amax = Float32(0.0)
        for vec in cutlass.range_constexpr(cfg.values_per_lane):
            value = Float32(values_bf16[vec])
            values[vec] = value
            magnitude = (
                value.bitcast(Int32) & Int32(0x7FFFFFFF)
            ).bitcast(Float32)
            local_amax = cute.arch.fmax(local_amax, magnitude, nan=True)
        block_amax = self._subwarp_amax(
            local_amax,
            cfg.threads_per_scale,
            cute.arch.lane_idx(),
        )
        raw_block_scale = block_amax / (Float32(F4_MAX) * tensor_scale)
        block_scale = self._e4m3_scale(raw_block_scale)
        # Preserve TorchAO/MSLK's operation order. At FP4 thresholds,
        # ``1 / (global * block)`` and ``(1 / global) / block`` can round to
        # different E2M1 values even though they are algebraically identical.
        reciprocal = (Float32(1.0) / tensor_scale) / Float32(block_scale)
        packed_row = quantized_packed[row, None]
        for pair in cutlass.range_constexpr((cfg.values_per_lane + 1) // 2):
            value0_idx = pair * 2
            value1_idx = cutlass.min(
                value0_idx + 1,
                cfg.values_per_lane - 1,
            )
            packed = self._quantize_pair(
                values[value0_idx] * reciprocal,
                values[value1_idx] * reciprocal,
            )
            if cutlass.const_expr(cfg.values_per_lane == 1):
                # A one-value lane shares its destination byte with its
                # immediate neighbor. Keep this legal baseline simple; wider
                # candidates use every converter lane and one full-byte store.
                nibble = Uint8(packed & Int16(0xF))
                byte_index = (k_base + value0_idx) // 2
                old = packed_row[byte_index]
                if (k_base + value0_idx) % 2 == 0:
                    packed_row[byte_index] = (old & Uint8(0xF0)) | nibble
                else:
                    packed_row[byte_index] = (old & Uint8(0x0F)) | (nibble << 4)
            else:
                packed_row[(k_base + value0_idx) // 2] = Uint8(packed)
        if lane_in_scale == 0:
            scales[row, scale_block] = block_scale

    @cute.kernel
    def kernel(
        self,
        src: cute.Tensor,
        quantized_packed: cute.Tensor,
        scales: cute.Tensor,
        tensor_scale: cute.Tensor,
    ):
        cfg = self.config
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block_idx, _, _ = cute.arch.block_idx()
        warp_linear = block_idx * cfg.num_warps + warp_idx
        warp_stride = self.grid_ctas * cfg.num_warps
        block_in_warp = lane_idx // cfg.threads_per_scale
        lane_in_scale = lane_idx % cfg.threads_per_scale
        blocks_per_row = self.k // NVFP4_SF_VEC_SIZE
        global_scale = Float32(tensor_scale[0])
        for task_group in cutlass.range(
            warp_linear,
            self.task_groups,
            warp_stride,
            unroll=1,
        ):
            linear_block = task_group * cfg.blocks_per_warp + block_in_warp
            row = linear_block // blocks_per_row
            scale_block = linear_block % blocks_per_row
            self._quantize_task(
                src,
                quantized_packed,
                scales,
                global_scale,
                row,
                scale_block,
                lane_in_scale,
            )


class NVFP4DualQuantKernel(NVFP4QuantKernel):
    """Quantize activation and weight matrices in one persistent launch."""

    def __init__(
        self,
        x_rows: int,
        weight_rows: int,
        k: int,
        config: NVFP4QuantConfig,
    ):
        for rows in (x_rows, weight_rows):
            rejection = config.rejection(rows, k)
            if rejection is not None:
                raise ValueError(f"illegal dual NVFP4 quantizer: {rejection}")
        self.x_rows = x_rows
        self.weight_rows = weight_rows
        self.k = k
        self.config = config
        blocks_per_row = k // NVFP4_SF_VEC_SIZE
        self.x_task_groups = (
            x_rows * blocks_per_row // config.blocks_per_warp
        )
        self.weight_task_groups = (
            weight_rows * blocks_per_row // config.blocks_per_warp
        )
        self.task_groups = self.x_task_groups + self.weight_task_groups
        sm_count = utils.HardwareInfo().get_device_multiprocessor_count()
        self.grid_ctas = min(
            cute.ceil_div(self.task_groups, config.num_warps),
            sm_count * config.persistent_waves,
        )

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        qx: cute.Tensor,
        qw: cute.Tensor,
        sx: cute.Tensor,
        sw: cute.Tensor,
        x_tensor_scale: cute.Tensor,
        weight_tensor_scale: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.dual_kernel(
            x,
            weight,
            qx,
            qw,
            sx,
            sw,
            x_tensor_scale,
            weight_tensor_scale,
        ).launch(
            grid=(self.grid_ctas, 1, 1),
            block=(self.config.num_warps * 32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def dual_kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        qx: cute.Tensor,
        qw: cute.Tensor,
        sx: cute.Tensor,
        sw: cute.Tensor,
        x_tensor_scale: cute.Tensor,
        weight_tensor_scale: cute.Tensor,
    ):
        cfg = self.config
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block_idx, _, _ = cute.arch.block_idx()
        warp_linear = block_idx * cfg.num_warps + warp_idx
        warp_stride = self.grid_ctas * cfg.num_warps
        block_in_warp = lane_idx // cfg.threads_per_scale
        lane_in_scale = lane_idx % cfg.threads_per_scale
        blocks_per_row = self.k // NVFP4_SF_VEC_SIZE
        x_global_scale = Float32(x_tensor_scale[0])
        w_global_scale = Float32(weight_tensor_scale[0])
        for task_group in cutlass.range(
            warp_linear,
            self.task_groups,
            warp_stride,
            unroll=1,
        ):
            if task_group < self.x_task_groups:
                linear_block = (
                    task_group * cfg.blocks_per_warp + block_in_warp
                )
                self._quantize_task(
                    x,
                    qx,
                    sx,
                    x_global_scale,
                    linear_block // blocks_per_row,
                    linear_block % blocks_per_row,
                    lane_in_scale,
                )
            else:
                local_group = task_group - self.x_task_groups
                linear_block = (
                    local_group * cfg.blocks_per_warp + block_in_warp
                )
                self._quantize_task(
                    weight,
                    qw,
                    sw,
                    w_global_scale,
                    linear_block // blocks_per_row,
                    linear_block % blocks_per_row,
                    lane_in_scale,
                )


class NVFP4BlockQuantKernel(NVFP4QuantKernel):
    """Block-only quantizer with the unit outer scale removed from its ABI."""

    @cute.jit
    def __call__(
        self,
        src: cute.Tensor,
        quantized_packed: cute.Tensor,
        scales: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.block_kernel(src, quantized_packed, scales).launch(
            grid=(self.grid_ctas, 1, 1),
            block=(self.config.num_warps * 32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def block_kernel(
        self,
        src: cute.Tensor,
        quantized_packed: cute.Tensor,
        scales: cute.Tensor,
    ):
        cfg = self.config
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block_idx, _, _ = cute.arch.block_idx()
        warp_linear = block_idx * cfg.num_warps + warp_idx
        warp_stride = self.grid_ctas * cfg.num_warps
        block_in_warp = lane_idx // cfg.threads_per_scale
        lane_in_scale = lane_idx % cfg.threads_per_scale
        blocks_per_row = self.k // NVFP4_SF_VEC_SIZE
        for task_group in cutlass.range(
            warp_linear, self.task_groups, warp_stride, unroll=1
        ):
            linear_block = task_group * cfg.blocks_per_warp + block_in_warp
            self._quantize_task(
                src,
                quantized_packed,
                scales,
                Float32(1.0),
                linear_block // blocks_per_row,
                linear_block % blocks_per_row,
                lane_in_scale,
            )


class NVFP4BlockDualQuantKernel(NVFP4DualQuantKernel):
    """Dual block-only quantizer without outer-scale pointer arguments."""

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
        self.block_dual_kernel(x, weight, qx, qw, sx, sw).launch(
            grid=(self.grid_ctas, 1, 1),
            block=(self.config.num_warps * 32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def block_dual_kernel(
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
        block_in_warp = lane_idx // cfg.threads_per_scale
        lane_in_scale = lane_idx % cfg.threads_per_scale
        blocks_per_row = self.k // NVFP4_SF_VEC_SIZE
        for task_group in cutlass.range(
            warp_linear, self.task_groups, warp_stride, unroll=1
        ):
            if task_group < self.x_task_groups:
                linear_block = task_group * cfg.blocks_per_warp + block_in_warp
                self._quantize_task(
                    x,
                    qx,
                    sx,
                    Float32(1.0),
                    linear_block // blocks_per_row,
                    linear_block % blocks_per_row,
                    lane_in_scale,
                )
            else:
                local_group = task_group - self.x_task_groups
                linear_block = local_group * cfg.blocks_per_warp + block_in_warp
                self._quantize_task(
                    weight,
                    qw,
                    sw,
                    Float32(1.0),
                    linear_block // blocks_per_row,
                    linear_block % blocks_per_row,
                    lane_in_scale,
                )


def _fake_bf16(rows: int, k: int):
    return cute.runtime.make_fake_tensor(
        BFloat16, (rows, k), stride=(k, 1), assumed_align=16
    )


def _fake_packed(rows: int, k: int):
    return cute.runtime.make_fake_tensor(
        Uint8, (rows, k // 2), stride=(k // 2, 1), assumed_align=16
    )


def _fake_scales(rows: int, k: int):
    return cute.runtime.make_fake_tensor(
        Float8E4M3FN,
        (rows, k // NVFP4_SF_VEC_SIZE),
        stride=(k // NVFP4_SF_VEC_SIZE, 1),
        assumed_align=16,
    )


def _fake_tensor_scale():
    return cute.runtime.make_fake_tensor(
        Float32, (1,), stride=(1,), assumed_align=4
    )


@lru_cache(maxsize=None)
def compile_nvfp4_quant(
    rows: int,
    k: int,
    config: NVFP4QuantConfig = NVFP4QuantConfig(),
):
    kernel = NVFP4QuantKernel(rows, k, config)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        _fake_bf16(rows, k),
        _fake_packed(rows, k),
        _fake_scales(rows, k),
        _fake_tensor_scale(),
        stream,
        options=(
            "--enable-tvm-ffi --opt-level 3 "
            f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"
        ),
    )


@lru_cache(maxsize=None)
def compile_nvfp4_dual_quant(
    x_rows: int,
    weight_rows: int,
    k: int,
    config: NVFP4QuantConfig = NVFP4QuantConfig(),
):
    kernel = NVFP4DualQuantKernel(x_rows, weight_rows, k, config)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        _fake_bf16(x_rows, k),
        _fake_bf16(weight_rows, k),
        _fake_packed(x_rows, k),
        _fake_packed(weight_rows, k),
        _fake_scales(x_rows, k),
        _fake_scales(weight_rows, k),
        _fake_tensor_scale(),
        _fake_tensor_scale(),
        stream,
        options=(
            "--enable-tvm-ffi --opt-level 3 "
            f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"
        ),
    )


@lru_cache(maxsize=None)
def compile_nvfp4_block_quant(
    rows: int,
    k: int,
    config: NVFP4QuantConfig = NVFP4QuantConfig(),
):
    kernel = NVFP4BlockQuantKernel(rows, k, config)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        _fake_bf16(rows, k),
        _fake_packed(rows, k),
        _fake_scales(rows, k),
        stream,
        options=(
            "--enable-tvm-ffi --opt-level 3 "
            f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"
        ),
    )


@lru_cache(maxsize=None)
def compile_nvfp4_block_dual_quant(
    x_rows: int,
    weight_rows: int,
    k: int,
    config: NVFP4QuantConfig = NVFP4QuantConfig(),
):
    kernel = NVFP4BlockDualQuantKernel(x_rows, weight_rows, k, config)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        _fake_bf16(x_rows, k),
        _fake_bf16(weight_rows, k),
        _fake_packed(x_rows, k),
        _fake_packed(weight_rows, k),
        _fake_scales(x_rows, k),
        _fake_scales(weight_rows, k),
        stream,
        options=(
            "--enable-tvm-ffi --opt-level 3 "
            f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"
        ),
    )


__all__ = [
    "NVFP4DualQuantKernel",
    "NVFP4QuantConfig",
    "NVFP4QuantKernel",
    "compile_nvfp4_dual_quant",
    "compile_nvfp4_block_dual_quant",
    "compile_nvfp4_block_quant",
    "compile_nvfp4_quant",
]
