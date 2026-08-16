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
        self.storage_k = (
            (k + NVFP4_SF_VEC_SIZE - 1)
            // NVFP4_SF_VEC_SIZE
            * NVFP4_SF_VEC_SIZE
        )
        self.config = config
        blocks = rows * (self.storage_k // NVFP4_SF_VEC_SIZE)
        self.task_groups = cute.ceil_div(blocks, config.blocks_per_warp)
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
    def _tensor_scale_from_amax(
        self,
        amax: Float32,
        tensor_scale_mode: cutlass.Constexpr,
    ):
        """Return the delayed outer scale without logarithms or division."""

        if cutlass.const_expr(tensor_scale_mode == "exact"):
            scale = amax * Float32(1.0 / 2688.0)
            if amax == Float32(0.0):
                scale = Float32(1.0)
            return scale

        target = cute.arch.fmax(
            amax * Float32(1.0 / 2688.0),
            Float32(2.0**-126),
            nan=True,
        )
        bits = target.bitcast(Int32)
        exponent_bits = bits & Int32(0x7F800000)
        if (bits & Int32(0x007FFFFF)) != 0:
            exponent_bits += Int32(1 << 23)
        scale = exponent_bits.bitcast(Float32)
        if amax == Float32(0.0):
            scale = Float32(1.0)
        return scale

    @cute.jit
    def _block_reciprocal(
        self,
        block_scale: Float8E4M3FN,
        tensor_scale: Float32,
    ):
        if cutlass.const_expr(self.config.scale_reciprocal == "e4m3_lut"):
            # E4M3 normal values are (1 + mantissa/8) * 2**(exponent-7).
            # Reconstruct the correctly rounded reciprocal from the encoded
            # exponent and an eight-entry mantissa LUT. Block-only execution
            # folds the final division by tensor_scale=1 out of device IR.
            reciprocal = nvvm.inline_ptx_hl(
                "{.reg .u32 code, exp, mant; "
                ".reg .b32 exp_scale, mrec; .reg .f32 tmp; .reg .pred p; "
                "mov.b32 code, {$r0}; "
                "shr.u32 exp, code, 3; and.b32 exp, exp, 15; "
                "sub.u32 exp, 134, exp; shl.b32 exp_scale, exp, 23; "
                "and.b32 mant, code, 7; mov.b32 mrec, 0f3f800000; "
                "setp.eq.u32 p, mant, 1; selp.b32 mrec, 0f3f638e39, mrec, p; "
                "setp.eq.u32 p, mant, 2; selp.b32 mrec, 0f3f4ccccd, mrec, p; "
                "setp.eq.u32 p, mant, 3; selp.b32 mrec, 0f3f3a2e8c, mrec, p; "
                "setp.eq.u32 p, mant, 4; selp.b32 mrec, 0f3f2aaaab, mrec, p; "
                "setp.eq.u32 p, mant, 5; selp.b32 mrec, 0f3f1d89d9, mrec, p; "
                "setp.eq.u32 p, mant, 6; selp.b32 mrec, 0f3f124925, mrec, p; "
                "setp.eq.u32 p, mant, 7; selp.b32 mrec, 0f3f088889, mrec, p; "
                "mul.rn.f32 {$w0}, exp_scale, mrec;}",
                write_only_types=[Float32],
                read_only_args=[Int32(block_scale.bitcast(Uint8))],
            )
            return reciprocal / tensor_scale
        if cutlass.const_expr(self.config.scale_reciprocal == "rcp_approx"):
            denominator = tensor_scale * Float32(block_scale)
            return nvvm.inline_ptx_hl(
                "rcp.approx.f32 {$w0}, {$r0};",
                write_only_types=[Float32],
                read_only_args=[denominator],
            )
        return (Float32(1.0) / tensor_scale) / Float32(block_scale)

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
            global_k_base = k_base + value_base
            if cutlass.const_expr(values_per_load == 1):
                if global_k_base < self.k:
                    values_bf16[value_base] = src_row[global_k_base]
            else:
                load_bytes = cfg.load_bits // 8
                naturally_aligned = (
                    (row * self.k + global_k_base) * (BFloat16.width // 8)
                ) % load_bytes == 0
                if (
                    global_k_base + values_per_load <= self.k
                    and naturally_aligned
                ):
                    loaded = nvvm.load_ext(
                        src_row.iterator + src_row.layout(global_k_base),
                        dtype=Uint16,
                        count=values_per_load,
                    ).bitcast(BFloat16)
                    for vec in cutlass.range_constexpr(values_per_load):
                        values_bf16[value_base + vec] = loaded[vec]
                else:
                    for vec in cutlass.range_constexpr(values_per_load):
                        scalar_k = global_k_base + vec
                        if scalar_k < self.k:
                            values_bf16[value_base + vec] = src_row[scalar_k]

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
        if cutlass.const_expr(cfg.scale_compute == "leader_broadcast"):
            block_scale_bits = Int32(0)
            reciprocal = Float32(0.0)
            if lane_in_scale == 0:
                raw_block_scale = block_amax / (
                    Float32(F4_MAX) * tensor_scale
                )
                leader_scale = self._e4m3_scale(raw_block_scale)
                block_scale_bits = Int32(leader_scale.bitcast(Uint8))
                reciprocal = self._block_reciprocal(
                    leader_scale, tensor_scale
                )
            group_base = cute.arch.lane_idx() & Int32(
                ~(cfg.threads_per_scale - 1)
            )
            block_scale_bits = cute.arch.shuffle_sync(
                block_scale_bits, group_base
            )
            reciprocal = cute.arch.shuffle_sync(reciprocal, group_base)
            block_scale = Uint8(block_scale_bits).bitcast(Float8E4M3FN)
        else:
            raw_block_scale = block_amax / (
                Float32(F4_MAX) * tensor_scale
            )
            block_scale = self._e4m3_scale(raw_block_scale)
            # Preserve TorchAO/MSLK's operation order. At FP4 thresholds,
            # ``1 / (global * block)`` and ``(1 / global) / block`` can round
            # to different E2M1 values despite being algebraically identical.
            reciprocal = self._block_reciprocal(block_scale, tensor_scale)
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
            if cutlass.const_expr(cfg.scale_layout == "row_major"):
                scales[row, scale_block] = block_scale
            else:
                scale_in_tile = scale_block % 8
                physical = (
                    (row % 32) * 16
                    + ((row // 32) % 4) * 4
                    + scale_in_tile % 4
                    + (scale_in_tile // 4) * 512
                )
                scales[row // 128, scale_block // 8, physical] = block_scale
        return block_amax

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
        blocks_per_row = self.storage_k // NVFP4_SF_VEC_SIZE
        global_scale = Float32(tensor_scale[0])
        for task_group in cutlass.range(
            warp_linear,
            self.task_groups,
            warp_stride,
            unroll=1,
        ):
            linear_block = task_group * cfg.blocks_per_warp + block_in_warp
            if linear_block < self.rows * blocks_per_row:
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
        self.storage_k = (
            (k + NVFP4_SF_VEC_SIZE - 1)
            // NVFP4_SF_VEC_SIZE
            * NVFP4_SF_VEC_SIZE
        )
        self.config = config
        blocks_per_row = self.storage_k // NVFP4_SF_VEC_SIZE
        self.x_task_groups = cute.ceil_div(
            x_rows * blocks_per_row, config.blocks_per_warp
        )
        self.weight_task_groups = cute.ceil_div(
            weight_rows * blocks_per_row, config.blocks_per_warp
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
        blocks_per_row = self.storage_k // NVFP4_SF_VEC_SIZE
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
                if linear_block < self.x_rows * blocks_per_row:
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
                if linear_block < self.weight_rows * blocks_per_row:
                    self._quantize_task(
                        weight,
                        qw,
                        sw,
                        w_global_scale,
                        linear_block // blocks_per_row,
                        linear_block % blocks_per_row,
                        lane_in_scale,
                    )


class NVFP4DelayedDualQuantKernel(NVFP4DualQuantKernel):
    """Dual quantizer using prior amax history and emitting its successor.

    Quantization and current-amax observation share the single BF16 read. Each
    warp contributes at most one atomic maximum per operand, while CTA zero
    rotates the immutable older history and publishes the GEMM output scale.
    """

    def __init__(
        self,
        x_rows: int,
        weight_rows: int,
        k: int,
        config: NVFP4QuantConfig,
        history_len: int,
        history_algo: str,
        tensor_scale_mode: str,
    ):
        super().__init__(x_rows, weight_rows, k, config)
        if history_len not in (1, 4, 16, 64):
            raise ValueError("delayed NVFP4 history must have 1, 4, 16, or 64 values")
        if history_algo not in ("most_recent", "window_max"):
            raise ValueError("unknown delayed NVFP4 history algorithm")
        if tensor_scale_mode not in ("power2", "exact"):
            raise ValueError("unknown delayed NVFP4 tensor-scale mode")
        self.history_len = history_len
        self.history_algo = history_algo
        self.tensor_scale_mode = tensor_scale_mode

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        qx: cute.Tensor,
        qw: cute.Tensor,
        sx: cute.Tensor,
        sw: cute.Tensor,
        x_history: cute.Tensor,
        weight_history: cute.Tensor,
        next_x_history: cute.Tensor,
        next_weight_history: cute.Tensor,
        output_scale: cute.Tensor,
        stream: cuda.CUstream,
    ):
        self.delayed_dual_kernel(
            x,
            weight,
            qx,
            qw,
            sx,
            sw,
            x_history,
            weight_history,
            next_x_history,
            next_weight_history,
            output_scale,
        ).launch(
            grid=(self.grid_ctas, 1, 1),
            block=(self.config.num_warps * 32, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def delayed_dual_kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        qx: cute.Tensor,
        qw: cute.Tensor,
        sx: cute.Tensor,
        sw: cute.Tensor,
        x_history: cute.Tensor,
        weight_history: cute.Tensor,
        next_x_history: cute.Tensor,
        next_weight_history: cute.Tensor,
        output_scale: cute.Tensor,
    ):
        cfg = self.config
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block_idx, _, _ = cute.arch.block_idx()

        history_entries = self.history_len
        if cutlass.const_expr(self.history_algo == "most_recent"):
            history_entries = 1
        prior_x_amax = Float32(0.0)
        prior_weight_amax = Float32(0.0)
        for history_idx in cutlass.range(
            lane_idx, history_entries, 32, unroll=1
        ):
            prior_x_amax = cute.arch.fmax(
                prior_x_amax, Float32(x_history[history_idx]), nan=True
            )
            prior_weight_amax = cute.arch.fmax(
                prior_weight_amax,
                Float32(weight_history[history_idx]),
                nan=True,
            )
        prior_x_amax = nvvm.redux_sync(
            prior_x_amax.bitcast(Int32),
            nvvm.ReductionKind.UMAX,
            Int32(0xFFFFFFFF),
        ).bitcast(Float32)
        prior_weight_amax = nvvm.redux_sync(
            prior_weight_amax.bitcast(Int32),
            nvvm.ReductionKind.UMAX,
            Int32(0xFFFFFFFF),
        ).bitcast(Float32)
        x_global_scale = self._tensor_scale_from_amax(
            prior_x_amax, self.tensor_scale_mode
        )
        weight_global_scale = self._tensor_scale_from_amax(
            prior_weight_amax, self.tensor_scale_mode
        )

        if block_idx == 0 and warp_idx == 0:
            if lane_idx == 0:
                output_scale[0] = x_global_scale * weight_global_scale
            for history_idx in cutlass.range(
                lane_idx + 1, self.history_len, 32, unroll=1
            ):
                next_x_history[history_idx] = x_history[history_idx - 1]
                next_weight_history[history_idx] = weight_history[
                    history_idx - 1
                ]

        warp_linear = block_idx * cfg.num_warps + warp_idx
        warp_stride = self.grid_ctas * cfg.num_warps
        block_in_warp = lane_idx // cfg.threads_per_scale
        lane_in_scale = lane_idx % cfg.threads_per_scale
        blocks_per_row = self.storage_k // NVFP4_SF_VEC_SIZE
        observed_x_amax = Float32(0.0)
        observed_weight_amax = Float32(0.0)
        for task_group in cutlass.range(
            warp_linear, self.task_groups, warp_stride, unroll=1
        ):
            if task_group < self.x_task_groups:
                linear_block = task_group * cfg.blocks_per_warp + block_in_warp
                if linear_block < self.x_rows * blocks_per_row:
                    block_amax = self._quantize_task(
                        x,
                        qx,
                        sx,
                        x_global_scale,
                        linear_block // blocks_per_row,
                        linear_block % blocks_per_row,
                        lane_in_scale,
                    )
                    observed_x_amax = cute.arch.fmax(
                        observed_x_amax, block_amax, nan=True
                    )
            else:
                local_group = task_group - self.x_task_groups
                linear_block = local_group * cfg.blocks_per_warp + block_in_warp
                if linear_block < self.weight_rows * blocks_per_row:
                    block_amax = self._quantize_task(
                        weight,
                        qw,
                        sw,
                        weight_global_scale,
                        linear_block // blocks_per_row,
                        linear_block % blocks_per_row,
                        lane_in_scale,
                    )
                    observed_weight_amax = cute.arch.fmax(
                        observed_weight_amax, block_amax, nan=True
                    )

        observed_x_amax = nvvm.redux_sync(
            observed_x_amax.bitcast(Int32),
            nvvm.ReductionKind.UMAX,
            Int32(0xFFFFFFFF),
        ).bitcast(Float32)
        observed_weight_amax = nvvm.redux_sync(
            observed_weight_amax.bitcast(Int32),
            nvvm.ReductionKind.UMAX,
            Int32(0xFFFFFFFF),
        ).bitcast(Float32)
        if lane_idx == 0:
            cute.arch.atomic_fmax(
                next_x_history.iterator + next_x_history.layout(0),
                observed_x_amax,
                sign_bit=False,
                scope="gpu",
            )
            cute.arch.atomic_fmax(
                next_weight_history.iterator + next_weight_history.layout(0),
                observed_weight_amax,
                sign_bit=False,
                scope="gpu",
            )


class NVFP4JITRegionDualQuantKernel(NVFP4DualQuantKernel):
    """Observe and quantize independently scaled row regions in one launch.

    One CTA owns a complete region, so its current amax needs no inter-CTA
    communication.  The first pass is a cooperative BF16 read/reduction; the
    second pass reuses the warm cache lines while producing packed E2M1 values
    and native E4M3 1x16 scales.  The compact FP32 scale array is consumed by
    the GEMM epilogue and never materialized through eager Torch operations.
    """

    def __init__(
        self,
        x_rows: int,
        weight_rows: int,
        k: int,
        config: NVFP4QuantConfig,
        x_region_rows: int,
        weight_region_rows: int,
        tensor_scale_mode: str,
        amax_load_bits: int,
        amax_unroll: int,
        region_waves: int,
        region_order: str,
        region_ownership: str,
        use_pdl: bool = False,
    ):
        super().__init__(x_rows, weight_rows, k, config)
        if x_region_rows < 1 or weight_region_rows < 1:
            raise ValueError("JIT NVFP4 region rows must be positive")
        if tensor_scale_mode not in ("power2", "exact"):
            raise ValueError("unknown JIT NVFP4 tensor-scale mode")
        if amax_load_bits not in (16, 32, 64, 128):
            raise ValueError("JIT NVFP4 amax loads must be 16, 32, 64, or 128 bits")
        if amax_unroll not in (1, 2, 4, 8):
            raise ValueError("JIT NVFP4 amax unroll must be 1, 2, 4, or 8")
        if region_waves not in (1, 2, 3, 4, 6, 8):
            raise ValueError("JIT NVFP4 region waves must be 1, 2, 3, 4, 6, or 8")
        if region_order not in ("x_first", "weight_first"):
            raise ValueError("unknown JIT NVFP4 region order")
        if region_ownership not in ("warp", "cta"):
            raise ValueError("unknown JIT NVFP4 region ownership")
        self.x_region_rows = x_region_rows
        self.weight_region_rows = weight_region_rows
        self.x_regions = cute.ceil_div(x_rows, x_region_rows)
        self.weight_regions = cute.ceil_div(weight_rows, weight_region_rows)
        self.total_regions = self.x_regions + self.weight_regions
        self.tensor_scale_mode = tensor_scale_mode
        self.amax_load_bits = amax_load_bits
        self.amax_unroll = amax_unroll
        self.region_order = region_order
        self.region_ownership = region_ownership
        self.use_pdl = use_pdl
        sm_count = utils.HardwareInfo().get_device_multiprocessor_count()
        natural_ctas = (
            cute.ceil_div(self.total_regions, config.num_warps)
            if region_ownership == "warp"
            else self.total_regions
        )
        self.grid_ctas = min(natural_ctas, sm_count * region_waves)

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        qx: cute.Tensor,
        qw: cute.Tensor,
        sx: cute.Tensor,
        sw: cute.Tensor,
        region_scales: cute.Tensor,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            warp_amax: cute.struct.Align[
                cute.struct.MemRange[Float32, self.config.num_warps], 16
            ]
            outer_scale: cute.struct.Align[
                cute.struct.MemRange[Float32, 1], 4
            ]

        self.shared_storage = SharedStorage
        if cutlass.const_expr(self.region_ownership == "warp"):
            self.jit_region_warp_dual_kernel(
                x, weight, qx, qw, sx, sw, region_scales
            ).launch(
                grid=(self.grid_ctas, 1, 1),
                block=(self.config.num_warps * 32, 1, 1),
                stream=stream,
                use_pdl=self.use_pdl,
            )
        else:
            self.jit_region_dual_kernel(
                x, weight, qx, qw, sx, sw, region_scales
            ).launch(
                grid=(self.grid_ctas, 1, 1),
                block=(self.config.num_warps * 32, 1, 1),
                stream=stream,
                use_pdl=self.use_pdl,
            )

    @cute.jit
    def _region_amax(
        self,
        src: cute.Tensor,
        row_begin: Int32,
        row_end: Int32,
        warp_amax: cute.Tensor,
        outer_scale: cute.Tensor,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        threads = self.config.num_warps * 32
        values_per_load = self.amax_load_bits // BFloat16.width
        region_values = (row_end - row_begin) * self.k
        src_flat = cute.make_tensor(
            src.iterator,
            cute.make_layout((cute.size(src),), stride=(1,)),
        )
        local_amax = Float32(0.0)
        for value_base in cutlass.range(
            tidx * values_per_load,
            region_values,
            threads * values_per_load,
            unroll=self.amax_unroll,
        ):
            global_value = row_begin * self.k + value_base
            remaining = region_values - value_base
            load_bytes = self.amax_load_bits // 8
            aligned = (global_value * (BFloat16.width // 8)) % load_bytes == 0
            if cutlass.const_expr(values_per_load == 1):
                magnitude = (
                    Float32(src_flat[global_value]).bitcast(Int32)
                    & Int32(0x7FFFFFFF)
                ).bitcast(Float32)
                local_amax = cute.arch.fmax(local_amax, magnitude, nan=True)
            elif remaining >= values_per_load and aligned:
                loaded = nvvm.load_ext(
                    src_flat.iterator + src_flat.layout(global_value),
                    dtype=Uint16,
                    count=values_per_load,
                ).bitcast(BFloat16)
                for vec in cutlass.range_constexpr(values_per_load):
                    magnitude = (
                        Float32(loaded[vec]).bitcast(Int32)
                        & Int32(0x7FFFFFFF)
                    ).bitcast(Float32)
                    local_amax = cute.arch.fmax(local_amax, magnitude, nan=True)
            else:
                for vec in cutlass.range_constexpr(values_per_load):
                    if value_base + vec < region_values:
                        magnitude = (
                            Float32(src_flat[global_value + vec]).bitcast(Int32)
                            & Int32(0x7FFFFFFF)
                        ).bitcast(Float32)
                        local_amax = cute.arch.fmax(
                            local_amax, magnitude, nan=True
                        )
        warp_max = nvvm.redux_sync(
            local_amax.bitcast(Int32),
            nvvm.ReductionKind.UMAX,
            Int32(0xFFFFFFFF),
        ).bitcast(Float32)
        if lane_idx == 0:
            warp_amax[warp_idx] = warp_max
        cute.arch.sync_threads()
        if warp_idx == 0:
            cta_max = Float32(0.0)
            if lane_idx < self.config.num_warps:
                cta_max = Float32(warp_amax[lane_idx])
            cta_max = nvvm.redux_sync(
                cta_max.bitcast(Int32),
                nvvm.ReductionKind.UMAX,
                Int32(0xFFFFFFFF),
            ).bitcast(Float32)
            if lane_idx == 0:
                outer_scale[0] = self._tensor_scale_from_amax(
                    cta_max, self.tensor_scale_mode
                )
        cute.arch.sync_threads()

    @cute.jit
    def _quantize_region(
        self,
        src: cute.Tensor,
        packed: cute.Tensor,
        scales: cute.Tensor,
        row_begin: Int32,
        row_end: Int32,
        tensor_scale: Float32,
    ):
        cfg = self.config
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        block_in_warp = lane_idx // cfg.threads_per_scale
        lane_in_scale = lane_idx % cfg.threads_per_scale
        blocks_per_row = self.storage_k // NVFP4_SF_VEC_SIZE
        region_blocks = (row_end - row_begin) * blocks_per_row
        for task_group in cutlass.range(
            warp_idx,
            cute.ceil_div(region_blocks, cfg.blocks_per_warp),
            cfg.num_warps,
            unroll=1,
        ):
            local_block = task_group * cfg.blocks_per_warp + block_in_warp
            if local_block < region_blocks:
                self._quantize_task(
                    src,
                    packed,
                    scales,
                    tensor_scale,
                    row_begin + local_block // blocks_per_row,
                    local_block % blocks_per_row,
                    lane_in_scale,
                )

    @cute.jit
    def _warp_region_amax(
        self,
        src: cute.Tensor,
        row_begin: Int32,
        row_end: Int32,
    ):
        lane_idx = cute.arch.lane_idx()
        values_per_load = self.amax_load_bits // BFloat16.width
        region_values = (row_end - row_begin) * self.k
        src_flat = cute.make_tensor(
            src.iterator,
            cute.make_layout((cute.size(src),), stride=(1,)),
        )
        local_amax = Float32(0.0)
        for value_base in cutlass.range(
            lane_idx * values_per_load,
            region_values,
            32 * values_per_load,
            unroll=self.amax_unroll,
        ):
            global_value = row_begin * self.k + value_base
            remaining = region_values - value_base
            load_bytes = self.amax_load_bits // 8
            aligned = (global_value * (BFloat16.width // 8)) % load_bytes == 0
            if cutlass.const_expr(values_per_load == 1):
                magnitude = (
                    Float32(src_flat[global_value]).bitcast(Int32)
                    & Int32(0x7FFFFFFF)
                ).bitcast(Float32)
                local_amax = cute.arch.fmax(local_amax, magnitude, nan=True)
            elif remaining >= values_per_load and aligned:
                loaded = nvvm.load_ext(
                    src_flat.iterator + src_flat.layout(global_value),
                    dtype=Uint16,
                    count=values_per_load,
                ).bitcast(BFloat16)
                for vec in cutlass.range_constexpr(values_per_load):
                    magnitude = (
                        Float32(loaded[vec]).bitcast(Int32)
                        & Int32(0x7FFFFFFF)
                    ).bitcast(Float32)
                    local_amax = cute.arch.fmax(local_amax, magnitude, nan=True)
            else:
                for vec in cutlass.range_constexpr(values_per_load):
                    if value_base + vec < region_values:
                        magnitude = (
                            Float32(src_flat[global_value + vec]).bitcast(Int32)
                            & Int32(0x7FFFFFFF)
                        ).bitcast(Float32)
                        local_amax = cute.arch.fmax(
                            local_amax, magnitude, nan=True
                        )
        region_amax = nvvm.redux_sync(
            local_amax.bitcast(Int32),
            nvvm.ReductionKind.UMAX,
            Int32(0xFFFFFFFF),
        ).bitcast(Float32)
        scale = Float32(0.0)
        if lane_idx == 0:
            scale = self._tensor_scale_from_amax(
                region_amax, self.tensor_scale_mode
            )
        return cute.arch.shuffle_sync(scale, 0)

    @cute.jit
    def _quantize_region_warp(
        self,
        src: cute.Tensor,
        packed: cute.Tensor,
        scales: cute.Tensor,
        row_begin: Int32,
        row_end: Int32,
        tensor_scale: Float32,
    ):
        cfg = self.config
        lane_idx = cute.arch.lane_idx()
        block_in_warp = lane_idx // cfg.threads_per_scale
        lane_in_scale = lane_idx % cfg.threads_per_scale
        blocks_per_row = self.storage_k // NVFP4_SF_VEC_SIZE
        region_blocks = (row_end - row_begin) * blocks_per_row
        for task_group in cutlass.range(
            cute.ceil_div(region_blocks, cfg.blocks_per_warp), unroll=1
        ):
            local_block = task_group * cfg.blocks_per_warp + block_in_warp
            if local_block < region_blocks:
                self._quantize_task(
                    src,
                    packed,
                    scales,
                    tensor_scale,
                    row_begin + local_block // blocks_per_row,
                    local_block % blocks_per_row,
                    lane_in_scale,
                )

    @cute.kernel
    def jit_region_warp_dual_kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        qx: cute.Tensor,
        qw: cute.Tensor,
        sx: cute.Tensor,
        sw: cute.Tensor,
        region_scales: cute.Tensor,
    ):
        block_idx, _, _ = cute.arch.block_idx()
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_idx = cute.arch.lane_idx()
        warp_linear = block_idx * self.config.num_warps + warp_idx
        warp_stride = self.grid_ctas * self.config.num_warps
        for region_task in cutlass.range(
            warp_linear, self.total_regions, warp_stride, unroll=1
        ):
            is_x = region_task < self.x_regions
            local_region = region_task
            scale_index = region_task
            if cutlass.const_expr(self.region_order == "weight_first"):
                is_x = region_task >= self.weight_regions
                if is_x:
                    local_region = region_task - self.weight_regions
                    scale_index = local_region
                else:
                    local_region = region_task
                    scale_index = self.x_regions + local_region
            elif not is_x:
                local_region = region_task - self.x_regions

            if is_x:
                row_begin = local_region * self.x_region_rows
                row_end = cutlass.min(row_begin + self.x_region_rows, self.x_rows)
                scale = self._warp_region_amax(x, row_begin, row_end)
                if lane_idx == 0:
                    region_scales[scale_index] = scale
                self._quantize_region_warp(
                    x, qx, sx, row_begin, row_end, scale
                )
            else:
                row_begin = local_region * self.weight_region_rows
                row_end = cutlass.min(
                    row_begin + self.weight_region_rows, self.weight_rows
                )
                scale = self._warp_region_amax(weight, row_begin, row_end)
                if lane_idx == 0:
                    region_scales[scale_index] = scale
                self._quantize_region_warp(
                    weight, qw, sw, row_begin, row_end, scale
                )
        if cutlass.const_expr(self.use_pdl):
            if nvvm.elect_sync():
                nvvm.griddepcontrol("launch_dependents")

    @cute.kernel
    def jit_region_dual_kernel(
        self,
        x: cute.Tensor,
        weight: cute.Tensor,
        qx: cute.Tensor,
        qw: cute.Tensor,
        sx: cute.Tensor,
        sw: cute.Tensor,
        region_scales: cute.Tensor,
    ):
        block_idx, _, _ = cute.arch.block_idx()
        storage = cutlass.utils.SmemAllocator().allocate(self.shared_storage)
        warp_amax = storage.warp_amax.get_tensor(
            cute.make_layout((self.config.num_warps,), stride=(1,))
        )
        outer_scale = storage.outer_scale.get_tensor(
            cute.make_layout((1,), stride=(1,))
        )
        for region_task in cutlass.range(
            block_idx, self.total_regions, self.grid_ctas, unroll=1
        ):
            is_x = region_task < self.x_regions
            local_region = region_task
            scale_index = region_task
            if cutlass.const_expr(self.region_order == "weight_first"):
                is_x = region_task >= self.weight_regions
                if is_x:
                    local_region = region_task - self.weight_regions
                    scale_index = local_region
                else:
                    local_region = region_task
                    scale_index = self.x_regions + local_region
            elif not is_x:
                local_region = region_task - self.x_regions

            if is_x:
                row_begin = local_region * self.x_region_rows
                row_end = cutlass.min(row_begin + self.x_region_rows, self.x_rows)
                self._region_amax(
                    x, row_begin, row_end, warp_amax, outer_scale
                )
                scale = Float32(outer_scale[0])
                if cute.arch.thread_idx()[0] == 0:
                    region_scales[scale_index] = scale
                self._quantize_region(x, qx, sx, row_begin, row_end, scale)
            else:
                row_begin = local_region * self.weight_region_rows
                row_end = cutlass.min(
                    row_begin + self.weight_region_rows, self.weight_rows
                )
                self._region_amax(
                    weight,
                    row_begin,
                    row_end,
                    warp_amax,
                    outer_scale,
                )
                scale = Float32(outer_scale[0])
                if cute.arch.thread_idx()[0] == 0:
                    region_scales[scale_index] = scale
                self._quantize_region(weight, qw, sw, row_begin, row_end, scale)
            cute.arch.sync_threads()
        if cutlass.const_expr(self.use_pdl):
            if nvvm.elect_sync():
                nvvm.griddepcontrol("launch_dependents")


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
        blocks_per_row = self.storage_k // NVFP4_SF_VEC_SIZE
        for task_group in cutlass.range(
            warp_linear, self.task_groups, warp_stride, unroll=1
        ):
            linear_block = task_group * cfg.blocks_per_warp + block_in_warp
            if linear_block < self.rows * blocks_per_row:
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
        blocks_per_row = self.storage_k // NVFP4_SF_VEC_SIZE
        for task_group in cutlass.range(
            warp_linear, self.task_groups, warp_stride, unroll=1
        ):
            if task_group < self.x_task_groups:
                linear_block = task_group * cfg.blocks_per_warp + block_in_warp
                if linear_block < self.x_rows * blocks_per_row:
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
                if linear_block < self.weight_rows * blocks_per_row:
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
    storage_k = (
        (k + NVFP4_SF_VEC_SIZE - 1)
        // NVFP4_SF_VEC_SIZE
        * NVFP4_SF_VEC_SIZE
    )
    return cute.runtime.make_fake_tensor(
        Uint8,
        (rows, storage_k // 2),
        stride=(storage_k // 2, 1),
        assumed_align=16,
    )


def _fake_scales(rows: int, k: int, scale_layout: str = "row_major"):
    if scale_layout == "row_major":
        storage_k = (
            (k + NVFP4_SF_VEC_SIZE - 1)
            // NVFP4_SF_VEC_SIZE
            * NVFP4_SF_VEC_SIZE
        )
        return cute.runtime.make_fake_tensor(
            Float8E4M3FN,
            (rows, storage_k // NVFP4_SF_VEC_SIZE),
            stride=(storage_k // NVFP4_SF_VEC_SIZE, 1),
            assumed_align=16,
        )
    return cute.runtime.make_fake_tensor(
        Float8E4M3FN,
        (rows // 128, k // 128, 1024),
        stride=(k // 128 * 1024, 1024, 1),
        assumed_align=16,
    )


def _fake_tensor_scale():
    return cute.runtime.make_fake_tensor(
        Float32, (1,), stride=(1,), assumed_align=4
    )


def _fake_amax_history(values: int):
    return cute.runtime.make_fake_tensor(
        Float32, (values,), stride=(1,), assumed_align=4
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
        _fake_scales(rows, k, config.scale_layout),
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
        _fake_scales(x_rows, k, config.scale_layout),
        _fake_scales(weight_rows, k, config.scale_layout),
        _fake_tensor_scale(),
        _fake_tensor_scale(),
        stream,
        options=(
            "--enable-tvm-ffi --opt-level 3 "
            f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"
        ),
    )


@lru_cache(maxsize=None)
def compile_nvfp4_delayed_dual_quant(
    x_rows: int,
    weight_rows: int,
    k: int,
    config: NVFP4QuantConfig = NVFP4QuantConfig(),
    history_len: int = 16,
    history_algo: str = "window_max",
    tensor_scale_mode: str = "power2",
):
    kernel = NVFP4DelayedDualQuantKernel(
        x_rows,
        weight_rows,
        k,
        config,
        history_len,
        history_algo,
        tensor_scale_mode,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    history = _fake_amax_history(history_len)
    return cute.compile(
        kernel,
        _fake_bf16(x_rows, k),
        _fake_bf16(weight_rows, k),
        _fake_packed(x_rows, k),
        _fake_packed(weight_rows, k),
        _fake_scales(x_rows, k, config.scale_layout),
        _fake_scales(weight_rows, k, config.scale_layout),
        history,
        history,
        history,
        history,
        _fake_tensor_scale(),
        stream,
        options=(
            "--enable-tvm-ffi --opt-level 3 "
            f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"
        ),
    )


@lru_cache(maxsize=None)
def compile_nvfp4_jit_region_dual_quant(
    x_rows: int,
    weight_rows: int,
    k: int,
    config: NVFP4QuantConfig = NVFP4QuantConfig(),
    x_region_rows: int = 1,
    weight_region_rows: int = 1,
    tensor_scale_mode: str = "power2",
    amax_load_bits: int = 128,
    amax_unroll: int = 1,
    region_waves: int = 4,
    region_order: str = "x_first",
    region_ownership: str = "warp",
    use_pdl: bool = False,
):
    kernel = NVFP4JITRegionDualQuantKernel(
        x_rows,
        weight_rows,
        k,
        config,
        x_region_rows,
        weight_region_rows,
        tensor_scale_mode,
        amax_load_bits,
        amax_unroll,
        region_waves,
        region_order,
        region_ownership,
        use_pdl,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    region_scales = cute.runtime.make_fake_tensor(
        Float32,
        (
            cute.ceil_div(x_rows, x_region_rows)
            + cute.ceil_div(weight_rows, weight_region_rows),
        ),
        stride=(1,),
        assumed_align=4,
    )
    return cute.compile(
        kernel,
        _fake_bf16(x_rows, k),
        _fake_bf16(weight_rows, k),
        _fake_packed(x_rows, k),
        _fake_packed(weight_rows, k),
        _fake_scales(x_rows, k, config.scale_layout),
        _fake_scales(weight_rows, k, config.scale_layout),
        region_scales,
        stream,
        options=(
            "--enable-tvm-ffi --opt-level 3 "
            f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"
        ),
    )


@lru_cache(maxsize=None)
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
        _fake_scales(rows, k, config.scale_layout),
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
        _fake_scales(x_rows, k, config.scale_layout),
        _fake_scales(weight_rows, k, config.scale_layout),
        stream,
        options=(
            "--enable-tvm-ffi --opt-level 3 "
            f"--ptxas-options '-O3 -v --maxrregcount={config.maxrregcount}'"
        ),
    )


__all__ = [
    "NVFP4DelayedDualQuantKernel",
    "NVFP4DualQuantKernel",
    "NVFP4JITRegionDualQuantKernel",
    "NVFP4QuantConfig",
    "NVFP4QuantKernel",
    "compile_nvfp4_dual_quant",
    "compile_nvfp4_delayed_dual_quant",
    "compile_nvfp4_jit_region_dual_quant",
    "compile_nvfp4_block_dual_quant",
    "compile_nvfp4_block_quant",
    "compile_nvfp4_quant",
]
