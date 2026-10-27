"""Experimental scale-only post-hoc alignment, not a public backward policy.

The producer supplies corrected BF16 scales and their FP32 maximum. Uniform
samples are caller-owned: RNG generation/replay is deliberately not implicit.
No packed operand is read or modified by this kernel.
"""
from dataclasses import dataclass
from functools import lru_cache
import os

os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")
import cuda.bindings.driver as cuda
import cutlass
import cutlass.utils
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Float8E4M3FN, Int16, Int32, Uint8, Uint32
from cutlass.experimental.primitives import nvvm_wrapper as nvvm
from cutlass.cute.nvgpu import cpasync
from rtx.kernels.nvfp4_scale_bits import producer_scale_rtz4


@dataclass(frozen=True)
class PosthocScaleConfig:
    threads: int = 128
    blocks: int = 128
    layout: str = "row_major"
    probability_math: str = "divide"

    def __post_init__(self):
        if self.threads not in (64,128,256,512) or type(self.blocks) is not int or self.blocks < 1:
            raise ValueError("invalid post-hoc launch geometry")
        if self.layout not in ("row_major", "mma128"):
            raise ValueError("invalid post-hoc scale layout")
        if self.probability_math not in ("divide", "power2"):
            raise ValueError("invalid probability math")


class PosthocScaleKernel:
    def __init__(self, rows, groups, config, partial_count=1):
        if rows <= 0 or groups <= 0:
            raise ValueError("positive scale shape required")
        if config.layout == "mma128" and (rows % 128 or groups % 8):
            raise ValueError("mma128 requires 128 rows and 8 groups")
        self.rows, self.groups, self.config = rows, groups, config
        if partial_count < 1:
            raise ValueError("positive partial maximum count required")
        self.partial_count = partial_count
        self.blocks = min(config.blocks, (rows*groups+config.threads-1)//config.threads)

    @cute.jit
    def sample_uniform(self, uniforms: cute.Tensor, index):
        # Overridden by the explicit-seed Philox finalizer. Keep caller-owned
        # uniform semantics here, including its unchanged conversion grid.
        return uniforms[index]

    @cute.jit
    def tensor_scale(self,maximum:cute.Tensor):
        largest=Float32(0.)
        if cutlass.const_expr(self.partial_count==1):
            largest=maximum[0]
        else:
            lane=cute.arch.lane_idx()
            for j in cutlass.range(lane,self.partial_count,32,unroll=1):
                largest=cute.arch.fmax(largest,maximum[j])
            for level in cutlass.range_constexpr(5):
                largest=cute.arch.fmax(largest,cute.arch.shuffle_sync_bfly(largest,16 >> level))
        scale=Float32(1.)
        if largest>Float32(0.):
            scale=largest*Float32(1./448.)
        return scale

    @cute.jit
    def rounding_probability(self,value,low,high):
        probability=Float32(0.)
        if high > low:
            numerator=value-low
            gap=high-low
            if cutlass.const_expr(self.config.probability_math=="power2"):
                bits=gap.bitcast(Int32)
                # Adjacent positive E4M3 gaps are exactly 2^-9 through 2^5.
                # Keep division for every other gap instead of approximating.
                if ((bits & Int32(0x007fffff))==0 and
                    bits>=Int32(0x3b000000) and bits<=Int32(0x42000000)):
                    inverse=(Int32(0x7f000000)-bits).bitcast(Float32)
                    # Exact reciprocal, one RN operation, no FMA contraction.
                    probability=nvvm.inline_ptx_hl(
                        "mul.rn.f32 {$w0}, {$r0}, {$r1};",
                        write_only_types=[Float32],read_only_args=[numerator,inverse])
                else:
                    probability=numerator/gap
            else:
                probability=numerator/gap
        return probability

    @cute.jit
    def finalize_one(self,src:cute.Tensor,dst:cute.Tensor,i,scale,uniform):
        value = cute.arch.fmin(src[i].to(Float32)/scale, Float32(448.))
        packed = nvvm.inline_ptx_hl(
            "cvt.rn.satfinite.e4m3x2.f32 {$w0}, {$r0}, {$r0};",
            write_only_types=[Int16], read_only_args=[value])
        code = Int32(packed & Int16(255))
        nearest = Uint8(code).bitcast(Float8E4M3FN).to(Float32)
        lo, hi = code, code
        if nearest > value:
            lo = code-1
        if nearest < value:
            hi = code+1
        low = Uint8(lo).bitcast(Float8E4M3FN).to(Float32)
        high = Uint8(hi).bitcast(Float8E4M3FN).to(Float32)
        probability = self.rounding_probability(value,low,high)
        selected = lo
        if uniform < probability:
            selected = hi
        offset = i
        if cutlass.const_expr(self.config.layout == "mma128"):
            row, group = i//self.groups, i%self.groups
            offset = ((row//128)*(self.groups//8)+group//8)*1024
            offset += (group%8//4)*512+(row%32)*16+(row//32%4)*4+group%4
        dst[offset] = Uint8(selected).bitcast(Float8E4M3FN)

    @cute.jit
    def __call__(self, src: cute.Tensor, maximum: cute.Tensor, uniforms: cute.Tensor,
                 dst: cute.Tensor, outer: cute.Tensor, stream: cuda.CUstream):
        self.kernel(src, maximum, uniforms, dst, outer).launch(
            grid=(self.blocks,1,1), block=(self.config.threads,1,1), stream=stream)

    @cute.kernel
    def kernel(self, src: cute.Tensor, maximum: cute.Tensor, uniforms: cute.Tensor,
               dst: cute.Tensor, outer: cute.Tensor):
        thread,_,_ = cute.arch.thread_idx()
        block,_,_ = cute.arch.block_idx()
        scale=self.tensor_scale(maximum)
        index = block*self.config.threads+thread
        if index == 0:
            outer[0] = scale
        for i in cutlass.range(index, self.rows*self.groups,
                               self.blocks*self.config.threads, unroll=1):
            self.finalize_one(src,dst,i,scale,self.sample_uniform(uniforms,i))


@lru_cache(maxsize=64)
def compile_posthoc_scales(rows, groups, config=PosthocScaleConfig(), partial_count=1):
    kernel = PosthocScaleKernel(rows,groups,config,partial_count)
    def fake(dtype,size):
        return cute.runtime.make_fake_tensor(dtype,(size,),stride=(1,),assumed_align=4)
    return cute.compile(kernel, fake(BFloat16,rows*groups), fake(Float32,partial_count),
        fake(Float32,rows*groups), fake(Float8E4M3FN,rows*groups), fake(Float32,1),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 3")


class PartialMaximumKernel:
    """Optional staged reduction of nonnegative FP32 producer maxima."""
    def __init__(self, count):
        if type(count) is not int or count < 1:
            raise ValueError("positive partial count required")
        self.count = count

    @cute.jit
    def __call__(self, partials: cute.Tensor, maximum: cute.Tensor, stream: cuda.CUstream):
        self.kernel(partials, maximum).launch(grid=(1,1,1),block=(32,1,1),stream=stream)

    @cute.kernel
    def kernel(self, partials: cute.Tensor, maximum: cute.Tensor):
        lane=cute.arch.lane_idx()
        largest=Float32(0.)
        for index in cutlass.range(lane,self.count,32,unroll=1):
            largest=cute.arch.fmax(largest,partials[index])
        for level in cutlass.range_constexpr(5):
            largest=cute.arch.fmax(largest,cute.arch.shuffle_sync_bfly(largest,16 >> level))
        if lane == 0:
            maximum[0]=largest


@lru_cache(maxsize=64)
def compile_partial_maximum(count):
    src=cute.runtime.make_fake_tensor(Float32,(count,),stride=(1,),assumed_align=4)
    dst=cute.runtime.make_fake_tensor(Float32,(1,),stride=(1,),assumed_align=4)
    return cute.compile(PartialMaximumKernel(count),src,dst,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 3")


class ExtendedPreparationKernel:
    """First pass over already rotated FP32 input; no global atomics/reset."""
    def __init__(self, rows, groups, config, mantissa_bits):
        if rows <= 0 or groups <= 0 or mantissa_bits not in (3,4):
            raise ValueError("invalid extended preparation shape or mantissa policy")
        self.count = rows*groups
        self.config, self.mantissa_bits = config,mantissa_bits
        self.blocks = min(config.blocks,(self.count+config.threads-1)//config.threads)
        self.partial_count = self.blocks*(config.threads//32)

    @cute.jit
    def __call__(self, src: cute.Tensor, packed: cute.Tensor, scales: cute.Tensor,
                 partials: cute.Tensor, stream: cuda.CUstream):
        self.kernel(src,packed,scales,partials).launch(grid=(self.blocks,1,1),
            block=(self.config.threads,1,1),stream=stream)

    @cute.jit
    def decode(self, code: Int32):
        magnitude = (Uint32(0xC8643210) >> Uint32((code & 7)*4)) & Uint32(15)
        value = Float32(magnitude)*Float32(.5)
        if (code & 8) != 0:
            value = -value
        return value

    @cute.jit
    def decode_pair(self, bits: Int32, mode: cutlass.Constexpr):
        if cutlass.const_expr(mode == "native"):
            # All E2M1 values are exactly representable in FP16 and FP32,
            # including negative zero. Only the decode goes through FP16;
            # Hadamard, correction products and reductions remain FP32.
            qa, qb = nvvm.inline_ptx_hl(
                "{.reg .b8 b0,b1,b2,b3; .reg .b32 h2; .reg .b16 h0,h1; "
                "mov.b32 {b0,b1,b2,b3}, {$r0}; "
                "cvt.rn.f16x2.e2m1x2 h2, b0; "
                "mov.b32 {h0,h1}, h2; "
                "cvt.f32.f16 {$w0}, h0; cvt.f32.f16 {$w1}, h1;}",
                write_only_types=[Float32, Float32], read_only_args=[bits])
        else:
            qa, qb = self.decode(bits & 15), self.decode((bits >> 4) & 15)
        return qa, qb

    @cute.kernel
    def kernel(self, src: cute.Tensor, packed: cute.Tensor, scales: cute.Tensor, partials: cute.Tensor):
        thread,_,_ = cute.arch.thread_idx()
        block,_,_ = cute.arch.block_idx()
        largest = Float32(0.)
        for i in cutlass.range(block*self.config.threads+thread,self.count,
                               self.blocks*self.config.threads,unroll=1):
            values = cute.make_rmem_tensor((16,),Float32)
            maximum = Float32(0.)
            for j in cutlass.range_constexpr(16):
                values[j] = src[i*16+j]
                absolute = (values[j].bitcast(Int32) & Int32(0x7fffffff)).bitcast(Float32)
                maximum = cute.arch.fmax(maximum,absolute)
            scale = maximum / Float32(6.)
            scale = (scale.bitcast(Int32) & Int32(~((1 << (23-self.mantissa_bits))-1))).bitcast(Float32)
            if scale < Float32(1.1754943508222875e-38):
                scale = Float32(0.)
            xx, xy = Float32(0.), Float32(0.)
            for j in cutlass.range_constexpr(8):
                a,b = Float32(0.),Float32(0.)
                if scale > Float32(0.):
                    a,b = values[j*2]/scale,values[j*2+1]/scale
                bits = nvvm.inline_ptx_hl(
                    "{.reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, {$r1}, {$r0}; mov.b16 {$w0}, {b, 0};}",
                    write_only_types=[Int16],read_only_args=[a,b])
                packed[i*8+j] = Uint8(bits)
                qa,qb = self.decode(Int32(bits)&15),self.decode((Int32(bits)>>4)&15)
                xx += a*a+b*b
                xy += a*qa+b*qb
            correction = Float32(1.)
            if xy > Float32(0.):
                correction = xx/xy
            corrected = (scale*correction).to(BFloat16)
            scales[i] = corrected
            largest = cute.arch.fmax(largest,corrected.to(Float32))
        for level in cutlass.range_constexpr(5):
            largest = cute.arch.fmax(largest,cute.arch.shuffle_sync_bfly(largest,16 >> level))
        if thread%32 == 0:
            partials[block*(self.config.threads//32)+thread//32] = largest


@lru_cache(maxsize=64)
def compile_extended_preparation(rows, groups, config=PosthocScaleConfig(), mantissa_bits=4):
    kernel = ExtendedPreparationKernel(rows,groups,config,mantissa_bits)
    def fake(dtype,size):
        return cute.runtime.make_fake_tensor(dtype,(size,),stride=(1,),assumed_align=4)
    compiled = cute.compile(kernel,fake(Float32,rows*groups*16),fake(Uint8,rows*groups*8),
        fake(BFloat16,rows*groups),fake(Float32,kernel.partial_count),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 3")
    return compiled,kernel.partial_count


class HadamardPreparationKernel(ExtendedPreparationKernel):
    """Warp Hadamard-128 plus EDEN preparation from original BF16 storage.

    Caller supplies 128 shared Rademacher signs. Both GEMM operands must use
    the same transform. This is a numerical policy, not a scheduling alias.
    Tile order changes warp ownership only. Reduction-major visits neighboring
    logical rows together, encouraging cross-warp source-line reuse for logical
    transpose; it does not coalesce each warp's strided loads or transpose data.
    """
    def __init__(self, rows, cols, config, mantissa_bits, transpose, packed_source=False, tile_order="row_major", load_mode="direct", signs_mode="reload", stages=1, async_copy=False, correction_decode="lut", partial_max_scope="warp", normalization_math="divide", producer_copy_bits=32, producer_unroll_rows=True, producer_row_tile_batch=1, producer_row_load_bits=16, producer_values_per_lane=4, producer_transpose_values_per_lane=4, producer_scale_math="divide", producer_grouped_store_bits=8):
        if type(producer_values_per_lane) is not int or producer_values_per_lane not in (4,8,16):
            raise ValueError("producer values per lane must be 4, 8, or 16")
        if not transpose and producer_values_per_lane>4:
            if packed_source:
                raise ValueError("grouped row preparation requires original BF16 source")
            if normalization_math!="divide" or producer_row_tile_batch!=1:
                raise ValueError("grouped row preparation requires divide normalization and row tile batch 1")
        if type(producer_transpose_values_per_lane) is not int or producer_transpose_values_per_lane not in (4,8,16):
            raise ValueError("producer transpose values per lane must be 4, 8, or 16")
        if transpose and producer_transpose_values_per_lane>4:
            if packed_source or load_mode!="coalesced_x8" or normalization_math!="divide":
                raise ValueError("grouped transpose preparation requires original BF16, coalesced_x8 and divide normalization")
        if producer_scale_math not in ("divide", "integer_rtz4"):
            raise ValueError("unknown producer scale math")
        if producer_scale_math=="integer_rtz4" and mantissa_bits!=4:
            raise ValueError("integer_rtz4 producer scale requires mantissa_bits=4")
        self.producer_scale_math=producer_scale_math
        self.grouped_refinement=False
        self.grouped_refinement_streamed=False
        self.grouped_refinement_lut=False
        self.grouped_scale_free=False
        self.grouped_scale_free_streamed=False
        if type(producer_grouped_store_bits) is not int or producer_grouped_store_bits not in (8,16,32,64):
            raise ValueError("grouped store width must be 8, 16, 32, or 64 bits")
        self.values_per_lane=producer_values_per_lane if not transpose else producer_transpose_values_per_lane
        self.grouped_store_bits=min(producer_grouped_store_bits,self.values_per_lane*4) if self.values_per_lane>4 else 8
        self.groups_per_warp=self.values_per_lane//4
        self.lanes_per_group=128//self.values_per_lane
        if type(producer_row_load_bits) is not int or producer_row_load_bits not in (16,32,64):
            raise ValueError("producer row load bits must be 16, 32, or 64")
        if packed_source and not transpose and producer_row_load_bits>16:
            raise ValueError("vector row loads require original BF16 row source")
        self.producer_row_load_bits=producer_row_load_bits
        if type(producer_row_tile_batch) is not int or producer_row_tile_batch not in (1,2,4):
            raise ValueError("producer row tile batch must be 1, 2, or 4")
        if packed_source and not transpose and producer_row_tile_batch>1:
            raise ValueError("row tile batching requires original BF16 row source")
        self.producer_row_tile_batch=producer_row_tile_batch
        if type(producer_unroll_rows) is not bool:
            raise ValueError("producer unroll rows must be boolean")
        self.producer_unroll_rows=producer_unroll_rows
        if type(producer_copy_bits) is not int or producer_copy_bits not in (32,64,128):
            raise ValueError("producer copy bits must be 32, 64, or 128")
        if packed_source and producer_copy_bits!=32:
            raise ValueError("wide asynchronous copies require BF16 source")
        self.producer_copy_bits=producer_copy_bits
        self.copy_elements=producer_copy_bits//16
        if normalization_math not in ("divide", "refine", "threshold_scale_free", "refine_warp_lut", "guarded_scale_free"):
            raise ValueError("unknown normalization math")
        if normalization_math in ("refine_warp_lut","guarded_scale_free") and self.values_per_lane!=4:
            raise ValueError("warp reciprocal modes currently require VPL4 preparation")
        self.normalization_math = normalization_math
        if partial_max_scope not in ("warp", "cta"):
            raise ValueError("unknown partial maximum scope")
        self.partial_max_scope = partial_max_scope
        if correction_decode not in ("lut", "native"):
            raise ValueError("unknown correction decode mode")
        self.correction_decode = correction_decode
        if type(stages) is not int or stages not in (1, 2, 3):
            raise ValueError("preparation stages must be 1, 2, or 3")
        if type(async_copy) is not bool:
            raise ValueError("preparation async_copy must be boolean")
        if stages > 1 and not async_copy:
            raise ValueError("multiple preparation stages require async_copy")
        if async_copy and packed_source:
            raise ValueError("async preparation requires original BF16 source")
        if tile_order not in ("row_major", "reduction_major"):
            raise ValueError("unknown preparation tile order")
        self.tile_order = tile_order
        if signs_mode not in ("reload", "register"):
            raise ValueError("unknown sign reuse mode")
        self.signs_mode=signs_mode
        if load_mode not in ("direct", "coalesced", "coalesced_wide", "coalesced_x4", "coalesced_x8"):
            raise ValueError("unknown preparation load mode")
        self.coalesced = load_mode != "direct" and transpose
        # Wider contiguous source-column ownership amortizes staging barriers
        # and avoids repeatedly fetching narrow slices of the same cache line.
        self.rows_per_warp = ({"coalesced_wide": 2, "coalesced_x4": 4,
                               "coalesced_x8": 8}.get(load_mode, 1) if transpose else 1)
        self.source_tile_rows = config.threads//32*self.rows_per_warp
        # A whole-gradient schedule also reaches its row-major operand. Keep
        # that path unchanged; only transposed coalesced operands are staged.
        self.async_copy = async_copy and self.coalesced
        self.stages = stages if self.async_copy else 1
        if async_copy and transpose and not self.coalesced:
            raise ValueError("async preparation requires a coalesced transpose loader")
        if producer_copy_bits!=32 and transpose and not self.async_copy:
            raise ValueError("wide copies require asynchronous coalesced transpose loads")
        if self.async_copy and self.source_tile_rows % 2:
            raise ValueError("async preparation requires aligned BF16 pairs")
        if self.async_copy and (self.source_tile_rows%self.copy_elements or cols%self.copy_elements):
            raise ValueError("asynchronous copy width requires aligned source row groups and row stride")
        # Padding preserves vector alignment of every SMEM row and stage.
        # Wider padding may worsen read-bank conflicts: this is a tuning axis,
        # not an assertion that fewer copy instructions improve total latency.
        self.async_row_stride=self.source_tile_rows+self.copy_elements
        self.packed_source = packed_source
        if packed_source and cols % 16:
            raise ValueError("packed source requires full row-major 16-element blocks")
        self.source_rows,self.source_cols,self.transpose=rows,cols,transpose
        self.rows,self.cols=(cols,rows) if transpose else (rows,cols)
        if self.coalesced and self.rows % self.source_tile_rows:
            raise ValueError("coalesced loader requires complete row groups")
        if self.cols % 128:
            raise ValueError("Hadamard reduction dimension must be divisible by 128")
        super().__init__(self.rows,self.cols//16,config,mantissa_bits)
        self.tiles=self.rows*(self.cols//128)
        self.blocks=min(config.blocks,(self.tiles+config.threads//32-1)//(config.threads//32))
        # Output aggregation must never change which warp owns each work tile.
        self.work_warp_count=self.blocks*(config.threads//32)
        self.partial_count=self.blocks if partial_max_scope=="cta" else self.work_warp_count

    @cute.jit
    def normalization_parameters(self, scale: Float32, need_reciprocal: cutlass.Constexpr = True):
        """One precise reciprocal shared by the fragment's four divisions.

        The proof requires positive normal scales with at most five significant
        bits. Exponent 254 would require a subnormal inverse power of two, so
        it deliberately falls back. No approximate reciprocal is used.
        """
        bits = scale.bitcast(Int32)
        exponent = (bits >> 23) & 255
        eligible = (bits > 0) & (exponent >= 1) & (exponent <= 253) & ((bits & 0x7ffff) == 0)
        normalized_scale = ((bits & Int32(0x7fffff)) | Int32(0x3f800000)).bitcast(Float32)
        inverse_power, reciprocal = Float32(1.), Float32(1.)
        if eligible:
            inverse_power = ((Int32(254) - exponent) << 23).bitcast(Float32)
            if cutlass.const_expr(need_reciprocal):
                reciprocal = nvvm.inline_ptx_hl("div.rn.f32 {$w0}, 0f3f800000, {$r0};",
                    write_only_types=[Float32], read_only_args=[normalized_scale])
        return eligible, normalized_scale, inverse_power, reciprocal

    @cute.jit
    def warp_reciprocal_entry(self):
        """One exact reciprocal word per lane, initialized before tile loops.

        All 32 lanes participate; lanes 16..31 duplicate the first 16 words.
        Keep the literal one in PTX, not an inline-asm constant read operand.
        """
        lane=cute.arch.lane_idx()
        denominator=(Int32(0x3f800000)|((lane&15)<<19)).bitcast(Float32)
        return nvvm.inline_ptx_hl("div.rn.f32 {$w0}, 0f3f800000, {$r0};",
            write_only_types=[Float32],read_only_args=[denominator])

    @cute.jit
    def warp_reciprocal_lookup(self, cached_reciprocal: Float32, scale: Float32):
        # This must execute before eligibility/zero branches. Even an invalid
        # source lane owns a valid cached table word requested by other lanes.
        index=(scale.bitcast(Int32)>>19)&15
        return cute.arch.shuffle_sync(cached_reciprocal,index)

    @cute.jit
    def guarded_scale_free_proposal(self, normalized: Float32, original: Float32,
                                    scale: Float32, reciprocal: Float32):
        """Packing proposal only; never use its value in EDEN statistics.

        Caller retains threshold_scale_free's safe domain. Every E2M1 midpoint
        word is a multiple of 2^21. The W=16 window conservatively covers the
        RN-reciprocal/multiply error relative to RN(original/scale), including
        binade crossings. Boundary cases use the original precise division.
        """
        proposal=nvvm.inline_ptx_hl("mul.rn.f32 {$w0}, {$r0}, {$r1};",
            write_only_types=[Float32],read_only_args=[normalized,reciprocal])
        magnitude=proposal.bitcast(Int32)&Int32(0x7fffffff)
        # Signed zero already has its exact quotient sign for a positive scale.
        # Skip an unnecessary division without changing the conservative window.
        if (magnitude!=0) & (((magnitude+Int32(16))&Int32(0x1fffff))<=Int32(32)):
            proposal=nvvm.inline_ptx_hl("div.rn.f32 {$w0}, {$r0}, {$r1};",
                write_only_types=[Float32],read_only_args=[original,scale])
        return proposal

    @cute.jit
    def threshold_parameters(self, normalized_scale: Float32):
        # Five-significant-bit m times these short dyadic midpoints is exact.
        return (normalized_scale*Float32(.25), normalized_scale*Float32(.75),
                normalized_scale*Float32(1.25), normalized_scale*Float32(1.75),
                normalized_scale*Float32(2.5), normalized_scale*Float32(3.5),
                normalized_scale*Float32(5.))

    @cute.jit
    def threshold_code(self, value: Float32, t0, t1, t2, t3, t4, t5, t6):
        """Exact E2M1 ties-to-even bisection: three comparisons, no branches.

        Domain: finite normalized value and a positive <=5-significant-bit
        normalized scale. FP32 division cannot create an additional tie input:
        adjacent input spacing / m is >=16/31 quotient ULP, strictly above .5.
        """
        absolute = (value.bitcast(Int32) & Int32(0x7fffffff)).bitcast(Float32)
        code = nvvm.inline_ptx_hl(
            "{.reg .pred p2,p1,p0; .reg .f32 t,lo,hi; .reg .b32 c2,c1,c0; "
            "setp.ge.f32 p2, {$r0}, {$r4}; "
            "selp.f32 t, {$r6}, {$r2}, p2; setp.ge.f32 p1, {$r0}, t; "
            "selp.f32 lo, {$r5}, {$r1}, p2; selp.f32 hi, {$r7}, {$r3}, p2; "
            "selp.f32 t, hi, lo, p1; setp.gt.f32 p0, {$r0}, t; "
            "selp.b32 c2, 4, 0, p2; selp.b32 c1, 2, 0, p1; selp.b32 c0, 1, 0, p0; "
            "or.b32 c2, c2, c1; or.b32 {$w0}, c2, c0;}",
            write_only_types=[Int32], read_only_args=[absolute,t0,t1,t2,t3,t4,t5,t6])
        return code | ((value.bitcast(Int32) >> 28) & Int32(8))

    @cute.jit
    def normalize_refined(self, value: Float32, scale: Float32, eligible,
                          normalized_scale: Float32, inverse_power: Float32, reciprocal: Float32):
        """Correctly-rounded x/scale on the guarded domain, exact div otherwise.

        See the proof and CPU evidence in ms_eden_speed_target_20261026.md.
        Explicit PTX prevents contraction/reordering and preserves signed zero.
        """
        absolute = value.bitcast(Int32) & Int32(0x7fffffff)
        normalized = Float32(0.)
        fast = cutlass.Boolean(False)
        if eligible:
            normalized = nvvm.inline_ptx_hl("mul.rn.f32 {$w0}, {$r0}, {$r1};",
                write_only_types=[Float32], read_only_args=[value, inverse_power])
            normalized_bits = normalized.bitcast(Int32) & Int32(0x7fffffff)
            fast = (absolute < 0x7f800000) & (normalized_bits >= 0x12800000) & (normalized_bits <= 0x41800000)
        result = Float32(0.)
        if eligible & (absolute == 0):
            result = value
        elif fast:
            estimate = nvvm.inline_ptx_hl("mul.rn.f32 {$w0}, {$r0}, {$r1};",
                write_only_types=[Float32], read_only_args=[normalized, reciprocal])
            residual = nvvm.inline_ptx_hl("fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};",
                write_only_types=[Float32], read_only_args=[-estimate, normalized_scale, normalized])
            result = nvvm.inline_ptx_hl("fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};",
                write_only_types=[Float32], read_only_args=[residual, reciprocal, estimate])
        else:
            result = nvvm.inline_ptx_hl("div.rn.f32 {$w0}, {$r0}, {$r1};",
                write_only_types=[Float32], read_only_args=[value, scale])
        return result

    @cute.jit
    def write_partial_maximum(self, largest, partials, warp_maxima):
        thread,_,_=cute.arch.thread_idx()
        block,_,_=cute.arch.block_idx()
        lane=cute.arch.lane_idx()
        for level in cutlass.range_constexpr(5):
            largest=cute.arch.fmax(largest,cute.arch.shuffle_sync_bfly(largest,16 >> level))
        if cutlass.const_expr(self.partial_max_scope=="cta"):
            if lane==0:
                warp_maxima[thread//32]=largest
            # Includes idle warps/CTAs: their +0 accumulator is published too.
            cute.arch.sync_threads()
            if thread<32:
                largest=Float32(0.)
                if lane < self.config.threads//32:
                    largest=warp_maxima[lane]
                for level in cutlass.range_constexpr(5):
                    largest=cute.arch.fmax(largest,cute.arch.shuffle_sync_bfly(largest,16 >> level))
                if lane==0:
                    partials[block]=largest
        else:
            if lane==0:
                partials[block*(self.config.threads//32)+thread//32]=largest

    @cute.jit
    def __call__(self, src: cute.Tensor, signs: cute.Tensor, packed: cute.Tensor,
                 scales: cute.Tensor, partials: cute.Tensor, stream: cuda.CUstream):
        if cutlass.const_expr(not self.transpose and self.values_per_lane>4):
            self.grouped_row_kernel(src,signs,packed,scales,partials).launch(
                grid=(self.blocks,1,1),block=(self.config.threads,1,1),stream=stream)
        elif cutlass.const_expr(self.async_copy):
            self.buffered_rotated_kernel(src,signs,packed,scales,partials).launch(
                grid=(self.blocks,1,1),block=(self.config.threads,1,1),stream=stream)
        else:
            self.rotated_kernel(src,signs,packed,scales,partials,signs,signs).launch(
                grid=(self.blocks,1,1),block=(self.config.threads,1,1),stream=stream)

    @cute.jit
    def grouped_pair_statistics(self,a,b,qa,qb):
        # Literal tree emitted by the existing native row producer: round b*b
        # first, then fuse a*a into that product (likewise for the dot product).
        bb=nvvm.inline_ptx_hl("mul.rn.f32 {$w0}, {$r0}, {$r0};",
            write_only_types=[Float32],read_only_args=[b])
        bqb=nvvm.inline_ptx_hl("mul.rn.f32 {$w0}, {$r0}, {$r1};",
            write_only_types=[Float32],read_only_args=[b,qb])
        xx=nvvm.inline_ptx_hl("fma.rn.f32 {$w0}, {$r0}, {$r0}, {$r1};",
            write_only_types=[Float32],read_only_args=[a,bb])
        xy=nvvm.inline_ptx_hl("fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};",
            write_only_types=[Float32],read_only_args=[a,qa,bqb])
        return xx,xy

    @cute.jit
    def grouped_add(self,a,b):
        return nvvm.inline_ptx_hl("add.rn.f32 {$w0}, {$r0}, {$r1};",
            write_only_types=[Float32],read_only_args=[a,b])

    @cute.jit
    def grouped_add_zero(self,a):
        # Encode the literal in PTX: passing a Float32 literal as an inline-asm
        # read_only_arg hits the installed NVVM wrapper's constant-operand bug.
        return nvvm.inline_ptx_hl("add.rn.f32 {$w0}, {$r0}, 0f00000000;",
            write_only_types=[Float32],read_only_args=[a])

    @cute.kernel
    def grouped_row_kernel(self,src:cute.Tensor,signs:cute.Tensor,packed:cute.Tensor,
                           scales:cute.Tensor,partials:cute.Tensor):
        thread,_,_=cute.arch.thread_idx()
        block,_,_=cute.arch.block_idx()
        lane=cute.arch.lane_idx()
        warp=block*(self.config.threads//32)+thread//32
        subgroup=lane//self.lanes_per_group
        local_lane=lane%self.lanes_per_group
        largest=Float32(0.)
        cached_reciprocal=Float32(1.)
        if cutlass.const_expr(self.grouped_refinement_lut):
            cached_reciprocal=self.warp_reciprocal_entry()
        if cutlass.const_expr(self.signs_mode=="register"):
            cached_signs=cute.make_rmem_tensor((self.values_per_lane,),Float32)
            for j in cutlass.range_constexpr(self.values_per_lane):
                cached_signs[j]=signs[local_lane*self.values_per_lane+j]
        smem=cutlass.utils.SmemAllocator()
        warp_maxima=partials
        if cutlass.const_expr(self.partial_max_scope=="cta"):
            warp_maxima=smem.allocate_tensor(Float32,cute.make_layout((self.config.threads//32,)),byte_alignment=4)
        # Uniform physical-warp loop bounds keep every subgroup participating in
        # shuffles, even when a final subgroup has no tile. Its values are +0.
        for first in cutlass.range(warp,self.tiles,self.work_warp_count*self.groups_per_warp,unroll=1):
            work_tile=first+subgroup*self.work_warp_count
            active=work_tile<self.tiles
            tile=work_tile
            if cutlass.const_expr(self.tile_order=="reduction_major"):
                tile=(work_tile%self.rows)*(self.cols//128)+work_tile//self.rows
            values=cute.make_rmem_tensor((self.values_per_lane,),Float32)
            for j in cutlass.range_constexpr(self.values_per_lane):values[j]=Float32(0.)
            if active:
                offset=tile*128+local_lane*self.values_per_lane
                if cutlass.const_expr(self.producer_row_load_bits>16):
                    raw=cute.make_rmem_tensor((self.values_per_lane,),BFloat16)
                    atom=cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),BFloat16,
                        num_bits_per_copy=self.producer_row_load_bits)
                    aligned=cute.assume(offset,divby=self.producer_row_load_bits//16)
                    cute.copy(atom,cute.make_tensor(src.iterator+aligned,
                        cute.make_layout((self.values_per_lane,),stride=(1,))),raw)
                for j in cutlass.range_constexpr(self.values_per_lane):
                    value=(raw[j].to(Float32) if cutlass.const_expr(self.producer_row_load_bits>16)
                           else src[offset+j].to(Float32))
                    sign=(cached_signs[j] if cutlass.const_expr(self.signs_mode=="register")
                          else signs[local_lane*self.values_per_lane+j])
                    values[j]=value*sign
            largest=self.grouped_prepare_values(values,packed,scales,tile,local_lane,active,largest,cached_reciprocal)
        self.write_partial_maximum(largest,partials,warp_maxima)

    @cute.jit
    def grouped_transform_values(self,values,local_lane):
        # Same seven butterfly levels: the first 3/4 are now register-local.
        for level in cutlass.range_constexpr(self.values_per_lane.bit_length()-1):
            distance=1 << level
            for group in cutlass.range_constexpr(self.values_per_lane//(2*distance)):
                for j in cutlass.range_constexpr(distance):
                    lo=group*2*distance+j
                    a,b=values[lo],values[lo+distance]
                    values[lo],values[lo+distance]=a+b,a-b
        for level in cutlass.range_constexpr(self.lanes_per_group.bit_length()-1):
            for j in cutlass.range_constexpr(self.values_per_lane):
                own=values[j]
                other=cute.arch.shuffle_sync_bfly(own,1 << level)
                values[j]=own+other if (local_lane & (1 << level))==0 else other-own

    @cute.jit
    def grouped_prepare_values(self,values,packed,scales,tile,local_lane,active,largest,cached_reciprocal):
        self.grouped_transform_values(values,local_lane)
        maximum=Float32(0.)
        for j in cutlass.range_constexpr(self.values_per_lane):
            values[j]=values[j]*Float32(0.08838834764831845)
            absolute=(values[j].bitcast(Int32)&Int32(0x7fffffff)).bitcast(Float32)
            maximum=cute.arch.fmax(maximum,absolute)
        for level in cutlass.range_constexpr((16//self.values_per_lane).bit_length()-1):
            maximum=cute.arch.fmax(maximum,cute.arch.shuffle_sync_bfly(maximum,1 << level))
        if cutlass.const_expr(self.producer_scale_math=="integer_rtz4"):
            scale=producer_scale_rtz4(maximum)
        else:
            scale=maximum/Float32(6.)
            scale=(scale.bitcast(Int32)&Int32(~((1 << (23-self.mantissa_bits))-1))).bitcast(Float32)
            if scale<Float32(1.1754943508222875e-38):scale=Float32(0.)
        quarter_x=cute.make_rmem_tensor((self.values_per_lane//4,),Float32)
        quarter_y=cute.make_rmem_tensor((self.values_per_lane//4,),Float32)
        if cutlass.const_expr(self.grouped_store_bits>8):
            output_codes=cute.make_rmem_tensor((self.values_per_lane//2,),Uint8)
        if cutlass.const_expr(self.grouped_scale_free):
            eligible,normalized_scale,inverse_power,unused=self.normalization_parameters(scale,False)
            if cutlass.const_expr(not self.grouped_scale_free_streamed):
                normalized=cute.make_rmem_tensor((self.values_per_lane,),Float32)
            safe=Int32(1) if eligible else Int32(0)
            for j in cutlass.range_constexpr(self.values_per_lane):
                normalized_value=nvvm.inline_ptx_hl("mul.rn.f32 {$w0}, {$r0}, {$r1};",
                    write_only_types=[Float32],read_only_args=[values[j],inverse_power])
                if cutlass.const_expr(not self.grouped_scale_free_streamed):
                    normalized[j]=normalized_value
                absolute=normalized_value.bitcast(Int32)&Int32(0x7fffffff)
                original_absolute=values[j].bitcast(Int32)&Int32(0x7fffffff)
                if ((original_absolute>=0x7f800000) | (absolute>0x41800000)
                    | ((absolute!=0)&(absolute<0x21800000))
                    | ((original_absolute!=0)&(absolute==0))):
                    safe=Int32(0)
            for level in cutlass.range_constexpr((16//self.values_per_lane).bit_length()-1):
                safe &= cute.arch.shuffle_sync_bfly(safe,1<<level)
            normalized_maximum=nvvm.inline_ptx_hl("mul.rn.f32 {$w0}, {$r0}, {$r1};",
                write_only_types=[Float32],read_only_args=[maximum,inverse_power])
            if ((normalized_maximum<normalized_scale*Float32(6.))
                | (normalized_maximum>normalized_scale*Float32(6.*(1.+2.**-self.mantissa_bits)))):
                safe=Int32(0)
            scale_free=safe!=0
            thresholds=self.threshold_parameters(normalized_scale)
        if cutlass.const_expr(self.grouped_refinement):
            if cutlass.const_expr(self.grouped_refinement_lut):
                reciprocal=self.warp_reciprocal_lookup(cached_reciprocal,scale)
                eligible,normalized_scale,inverse_power,unused=self.normalization_parameters(scale,False)
            else:
                eligible,normalized_scale,inverse_power,reciprocal=self.normalization_parameters(scale)
            if cutlass.const_expr(not self.grouped_refinement_streamed):
                normalized=cute.make_rmem_tensor((self.values_per_lane,),Float32)
                safe=eligible
                for j in cutlass.range_constexpr(self.values_per_lane):
                    normalized[j]=nvvm.inline_ptx_hl("mul.rn.f32 {$w0}, {$r0}, {$r1};",
                        write_only_types=[Float32],read_only_args=[values[j],inverse_power])
                    absolute=values[j].bitcast(Int32)&Int32(0x7fffffff)
                    magnitude=normalized[j].bitcast(Int32)&Int32(0x7fffffff)
                    safe &= (absolute==0) | ((absolute<Int32(0x7f800000)) &
                        (magnitude>=Int32(0x12800000)) & (magnitude<=Int32(0x41800000)))
        for quarter in cutlass.range_constexpr(self.values_per_lane//4):
            xx,xy=Float32(0.),Float32(0.)
            for pair in cutlass.range_constexpr(2):
                j=quarter*4+pair*2
                a,b=Float32(0.),Float32(0.)
                if scale>Float32(0.):
                    if cutlass.const_expr(self.grouped_scale_free):
                        if scale_free:
                            if cutlass.const_expr(self.grouped_scale_free_streamed):
                                a=nvvm.inline_ptx_hl("mul.rn.f32 {$w0}, {$r0}, {$r1};",
                                    write_only_types=[Float32],read_only_args=[values[j],inverse_power])
                                b=nvvm.inline_ptx_hl("mul.rn.f32 {$w0}, {$r0}, {$r1};",
                                    write_only_types=[Float32],read_only_args=[values[j+1],inverse_power])
                            else:
                                a,b=normalized[j],normalized[j+1]
                        else:
                            a,b=values[j]/scale,values[j+1]/scale
                    elif cutlass.const_expr(self.grouped_refinement):
                        if cutlass.const_expr(self.grouped_refinement_streamed):
                            a=self.normalize_refined(values[j],scale,eligible,normalized_scale,inverse_power,reciprocal)
                            b=self.normalize_refined(values[j+1],scale,eligible,normalized_scale,inverse_power,reciprocal)
                        else:
                            if safe:
                                a=self.grouped_refined_quotient(normalized[j],normalized_scale,reciprocal)
                                b=self.grouped_refined_quotient(normalized[j+1],normalized_scale,reciprocal)
                            else:
                                a,b=values[j]/scale,values[j+1]/scale
                    else:
                        a,b=values[j]/scale,values[j+1]/scale
                code=Int16(0)
                if cutlass.const_expr(self.grouped_scale_free):
                    if scale_free:
                        code=Int16(self.threshold_code(a,*thresholds)|(self.threshold_code(b,*thresholds)<<4))
                    else:
                        code=nvvm.inline_ptx_hl(
                            "{.reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, {$r1}, {$r0}; mov.b16 {$w0}, {b, 0};}",
                            write_only_types=[Int16],read_only_args=[a,b])
                else:
                    code=nvvm.inline_ptx_hl(
                        "{.reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, {$r1}, {$r0}; mov.b16 {$w0}, {b, 0};}",
                        write_only_types=[Int16],read_only_args=[a,b])
                if cutlass.const_expr(self.grouped_store_bits>8):
                    output_codes[j//2]=Uint8(code)
                else:
                    if active:packed[tile*64+local_lane*(self.values_per_lane//2)+j//2]=Uint8(code)
                qa,qb=self.decode_pair(Int32(code),self.correction_decode)
                px,py=self.grouped_pair_statistics(a,b,qa,qb)
                if cutlass.const_expr(pair==0):
                    xx,xy=self.grouped_add_zero(px),self.grouped_add_zero(py)
                else:
                    xx,xy=self.grouped_add(xx,px),self.grouped_add(xy,py)
            quarter_x[quarter],quarter_y[quarter]=xx,xy
        if cutlass.const_expr(self.grouped_store_bits>8):
            if active:
                offset=cute.assume(tile*64+local_lane*(self.values_per_lane//2),divby=self.grouped_store_bits//8)
                destination=cute.make_tensor(packed.iterator+offset,cute.make_layout((self.values_per_lane//2,),stride=(1,)))
                atom=cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),Uint8,num_bits_per_copy=self.grouped_store_bits)
                cute.copy(atom,output_codes,destination)
        xx=self.grouped_add(quarter_x[0],quarter_x[1])
        xy=self.grouped_add(quarter_y[0],quarter_y[1])
        if cutlass.const_expr(self.values_per_lane==8):
            xx=self.grouped_add(xx,cute.arch.shuffle_sync_bfly(xx,1))
            xy=self.grouped_add(xy,cute.arch.shuffle_sync_bfly(xy,1))
        else:
            xx=self.grouped_add(xx,self.grouped_add(quarter_x[2],quarter_x[3]))
            xy=self.grouped_add(xy,self.grouped_add(quarter_y[2],quarter_y[3]))
        correction=Float32(1.)
        if xy>Float32(0.):correction=xx/xy
        output_scale=scale
        if cutlass.const_expr(self.grouped_scale_free):
            if scale_free:
                output_scale=(scale.bitcast(Int32)&Int32(0x7f800000)).bitcast(Float32)
        corrected=(output_scale*correction).to(BFloat16)
        if active and local_lane%(16//self.values_per_lane)==0:
            scales[tile*8+local_lane*self.values_per_lane//16]=corrected
        largest=cute.arch.fmax(largest,corrected.to(Float32))
        return largest

    @cute.jit
    def grouped_refined_quotient(self,value,scale,reciprocal):
        estimate=nvvm.inline_ptx_hl("mul.rn.f32 {$w0}, {$r0}, {$r1};",
            write_only_types=[Float32],read_only_args=[value,reciprocal])
        residual=nvvm.inline_ptx_hl("fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};",
            write_only_types=[Float32],read_only_args=[-estimate,scale,value])
        result=nvvm.inline_ptx_hl("fma.rn.f32 {$w0}, {$r0}, {$r1}, {$r2};",
            write_only_types=[Float32],read_only_args=[residual,reciprocal,estimate])
        # Preserve the ordinary quotient's signed-zero behavior explicitly.
        if (value.bitcast(Int32)&Int32(0x7fffffff))==0:result=value
        return result

    @cute.jit
    def stage_async(self, src, staged, work_tile, stage):
        thread,_,_=cute.arch.thread_idx()
        row_group=work_tile//(self.cols//128)
        chunk=work_tile%(self.cols//128)
        if cutlass.const_expr(self.tile_order == "reduction_major"):
            row_group=work_tile%(self.rows//self.source_tile_rows)
            chunk=work_tile//(self.rows//self.source_tile_rows)
        atom=cute.make_copy_atom(cpasync.CopyG2SOp(),BFloat16,num_bits_per_copy=self.producer_copy_bits)
        vectors=128*self.source_tile_rows//self.copy_elements
        for j in cutlass.range_constexpr((vectors+self.config.threads-1)//self.config.threads):
            vector=thread+j*self.config.threads
            if vector < vectors:
                index=vector*self.copy_elements
                source_row=chunk*128+index//self.source_tile_rows
                source_col=row_group*self.source_tile_rows+index%self.source_tile_rows
                offset=cute.assume(source_row*self.source_cols+source_col,divby=self.copy_elements)
                target=cute.assume(staged.layout((index//self.source_tile_rows,index%self.source_tile_rows,stage)),divby=self.copy_elements)
                cute.copy(atom,
                    cute.make_tensor(src.iterator+offset,cute.make_layout((self.copy_elements,))),
                    cute.make_tensor(staged.iterator+target,cute.make_layout((self.copy_elements,))))

    @cute.kernel
    def buffered_rotated_kernel(self, src: cute.Tensor, signs: cute.Tensor,
                                packed: cute.Tensor, scales: cute.Tensor,
                                partials: cute.Tensor):
        thread,_,_=cute.arch.thread_idx()
        block,_,_=cute.arch.block_idx()
        lane=cute.arch.lane_idx()
        warp=block*(self.config.threads//32)+thread//32
        largest=Float32(0.)
        cached_reciprocal=Float32(1.)
        if cutlass.const_expr(self.grouped_refinement_lut or self.normalization_math in ("refine_warp_lut","guarded_scale_free")):
            cached_reciprocal=self.warp_reciprocal_entry()
        if cutlass.const_expr(self.signs_mode == "register"):
            if cutlass.const_expr(self.values_per_lane>4):
                cached_signs=cute.make_rmem_tensor((self.values_per_lane,),Float32)
                for j in cutlass.range_constexpr(self.values_per_lane):
                    cached_signs[j]=signs[(lane%self.lanes_per_group)*self.values_per_lane+j]
            else:
                cached_signs=cute.make_rmem_tensor((4,),Float32)
                for j in cutlass.range_constexpr(4):
                    cached_signs[j]=signs[lane*4+j]
        smem=cutlass.utils.SmemAllocator()
        warp_maxima=partials
        if cutlass.const_expr(self.partial_max_scope=="cta"):
            warp_maxima=smem.allocate_tensor(Float32,cute.make_layout((self.config.threads//32,)),byte_alignment=4)
        staged=smem.allocate_tensor(
            BFloat16,cute.make_layout((128,self.source_tile_rows,self.stages),
                stride=(self.async_row_stride,1,128*self.async_row_stride)),
            byte_alignment=16)
        limit=self.tiles//self.source_tile_rows
        # All threads commit exactly one group per stage, including empty tail
        # groups. Thus wait_group(stages-1) always completes the oldest slot.
        # CTA-uniform predicates avoid divergent group histories.
        for stage in cutlass.range_constexpr(self.stages):
            work_tile=block+stage*self.blocks
            if work_tile < limit:
                self.stage_async(src,staged,work_tile,stage)
            cute.arch.cp_async_commit_group()
        for iteration in cutlass.range(0,(limit+self.blocks-1-block)//self.blocks,unroll=1):
            slot=iteration%self.stages
            work_tile=block+iteration*self.blocks
            cute.arch.cp_async_wait_group(self.stages-1)
            cute.arch.sync_threads()
            row_group=work_tile//(self.cols//128)
            chunk=work_tile%(self.cols//128)
            if cutlass.const_expr(self.tile_order == "reduction_major"):
                row_group=work_tile%(self.rows//self.source_tile_rows)
                chunk=work_tile//(self.rows//self.source_tile_rows)
            tile=(row_group*self.source_tile_rows+thread//32)*(self.cols//128)+chunk
            largest=self.prepare_rows(src,cached_signs if cutlass.const_expr(self.signs_mode == "register") else signs,
                packed,scales,signs,signs,staged[None,None,slot],tile,largest,cached_reciprocal)
            # No thread may replace a slot until every warp has finished its
            # reads. The refill then overlaps arithmetic on the other slots.
            cute.arch.sync_threads()
            future=work_tile+self.stages*self.blocks
            if future < limit:
                self.stage_async(src,staged,future,slot)
            cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        self.write_partial_maximum(largest,partials,warp_maxima)

    @cute.kernel
    def rotated_kernel(self, src: cute.Tensor, signs: cute.Tensor, packed: cute.Tensor,
                       scales: cute.Tensor, partials: cute.Tensor,
                       source_scales: cute.Tensor, source_outer: cute.Tensor):
        thread,_,_=cute.arch.thread_idx()
        block,_,_=cute.arch.block_idx()
        lane=cute.arch.lane_idx()
        warp=block*(self.config.threads//32)+thread//32
        largest=Float32(0.)
        cached_reciprocal=Float32(1.)
        if cutlass.const_expr(self.grouped_refinement_lut or self.normalization_math in ("refine_warp_lut","guarded_scale_free")):
            cached_reciprocal=self.warp_reciprocal_entry()
        if cutlass.const_expr(self.signs_mode == "register"):
            if cutlass.const_expr(self.values_per_lane>4):
                cached_signs=cute.make_rmem_tensor((self.values_per_lane,),Float32)
                for j in cutlass.range_constexpr(self.values_per_lane):
                    cached_signs[j]=signs[(lane%self.lanes_per_group)*self.values_per_lane+j]
            else:
                cached_signs=cute.make_rmem_tensor((4,),Float32)
                for j in cutlass.range_constexpr(4):
                    cached_signs[j]=signs[lane*4+j]
        smem=cutlass.utils.SmemAllocator()
        warp_maxima=partials
        if cutlass.const_expr(self.partial_max_scope=="cta"):
            warp_maxima=smem.allocate_tensor(Float32,cute.make_layout((self.config.threads//32,)),byte_alignment=4)
        first,limit,stride=warp,self.tiles,self.work_warp_count
        if cutlass.const_expr(not self.transpose and self.producer_row_tile_batch>1):
            stride*=self.producer_row_tile_batch
        if cutlass.const_expr(self.coalesced):
            staged=smem.allocate_tensor(
                BFloat16,cute.make_layout((128,self.source_tile_rows),
                                         stride=(self.source_tile_rows+1,1)),byte_alignment=16)
            first,limit,stride=block,self.tiles//self.source_tile_rows,self.blocks
        for work_tile in cutlass.range(first,limit,stride,unroll=1):
            tile = work_tile
            if cutlass.const_expr(self.tile_order == "reduction_major"):
                tile = (work_tile % self.rows)*(self.cols//128)+work_tile//self.rows
            if cutlass.const_expr(self.coalesced):
                row_group=work_tile//(self.cols//128)
                chunk=work_tile%(self.cols//128)
                if cutlass.const_expr(self.tile_order == "reduction_major"):
                    row_group=work_tile%(self.rows//self.source_tile_rows)
                    chunk=work_tile//(self.rows//self.source_tile_rows)
                tile=(row_group*self.source_tile_rows+thread//32)*(self.cols//128)+chunk
                for j in cutlass.range_constexpr(4*self.rows_per_warp):
                    index=thread+j*self.config.threads
                    source_row=chunk*128+index//self.source_tile_rows
                    source_col=row_group*self.source_tile_rows+index%self.source_tile_rows
                    offset=source_row*self.source_cols+source_col
                    if cutlass.const_expr(self.packed_source):
                        byte=Int32(src[offset//2])
                        code=(byte >> ((offset%2)*4)) & 15
                        value=self.decode(code)*source_scales[offset//16].to(Float32)*source_outer[0]
                        staged[index//self.source_tile_rows,index%self.source_tile_rows]=value.to(BFloat16)
                    else:
                        staged[index//self.source_tile_rows,index%self.source_tile_rows]=src[offset]
                cute.arch.sync_threads()
            if cutlass.const_expr(not self.transpose and self.producer_row_tile_batch>1):
                largest=self.prepare_row_batch(src,cached_signs if cutlass.const_expr(self.signs_mode == "register") else signs,
                    packed,scales,source_scales,source_outer,work_tile,largest,cached_reciprocal)
            else:
                largest=self.prepare_rows(src,cached_signs if cutlass.const_expr(self.signs_mode == "register") else signs,
                    packed,scales,source_scales,source_outer,
                    staged if cutlass.const_expr(self.coalesced) else src,tile,largest,cached_reciprocal)
            if cutlass.const_expr(self.coalesced):
                cute.arch.sync_threads()
        self.write_partial_maximum(largest,partials,warp_maxima)

    @cute.jit
    def prepare_row_batch(self,src,signs,packed,scales,source_scales,source_outer,first,largest,cached_reciprocal):
        # Keep each warp's exact original tile sequence. Only expose several
        # independent tiles to instruction scheduling before the loop backedge.
        for offset in cutlass.range_constexpr(self.producer_row_tile_batch):
            work_tile=first+offset*self.work_warp_count
            if work_tile<self.tiles:
                tile=work_tile
                if cutlass.const_expr(self.tile_order=="reduction_major"):
                    tile=(work_tile%self.rows)*(self.cols//128)+work_tile//self.rows
                largest=self.prepare_tile(src,signs,packed,scales,source_scales,source_outer,
                    src,tile,0,largest,cached_reciprocal)
        return largest

    @cute.jit
    def prepare_transpose_group(self,staged,signs,packed,scales,tile,base,largest,cached_reciprocal):
        thread,_,_=cute.arch.thread_idx()
        lane=cute.arch.lane_idx()
        local_lane=lane%self.lanes_per_group
        subrow=base+lane//self.lanes_per_group
        column=thread//32+subrow*(self.config.threads//32)
        local_tile=tile+subrow*(self.config.threads//32)*(self.cols//128)
        values=cute.make_rmem_tensor((self.values_per_lane,),Float32)
        for j in cutlass.range_constexpr(self.values_per_lane):
            sign=(signs[j] if cutlass.const_expr(self.signs_mode=="register")
                  else signs[local_lane*self.values_per_lane+j])
            values[j]=staged[local_lane*self.values_per_lane+j,column].to(Float32)*sign
        return self.grouped_prepare_values(values,packed,scales,local_tile,local_lane,True,largest,cached_reciprocal)

    @cute.jit
    def prepare_rows(self,src,signs,packed,scales,source_scales,source_outer,staged,tile,largest,cached_reciprocal):
        # Default retains the original constexpr expansion. The compact option
        # changes only code generation; input ownership and row order match.
        if cutlass.const_expr(self.transpose and self.values_per_lane>4):
            if cutlass.const_expr(self.producer_unroll_rows):
                for group in cutlass.range_constexpr(self.rows_per_warp//self.groups_per_warp):
                    largest=self.prepare_transpose_group(staged,signs,packed,scales,tile,
                        group*self.groups_per_warp,largest,cached_reciprocal)
            else:
                for group in cutlass.range(0,self.rows_per_warp//self.groups_per_warp,unroll=1):
                    largest=self.prepare_transpose_group(staged,signs,packed,scales,tile,
                        group*self.groups_per_warp,largest,cached_reciprocal)
        elif cutlass.const_expr(self.producer_unroll_rows):
            for subrow in cutlass.range_constexpr(self.rows_per_warp):
                local_tile=tile+subrow*(self.config.threads//32)*(self.cols//128)
                largest=self.prepare_tile(src,signs,packed,scales,source_scales,source_outer,
                    staged,local_tile,subrow,largest,cached_reciprocal)
        else:
            for subrow in cutlass.range(0,self.rows_per_warp,unroll=1):
                local_tile=tile+subrow*(self.config.threads//32)*(self.cols//128)
                largest=self.prepare_tile(src,signs,packed,scales,source_scales,source_outer,
                    staged,local_tile,subrow,largest,cached_reciprocal)
        return largest

    @cute.jit
    def prepare_tile(self, src, signs, packed, scales, source_scales, source_outer,
                 staged, tile, subrow, largest,cached_reciprocal):
        thread,_,_=cute.arch.thread_idx()
        lane=cute.arch.lane_idx()
        row=tile//(self.cols//128)
        column=(tile%(self.cols//128))*128+lane*4
        values=cute.make_rmem_tensor((4,),Float32)
        if cutlass.const_expr(not self.transpose and not self.packed_source and self.producer_row_load_bits>16):
            row_values=cute.make_rmem_tensor((4,),BFloat16)
            atom=cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(),BFloat16,
                num_bits_per_copy=self.producer_row_load_bits)
            offset=cute.assume(row*self.cols+column,divby=self.producer_row_load_bits//16)
            cute.copy(atom,cute.make_tensor(src.iterator+offset,cute.make_layout((4,),stride=(1,))),row_values)
        for j in cutlass.range_constexpr(4):
            offset=row*self.cols+column+j
            if cutlass.const_expr(self.transpose):
                offset=(column+j)*self.source_cols+row
            if cutlass.const_expr(self.coalesced):
                value=staged[lane*4+j,thread//32+subrow*(self.config.threads//32)].to(Float32)
            elif cutlass.const_expr(not self.transpose and self.producer_row_load_bits>16):
                value=row_values[j].to(Float32)
            elif cutlass.const_expr(self.packed_source):
                byte = Int32(src[offset//2])
                code = (byte >> ((offset%2)*4)) & 15
                value = self.decode(code)*source_scales[offset//16].to(Float32)*source_outer[0]
                # Explicit round-trip numerical contract, identical to
                # decoding saved forward operands to BF16 before backward.
                value = value.to(BFloat16).to(Float32)
            else:
                value = src[offset].to(Float32)
            values[j]=value*signs[j if cutlass.const_expr(self.signs_mode == "register") else lane*4+j]
        self.transform_values(values)
        maximum=Float32(0.)
        for j in cutlass.range_constexpr(4):
            values[j]=values[j]*Float32(0.08838834764831845)
            absolute=(values[j].bitcast(Int32)&Int32(0x7fffffff)).bitcast(Float32)
            maximum=cute.arch.fmax(maximum,absolute)
        for shift in cutlass.range_constexpr(2):
            maximum=cute.arch.fmax(maximum,cute.arch.shuffle_sync_bfly(maximum,1 << shift))
        if cutlass.const_expr(self.producer_scale_math=="integer_rtz4"):
            scale=producer_scale_rtz4(maximum)
        else:
            scale=maximum/Float32(6.)
            scale=(scale.bitcast(Int32)&Int32(~((1 << (23-self.mantissa_bits))-1))).bitcast(Float32)
            if scale < Float32(1.1754943508222875e-38):
                scale=Float32(0.)
        if cutlass.const_expr(self.normalization_math == "refine"):
            eligible, normalized_scale, inverse_power, reciprocal = self.normalization_parameters(scale)
        if cutlass.const_expr(self.normalization_math == "refine_warp_lut"):
            # Full-warp lookup is deliberately outside all eligibility and
            # scale>0 branches; no runtime reciprocal or extra tensor exists.
            reciprocal=self.warp_reciprocal_lookup(cached_reciprocal,scale)
            eligible, normalized_scale, inverse_power, unused_reciprocal = self.normalization_parameters(scale,False)
        if cutlass.const_expr(self.normalization_math == "guarded_scale_free"):
            # All lanes participate before eligibility/zero/fallback branches.
            guarded_reciprocal=self.warp_reciprocal_lookup(cached_reciprocal,scale)
        if cutlass.const_expr(self.normalization_math in ("threshold_scale_free","guarded_scale_free")):
            eligible, normalized_scale, inverse_power, unused_reciprocal = self.normalization_parameters(scale, False)
            normalized_values = cute.make_rmem_tensor((4,), Float32)
            safe = Int32(0)
            if eligible:
                safe = Int32(1)
            for j in cutlass.range_constexpr(4):
                normalized_values[j] = nvvm.inline_ptx_hl("mul.rn.f32 {$w0}, {$r0}, {$r1};",
                    write_only_types=[Float32], read_only_args=[values[j],inverse_power])
                absolute = normalized_values[j].bitcast(Int32) & Int32(0x7fffffff)
                original_absolute = values[j].bitcast(Int32) & Int32(0x7fffffff)
                if ((original_absolute >= 0x7f800000) | (absolute > 0x41800000)
                    | ((absolute != 0) & (absolute < 0x21800000))):
                    safe = Int32(0)
                # A nonzero source that underflowed during normalization is not
                # a signed-zero fast path; preserve original division instead.
                if (original_absolute != 0) & (absolute == 0):
                    safe = Int32(0)
            for shift in cutlass.range_constexpr(2):
                safe &= cute.arch.shuffle_sync_bfly(safe,1 << shift)
            normalized_maximum = nvvm.inline_ptx_hl("mul.rn.f32 {$w0}, {$r0}, {$r1};",
                write_only_types=[Float32], read_only_args=[maximum,inverse_power])
            # Verify the producer-derived RTZ scale regime. This gives a
            # nonzero denominator and bounds restored FP32/BF16 scale errors.
            if ((normalized_maximum < normalized_scale*Float32(6.))
                | (normalized_maximum > normalized_scale*Float32(6.*(1.+2.**-self.mantissa_bits)))):
                safe = Int32(0)
            scale_free = safe != 0
            if cutlass.const_expr(self.normalization_math == "threshold_scale_free"):
                thresholds = self.threshold_parameters(normalized_scale)
        xx,xy=Float32(0.),Float32(0.)
        for j in cutlass.range_constexpr(2):
            a,b=Float32(0.),Float32(0.)
            bits=Int16(0)
            if scale > Float32(0.):
                if cutlass.const_expr(self.normalization_math in ("threshold_scale_free","guarded_scale_free")):
                    if scale_free:
                        a,b=normalized_values[2*j],normalized_values[2*j+1]
                    else:
                        a,b=values[2*j]/scale,values[2*j+1]/scale
                elif cutlass.const_expr(self.normalization_math in ("refine","refine_warp_lut")):
                    a = self.normalize_refined(values[2*j], scale, eligible, normalized_scale, inverse_power, reciprocal)
                    b = self.normalize_refined(values[2*j+1], scale, eligible, normalized_scale, inverse_power, reciprocal)
                else:
                    a,b=values[2*j]/scale,values[2*j+1]/scale
            if cutlass.const_expr(self.normalization_math in ("threshold_scale_free","guarded_scale_free")):
                if scale_free:
                    if cutlass.const_expr(self.normalization_math == "guarded_scale_free"):
                        # a,b remain the exact power-normalized values consumed
                        # below by the UNCHANGED scale-free xx/xy expressions.
                        packing_a=self.guarded_scale_free_proposal(a,values[2*j],scale,guarded_reciprocal)
                        packing_b=self.guarded_scale_free_proposal(b,values[2*j+1],scale,guarded_reciprocal)
                        bits=nvvm.inline_ptx_hl(
                            "{.reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, {$r1}, {$r0}; mov.b16 {$w0}, {b, 0};}",
                            write_only_types=[Int16],read_only_args=[packing_a,packing_b])
                    else:
                        bits=Int16(self.threshold_code(a,*thresholds) | (self.threshold_code(b,*thresholds) << 4))
                else:
                    bits=nvvm.inline_ptx_hl(
                        "{.reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, {$r1}, {$r0}; mov.b16 {$w0}, {b, 0};}",
                        write_only_types=[Int16],read_only_args=[a,b])
            else:
                bits=nvvm.inline_ptx_hl(
                    "{.reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, {$r1}, {$r0}; mov.b16 {$w0}, {b, 0};}",
                    write_only_types=[Int16],read_only_args=[a,b])
            packed[tile*64+lane*2+j]=Uint8(bits)
            qa,qb=self.decode_pair(Int32(bits), self.correction_decode)
            xx+=a*a+b*b
            xy+=a*qa+b*qb
        for shift in cutlass.range_constexpr(2):
            xx+=cute.arch.shuffle_sync_bfly(xx,1 << shift)
            xy+=cute.arch.shuffle_sync_bfly(xy,1 << shift)
        correction=Float32(1.)
        if xy > Float32(0.):
            correction=xx/xy
        output_scale=scale
        if cutlass.const_expr(self.normalization_math in ("threshold_scale_free","guarded_scale_free")):
            if scale_free:
                output_scale=(scale.bitcast(Int32) & Int32(0x7f800000)).bitcast(Float32)
        corrected=(output_scale*correction).to(BFloat16)
        if lane%4 == 0:
            scales[tile*8+lane//4]=corrected
        largest=cute.arch.fmax(largest,corrected.to(Float32))
        return largest

    @cute.jit
    def transform_values(self,values):
        """Unnormalized scalar H128; experimental subclasses may override it.

        Inputs are already sign-multiplied FP32. Keep the original butterfly
        order and arithmetic here; normalization and EDEN stay in prepare_tile.
        """
        lane=cute.arch.lane_idx()
        a,b,c,d=values[0],values[1],values[2],values[3]
        values[0],values[1],values[2],values[3]=a+b,a-b,c+d,c-d
        a,b,c,d=values[0],values[1],values[2],values[3]
        values[0],values[1],values[2],values[3]=a+c,b+d,a-c,b-d
        for level in cutlass.range_constexpr(5):
            for j in cutlass.range_constexpr(4):
                other=cute.arch.shuffle_sync_bfly(values[j],1 << level)
                own=values[j]
                if (lane & (1 << level)) == 0:
                    values[j]=own+other
                else:
                    values[j]=other-own


@lru_cache(maxsize=64)
def compile_hadamard_preparation(rows, cols, config=PosthocScaleConfig(), mantissa_bits=4, transpose=False, tile_order="row_major", load_mode="direct", signs_mode="reload", stages=1, async_copy=False, correction_decode="lut", partial_max_scope="warp", normalization_math="divide", producer_copy_bits=32, producer_unroll_rows=True, producer_row_tile_batch=1, producer_row_load_bits=16, producer_values_per_lane=4, producer_transpose_values_per_lane=4, producer_scale_math="divide", producer_grouped_store_bits=8):
    """Compile original-BF16 preparation; native pair decode is research opt-in.

    correction_decode affects only the local decoded values used by correction,
    not source dtype, transform arithmetic, packing, reductions or finalization.
    """
    kernel=HadamardPreparationKernel(rows,cols,config,mantissa_bits,transpose,tile_order=tile_order,load_mode=load_mode,signs_mode=signs_mode,stages=stages,async_copy=async_copy,correction_decode=correction_decode,partial_max_scope=partial_max_scope,normalization_math=normalization_math,producer_copy_bits=producer_copy_bits,producer_unroll_rows=producer_unroll_rows,producer_row_tile_batch=producer_row_tile_batch,producer_row_load_bits=producer_row_load_bits,producer_values_per_lane=producer_values_per_lane,producer_transpose_values_per_lane=producer_transpose_values_per_lane,producer_scale_math=producer_scale_math,producer_grouped_store_bits=producer_grouped_store_bits)
    def fake(dtype,size):
        return cute.runtime.make_fake_tensor(dtype,(size,),stride=(1,),assumed_align=4)
    alignment=max(4,producer_copy_bits//8) if kernel.async_copy else max(4,producer_row_load_bits//8) if not transpose else 4
    source=cute.runtime.make_fake_tensor(BFloat16,(rows*cols,),stride=(1,),assumed_align=alignment)
    compiled=cute.compile(kernel,source,fake(Float32,128),
        cute.runtime.make_fake_tensor(Uint8,(rows*cols//2,),stride=(1,),assumed_align=max(4,kernel.grouped_store_bits//8)),fake(BFloat16,rows*cols//16),fake(Float32,kernel.partial_count),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 3")
    return compiled,kernel.partial_count


class PackedHadamardPreparationKernel(HadamardPreparationKernel):
    """Decode saved forward FP4 in registers, then rotate and requantize.

    Row-major E4M3 blocks plus one FP32 tensor scale. This is explicitly the
    BF16-roundtrip source policy, not original-BF16 backward and not a native
    scale-layout reader. No decoded or transposed GMEM intermediate is made.
    """
    @cute.jit
    def __call__(self, src: cute.Tensor, source_scales: cute.Tensor,
                 source_outer: cute.Tensor, signs: cute.Tensor, packed: cute.Tensor,
                 scales: cute.Tensor, partials: cute.Tensor, stream: cuda.CUstream):
        self.rotated_kernel(src,signs,packed,scales,partials,source_scales,source_outer).launch(
            grid=(self.blocks,1,1),block=(self.config.threads,1,1),stream=stream)


@lru_cache(maxsize=64)
def compile_packed_hadamard_preparation(rows, cols, config=PosthocScaleConfig(), mantissa_bits=4, transpose=False, tile_order="row_major", load_mode="direct", signs_mode="reload", stages=1, async_copy=False, correction_decode="lut", partial_max_scope="warp", normalization_math="divide", producer_copy_bits=32, producer_unroll_rows=True, producer_row_tile_batch=1, producer_row_load_bits=16, producer_values_per_lane=4, producer_transpose_values_per_lane=4, producer_scale_math="divide", producer_grouped_store_bits=8):
    kernel=PackedHadamardPreparationKernel(rows,cols,config,mantissa_bits,transpose,packed_source=True,tile_order=tile_order,load_mode=load_mode,signs_mode=signs_mode,stages=stages,async_copy=async_copy,correction_decode=correction_decode,partial_max_scope=partial_max_scope,normalization_math=normalization_math,producer_copy_bits=producer_copy_bits,producer_unroll_rows=producer_unroll_rows,producer_row_tile_batch=producer_row_tile_batch,producer_row_load_bits=producer_row_load_bits,producer_values_per_lane=producer_values_per_lane,producer_transpose_values_per_lane=producer_transpose_values_per_lane,producer_scale_math=producer_scale_math,producer_grouped_store_bits=producer_grouped_store_bits)
    def fake(dtype,size):
        return cute.runtime.make_fake_tensor(dtype,(size,),stride=(1,),assumed_align=4)
    compiled=cute.compile(kernel,fake(Uint8,rows*cols//2),
        fake(Float8E4M3FN,rows*cols//16),fake(Float32,1),fake(Float32,128),
        fake(Uint8,rows*cols//2),fake(BFloat16,rows*cols//16),fake(Float32,kernel.partial_count),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 3")
    return compiled,kernel.partial_count


class PackedBF16DecodeKernel(ExtendedPreparationKernel):
    """Explicit staging alternative to the register-fused packed reader."""
    @cute.jit
    def __call__(self, src: cute.Tensor, scales: cute.Tensor, outer: cute.Tensor,
                 out: cute.Tensor, stream: cuda.CUstream):
        self.decode_kernel(src,scales,outer,out).launch(
            grid=(self.config.blocks,1,1),block=(self.config.threads,1,1),stream=stream)

    @cute.kernel
    def decode_kernel(self, src: cute.Tensor, scales: cute.Tensor, outer: cute.Tensor, out: cute.Tensor):
        thread,_,_=cute.arch.thread_idx()
        block,_,_=cute.arch.block_idx()
        for index in cutlass.range(block*self.config.threads+thread,self.count*16,
                                   self.config.blocks*self.config.threads,unroll=1):
            code=(Int32(src[index//2]) >> ((index%2)*4)) & 15
            out[index]=(self.decode(code)*scales[index//16].to(Float32)*outer[0]).to(BFloat16)


@lru_cache(maxsize=64)
def compile_packed_bf16_decode(rows,cols,config=PosthocScaleConfig()):
    if cols%16:
        raise ValueError("packed decode requires complete 16-element blocks")
    kernel=PackedBF16DecodeKernel(rows,cols//16,config,4)
    def fake(dtype,size):
        return cute.runtime.make_fake_tensor(dtype,(size,),stride=(1,),assumed_align=4)
    return cute.compile(kernel,fake(Uint8,rows*cols//2),fake(Float8E4M3FN,rows*cols//16),
        fake(Float32,1),fake(BFloat16,rows*cols),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 3")


class ScaleProductKernel:
    @cute.jit
    def __call__(self, a: cute.Tensor, b: cute.Tensor, out: cute.Tensor, stream: cuda.CUstream):
        self.kernel(a,b,out).launch(grid=(1,1,1),block=(32,1,1),stream=stream)

    @cute.kernel
    def kernel(self, a: cute.Tensor, b: cute.Tensor, out: cute.Tensor):
        thread,_,_=cute.arch.thread_idx()
        if thread == 0:
            out[0]=a[0]*b[0]


@lru_cache(maxsize=1)
def compile_posthoc_scale_product():
    scalar=cute.runtime.make_fake_tensor(Float32,(1,),stride=(1,),assumed_align=4)
    return cute.compile(ScaleProductKernel(),scalar,scalar,scalar,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi --opt-level 3")
