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


@dataclass(frozen=True)
class PosthocScaleConfig:
    threads: int = 128
    blocks: int = 128
    layout: str = "row_major"

    def __post_init__(self):
        if self.threads not in (64,128,256,512) or type(self.blocks) is not int or self.blocks < 1:
            raise ValueError("invalid post-hoc launch geometry")
        if self.layout not in ("row_major", "mma128"):
            raise ValueError("invalid post-hoc scale layout")


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
    def __call__(self, src: cute.Tensor, maximum: cute.Tensor, uniforms: cute.Tensor,
                 dst: cute.Tensor, outer: cute.Tensor, stream: cuda.CUstream):
        self.kernel(src, maximum, uniforms, dst, outer).launch(
            grid=(self.blocks,1,1), block=(self.config.threads,1,1), stream=stream)

    @cute.kernel
    def kernel(self, src: cute.Tensor, maximum: cute.Tensor, uniforms: cute.Tensor,
               dst: cute.Tensor, outer: cute.Tensor):
        thread,_,_ = cute.arch.thread_idx()
        block,_,_ = cute.arch.block_idx()
        largest = Float32(0.)
        if cutlass.const_expr(self.partial_count == 1):
            largest = maximum[0]
        else:
            lane = cute.arch.lane_idx()
            for j in cutlass.range(lane,self.partial_count,32,unroll=1):
                largest = cute.arch.fmax(largest,maximum[j])
            for level in cutlass.range_constexpr(5):
                largest = cute.arch.fmax(largest,cute.arch.shuffle_sync_bfly(largest,16 >> level))
        scale = Float32(1.)
        if largest > Float32(0.):
            scale = largest * Float32(1. / 448.)
        index = block*self.config.threads+thread
        if index == 0:
            outer[0] = scale
        for i in cutlass.range(index, self.rows*self.groups,
                               self.blocks*self.config.threads, unroll=1):
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
            probability = Float32(0.)
            if high > low:
                probability = (value-low)/(high-low)
            selected = lo
            if uniforms[i] < probability:
                selected = hi
            offset = i
            if cutlass.const_expr(self.config.layout == "mma128"):
                row, group = i//self.groups, i%self.groups
                offset = ((row//128)*(self.groups//8)+group//8)*1024
                offset += (group%8//4)*512+(row%32)*16+(row//32%4)*4+group%4
            dst[offset] = Uint8(selected).bitcast(Float8E4M3FN)


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
    def __init__(self, rows, cols, config, mantissa_bits, transpose, packed_source=False, tile_order="row_major", load_mode="direct"):
        if tile_order not in ("row_major", "reduction_major"):
            raise ValueError("unknown preparation tile order")
        self.tile_order = tile_order
        if load_mode not in ("direct", "coalesced"):
            raise ValueError("unknown preparation load mode")
        self.coalesced = load_mode == "coalesced" and transpose
        self.packed_source = packed_source
        if packed_source and cols % 16:
            raise ValueError("packed source requires full row-major 16-element blocks")
        self.source_rows,self.source_cols,self.transpose=rows,cols,transpose
        self.rows,self.cols=(cols,rows) if transpose else (rows,cols)
        if self.coalesced and self.rows % (config.threads//32):
            raise ValueError("coalesced loader requires complete row groups")
        if self.cols % 128:
            raise ValueError("Hadamard reduction dimension must be divisible by 128")
        super().__init__(self.rows,self.cols//16,config,mantissa_bits)
        self.tiles=self.rows*(self.cols//128)
        self.blocks=min(config.blocks,(self.tiles+config.threads//32-1)//(config.threads//32))
        self.partial_count=self.blocks*(config.threads//32)

    @cute.jit
    def __call__(self, src: cute.Tensor, signs: cute.Tensor, packed: cute.Tensor,
                 scales: cute.Tensor, partials: cute.Tensor, stream: cuda.CUstream):
        self.rotated_kernel(src,signs,packed,scales,partials,signs,signs).launch(
            grid=(self.blocks,1,1),block=(self.config.threads,1,1),stream=stream)

    @cute.kernel
    def rotated_kernel(self, src: cute.Tensor, signs: cute.Tensor, packed: cute.Tensor,
                       scales: cute.Tensor, partials: cute.Tensor,
                       source_scales: cute.Tensor, source_outer: cute.Tensor):
        thread,_,_=cute.arch.thread_idx()
        block,_,_=cute.arch.block_idx()
        lane=cute.arch.lane_idx()
        warp=block*(self.config.threads//32)+thread//32
        largest=Float32(0.)
        first,limit,stride=warp,self.tiles,self.partial_count
        if cutlass.const_expr(self.coalesced):
            staged=cutlass.utils.SmemAllocator().allocate_tensor(
                BFloat16,cute.make_layout((128,self.config.threads//32),
                                         stride=(self.config.threads//32+1,1)),byte_alignment=16)
            first,limit,stride=block,self.tiles//(self.config.threads//32),self.blocks
        for work_tile in cutlass.range(first,limit,stride,unroll=1):
            tile = work_tile
            if cutlass.const_expr(self.tile_order == "reduction_major"):
                tile = (work_tile % self.rows)*(self.cols//128)+work_tile//self.rows
            if cutlass.const_expr(self.coalesced):
                row_group=work_tile//(self.cols//128)
                chunk=work_tile%(self.cols//128)
                if cutlass.const_expr(self.tile_order == "reduction_major"):
                    row_group=work_tile%(self.rows//(self.config.threads//32))
                    chunk=work_tile//(self.rows//(self.config.threads//32))
                tile=(row_group*(self.config.threads//32)+thread//32)*(self.cols//128)+chunk
                for j in cutlass.range_constexpr(4):
                    index=thread+j*self.config.threads
                    source_row=chunk*128+index//(self.config.threads//32)
                    source_col=row_group*(self.config.threads//32)+index%(self.config.threads//32)
                    offset=source_row*self.source_cols+source_col
                    if cutlass.const_expr(self.packed_source):
                        byte=Int32(src[offset//2])
                        code=(byte >> ((offset%2)*4)) & 15
                        value=self.decode(code)*source_scales[offset//16].to(Float32)*source_outer[0]
                        staged[index//(self.config.threads//32),index%(self.config.threads//32)]=value.to(BFloat16)
                    else:
                        staged[index//(self.config.threads//32),index%(self.config.threads//32)]=src[offset]
                cute.arch.sync_threads()
            row=tile//(self.cols//128)
            column=(tile%(self.cols//128))*128+lane*4
            values=cute.make_rmem_tensor((4,),Float32)
            for j in cutlass.range_constexpr(4):
                offset=row*self.cols+column+j
                if cutlass.const_expr(self.transpose):
                    offset=(column+j)*self.source_cols+row
                if cutlass.const_expr(self.coalesced):
                    value=staged[lane*4+j,thread//32].to(Float32)
                elif cutlass.const_expr(self.packed_source):
                    byte = Int32(src[offset//2])
                    code = (byte >> ((offset%2)*4)) & 15
                    value = self.decode(code)*source_scales[offset//16].to(Float32)*source_outer[0]
                    # Explicit round-trip numerical contract, identical to
                    # decoding saved forward operands to BF16 before backward.
                    value = value.to(BFloat16).to(Float32)
                else:
                    value = src[offset].to(Float32)
                values[j]=value*signs[lane*4+j]
            if cutlass.const_expr(self.coalesced):
                cute.arch.sync_threads()
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
            maximum=Float32(0.)
            for j in cutlass.range_constexpr(4):
                values[j]=values[j]*Float32(0.08838834764831845)
                absolute=(values[j].bitcast(Int32)&Int32(0x7fffffff)).bitcast(Float32)
                maximum=cute.arch.fmax(maximum,absolute)
            for shift in cutlass.range_constexpr(2):
                maximum=cute.arch.fmax(maximum,cute.arch.shuffle_sync_bfly(maximum,1 << shift))
            scale=maximum/Float32(6.)
            scale=(scale.bitcast(Int32)&Int32(~((1 << (23-self.mantissa_bits))-1))).bitcast(Float32)
            if scale < Float32(1.1754943508222875e-38):
                scale=Float32(0.)
            xx,xy=Float32(0.),Float32(0.)
            for j in cutlass.range_constexpr(2):
                a,b=Float32(0.),Float32(0.)
                if scale > Float32(0.):
                    a,b=values[2*j]/scale,values[2*j+1]/scale
                bits=nvvm.inline_ptx_hl(
                    "{.reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, {$r1}, {$r0}; mov.b16 {$w0}, {b, 0};}",
                    write_only_types=[Int16],read_only_args=[a,b])
                packed[tile*64+lane*2+j]=Uint8(bits)
                qa,qb=self.decode(Int32(bits)&15),self.decode((Int32(bits)>>4)&15)
                xx+=a*a+b*b
                xy+=a*qa+b*qb
            for shift in cutlass.range_constexpr(2):
                xx+=cute.arch.shuffle_sync_bfly(xx,1 << shift)
                xy+=cute.arch.shuffle_sync_bfly(xy,1 << shift)
            correction=Float32(1.)
            if xy > Float32(0.):
                correction=xx/xy
            corrected=(scale*correction).to(BFloat16)
            if lane%4 == 0:
                scales[tile*8+lane//4]=corrected
            largest=cute.arch.fmax(largest,corrected.to(Float32))
        for level in cutlass.range_constexpr(5):
            largest=cute.arch.fmax(largest,cute.arch.shuffle_sync_bfly(largest,16 >> level))
        if lane == 0:
            partials[warp]=largest


@lru_cache(maxsize=64)
def compile_hadamard_preparation(rows, cols, config=PosthocScaleConfig(), mantissa_bits=4, transpose=False, tile_order="row_major", load_mode="direct"):
    kernel=HadamardPreparationKernel(rows,cols,config,mantissa_bits,transpose,tile_order=tile_order,load_mode=load_mode)
    def fake(dtype,size):
        return cute.runtime.make_fake_tensor(dtype,(size,),stride=(1,),assumed_align=4)
    compiled=cute.compile(kernel,fake(BFloat16,rows*cols),fake(Float32,128),
        fake(Uint8,rows*cols//2),fake(BFloat16,rows*cols//16),fake(Float32,kernel.partial_count),
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
def compile_packed_hadamard_preparation(rows, cols, config=PosthocScaleConfig(), mantissa_bits=4, transpose=False, tile_order="row_major", load_mode="direct"):
    kernel=PackedHadamardPreparationKernel(rows,cols,config,mantissa_bits,transpose,packed_source=True,tile_order=tile_order,load_mode=load_mode)
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
