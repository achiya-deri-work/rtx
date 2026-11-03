"""Research row-only H16 preparation. No transformed GMEM intermediate.

Original BF16 inputs, FP32 transform/statistics, BF16 temporary scales and
packed FP4 output. Only guarded transform association differs from scalar.
"""
from functools import lru_cache
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16,Float32,Int16,Int32,Uint8,Uint16
from cutlass.cute.nvgpu.warp import MmaF16BF16Op
from cutlass.experimental.primitives import nvvm_wrapper as nvvm
from rtx.kernels.nvfp4_posthoc import HadamardPreparationKernel,PosthocScaleConfig


class H16PreparationKernel(HadamardPreparationKernel):
    def __init__(self,rows,cols,config=PosthocScaleConfig()):
        super().__init__(rows,cols,config,4,False,signs_mode='reload',correction_decode='native')
        self.pairs=(self.tiles+1)//2
        self.blocks=min(config.blocks,(self.pairs+config.threads//32-1)//(config.threads//32))
        self.work_warp_count=self.blocks*(config.threads//32)
        self.partial_count=self.work_warp_count

    @cute.jit
    def __call__(self,src:cute.Tensor,signs:cute.Tensor,packed:cute.Tensor,
                 scales:cute.Tensor,partials:cute.Tensor,stream:cuda.CUstream):
        self.kernel(src,signs,packed,scales,partials).launch(
            grid=(self.blocks,1,1),block=(self.config.threads,1,1),stream=stream)

    @cute.jit
    def quantize_block(self,values,part,tile,active,packed,scales,largest):
        lane=cute.arch.lane_idx()
        normalized=cute.make_rmem_tensor((4,),Float32)
        maximum=Float32(0.)
        for half in cutlass.range_constexpr(2):
            for j in cutlass.range_constexpr(2):
                value=values[part*2+j,half]*Float32(.08838834764831845)
                normalized[half*2+j]=value
                maximum=cute.arch.fmax(maximum,(value.bitcast(Int32)&Int32(0x7fffffff)).bitcast(Float32))
        for shift in cutlass.range_constexpr(2):
            maximum=cute.arch.fmax(maximum,cute.arch.shuffle_sync_bfly(maximum,1<<shift))
        scale=maximum/Float32(6.)
        scale=(scale.bitcast(Int32)&Int32(~((1<<19)-1))).bitcast(Float32)
        if scale<Float32(1.1754943508222875e-38):scale=Float32(0.)
        norms=cute.make_rmem_tensor((2,),Float32)
        dots=cute.make_rmem_tensor((2,),Float32)
        for half in cutlass.range_constexpr(2):
            a,b=Float32(0.),Float32(0.)
            if scale>Float32(0.):
                a,b=normalized[half*2]/scale,normalized[half*2+1]/scale
            code=nvvm.inline_ptx_hl(
                '{.reg .b8 b; cvt.rn.satfinite.e2m1x2.f32 b, {$r1}, {$r0}; mov.b16 {$w0}, {b, 0};}',
                write_only_types=[Int16],read_only_args=[a,b])
            if active:packed[tile*64+(lane//4)*8+(lane%4)+4*half]=Uint8(code)
            qa,qb=self.decode_pair(Int32(code),'native')
            xx,xy=self.grouped_pair_statistics(a,b,qa,qb)
            # Four lanes own pairs 0/1,2/3,4/5,6/7 within each eight-value
            # half. Reduce adjacent pairs before adding the two halves.
            for shift in cutlass.range_constexpr(2):
                xx=self.grouped_add(xx,cute.arch.shuffle_sync_bfly(xx,1<<shift))
                xy=self.grouped_add(xy,cute.arch.shuffle_sync_bfly(xy,1<<shift))
            norms[half],dots[half]=xx,xy
        xx=self.grouped_add(norms[0],norms[1])
        xy=self.grouped_add(dots[0],dots[1])
        correction=Float32(1.)
        if xy>Float32(0.):correction=xx/xy
        corrected=(scale*correction).to(BFloat16)
        if active and lane%4==0:scales[tile*8+lane//4]=corrected
        if active:largest=cute.arch.fmax(largest,corrected.to(Float32))
        return largest

    @cute.kernel
    def kernel(self,src:cute.Tensor,signs:cute.Tensor,packed:cute.Tensor,
               scales:cute.Tensor,partials:cute.Tensor):
        thread,_,_=cute.arch.thread_idx()
        block,_,_=cute.arch.block_idx()
        lane=cute.arch.lane_idx()
        warp=block*(self.config.threads//32)+thread//32
        largest=Float32(0.)
        cached_signs=cute.make_rmem_tensor((8,),Float32)
        valid=Int32(1)
        for reg in cutlass.range_constexpr(8):
            index=16*(lane//4)+2*(lane%4)+(reg&1)+8*(reg>>2)
            cached_signs[reg]=signs[index]
            if (cached_signs[reg].bitcast(Int32)&Int32(0x7fffffff))!=Int32(0x3f800000):valid=Int32(0)
        rademacher=cute.arch.vote_all_sync(valid!=0)
        mma=cute.make_mma_atom(MmaF16BF16Op(BFloat16,Float32,(16,8,16)))
        coefficients=cute.make_rmem_tensor((4,2),BFloat16)
        for half in cutlass.range_constexpr(2):
            for reg in cutlass.range_constexpr(4):
                k=2*(lane&3)+(reg&1)+8*(reg>>1)
                parity=cute.arch.popc(k&((lane>>2)+8*half))&1
                coefficients[reg,half]=Uint16(0x3f80|(parity<<15)).bitcast(BFloat16)
        for pair in cutlass.range(warp,self.pairs,self.work_warp_count,unroll=1):
            a=cute.make_rmem_tensor((8,1,1),BFloat16)
            valid=Int32(1)
            for reg in cutlass.range_constexpr(8):
                index=16*(lane//4)+2*(lane%4)+(reg&1)+128*((reg>>1)&1)+8*(reg>>2)
                value=Float32(0.)
                if pair*256+index<self.rows*self.cols:value=src[pair*256+index].to(Float32)
                absolute=value.bitcast(Int32)&Int32(0x7fffffff)
                if (absolute!=0)&((absolute<Int32(0x0f800000))|(absolute>Int32(0x6f800000))):valid=Int32(0)
                a[reg,0,0]=(value*cached_signs[reg]).to(BFloat16)
            fast=rademacher&cute.arch.vote_all_sync(valid!=0)
            if fast:
                values=cute.make_rmem_tensor((4,2),Float32)
                for half in cutlass.range_constexpr(2):
                    b=cute.make_rmem_tensor((4,1,1),BFloat16)
                    for reg in cutlass.range_constexpr(4):b[reg,0,0]=coefficients[reg,half]
                    acc=cute.make_rmem_tensor((4,1,1),Float32)
                    acc.fill(Float32(0.))
                    cute.gemm(mma,acc,a,b,acc)
                    for level in cutlass.range_constexpr(3):
                        for reg in cutlass.range_constexpr(4):
                            own=acc[reg,0,0]
                            other=cute.arch.shuffle_sync_bfly(own,1<<(level+2))
                            if (lane&(1<<(level+2)))==0:acc[reg,0,0]=own+other
                            else:acc[reg,0,0]=other-own
                    for reg in cutlass.range_constexpr(4):values[reg,half]=acc[reg,0,0]
                for part in cutlass.range_constexpr(2):
                    tile=pair*2+part
                    largest=self.quantize_block(values,part,tile,tile<self.tiles,packed,scales,largest)
            else:
                for part in cutlass.range_constexpr(2):
                    tile=pair*2+part
                    if tile<self.tiles:
                        largest=self.prepare_tile(src,signs,packed,scales,signs,signs,signs,
                                                  tile,0,largest,Float32(1.))
        self.write_partial_maximum(largest,partials,partials)


class H16GroupedPreparationKernel(H16PreparationKernel):
    """Four blocks per warp, register exchanges, then retained VPL16 quantizer."""
    def __init__(self,rows,cols,config=PosthocScaleConfig(),*,recompute_coefficients=False):
        HadamardPreparationKernel.__init__(self,rows,cols,config,4,False,
            signs_mode='reload',correction_decode='native',producer_values_per_lane=16,
            producer_grouped_store_bits=64)
        self.work_groups=(self.tiles+3)//4
        self.blocks=min(config.blocks,(self.work_groups+config.threads//32-1)//(config.threads//32))
        self.work_warp_count=self.blocks*(config.threads//32)
        self.partial_count=self.work_warp_count
        self.recompute_coefficients=recompute_coefficients

    @cute.jit
    def grouped_transform_values(self,values,local_lane):
        # The H16 MMA and remaining three stages already transformed values.
        return

    @cute.kernel
    def kernel(self,src:cute.Tensor,signs:cute.Tensor,packed:cute.Tensor,
               scales:cute.Tensor,partials:cute.Tensor):
        thread,_,_=cute.arch.thread_idx()
        block,_,_=cute.arch.block_idx()
        lane=cute.arch.lane_idx()
        warp=block*(self.config.threads//32)+thread//32
        largest=Float32(0.)
        cached_signs=cute.make_rmem_tensor((8,),Float32)
        valid=Int32(1)
        for reg in cutlass.range_constexpr(8):
            index=16*(lane//4)+2*(lane%4)+(reg&1)+8*(reg>>2)
            cached_signs[reg]=signs[index]
            if (cached_signs[reg].bitcast(Int32)&Int32(0x7fffffff))!=Int32(0x3f800000):valid=Int32(0)
        rademacher=cute.arch.vote_all_sync(valid!=0)
        mma=cute.make_mma_atom(MmaF16BF16Op(BFloat16,Float32,(16,8,16)))
        coefficients=cute.make_rmem_tensor((4,2),BFloat16)
        if cutlass.const_expr(not self.recompute_coefficients):
            for half in cutlass.range_constexpr(2):
                for reg in cutlass.range_constexpr(4):
                    k=2*(lane&3)+(reg&1)+8*(reg>>1)
                    parity=cute.arch.popc(k&((lane>>2)+8*half))&1
                    coefficients[reg,half]=Uint16(0x3f80|(parity<<15)).bitcast(BFloat16)
        for work in cutlass.range(warp,self.work_groups,self.work_warp_count,unroll=1):
            inputs=cute.make_rmem_tensor((8,2),BFloat16)
            valid=Int32(1)
            for pair in cutlass.range_constexpr(2):
                for reg in cutlass.range_constexpr(8):
                    index=16*(lane//4)+2*(lane%4)+(reg&1)+128*((reg>>1)&1)+8*(reg>>2)+256*pair
                    value=Float32(0.)
                    if work*512+index<self.rows*self.cols:value=src[work*512+index].to(Float32)
                    absolute=value.bitcast(Int32)&Int32(0x7fffffff)
                    if (absolute!=0)&((absolute<Int32(0x0f800000))|(absolute>Int32(0x6f800000))):valid=Int32(0)
                    inputs[reg,pair]=(value*cached_signs[reg]).to(BFloat16)
            fast=rademacher&cute.arch.vote_all_sync(valid!=0)
            if fast:
                values=cute.make_rmem_tensor((16,),Float32)
                if cutlass.const_expr(self.recompute_coefficients):
                    current_lane=nvvm.inline_ptx_hl(
                        '{.reg .u32 z; and.b32 z, {$r0}, 0; '
                        'xor.b32 {$w0}, {$r1}, z;}',
                        write_only_types=[Int32],read_only_args=[work,lane])
                for pair in cutlass.range_constexpr(2):
                    a=cute.make_rmem_tensor((8,1,1),BFloat16)
                    for reg in cutlass.range_constexpr(8):a[reg,0,0]=inputs[reg,pair]
                    for half in cutlass.range_constexpr(2):
                        b=cute.make_rmem_tensor((4,1,1),BFloat16)
                        for reg in cutlass.range_constexpr(4):
                            if cutlass.const_expr(self.recompute_coefficients):
                                # Read lane identity at use; work is an opaque
                                # dependency to keep the calculation in the loop.
                                recomputed_k=2*(current_lane&3)+(reg&1)+8*(reg>>1)
                                recomputed_parity=cute.arch.popc(recomputed_k&((current_lane>>2)+8*half))&1
                                b[reg,0,0]=Uint16(0x3f80|(recomputed_parity<<15)).bitcast(BFloat16)
                            else:b[reg,0,0]=coefficients[reg,half]
                        acc=cute.make_rmem_tensor((4,1,1),Float32)
                        acc.fill(Float32(0.))
                        cute.gemm(mma,acc,a,b,acc)
                        for level in cutlass.range_constexpr(3):
                            for reg in cutlass.range_constexpr(4):
                                own=acc[reg,0,0]
                                other=cute.arch.shuffle_sync_bfly(own,1<<(level+2))
                                if (lane&(1<<(level+2)))==0:acc[reg,0,0]=own+other
                                else:acc[reg,0,0]=other-own
                        for reg in cutlass.range_constexpr(4):values[reg+4*half+8*pair]=acc[reg,0,0]
                for swap in cutlass.range_constexpr(2):
                    reg_bit=1 if cutlass.const_expr(swap==0) else 3
                    lane_bit=swap
                    for reg in cutlass.range_constexpr(16):
                        if cutlass.const_expr((reg&(1<<reg_bit))==0):
                            a,b=values[reg],values[reg+(1<<reg_bit)]
                            other=cute.arch.shuffle_sync_bfly(b if (lane&(1<<lane_bit))==0 else a,1<<lane_bit)
                            if (lane&(1<<lane_bit))==0:
                                values[reg],values[reg+(1<<reg_bit)]=a,other
                            else:
                                values[reg],values[reg+(1<<reg_bit)]=other,b
                regrouped=cute.make_rmem_tensor((16,),Float32)
                for j in cutlass.range_constexpr(16):
                    regrouped[j]=values[(j&3)|((j&4)<<1)|((j&8)>>1)]
                tile=work*4+lane%4
                largest=self.grouped_prepare_values(regrouped,packed,scales,tile,lane//4,
                    tile<self.tiles,largest,Float32(1.))
            else:
                for part in cutlass.range_constexpr(4):
                    tile=work*4+part
                    if tile<self.tiles:
                        largest=self.prepare_tile(src,signs,packed,scales,signs,signs,signs,
                                                  tile,0,largest,Float32(1.))
        self.write_partial_maximum(largest,partials,partials)


@lru_cache(maxsize=64)
def compile_h16_preparation(rows,cols,config=PosthocScaleConfig(),*,grouped_output=False,recompute_coefficients=False):
    if type(grouped_output) is not bool:raise ValueError('grouped_output must be bool')
    if type(recompute_coefficients) is not bool:raise ValueError('recompute_coefficients must be bool')
    if recompute_coefficients and not grouped_output:raise ValueError('coefficient recomputation requires grouped output')
    kernel=(H16GroupedPreparationKernel(rows,cols,config,recompute_coefficients=recompute_coefficients)
            if grouped_output else H16PreparationKernel(rows,cols,config))
    def fake(dtype,size,alignment=4):return cute.runtime.make_fake_tensor(dtype,(size,),stride=(1,),assumed_align=alignment)
    run=cute.compile(kernel,fake(BFloat16,rows*cols),fake(Float32,128),
        fake(Uint8,rows*cols//2,8 if grouped_output else 4),fake(BFloat16,rows*cols//16),fake(Float32,kernel.partial_count),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options='--enable-tvm-ffi --opt-level 3')
    return run,kernel.partial_count
