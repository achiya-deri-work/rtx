"""Research-only paired staged maxima and post-hoc scale finalization.

Two independent operands retain their exact rounding, RNG arrays and layouts.
Each launch combines disjoint CTA work, not cross-operand numerical state.
"""
from functools import lru_cache
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16,Float32,Float8E4M3FN,Int16,Int32,Uint8
from cutlass.experimental.primitives import nvvm_wrapper as nvvm
from rtx.kernels.nvfp4_posthoc import PosthocScaleKernel,PosthocScaleConfig
from rtx.kernels.ms_eden_paired_finalizer import PairedScaleKernel


class FinalizerLeg(PosthocScaleKernel):
    def __init__(self,*args,pair_values=False,bit_brackets=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.pair_values=pair_values
        self.bit_brackets=bit_brackets

    @cute.jit
    def finalize_one(self,src,dst,i,scale,uniform):
        if cutlass.const_expr(not self.bit_brackets):
            PosthocScaleKernel.finalize_one(self,src,dst,i,scale,uniform)
        else:
            value=cute.arch.fmin(src[i].to(Float32)/scale,Float32(448.))
            if value.bitcast(Int32)>=Int32(0):
                low_code=Int32(0)
                low_value=Float32(0.)
                if value>=Float32(.015625):
                    low_code=(value.bitcast(Int32)>>20)-Int32(960)
                    low_value=((low_code+Int32(960))<<20).bitcast(Float32)
                else:
                    low_code=nvvm.inline_ptx_hl('cvt.rzi.s32.f32 {$w0}, {$r0};',
                        write_only_types=[Int32],read_only_args=[value*Float32(512.)])
                    low_value=low_code.to(Float32)*Float32(1./512.)
                high_code=low_code
                high_value=low_value
                if value>low_value:
                    high_code=low_code+Int32(1)
                    if high_code>=Int32(8):
                        high_value=((high_code+Int32(960))<<20).bitcast(Float32)
                    else:
                        high_value=high_code.to(Float32)*Float32(1./512.)
                probability=self.rounding_probability(value,low_value,high_value)
                selected=low_code
                if uniform<probability:selected=high_code
                offset=i
                if cutlass.const_expr(self.config.layout=='mma128'):
                    row,group=i//self.groups,i%self.groups
                    offset=((row//128)*(self.groups//8)+group//8)*1024
                    offset+=(group%8//4)*512+(row%32)*16+(row//32%4)*4+group%4
                dst[offset]=Uint8(selected).bitcast(Float8E4M3FN)
            else:
                # Preserve the pre-existing behavior outside positive scales.
                PosthocScaleKernel.finalize_one(self,src,dst,i,scale,uniform)

    @cute.jit
    def finalize_range(self,block,src,maximum,uniforms,dst,outer):
        scale=self.tensor_scale(maximum)
        self.finalize_scaled(block,src,scale,uniforms,dst,outer)

    @cute.jit
    def finalize_inline(self,block,src,partials,maximum,uniforms,dst,outer,count:cutlass.Constexpr,scope:cutlass.Constexpr):
        lane=cute.arch.lane_idx()
        thread,_,_=cute.arch.thread_idx()
        largest=Float32(0.)
        if cutlass.const_expr(scope=='cta'):
            for i in cutlass.range(thread,count,self.config.threads,unroll=1):
                largest=cute.arch.fmax(largest,partials[i])
        else:
            for i in cutlass.range(lane,count,32,unroll=1):
                largest=cute.arch.fmax(largest,partials[i])
        for level in cutlass.range_constexpr(5):
            largest=cute.arch.fmax(largest,cute.arch.shuffle_sync_bfly(largest,16>>level))
        if cutlass.const_expr(scope=='cta'):
            allocator=cutlass.utils.SmemAllocator()
            maxima=allocator.allocate_tensor(Float32,cute.make_layout((self.config.threads//32,)),byte_alignment=4)
            if lane==0:maxima[thread//32]=largest
            cute.arch.sync_threads()
            largest=Float32(0.)
            for warp in cutlass.range_constexpr(self.config.threads//32):
                largest=cute.arch.fmax(largest,maxima[warp])
        # Preserve the auxiliary maximum output. No CTA reads this output in
        # inline mode: the reduction is local to a warp/CTA, with no grid sync.
        if block==0 and thread==0:maximum[0]=largest
        scale=Float32(1.)
        if largest>Float32(0.):scale=largest*Float32(1./448.)
        self.finalize_scaled(block,src,scale,uniforms,dst,outer)

    @cute.jit
    def finalize_scaled(self,block,src,scale,uniforms,dst,outer):
        thread,_,_=cute.arch.thread_idx()
        index=block*self.config.threads+thread
        if index==0:outer[0]=scale
        if cutlass.const_expr(self.pair_values):
            for i in cutlass.range(index*2,self.rows*self.groups,
                                  self.blocks*self.config.threads*2,unroll=1):
                first=cute.arch.fmin(src[i].to(Float32)/scale,Float32(448.))
                second=Float32(0.)
                if i+1<self.rows*self.groups:
                    second=cute.arch.fmin(src[i+1].to(Float32)/scale,Float32(448.))
                packed=nvvm.inline_ptx_hl(
                    'cvt.rn.satfinite.e4m3x2.f32 {$w0}, {$r0}, {$r1};',
                    write_only_types=[Int16],read_only_args=[second,first])
                PairedScaleKernel.store_code(self,dst,i,first,Int32(packed&Int16(255)),uniforms[i])
                if i+1<self.rows*self.groups:
                    PairedScaleKernel.store_code(self,dst,i+1,second,
                        Int32((packed>>Int16(8))&Int16(255)),uniforms[i+1])
        else:
            for i in cutlass.range(index,self.rows*self.groups,
                                  self.blocks*self.config.threads,unroll=1):
                self.finalize_one(src,dst,i,scale,uniforms[i])


class DualFinalizer:
    def __init__(self,rows_a,groups_a,rows_b,groups_b,config_a,config_b,count_a,count_b,inline_max=False,inline_scope='warp',pair_values=False,bit_brackets=False):
        if type(pair_values) is not bool:raise ValueError('pair_values must be bool')
        if type(bit_brackets) is not bool:raise ValueError('bit_brackets must be bool')
        if pair_values and bit_brackets:raise ValueError('paired conversion and bit brackets are separate experiments')
        if type(inline_max) is not bool:raise ValueError('inline_max must be bool')
        if inline_scope not in ('warp','cta'):raise ValueError('invalid inline scope')
        if not inline_max and inline_scope!='warp':raise ValueError('CTA scope requires inline_max')
        if config_a.threads!=config_b.threads:
            raise ValueError('dual finalizer requires equal thread counts')
        if min(count_a,count_b)<1:
            raise ValueError('positive partial counts required')
        self.a=FinalizerLeg(rows_a,groups_a,config_a,1,pair_values=pair_values,bit_brackets=bit_brackets)
        self.b=FinalizerLeg(rows_b,groups_b,config_b,1,pair_values=pair_values,bit_brackets=bit_brackets)
        self.count_a,self.count_b=count_a,count_b
        self.inline_max=inline_max
        self.inline_scope=inline_scope

    @cute.jit
    def reduce_one(self,partials,maximum,count:cutlass.Constexpr):
        lane=cute.arch.lane_idx()
        largest=Float32(0.)
        for i in cutlass.range(lane,count,32,unroll=1):
            largest=cute.arch.fmax(largest,partials[i])
        for level in cutlass.range_constexpr(5):
            largest=cute.arch.fmax(largest,cute.arch.shuffle_sync_bfly(largest,16>>level))
        if lane==0:maximum[0]=largest

    @cute.kernel
    def reduce(self,pa:cute.Tensor,pb:cute.Tensor,ma:cute.Tensor,mb:cute.Tensor):
        block,_,_=cute.arch.block_idx()
        if block==0:self.reduce_one(pa,ma,self.count_a)
        else:self.reduce_one(pb,mb,self.count_b)

    @cute.kernel
    def finalize(self,sa:cute.Tensor,ma:cute.Tensor,ua:cute.Tensor,da:cute.Tensor,oa:cute.Tensor,
                 sb:cute.Tensor,mb:cute.Tensor,ub:cute.Tensor,db:cute.Tensor,ob:cute.Tensor):
        block,_,_=cute.arch.block_idx()
        if block<self.a.blocks:
            self.a.finalize_range(block,sa,ma,ua,da,oa)
        else:
            self.b.finalize_range(block-self.a.blocks,sb,mb,ub,db,ob)

    @cute.kernel
    def finalize_inline(self,sa:cute.Tensor,pa:cute.Tensor,ma:cute.Tensor,ua:cute.Tensor,da:cute.Tensor,oa:cute.Tensor,
                        sb:cute.Tensor,pb:cute.Tensor,mb:cute.Tensor,ub:cute.Tensor,db:cute.Tensor,ob:cute.Tensor):
        block,_,_=cute.arch.block_idx()
        if block<self.a.blocks:
            self.a.finalize_inline(block,sa,pa,ma,ua,da,oa,self.count_a,self.inline_scope)
        else:
            self.b.finalize_inline(block-self.a.blocks,sb,pb,mb,ub,db,ob,self.count_b,self.inline_scope)

    @cute.jit
    def __call__(self,sa:cute.Tensor,pa:cute.Tensor,ma:cute.Tensor,ua:cute.Tensor,da:cute.Tensor,oa:cute.Tensor,
                 sb:cute.Tensor,pb:cute.Tensor,mb:cute.Tensor,ub:cute.Tensor,db:cute.Tensor,ob:cute.Tensor,
                 stream:cuda.CUstream):
        if cutlass.const_expr(self.inline_max):
            self.finalize_inline(sa,pa,ma,ua,da,oa,sb,pb,mb,ub,db,ob).launch(
                grid=(self.a.blocks+self.b.blocks,1,1),block=(self.a.config.threads,1,1),stream=stream)
        else:
            self.reduce(pa,pb,ma,mb).launch(grid=(2,1,1),block=(32,1,1),stream=stream)
            self.finalize(sa,ma,ua,da,oa,sb,mb,ub,db,ob).launch(
                grid=(self.a.blocks+self.b.blocks,1,1),block=(self.a.config.threads,1,1),stream=stream)


@lru_cache(maxsize=64)
def compile_dual_finalizer(rows_a,groups_a,rows_b,groups_b,config_a=PosthocScaleConfig(),
                           config_b=PosthocScaleConfig(),count_a=1,count_b=1,*,inline_max=False,inline_scope='warp',pair_values=False,bit_brackets=False):
    kernel=DualFinalizer(rows_a,groups_a,rows_b,groups_b,config_a,config_b,count_a,count_b,inline_max,inline_scope,pair_values,bit_brackets)
    def fake(dtype,size):return cute.runtime.make_fake_tensor(dtype,(size,),stride=(1,),assumed_align=4)
    def operand(rows,groups,count):
        return (fake(BFloat16,rows*groups),fake(Float32,count),fake(Float32,1),
                fake(Float32,rows*groups),fake(Float8E4M3FN,rows*groups),fake(Float32,1))
    return cute.compile(kernel,*operand(rows_a,groups_a,count_a),*operand(rows_b,groups_b,count_b),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options='--enable-tvm-ffi --opt-level 3')
