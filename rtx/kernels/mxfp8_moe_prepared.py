"""Expert-local quantize-once portfolios and batched packed MXFP8 GEMMs.

BF16 transposes are layouts only. Newly quantized FP8 orientations have their
own 32-value blocks; neither leading-axis padding nor scales cross experts.
"""
from dataclasses import replace
from functools import lru_cache

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16,Float8E4M3FN,Float8E8M0FNU,Uint8

from .mxfp8_quant import MXFP8QuantKernel,MXFP8TransposedQuantKernel
from .mxfp8_gemm import MXFP8GemmKernel
from .mxfp8_glu_prequant import MXFP8PrequantGLUKernel
from .mxfp8_mlp import PrequantGLURecomputeDerivative


def _fake(dtype,rows,cols):
    return cute.runtime.make_fake_tensor(dtype,(rows,cols),stride=(cols,1),assumed_align=16)


def _scale_tensor(pointer,rows,k):
    return cute.make_tensor(pointer,cute.make_layout((rows//128,k//128,512),
        stride=(k//128*512,512,1)))


class ExpertQuantOperand(MXFP8QuantKernel):
    def __init__(self,rows,k,config,orientation,interleave):
        super().__init__(rows,k,config)
        self.orientation,self.interleave=orientation,interleave
        self.source_alignment=16 if rows*k%8==0 else 2

    @cute.jit
    def run(self,source,destination,scale,block,expert):
        lane=cute.arch.lane_idx()
        warp=cute.arch.make_warp_uniform(cute.arch.warp_idx())
        rows,k=self.rows,self.k
        layout=cute.make_layout((rows,k),stride=(k,1))
        if cutlass.const_expr(self.orientation=='transpose'):
            layout=cute.make_layout((rows,k),stride=(1,rows))
        if cutlass.const_expr(self.interleave):
            layout=cute.make_layout(((2,rows//2),k),stride=((rows//2*k,k),1))
        src=cute.make_tensor(source.iterator+expert*rows*k,layout)
        out=cute.make_tensor(destination.iterator+expert*rows*self.storage_k,
            cute.make_layout((rows,self.storage_k),stride=(self.storage_k,1)))
        sf=cute.make_tensor(scale.iterator+expert*rows*(self.storage_k//32),
            cute.make_layout((rows,self.storage_k//32),stride=(self.storage_k//32,1)))
        if cutlass.const_expr(self.scale_tile_rows==128):
            sf=_scale_tensor(sf.iterator,rows,self.storage_k)
        threads_per_scale=32//self.config.quant_vec
        scale_in_warp=lane//threads_per_scale
        lane_in_scale=lane%threads_per_scale
        blocks_per_row=self.storage_k//32
        for group in cutlass.range(block*self.config.num_warps+warp,
                                    self.task_groups,self.grid_ctas*self.config.num_warps,unroll=1):
            task=group*self.config.quant_vec+scale_in_warp
            if task<rows*blocks_per_row:
                row,sblock=task//blocks_per_row,task%blocks_per_row
                self._quantize_task(src,out,sf,row,sblock,lane_in_scale,
                    sblock*32+lane_in_scale*self.config.quant_vec,self.scale_tile_rows)


class ExpertTransposeOperand(MXFP8TransposedQuantKernel):
    @cute.jit
    def run(self,source,destination,scale,block,expert):
        cfg=self.config
        src=cute.make_tensor(source.iterator+expert*self.rows*self.k,
            cute.make_layout((self.rows,self.k),stride=(1,self.rows)))
        out=cute.make_tensor(destination.iterator+expert*self.rows*self.storage_k,
            cute.make_layout((self.rows,self.storage_k),stride=(self.storage_k,1)))
        sf=cute.make_tensor(scale.iterator+expert*self.rows*(self.storage_k//32),
            cute.make_layout((self.rows,self.storage_k//32),stride=(self.storage_k//32,1)))
        if cutlass.const_expr(self.scale_tile_rows==128):
            sf=_scale_tensor(sf.iterator,self.rows,self.storage_k)
        stride=cfg.transposed_tile_rows+cfg.transposed_smem_padding
        allocator=cutlass.utils.SmemAllocator()
        physical=allocator.allocate_tensor(BFloat16,
            cute.make_layout((cfg.transposed_tile_k,stride),stride=(stride,1)),byte_alignment=16)
        scratch=allocator.allocate_tensor(Float8E8M0FNU,
            cute.make_layout((cfg.transposed_tile_rows,cfg.transposed_tile_k//32),
                stride=(cfg.transposed_tile_k//32,1)),byte_alignment=16)
        logical=cute.make_tensor(physical.iterator,
            cute.make_layout((cfg.transposed_tile_rows,cfg.transposed_tile_k),stride=(1,stride)))
        for task in cutlass.range(block,self.task_groups,self.grid_ctas,unroll=1):
            self._quantize_transposed_cta(src,out,sf,physical,logical,scratch,task,self.scale_tile_rows,self.k)


class ExpertQuantPortfolio:
    def __init__(self,experts,specs,config,transpose_mode,grid_mode):
        if config.scale_layout not in ('row_major','mma128'):raise ValueError('Unsupported expert scale layout')
        self.experts,self.specs=experts,specs
        self.quantizers=[];self.offsets=[];offset=0
        self.mixed_batches=any(len(spec)==5 for spec in specs)
        self.batch_counts=[]
        for spec in specs:
            if len(spec) not in (4,5):raise ValueError('Quantizer spec needs four fields and optional batch extent')
            rows,k,orientation,interleave=spec[:4]
            batches=spec[4] if len(spec)==5 else experts
            if type(batches) is not int or batches<=0 or batches>experts:
                raise ValueError('Operand batch extent must be in [1,experts]')
            self.batch_counts.append(batches)
            # Global-token X scales remain row-addressable before routing.
            # Expert GEMMs can independently request native physical scales.
            operand_config=replace(config,scale_layout='row_major',native_scale_store='scalar') if len(spec)==5 and batches==1 else config
            if orientation not in ('row','transpose') or (interleave and orientation!='row'):
                raise ValueError('Invalid expert operand layout')
            if orientation=='transpose' and transpose_mode=='staged' and operand_config.transposed_rejection(rows,k) is None:
                q=ExpertTransposeOperand(rows,k,operand_config)
            else:
                q=ExpertQuantOperand(rows,k,replace(operand_config,load_bits=16) if orientation=='transpose' else operand_config,
                                     orientation,interleave)
            if len(spec)==5 and batches==1:
                q.source_alignment=2
            self.quantizers.append(q)
        if grid_mode not in ('operand','expert','portfolio'):raise ValueError('Unsupported quantizer grid policy')
        if grid_mode!='operand':
            sm_count=cutlass.utils.HardwareInfo().get_device_multiprocessor_count()
            budget=max(1,(sm_count*config.persistent_waves+experts-1)//experts)
            elements=sum(spec[0]*spec[1]*(batches if self.mixed_batches else 1)
                         for spec,batches in zip(specs,self.batch_counts))
            for q,spec,batches in zip(self.quantizers,specs,self.batch_counts):
                rows,k=spec[:2]
                if self.mixed_batches:
                    operand_budget=max(1,(sm_count*config.persistent_waves+batches-1)//batches)
                    cap=operand_budget if grid_mode=='expert' else max(1,
                        (sm_count*config.persistent_waves*rows*k+elements-1)//elements)
                else:
                    cap=budget if grid_mode=='expert' else max(1,(budget*rows*k+elements-1)//elements)
                q.grid_ctas=min(q.grid_ctas,cap)
        for q,batches in zip(self.quantizers,self.batch_counts):
            self.offsets.append(offset);offset+=q.grid_ctas*(batches if self.mixed_batches else 1)
        self.grid=offset
        self.threads=config.num_warps*32

    @cute.jit
    def __call__(self,anchor,sources,quantized,scales,stream):
        # CuTe 4.7 EnvStream discovery needs a top-level CUDA tensor; it does
        # not recurse through the portfolio tuples. No caller stream API.
        self.kernel(sources,quantized,scales).launch(grid=(self.grid,1,1 if self.mixed_batches else self.experts),
            block=(self.threads,1,1),stream=stream)

    @cute.kernel
    def kernel(self,sources,quantized,scales):
        block,_,expert=cute.arch.block_idx()
        for i in cutlass.range_constexpr(len(self.specs)):
            span=self.quantizers[i].grid_ctas*(self.batch_counts[i] if self.mixed_batches else 1)
            if block>=self.offsets[i] and block<self.offsets[i]+span:
                local=block-self.offsets[i]
                if cutlass.const_expr(self.mixed_batches):
                    self.quantizers[i].run(sources[i],quantized[i],scales[i],
                        local%self.quantizers[i].grid_ctas,local//self.quantizers[i].grid_ctas)
                else:
                    self.quantizers[i].run(sources[i],quantized[i],scales[i],local,expert)


@lru_cache(maxsize=128)
def compile_expert_quant(experts,specs,config,transpose_mode='staged',grid_mode='operand',_byte_scales=False):
    """Quantize expert operands, optionally together with shared token rows.

    Four-field specs retain the original all-experts layout. A fifth batch
    extent makes work heterogeneous; an explicit singleton emits row-major
    scales, allowing original-token X to be routed after quantization.
    """
    if transpose_mode not in ('direct','staged'):raise ValueError('Unknown transpose preparation mode')
    sources=[];quantized=[];scales=[]
    for spec in specs:
        rows,k,orientation,_=spec[:4]
        batches=spec[4] if len(spec)==5 else experts
        sources.append(cute.runtime.make_fake_tensor(BFloat16,
            (batches*(rows if orientation=='row' else k),k if orientation=='row' else rows),
            stride=(k if orientation=='row' else rows,1),
            assumed_align=2 if len(spec)==5 and batches==1 else 16))
        width=(k+31)//32*32
        quantized.append(_fake(Float8E4M3FN,batches*rows,width))
        scales.append(_fake(Uint8 if _byte_scales else Float8E8M0FNU,batches*rows,width//32))
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    kernel=ExpertQuantPortfolio(experts,specs,config,transpose_mode,grid_mode)
    if _byte_scales:
        from .mxfp8_moe_buffers import ByteQuantPortfolio
        kernel=ByteQuantPortfolio(kernel)
    return cute.compile(kernel,sources[0],tuple(sources),tuple(quantized),tuple(scales),stream,
        options='--enable-tvm-ffi --opt-level 3 '
        f"--ptxas-options '-O3 --maxrregcount={config.maxrregcount}'")


class TypedExpertScaleAux:
    """Keep Torch-visible scale buffers E8M0; reinterpret only inside CuTe."""
    def __init__(self,inner):self.inner=inner

    @cute.jit
    def __call__(self,qx,qw,sx,sw,out,aux,stream):
        raw=cute.make_tensor(cute.recast_ptr(aux.iterator,dtype=Uint8),aux.layout)
        self.inner(qx,qw,sx,sw,out,raw,stream)


@lru_cache(maxsize=128)
def compile_expert_packed_gemm(experts,problem,config,kind='linear',transport='tma',typed_aux_scales=False,_byte_scales=False):
    if config.scale_layout not in ('row_major','mma128'):
        raise ValueError('Batched packed GEMM requires row-major/native128 scales')
    if typed_aux_scales and kind!='glu':raise ValueError('Typed scale auxiliaries are only used by GLU')
    if typed_aux_scales and _byte_scales:raise ValueError('Scale auxiliary cannot be typed and byte-canonical together')
    if kind=='linear':
        cls=MXFP8GemmKernel
        if transport=='cooperative':
            from .mxfp8_moe_cooperative import CooperativeExpertGemm
            cls=CooperativeExpertGemm
        elif transport!='tma':raise ValueError('Unsupported packed transport')
        kernel=cls(problem,config)
        shape=(problem.m,problem.n);aux_shape=None
    elif kind in ('glu','derivative'):
        if transport!='tma':raise ValueError('GLU currently requires TMA transport')
        cls=MXFP8PrequantGLUKernel if kind=='glu' else PrequantGLURecomputeDerivative
        kernel=cls(problem,config,'silu','mxfp8' if kind=='glu' else 'bf16')
        h=problem.n//2
        shape=(problem.m,(h+31)//32*32 if kind=='glu' else problem.n)
        aux_shape=(problem.m,shape[1]//32 if kind=='glu' else h)
        if kind=='derivative':
            kernel.single_stage_recompute_handoff=True
            kernel.emit_hidden=False;kernel.emit_gradient_quant=False;kernel.emit_hidden_quant=False
            kernel.emit_gradient_transpose_quant=False;kernel.separate_hidden=False
            kernel.glu_scratch_elems=0
    else:raise ValueError('Unknown packed expert GEMM kind')
    kernel.batch_count=experts
    kernel.batch_output_shape=shape;kernel.batch_extra_shape=aux_shape
    qx=_fake(Float8E4M3FN,experts*problem.m,problem.k)
    qw=_fake(Float8E4M3FN,experts*problem.n,problem.k)
    sx=_fake(Uint8 if _byte_scales else Float8E8M0FNU,experts*problem.m,problem.k//32)
    sw=_fake(Uint8 if _byte_scales else Float8E8M0FNU,experts*problem.n,problem.k//32)
    out=_fake(Float8E4M3FN if kind=='glu' else BFloat16,experts*shape[0],shape[1])
    aux_dtype=(Float8E8M0FNU if typed_aux_scales else Uint8) if kind=='glu' else BFloat16
    aux=out if aux_shape is None else _fake(aux_dtype,
                                          experts*aux_shape[0],aux_shape[1])
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    if typed_aux_scales:kernel=TypedExpertScaleAux(kernel)
    if _byte_scales:
        from .mxfp8_moe_buffers import BytePackedGemm
        kernel=BytePackedGemm(kernel)
    return cute.compile(kernel,qx,qw,sx,sw,out,aux,stream,
        options='--enable-tvm-ffi --opt-level 3 '
        f"--ptxas-options '-O3 --maxrregcount={config.maxrregcount}'")


class ExpertJointRecompute(PrequantGLURecomputeDerivative):
    """One projection produces both gradient orientations and both hidden ones.

    Reuse the qualified FP32 activation/derivative recipe. Expert rebasing is
    local to this epilogue; the shared GEMM core never interprets nested buffers.
    """

    @cute.jit
    def _local(self,value,rows:cutlass.Constexpr,cols:cutlass.Constexpr,expert):
        return cute.make_tensor(value.iterator+expert*rows*cols,
            cute.make_layout((rows,cols),stride=(cols,1)))

    @cute.jit
    def _glu_epilogue(self,storage,accumulators,coordinates,out,aux,
                      block_m,block_n,tidx,tma_out,tma_out_tensor,out_layout,work_slot):
        expert=cute.arch.block_idx()[2]
        m,n=self.problem.m,self.problem.n
        h=n//2
        dh=self._local(aux[0],m,h,expert)
        qh=self._local(aux[1],m,h,expert)
        sh=self._local(aux[2],m,h//32,expert)
        qht=self._local(aux[3],h,m,expert)
        sht=self._local(aux[4],h,m//32,expert)
        qp=self._local(aux[5],2*m,h,expert)
        sp=self._local(aux[6],2*m,h//32,expert)
        qpt=self._local(aux[7],n,m,expert)
        spt=self._local(aux[8],n,m//32,expert)
        nested=((((dh,dh),qp,sp),((qht,qh,sh)),sht),qpt,spt)
        PrequantGLURecomputeDerivative._glu_epilogue(self,storage,accumulators,coordinates,out,nested,
            block_m,block_n,tidx,tma_out,tma_out_tensor,out_layout,work_slot)


@lru_cache(maxsize=128)
def compile_expert_joint_recompute(experts,problem,config,quant,_byte_scales=False):
    if problem.m%32 or problem.n%64 or config.scale_layout!='row_major':
        raise ValueError('Joint experts require 32-value blocks and row-major scales')
    kernel=ExpertJointRecompute(problem,config,'silu','bf16')
    kernel.single_stage_recompute_handoff=True
    kernel.emit_hidden=True;kernel.separate_hidden=True
    kernel.emit_gradient_quant=True;kernel.emit_hidden_quant=True
    kernel.emit_gradient_transpose_quant=True;kernel.emit_forward_hidden_quant=True
    kernel.gradient_transpose_swizzle=config.tile_n.bit_length()-2
    kernel.gradient_quant_policy=(quant.quant_math,quant.quant_amax)
    kernel.gradient_transpose_quant_policy=kernel.gradient_quant_policy
    kernel.hidden_quant_policy=kernel.gradient_quant_policy
    kernel.glu_scratch_elems=config.tile_m*config.tile_n//2
    kernel.batch_count=experts
    m,n,k=problem.m,problem.n,problem.k;h=n//2
    kernel.batch_output_shape=(m,h)
    qx=_fake(Float8E4M3FN,experts*m,k);qw=_fake(Float8E4M3FN,experts*n,k)
    sf_dtype=Uint8 if _byte_scales else Float8E8M0FNU
    sx=_fake(sf_dtype,experts*m,k//32);sw=_fake(sf_dtype,experts*n,k//32)
    dh=_fake(BFloat16,experts*m,h)
    aux=(dh,_fake(Float8E4M3FN,experts*m,h),_fake(sf_dtype,experts*m,h//32),
         _fake(Float8E4M3FN,experts*h,m),_fake(sf_dtype,experts*h,m//32),
         _fake(Float8E4M3FN,experts*2*m,h),_fake(sf_dtype,experts*2*m,h//32),
         _fake(Float8E4M3FN,experts*n,m),_fake(sf_dtype,experts*n,m//32))
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    if _byte_scales:
        from .mxfp8_moe_buffers import ByteJointRecompute
        kernel=ByteJointRecompute(kernel)
    return cute.compile(kernel,qx,qw,sx,sw,dh,aux,stream,
        options='--enable-tvm-ffi --opt-level 3 '
        f"--ptxas-options '-O3 --maxrregcount={config.maxrregcount}'")


class NativeHiddenScales:
    def __init__(self,e,rows,k,threads):
        self.e,self.rows,self.k,self.threads=e,rows,k,threads

    @cute.jit
    def __call__(self,source,destination,stream):
        self.kernel(source,destination).launch(
            grid=(cute.ceil_div(self.rows*self.k//32,self.threads),1,self.e),
            block=(self.threads,1,1),stream=stream)

    @cute.kernel
    def kernel(self,source,destination):
        block,_,expert=cute.arch.block_idx()
        index=block*self.threads+cute.arch.thread_idx()[0]
        if index<self.rows*self.k//32:
            row,sblock=index//(self.k//32),index%(self.k//32)
            physical=(row%32)*16+((row//32)%4)*4+sblock%4
            target=expert*self.rows*(self.k//32)+(row//128)*(self.k//128)*512+(sblock//4)*512+physical
            destination[target]=source[expert*self.rows*(self.k//32)+index]


@lru_cache(maxsize=64)
def compile_native_hidden_scales(e,rows,k,threads=128):
    fake=cute.runtime.make_fake_tensor(Float8E8M0FNU,(e*rows*k//32,),stride=(1,),assumed_align=16)
    stream=cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(NativeHiddenScales(e,rows,k,threads),fake,fake,stream,
                        options='--enable-tvm-ffi --opt-level 3')
