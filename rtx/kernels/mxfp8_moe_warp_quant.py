"""Warp-local BF16 logical-transpose quantization; no shared-memory transpose.

Each warp owns feature pairs and independent 32-token scale blocks. BF16
loads are coalesced across features; four-byte register transposes then give
coalesced FP8 stores along tokens. All scales belong to this orientation.
"""
import math
from functools import lru_cache

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16,Float32,Int16,Int32,Uint8,Uint16,Uint32
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

from .mxfp8_moe_input import rounded_prefix
from .mxfp8_quant import MXFP8QuantKernel


class ExpertWarpTransposeOperand(MXFP8QuantKernel):
    def __init__(self,rows,k,config,token_lanes=8,raster='features'):
        super().__init__(rows,k,config)
        if token_lanes not in (2,4,8) or raster not in ('features','tokens'):
            raise ValueError('Warp transpose needs 2/4/8 token lanes and features/tokens raster')
        if config.native_scale_store!='scalar':raise ValueError('Warp transpose currently emits scalar scale bytes')
        self.token_lanes,self.raster=token_lanes,raster
        self.feature_pairs=32//token_lanes
        self.member_mask=sum(1<<(i*self.feature_pairs) for i in range(token_lanes))
        self.features_per_warp=2*self.feature_pairs
        self.rounds=32//token_lanes
        self.words=self.rounds//4
        self.feature_tiles=cute.ceil_div(rows,self.features_per_warp)
        self.token_blocks=cute.ceil_div(k,32)
        self.task_groups=self.feature_tiles*self.token_blocks
        self.source_expert_stride=rows*k
        self.grid_ctas=min(cute.ceil_div(self.task_groups,config.num_warps),
            cutlass.utils.HardwareInfo().get_device_multiprocessor_count()*config.persistent_waves)

    @cute.jit
    def transpose_word(self,word,lane_token):
        peer=cute.arch.shuffle_sync_bfly(word,self.feature_pairs)
        value=Uint32(0)
        if (lane_token&1)==0:
            value=nvvm.inline_ptx_hl('prmt.b32 {$w0}, {$r0}, {$r1}, 0x5140;',
                write_only_types=[Uint32],read_only_args=[word,peer])
        else:
            value=nvvm.inline_ptx_hl('prmt.b32 {$w0}, {$r0}, {$r1}, 0x7362;',
                write_only_types=[Uint32],read_only_args=[peer,word])
        if cutlass.const_expr(self.token_lanes>=4):
            peer=cute.arch.shuffle_sync_bfly(value,2*self.feature_pairs)
            if (lane_token&2)==0:
                value=nvvm.inline_ptx_hl('prmt.b32 {$w0}, {$r0}, {$r1}, 0x5410;',
                    write_only_types=[Uint32],read_only_args=[value,peer])
            else:
                value=nvvm.inline_ptx_hl('prmt.b32 {$w0}, {$r0}, {$r1}, 0x7632;',
                    write_only_types=[Uint32],read_only_args=[peer,value])
        return value

    @cute.jit
    def maximum(self,value,lane_feature):
        if cutlass.const_expr(self.config.reduction=='redux'):
            mask=Int32(self.member_mask)<<lane_feature
            return nvvm.redux_sync(value,nvvm.ReductionKind.UMAX,mask)
        for level in cutlass.range_constexpr(int(math.log2(self.token_lanes))):
            value=cutlass.max(value,cute.arch.shuffle_sync_bfly(value,self.feature_pairs*(1<<level)))
        return value

    @cute.jit
    def load_pair(self,src,index,token,row,expert,mapping):
        bits=Uint32(0)
        if token<self.k and row<self.rows:
            pointer=src.iterator+index
            if row+1<self.rows and pointer.toint()%4==0:
                bits=nvvm.inline_ptx_hl('ld.global.u32 {$w0}, [{$r0}];',
                    write_only_types=[Uint32],read_only_args=[pointer])
            else:
                bits=Uint32(src[index].bitcast(Uint16))
                if row+1<self.rows:bits|=Uint32(src[index+1].bitcast(Uint16))<<16
        return bits

    @cute.jit
    def run(self,source,destination,scale,block,expert,mapping=None,active_blocks=None):
        lane=cute.arch.lane_idx();warp=cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane_feature=lane%self.feature_pairs;lane_token=lane//self.feature_pairs
        src=cute.make_tensor(source.iterator+expert*self.source_expert_stride,cute.make_layout((self.rows*self.k,)))
        sf=cute.make_tensor(scale.iterator+expert*self.rows*(self.storage_k//32),
                            cute.make_layout((self.rows*(self.storage_k//32),)))
        token_blocks=self.token_blocks
        if cutlass.const_expr(active_blocks is not None):
            token_blocks=active_blocks
        for task in cutlass.range(block*self.config.num_warps+warp,self.feature_tiles*token_blocks,
                                 self.grid_ctas*self.config.num_warps,unroll=1):
            if cutlass.const_expr(self.raster=='features'):
                feature_tile=task%self.feature_tiles;token_block=task//self.feature_tiles
            else:
                feature_tile=task//token_blocks;token_block=task%token_blocks
            row=feature_tile*self.features_per_warp+2*lane_feature
            data=cute.make_rmem_tensor((self.rounds,),Uint32);data.fill(Uint32(0))
            maximum_a=Int32(0);maximum_b=Int32(0)
            for item in cutlass.range_constexpr(self.rounds):
                token=token_block*32+lane_token+item*self.token_lanes
                index=token*self.rows+row
                bits=self.load_pair(src,index,token,row,expert,mapping)
                data[item]=bits
                maximum_a=cutlass.max(maximum_a,Int32(bits&Uint32(0x7fff)))
                maximum_b=cutlass.max(maximum_b,Int32((bits>>16)&Uint32(0x7fff)))
            maximum_a=self.maximum(maximum_a,lane_feature)
            maximum_b=self.maximum(maximum_b,lane_feature)
            sf_a,inv_a=self._scale_from_amax(Float32(Uint16(maximum_a).bitcast(BFloat16)))
            sf_b,inv_b=self._scale_from_amax(Float32(Uint16(maximum_b).bitcast(BFloat16)))
            qa=cute.make_rmem_tensor((self.words,),Uint32);qa.fill(Uint32(0))
            qb=cute.make_rmem_tensor((self.words,),Uint32);qb.fill(Uint32(0))
            for item in cutlass.range_constexpr(self.rounds):
                bits=data[item]
                if cutlass.const_expr(self.config.quant_math=='bf16x2'):
                    reciprocal=Int32(BFloat16(inv_a).bitcast(Uint16))|(Int32(BFloat16(inv_b).bitcast(Uint16))<<16)
                    scaled=nvvm.mul_bf16x2(Int32(bits),reciprocal)
                    packed=nvvm.inline_ptx_hl('cvt.rn.satfinite.e4m3x2.bf16x2 {$w0}, {$r0};',
                        write_only_types=[Int16],read_only_args=[scaled])
                    a=Uint32(packed&Int16(255));b=Uint32((packed>>8)&Int16(255))
                else:
                    va=Float32(Uint16(bits&Uint32(65535)).bitcast(BFloat16))*inv_a
                    vb=Float32(Uint16(bits>>16).bitcast(BFloat16))*inv_b
                    packed=nvvm.inline_ptx_hl('cvt.rn.satfinite.e4m3x2.f32 {$w0}, {$r0}, {$r1};',
                        write_only_types=[Int16],read_only_args=[va,vb])
                    a=Uint32((packed>>8)&Int16(255));b=Uint32(packed&Int16(255))
                qa[item//4]|=a<<((item%4)*8);qb[item//4]|=b<<((item%4)*8)
            position=lane_token*4
            if cutlass.const_expr(self.token_lanes>=4):
                order=((lane_token&1)<<1)|((lane_token>>1)&1)
                position=order*self.token_lanes+(lane_token//4)*4
            for word in cutlass.range_constexpr(self.words):
                a=self.transpose_word(qa[word],lane_token);b=self.transpose_word(qb[word],lane_token)
                token=token_block*32+word*4*self.token_lanes+position
                if row<self.rows:
                    nvvm.inline_ptx_hl('st.global.u32 [{$r0}], {$r1};',write_only_types=[],
                        read_only_args=[destination.iterator+expert*self.rows*self.storage_k+row*self.storage_k+token,a])
                if row+1<self.rows:
                    nvvm.inline_ptx_hl('st.global.u32 [{$r0}], {$r1};',write_only_types=[],
                        read_only_args=[destination.iterator+expert*self.rows*self.storage_k+(row+1)*self.storage_k+token,b])
            if lane_token==0:
                for feature in cutlass.range_constexpr(2):
                    r=row+feature
                    if r<self.rows:
                        offset=r*(self.storage_k//32)+token_block
                        if cutlass.const_expr(self.scale_tile_rows==128):
                            offset=((r//128)*(self.storage_k//128*512)
                                    +(token_block//4)*512+(r%32)*16+((r//32)%4)*4+token_block%4)
                        sf[offset]=sf_a if feature==0 else sf_b


class IndexedExpertTranspose(ExpertWarpTransposeOperand):
    """Load original BF16 X through validated routed token indices.

    The small compiler-owned map contains -1 for dummy/gap slots. Quantized
    32-slot blocks and their scales are identical to materialized gathering.
    """
    def __init__(self,n,d,e,storage_c,config,token_lanes,raster,grid_mode,active_tile_k=0):
        if type(active_tile_k) is not int or active_tile_k < 0 or active_tile_k % 32:
            raise ValueError('Invalid indexed X transpose consumer K tile')
        if active_tile_k and (storage_c % 32 or config.scale_layout != 'row_major'):
            raise ValueError('Bounded indexed X needs complete row-major scale blocks')
        super().__init__(d,storage_c,config,token_lanes,raster)
        self.active_tile_k=active_tile_k
        self.source_expert_stride=0;self.tokens=n;self.experts=e
        if grid_mode!='operand':
            cap=max(1,(cutlass.utils.HardwareInfo().get_device_multiprocessor_count()*config.persistent_waves+e-1)//e)
            self.grid_ctas=min(self.grid_ctas,cap)

    @cute.jit
    def load_pair(self,src,index,token,row,expert,mapping):
        original=Int32(-1)
        if token<self.k:original=mapping[expert*self.k+token,0]
        bits=Uint32(0)
        if original>=0 and original<self.tokens:
            bits=ExpertWarpTransposeOperand.load_pair(self,src,original*self.rows+row,
                                                     Int32(0),row,expert,mapping)
        return bits

    @cute.jit
    def __call__(self,x,mapping,out,sf,stream,active_capacity=None):
        self.kernel(x,mapping,out,sf,active_capacity).launch(grid=(self.grid_ctas,1,self.experts),
            block=(self.config.num_warps*32,1,1),stream=stream)

    @cute.kernel
    def kernel(self,x,mapping,out,sf,active_capacity):
        block,_,expert=cute.arch.block_idx()
        active_blocks=None
        if cutlass.const_expr(self.active_tile_k):
            # Every visited scale block remains at its original coordinates.
            # The producer of the index map defines all tokens through this
            # same K envelope (including -1 for the last tile's dummy rows).
            active_blocks=rounded_prefix(active_capacity[0],self.k,self.active_tile_k)//32
        self.run(x,out,sf,block,expert,mapping,active_blocks)


class ByteIndexedTranspose:
    def __init__(self,inner):self.inner=inner

    @cute.jit
    def __call__(self,x,mapping,out,sf,stream,active_capacity=None):
        typed=cute.make_tensor(cute.recast_ptr(sf.iterator,dtype=cutlass.Float8E8M0FNU),sf.layout)
        self.inner(x,mapping,out,typed,stream,active_capacity)


@lru_cache(maxsize=128)
def compile_indexed_expert_transpose(n,d,e,storage_c,config,token_lanes=8,raster='features',
                                      grid_mode='operand',active_tile_k=0):
    """Append a device i32 prefix only for a nonzero dW consumer tile K."""
    def fake(dtype,rows,cols,alignment=16):
        return cute.runtime.make_fake_tensor(dtype,(rows,cols),stride=(cols,1),assumed_align=alignment)
    extra=()
    if active_tile_k:
        from .mxfp8_moe_active import fake_prefix
        extra=(fake_prefix(),)
    inner=IndexedExpertTranspose(n,d,e,storage_c,config,token_lanes,raster,grid_mode,active_tile_k)
    return cute.compile(ByteIndexedTranspose(inner),
        fake(BFloat16,n,d,2),fake(Int32,e*storage_c,1),fake(cutlass.Float8E4M3FN,e*d,storage_c),
        fake(Uint8,e*d,storage_c//32),cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),*extra,
        options=f"--enable-tvm-ffi --opt-level 3 --ptxas-options '-O3 --maxrregcount={config.maxrregcount}'")
