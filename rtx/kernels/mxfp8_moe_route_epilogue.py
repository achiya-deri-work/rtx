"""Recomputed expert down projection -> FP32 route-gradient partials.

The ordinary BF16 output rounding is preserved before multiplying by dY.
The accumulator loader uses only shared row/column-warp partials. The
cooperative loader exchanges one tile of unweighted FP32 dY through shared
memory. Neither writes the E*C*D expert output to global memory. Final
cross-CTA summation is deterministic.
"""
from dataclasses import replace
from functools import lru_cache

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16,Float32,Float8E4M3FN,Int32,Int64,Uint8
from cutlass.cute.nvgpu import cpasync
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

from .mxfp8_gemm import MXFP8GemmKernel
from .mxfp8_moe_buffers import _e8


class ExpertRouteGradient(MXFP8GemmKernel):
    def __init__(self,problem,config,tokens,experts,logical_capacity,top_k,indexed=False,
                 loader='accumulator'):
        # Direct fused epilogue only; forward's down store schedule remains
        # independently active and may still use TMA.
        config=replace(config,epilogue='direct',store_vec=1,epilogue_stages=1)
        super().__init__(problem,config)
        self.tokens,self.logical_capacity,self.top_k=tokens,logical_capacity,top_k
        self.indexed=indexed
        if loader not in ('accumulator','shared_vec4','warp_vec2'):
            raise ValueError(f'Unknown route-gradient loader: {loader}')
        self.loader=loader
        # Reuse the core's custom epilogue hook, without its GLU interleave.
        self.glu=True;self.glu_interleaved_operands=True
        self.glu_scratch_elems=config.tile_m*config.atom_layout_n
        if loader=='shared_vec4':
            self.glu_scratch_elems+=config.tile_m*(config.tile_n+1)
        self.lanes_per_load_row=min(32,config.tile_n//4)
        self.batch_count=experts
        self.column_tiles=(problem.n+config.tile_n-1)//config.tile_n
        self.batch_output_shape=(problem.m,self.column_tiles)

    def _setup_static_layouts(self):
        super()._setup_static_layouts()
        representative=self.tiled_mma.get_slice(0).partition_C(
            cute.make_identity_tensor((self.config.tile_m,self.config.tile_n)))
        rows=tuple(int(representative[i][0]) for i in range(cute.size(representative)))
        self.row_groups=tuple(tuple(i for i,r in enumerate(rows) if r==row) for row in dict.fromkeys(rows))
        if self.loader=='warp_vec2':
            for indices in self.row_groups:
                cols=tuple(int(representative[i][1]) for i in indices)
                if len(cols)%2 or any(cols[j]%2 or cols[j+1]!=cols[j]+1 for j in range(0,len(cols),2)):
                    raise ValueError('Vector route-gradient loader needs adjacent accumulator column pairs')

    @cute.jit
    def _token(self,ids,inverse,expert,row):
        found=Int32(-1)
        if cutlass.const_expr(self.indexed):
            # The original gather has already validated the same route map.
            if row<self.problem.m:
                token=ids[expert*self.problem.m+row,0]
                if token>=0 and token<self.tokens:found=token
        else:
            if row<self.logical_capacity:
                token=Int32(ids[expert,row])
                if token>=0 and token<self.tokens:
                    for rank in cutlass.range_constexpr(self.top_k):
                        if inverse[token,rank]==expert*self.logical_capacity+row:found=token
        return found

    @cute.jit
    def _dy_offset(self,row,col):
        # Keep 16-byte groups intact but rotate banks by row inside the
        # 4-lane accumulator groups, whose logical rows share one warp.
        return row*self.config.tile_n+(col^((row%8)*4))

    @cute.jit
    def _stage_dy(self,ids,inverse,dy,staged,row_tokens,expert,bm,bn,thread):
        cfg=self.config
        lanes=self.lanes_per_load_row
        group_count=cfg.num_threads//lanes
        group=thread//lanes
        lane=cute.arch.lane_idx()
        load=cute.make_copy_atom(cpasync.CopyG2SOp(),Float32,num_bits_per_copy=128)
        vec=cute.make_layout(4)
        for visit in cutlass.range_constexpr(cute.ceil_div(cfg.tile_m,group_count)):
            row=group+visit*group_count
            token=Int32(-1)
            if row<cfg.tile_m and lane%lanes==0:
                token=self._token(ids,inverse,expert,bm*cfg.tile_m+row)
                row_tokens[row]=token
            token=cute.arch.shuffle_sync(token,lane-lane%lanes)
            if token>=0:
                for part in cutlass.range_constexpr(cfg.tile_n//(4*lanes)):
                    col=(part*lanes+lane%lanes)*4
                    feature=bn*cfg.tile_n+col
                    offset=cute.assume(self._dy_offset(row,col),divby=4)
                    if cutlass.const_expr(self.problem.n%4==0):
                        if feature<self.problem.n:
                            source=cute.assume(token*self.problem.n+feature,divby=4)
                            cute.copy(load,cute.make_tensor(dy.iterator+source,vec),
                                      cute.make_tensor(staged.iterator+offset,vec))
                    else:
                        for part_col in cutlass.range_constexpr(4):
                            if feature+part_col<self.problem.n:
                                staged[offset+part_col]=dy[token,feature+part_col]
        if cutlass.const_expr(self.problem.n%4==0):
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()

    @cute.jit
    def _dot(self,acc,coords,dy,staged,token,lr,bn,indices:cutlass.Constexpr):
        total=Float32(0)
        if cutlass.const_expr(self.loader=='warp_vec2' and self.problem.n%2==0):
            for j in cutlass.range_constexpr(0,len(indices),2):
                first,second=indices[j],indices[j+1]
                column=coords[first][1]
                if column<self.problem.n:
                    offset=cute.assume(token*self.problem.n+column,divby=2)
                    values=nvvm.load_ext(dy.iterator+offset,dtype=Float32,count=2)
                    total+=Float32(BFloat16(acc[first]))*values[0]
                    total+=Float32(BFloat16(acc[second]))*values[1]
        else:
            for j in cutlass.range_constexpr(len(indices)):
                item=indices[j];column=coords[item][1]
                if column<self.problem.n:
                    if cutlass.const_expr(self.loader=='shared_vec4'):
                        upstream=staged[self._dy_offset(lr,column-bn*self.config.tile_n)]
                    else:
                        upstream=dy[token,column]
                    # Keep the ordinary expert BF16 output rounding before
                    # the dot with unweighted FP32 upstream values.
                    total+=Float32(BFloat16(acc[item]))*upstream
        return total

    @cute.jit
    def _glu_epilogue(self,storage,acc,coords,out,aux,bm,bn,thread,
                      tma_out,tma_tensor,out_layout,work_slot):
        cfg=self.config
        cta,_,expert=cute.arch.block_idx()
        raw=cta*cfg.tiles_per_cta+work_slot
        if cutlass.const_expr(cfg.persistent_waves):raw=cta+work_slot*self.grid_ctas
        active=raw<self.total_tiles
        ids,inverse,dy=aux
        scratch=storage.glu_scratch.data_ptr()
        partials=cute.make_tensor(scratch,cute.make_layout((cfg.tile_m,cfg.atom_layout_n),
            stride=(cfg.atom_layout_n,1)))
        staged=partials  # Typed placeholder; unused for accumulator or warp loaders.
        row_tokens=partials
        if cutlass.const_expr(self.loader=='shared_vec4'):
            mapping_start=cfg.tile_m*cfg.atom_layout_n
            row_tokens=cute.make_tensor(cute.recast_ptr(scratch+mapping_start,dtype=Int32),
                                        cute.make_layout(cfg.tile_m))
            staged=cute.make_tensor(scratch+mapping_start+cfg.tile_m,
                                     cute.make_layout(cfg.tile_m*cfg.tile_n))
            self._stage_dy(ids,inverse,dy,staged,row_tokens,expert,bm,bn,thread)
        if thread<cfg.num_mma_warps*32:
            lane=cute.arch.lane_idx();warp=cute.arch.make_warp_uniform(cute.arch.warp_idx())
            for group in cutlass.range_constexpr(len(self.row_groups)):
                indices=self.row_groups[group]
                row=coords[indices[0]][0]
                lr=row-bm*cfg.tile_m
                total=Float32(0)
                if cutlass.const_expr(self.loader=='accumulator'):
                    # Keep the inherited per-accumulator loader as the
                    # reproducible control for both structural alternatives.
                    token=Int32(0);valid=Int32(0)
                    if cutlass.const_expr(self.indexed):
                        if row<self.problem.m:
                            token=ids[expert*self.problem.m+row,0]
                            if token>=0 and token<self.tokens:valid=Int32(1)
                    else:
                        if row<self.logical_capacity:
                            token=Int32(ids[expert,row])
                            if token>=0 and token<self.tokens:
                                for rank in cutlass.range_constexpr(self.top_k):
                                    if inverse[token,rank]==expert*self.logical_capacity+row:valid=Int32(1)
                    for j in cutlass.range_constexpr(len(indices)):
                        item=indices[j];column=coords[item][1]
                        if valid!=0 and column<self.problem.n:
                            total+=Float32(BFloat16(acc[item]))*dy[token,column]
                else:
                    if cutlass.const_expr(self.loader=='shared_vec4'):
                        token=row_tokens[lr]
                    else:
                        token=Int32(-1)
                        if lane%4==0:token=self._token(ids,inverse,expert,row)
                        token=cute.arch.shuffle_sync(token,lane-lane%4)
                    if token>=0:total=self._dot(acc,coords,dy,staged,token,lr,bn,indices)
                total+=cute.arch.shuffle_sync_bfly(total,1)
                total+=cute.arch.shuffle_sync_bfly(total,2)
                if lane%4==0:
                    partials[lr,warp//cfg.atom_layout_m]=total
        cute.arch.sync_threads()
        for row in cutlass.range(thread,cfg.tile_m,cfg.num_threads,unroll=1):
            global_row=bm*cfg.tile_m+row
            if global_row<self.problem.m and active:
                total=Float32(0)
                for part in cutlass.range_constexpr(cfg.atom_layout_n):total+=partials[row,part]
                out[global_row,bn]=total
        # Persistent output tiles may reuse this row-partial storage.
        cute.arch.sync_threads()


class ByteRouteGradient:
    def __init__(self,inner,fallback,input_alignment):
        self.inner,self.fallback,self.input_alignment=inner,fallback,input_alignment

    @cute.jit
    def __call__(self,qx,qw,sx,sw,out,ids,inverse,dy,stream,active_capacity=None):
        if cutlass.const_expr(self.fallback is None):
            self.inner(qx,qw,_e8(sx),_e8(sw),out,(ids,inverse,dy),stream,active_capacity)
        else:
            # Contiguous strides do not imply a zero storage offset. Inspect
            # pointer metadata at launch, never device data; the scalar
            # fallback has the same tile/partials ABI and needs no copy.
            if dy.iterator.toint()%self.input_alignment==0:
                aligned=cute.make_tensor(dy.iterator.align(self.input_alignment),dy.layout)
                self.inner(qx,qw,_e8(sx),_e8(sw),out,(ids,inverse,aligned),stream,active_capacity)
            else:
                self.fallback(qx,qw,_e8(sx),_e8(sw),out,(ids,inverse,dy),stream,active_capacity)


class RouteGradientReduce:
    def __init__(self,e,c,storage_c,columns,threads):
        self.e,self.c,self.storage_c,self.columns,self.threads=e,c,storage_c,columns,threads

    @cute.jit
    def __call__(self,source,out,stream,active_capacity=None):
        self.kernel(source,out,active_capacity).launch(grid=(cute.ceil_div(self.c,self.threads),1,self.e),
            block=(self.threads,1,1),stream=stream)

    @cute.kernel
    def kernel(self,source,out,active_capacity):
        block,_,expert=cute.arch.block_idx();row=block*self.threads+cute.arch.thread_idx()[0]
        if row<self.c:
            total=Float32(0)
            limit=self.c
            if cutlass.const_expr(active_capacity is not None):
                limit=active_capacity[0]
            if row<limit:
                for column in cutlass.range_constexpr(self.columns):
                    total+=source[expert*self.storage_c+row,column]
            out[expert,row]=total


def _fake(dtype,rows,cols,alignment=16):
    return cute.runtime.make_fake_tensor(dtype,(rows,cols),stride=(cols,1),assumed_align=alignment)


@lru_cache(maxsize=128)
def compile_expert_route_gradient(tokens,e,c,k,problem,config,indexed=False,active_prefix=False,
                                  loader='accumulator'):
    kernel=ExpertRouteGradient(problem,config,tokens,e,c,k,indexed,loader)
    alignment=(16 if loader=='shared_vec4' and problem.n%4==0 else
               8 if loader=='warp_vec2' and problem.n%2==0 else 4)
    fallback=ExpertRouteGradient(problem,config,tokens,e,c,k,indexed) if alignment>4 else None
    extra=()
    if active_prefix:
        if config.tiles_per_cta!=1 or config.persistent_waves:
            raise ValueError('Active route-gradient epilogue requires one tile per CTA')
        from .mxfp8_moe_active import fake_prefix
        kernel.active_m_alignment=1
        if fallback is not None:fallback.active_m_alignment=1
        extra=(fake_prefix(),)
    qx=_fake(Float8E4M3FN,e*problem.m,problem.k)
    qw=_fake(Float8E4M3FN,e*problem.n,problem.k)
    sx=_fake(Uint8,e*problem.m,problem.k//32);sw=_fake(Uint8,e*problem.n,problem.k//32)
    out=_fake(Float32,e*problem.m,kernel.column_tiles)
    ids=_fake(Int32,e*problem.m,1) if indexed else _fake(Int64,e,c)
    inverse=_fake(Int64,tokens,k);dy=_fake(Float32,tokens,problem.n,alignment=4)
    return cute.compile(ByteRouteGradient(kernel,fallback,alignment),qx,qw,sx,sw,out,ids,inverse,dy,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),*extra,
        options=f"--enable-tvm-ffi --opt-level 3 --ptxas-options '-O3 --maxrregcount={config.maxrregcount}'")


@lru_cache(maxsize=128)
def compile_route_gradient_reduce(e,c,storage_c,columns,threads,active_prefix=False):
    extra=()
    if active_prefix:
        from .mxfp8_moe_active import fake_prefix
        extra=(fake_prefix(),)
    return cute.compile(RouteGradientReduce(e,c,storage_c,columns,threads),
        _fake(Float32,e*storage_c,columns),_fake(Float32,e,c),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),*extra,
        options='--enable-tvm-ffi --opt-level 3')
