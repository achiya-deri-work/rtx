"""Research-only interleaved row/transpose dY preparation grid.

Each physical CTA executes exactly ONE original producer role. Both roles
complete before either finalizer may consume their outputs. Row CTAs pay the
common kernel's maximum register/shared-memory envelope; this does not promise
better occupancy or latency. Original BF16 data is read twice, not copied.
"""
from functools import lru_cache

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils
from cutlass import BFloat16,Float32,Uint8

from .nvfp4_posthoc import HadamardPreparationKernel


class DualDYPreparationKernel:
    """Bounded current-winner prototype, with no alternative numerical modes."""
    def __init__(self,row_kernel,transpose_kernel):
        for kernel in (row_kernel,transpose_kernel):
            if type(kernel) is not HadamardPreparationKernel:
                raise ValueError("dual dY requires original HadamardPreparationKernel instances")
            required=dict(packed_source=False,mantissa_bits=4,signs_mode="register",
                correction_decode="native",normalization_math="divide",partial_max_scope="warp",
                producer_scale_math="divide",producer_copy_bits=32,producer_row_tile_batch=1,
                producer_row_load_bits=16,producer_unroll_rows=True)
            for name,value in required.items():
                if type(getattr(kernel,name)) is not type(value) or getattr(kernel,name)!=value:
                    raise ValueError(f"dual dY unsupported {name}: expected {value!r}")
        if row_kernel.transpose or row_kernel.coalesced or row_kernel.values_per_lane!=4 or row_kernel.tile_order!="row_major":
            raise ValueError("dual dY row role requires row-major original VPL4")
        if (not transpose_kernel.transpose or not transpose_kernel.coalesced
                or transpose_kernel.rows_per_warp!=8 or not transpose_kernel.async_copy
                or transpose_kernel.stages!=1 or transpose_kernel.values_per_lane!=16
                or transpose_kernel.grouped_store_bits!=32 or transpose_kernel.tile_order!="reduction_major"):
            raise ValueError("dual dY transpose role requires reduction-major VPL16/coalesced_x8/async1/store32")
        if (row_kernel.source_rows,row_kernel.source_cols)!=(transpose_kernel.source_rows,transpose_kernel.source_cols):
            raise ValueError("dual dY roles must share source shape")
        if row_kernel.config.threads!=transpose_kernel.config.threads or row_kernel.blocks!=transpose_kernel.blocks:
            raise ValueError("dual dY requires equal role CTA counts and common threads")
        self.row,self.transposed=row_kernel,transpose_kernel
        self.rows,self.cols=row_kernel.source_rows,row_kernel.source_cols
        self.blocks=row_kernel.blocks
        self.threads=row_kernel.config.threads
        self.grid_blocks=2*self.blocks
        self.partial_counts=(row_kernel.partial_count,transpose_kernel.partial_count)
        # Explicit shared union: row scratch is zero for warp partial maxima.
        # One allocation, NOT two per-role allocators with cumulative offsets.
        self.shared_bytes=2*(127*transpose_kernel.async_row_stride+transpose_kernel.source_tile_rows)

    @cute.jit
    def __call__(self,source:cute.Tensor,row_signs:cute.Tensor,row_packed:cute.Tensor,
                 row_scales:cute.Tensor,row_partials:cute.Tensor,transpose_signs:cute.Tensor,
                 transpose_packed:cute.Tensor,transpose_scales:cute.Tensor,
                 transpose_partials:cute.Tensor,stream:cuda.CUstream):
        self.kernel(source,row_signs,row_packed,row_scales,row_partials,transpose_signs,
                    transpose_packed,transpose_scales,transpose_partials).launch(
            grid=(self.grid_blocks,1,1),block=(self.threads,1,1),stream=stream)

    @cute.kernel
    def kernel(self,source:cute.Tensor,row_signs:cute.Tensor,row_packed:cute.Tensor,
               row_scales:cute.Tensor,row_partials:cute.Tensor,transpose_signs:cute.Tensor,
               transpose_packed:cute.Tensor,transpose_scales:cute.Tensor,
               transpose_partials:cute.Tensor):
        physical,_,_=cute.arch.block_idx()
        logical=physical//2
        allocator=cutlass.utils.SmemAllocator()
        staged=allocator.allocate_tensor(BFloat16,
            cute.make_layout((128,self.transposed.source_tile_rows,1),
                stride=(self.transposed.async_row_stride,1,128*self.transposed.async_row_stride)),
            byte_alignment=16)
        # Predicate is CTA-uniform: never split roles between warps because
        # transpose staging uses full-CTA barriers and cp.async group histories.
        if physical%2==0:
            self.prepare_row(source,row_signs,row_packed,row_scales,row_partials,logical)
        else:
            self.prepare_transpose(source,transpose_signs,transpose_packed,transpose_scales,
                                   transpose_partials,logical,staged)

    @cute.jit
    def write_remapped_partial(self,largest,partials,logical):
        thread,_,_=cute.arch.thread_idx()
        lane=cute.arch.lane_idx()
        # Identical warp-local reduction to write_partial_maximum. The original
        # helper reads physical blockIdx internally, so cannot be called here.
        for level in cutlass.range_constexpr(5):
            largest=cute.arch.fmax(largest,cute.arch.shuffle_sync_bfly(largest,16 >> level))
        if lane==0:partials[logical*(self.threads//32)+thread//32]=largest

    @cute.jit
    def prepare_row(self,source,signs,packed,scales,partials,logical):
        thread,_,_=cute.arch.thread_idx()
        lane=cute.arch.lane_idx()
        warp=logical*(self.threads//32)+thread//32
        cached_signs=cute.make_rmem_tensor((4,),Float32)
        for j in cutlass.range_constexpr(4):cached_signs[j]=signs[lane*4+j]
        largest=Float32(0.)
        for tile in cutlass.range(warp,self.row.tiles,self.row.work_warp_count,unroll=1):
            largest=self.row.prepare_tile(source,cached_signs,packed,scales,signs,signs,
                                          source,tile,0,largest,Float32(1.))
        self.write_remapped_partial(largest,partials,logical)

    @cute.jit
    def prepare_transpose(self,source,signs,packed,scales,partials,logical,staged):
        thread,_,_=cute.arch.thread_idx()
        lane=cute.arch.lane_idx()
        cached_signs=cute.make_rmem_tensor((16,),Float32)
        for j in cutlass.range_constexpr(16):cached_signs[j]=signs[(lane%8)*16+j]
        largest=Float32(0.)
        limit=self.transposed.tiles//self.transposed.source_tile_rows
        # Stage1 ring and every barrier/commit/wait retain the original order.
        if logical<limit:self.transposed.stage_async(source,staged,logical,0)
        cute.arch.cp_async_commit_group()
        for iteration in cutlass.range(0,(limit+self.blocks-1-logical)//self.blocks,unroll=1):
            work_tile=logical+iteration*self.blocks
            cute.arch.cp_async_wait_group(0)
            cute.arch.sync_threads()
            row_group=work_tile%(self.transposed.rows//self.transposed.source_tile_rows)
            chunk=work_tile//(self.transposed.rows//self.transposed.source_tile_rows)
            tile=(row_group*self.transposed.source_tile_rows+thread//32)*(self.transposed.cols//128)+chunk
            largest=self.transposed.prepare_rows(source,cached_signs,packed,scales,signs,signs,
                staged[None,None,0],tile,largest,Float32(1.))
            cute.arch.sync_threads()
            future=work_tile+self.blocks
            if future<limit:self.transposed.stage_async(source,staged,future,0)
            cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        self.write_remapped_partial(largest,partials,logical)


@lru_cache(maxsize=32)
def compile_dual_dy_preparation(row_kernel,transpose_kernel):
    """Return (runner,(row_partial_count,transpose_partial_count)).

    runner(DY_flat, dx_signs, row_packed_u8, row_temporary_bf16, row_partials,
           dw_signs, transpose_packed_u8, transpose_temporary_bf16, transpose_partials)
    All tensors are caller-owned; each role keeps its original partial indices.
    Finalizers MUST wait for completion of the entire fused producer grid.
    """
    kernel=DualDYPreparationKernel(row_kernel,transpose_kernel)
    elements=kernel.rows*kernel.cols
    fake=lambda dtype,count:cute.runtime.make_fake_tensor(dtype,(count,),stride=(1,),assumed_align=4)
    compiled=cute.compile(kernel,fake(BFloat16,elements),fake(Float32,128),
        fake(Uint8,elements//2),fake(BFloat16,elements//16),fake(Float32,kernel.partial_counts[0]),
        fake(Float32,128),fake(Uint8,elements//2),fake(BFloat16,elements//16),fake(Float32,kernel.partial_counts[1]),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options="--enable-tvm-ffi --opt-level 3")
    return compiled,kernel.partial_counts


__all__=["DualDYPreparationKernel","compile_dual_dy_preparation"]
