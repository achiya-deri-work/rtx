"""Research TMA staging with retained FP32 H128/EDEN arithmetic.

Only original-BF16 transposed x8 preparation is supported. No default dispatch.
Each stage has its own generation-counted transaction barrier. All consumers
finish reading a slot before the elected producer refills it.
"""
from functools import lru_cache
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float32, Int64, Uint8
from cutlass.cute.nvgpu import cpasync
from .nvfp4_posthoc import HadamardPreparationKernel, PosthocScaleConfig


class TmaPreparationKernel(HadamardPreparationKernel):
    def __init__(self, *args, tma_swizzle='64b', consumer_order='retained', **kwargs):
        super().__init__(*args, **kwargs)
        if not (self.transpose and self.async_copy and self.config.threads == 128
                and self.source_tile_rows == 32 and self.values_per_lane == 16
                and not self.packed_source):
            raise ValueError('TMA probe requires original BF16 async x8 transpose, 128 threads, VPL16')
        if tma_swizzle not in ('none', '32b', '64b'):
            raise ValueError('unsupported TMA preparation swizzle')
        self.tma_swizzle = tma_swizzle
        if consumer_order not in ('retained','xor4','xor5','columns'):
            raise ValueError('unsupported TMA consumer order')
        self.consumer_order = consumer_order
        self.permutation_bits = (0,1,2) if consumer_order == 'columns' else (0,2) if consumer_order == 'xor5' else (2,3)

    @cute.jit
    def prepare_transpose_group(self, staged, signs, packed, scales, tile, base, largest, cached_reciprocal):
        if cutlass.const_expr(self.consumer_order == 'retained'):
            return HadamardPreparationKernel.prepare_transpose_group(
                self,staged,signs,packed,scales,tile,base,largest,cached_reciprocal)
        thread,_,_ = cute.arch.thread_idx()
        lane = cute.arch.lane_idx()
        local_lane = lane%8
        subrow = base+lane//8
        column = thread//32+subrow*4
        local_tile = tile+subrow*4*(self.cols//128)
        if cutlass.const_expr(self.consumer_order == 'columns'):
            warp = thread//32
            column = warp%2 + 2*(subrow%4) + 8*(warp//2) + 16*(subrow//4)
            local_tile = tile + (column-warp)*(self.cols//128)
        values = cute.make_rmem_tensor((16,),Float32)
        permutation = ((local_lane&1)*(1 << self.permutation_bits[0])
                       + ((local_lane>>1)&1)*(1 << self.permutation_bits[1]))
        if cutlass.const_expr(self.consumer_order == 'columns'):
            permutation = permutation + ((local_lane>>2)&1)*4
        for j in cutlass.range_constexpr(16):
            values[j] = staged[local_lane*16+(j^permutation),column].to(Float32)
        # Exact register permutation only: no sign changes, rounding or new
        # butterfly association. XOR5 uses row bits0/2: byte-address SW64
        # swizzles bits7/8 into4/5, so row bit3 in XOR4 does not change banks.
        for bit in cutlass.range_constexpr(len(self.permutation_bits)):
            distance = 1 << self.permutation_bits[bit]
            swap = (local_lane & (1 << bit)) != 0
            for group in cutlass.range_constexpr(16//(2*distance)):
                for j in cutlass.range_constexpr(distance):
                    lo = group*2*distance+j
                    a,b = values[lo],values[lo+distance]
                    values[lo] = b if swap else a
                    values[lo+distance] = a if swap else b
        for j in cutlass.range_constexpr(16):
            sign = signs[j] if cutlass.const_expr(self.signs_mode == 'register') else signs[local_lane*16+j]
            values[j] = values[j]*sign
        return self.grouped_prepare_values(values,packed,scales,local_tile,local_lane,
                                           True,largest,cached_reciprocal)

    @cute.jit
    def __call__(self, src: cute.Tensor, signs: cute.Tensor, packed: cute.Tensor,
                 scales: cute.Tensor, partials: cute.Tensor, stream: cuda.CUstream):
        kinds = {'none': cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_INTER,
                 '32b': cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW32,
                 '64b': cute.nvgpu.warpgroup.SmemLayoutAtomKind.K_SW64}
        layout = cute.tile_to_shape(cute.nvgpu.warpgroup.make_smem_layout_atom(
            kinds[self.tma_swizzle], BFloat16), (128,32,self.stages), order=(0,1,2))
        source = cute.make_tensor(src.iterator, cute.make_layout(
            (self.source_rows,self.source_cols), stride=(self.source_cols,1)))
        atom, tensor = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(),
            source, cute.slice_(layout,(None,None,0)), (128,32))
        self.kernel(atom,tensor,layout,signs,packed,scales,partials).launch(
            grid=(self.blocks,1,1), block=(128,1,1), stream=stream)

    @cute.jit
    def issue(self, atom, source, destination, barriers, work, slot):
        row_group = work//(self.cols//128)
        chunk = work%(self.cols//128)
        if cutlass.const_expr(self.tile_order == 'reduction_major'):
            row_group = work%(self.rows//32)
            chunk = work//(self.rows//32)
        # TMA copy lowering elects a lane with a full-warp mask. Enter it from
        # the whole producer warp, not a thread==0 divergent region.
        with cute.arch.elect_one():
            cute.arch.mbarrier_arrive_and_expect_tx(barriers.iterator+slot, 128*32*2)
        cute.copy(atom,source[None,chunk,row_group],destination[None,slot],
                  tma_bar_ptr=barriers.iterator+slot)

    @cute.kernel
    def kernel(self, atom: cute.CopyAtom, tensor: cute.Tensor, layout: cute.ComposedLayout,
               signs: cute.Tensor, packed: cute.Tensor, scales: cute.Tensor, partials: cute.Tensor):
        thread,_,_ = cute.arch.thread_idx()
        block,_,_ = cute.arch.block_idx()
        lane = cute.arch.lane_idx()
        smem = cutlass.utils.SmemAllocator()
        staged = smem.allocate_tensor(BFloat16,layout.outer,byte_alignment=1024,swizzle=layout.inner)
        barriers = smem.allocate_tensor(Int64,cute.make_layout((self.stages,)),byte_alignment=8)
        warp_maxima = partials
        if cutlass.const_expr(self.partial_max_scope == 'cta'):
            warp_maxima = smem.allocate_tensor(Float32,cute.make_layout((4,)),byte_alignment=4)
        if thread == 0:
            for stage in cutlass.range_constexpr(self.stages):
                cute.arch.mbarrier_init(barriers.iterator+stage,1)
        cute.arch.mbarrier_init_fence()
        cute.arch.sync_threads()
        tiled = cute.local_tile(tensor,(128,32),(None,None))
        destination,source = cpasync.tma_partition(atom,0,cute.make_layout(1),
            cute.group_modes(staged,0,2),cute.group_modes(tiled,0,2))
        cached_signs = cute.make_rmem_tensor((16,),Float32)
        if cutlass.const_expr(self.signs_mode == 'register'):
            for j in cutlass.range_constexpr(16):
                cached_signs[j] = signs[(lane%8)*16+j]
        limit = self.tiles//32
        if thread//32 == 0:
            for slot in cutlass.range_constexpr(self.stages):
                work = block+slot*self.blocks
                if work < limit:
                    self.issue(atom,source,destination,barriers,work,slot)
        largest = Float32(0.)
        for iteration in cutlass.range(0,(limit+self.blocks-1-block)//self.blocks,unroll=1):
            work = block+iteration*self.blocks
            slot = iteration%self.stages
            cute.arch.mbarrier_wait(barriers.iterator+slot,(iteration//self.stages)%2)
            row_group,chunk = work//(self.cols//128),work%(self.cols//128)
            if cutlass.const_expr(self.tile_order == 'reduction_major'):
                row_group,chunk = work%(self.rows//32),work//(self.rows//32)
            tile = (row_group*32+thread//32)*(self.cols//128)+chunk
            largest = self.prepare_rows(tensor,
                cached_signs if cutlass.const_expr(self.signs_mode == 'register') else signs,
                packed,scales,signs,signs,staged[None,None,slot],tile,largest,Float32(1.))
            cute.arch.sync_threads()
            future = work+self.stages*self.blocks
            if thread//32 == 0 and future < limit:
                self.issue(atom,source,destination,barriers,future,slot)
        self.write_partial_maximum(largest,partials,warp_maxima)


@lru_cache(maxsize=64)
def compile_tma_preparation(rows,cols,config=PosthocScaleConfig(),mantissa_bits=4,transpose=False,**kwargs):
    kernel = TmaPreparationKernel(rows,cols,config,mantissa_bits,transpose,**kwargs)
    def fake(dtype,size,align=4):
        return cute.runtime.make_fake_tensor(dtype,(size,),stride=(1,),assumed_align=align)
    run = cute.compile(kernel,fake(BFloat16,rows*cols,16),fake(Float32,128),
        fake(Uint8,rows*cols//2,max(4,kernel.grouped_store_bits//8)),
        fake(BFloat16,rows*cols//16),fake(Float32,kernel.partial_count),
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),options='--enable-tvm-ffi --opt-level 3')
    return run,kernel.partial_count
