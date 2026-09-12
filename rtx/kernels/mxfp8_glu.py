"""Experimental paired MXFP8 projection with an FP32 GLU epilogue.

No [M, 2H] global intermediate is allocated. Projection schedules are shared
with the fused MXFP8 linear; GLU adds register/exchange epilogues and optional
buffered TMA stores. Unsupported combinations are rejected explicitly.
"""
from functools import lru_cache

import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import cpasync
from cutlass import BFloat16, Float8E4M3FN, Float32, Uint8, Uint16, Int16
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

from ..configs.glu import DEFAULT_GLU_CONFIG, glu_rejection
from .mxfp8_fwd import MXFP8LinearFwdKernel


class MXFP8GLUKernel(MXFP8LinearFwdKernel):
    def __init__(self, problem, config, activation, output_format):
        problem.validate()
        reason = glu_rejection(problem, config, output_format)
        if reason:
            raise ValueError(reason)
        if activation not in ("silu", "gelu", "relu"):
            raise ValueError("activation must be silu, gelu, or relu")
        if output_format not in ("bf16", "mxfp8"):
            raise ValueError("output_format must be bf16 or mxfp8")
        super().__init__(problem, config)
        self.glu = True
        self.quant_down = False
        self.glu_tma_store = getattr(config,"glu_store","direct") == "tma"
        self.glu_store_stages = getattr(config,"glu_store_stages",1)
        self.activation = activation
        self.output_format = output_format
        self.register_epilogue = getattr(config, "glu_epilogue", "exchange") == "registers"
        self.glu_scratch_elems = config.tile_m * config.tile_n
        if self.register_epilogue:
            self.glu_scratch_elems = 0 if output_format == "bf16" else self.glu_scratch_elems // 2
        self.reuse_glu_scratch = config.glu_scratch == "reuse_operands"
        self.warp_quant = config.glu_quant_epilogue == "fragment_registers"
        if self.warp_quant:
            self.glu_scratch_elems = config.tile_m*config.tile_n*config.atom_layout_n//64 if output_format == "mxfp8" else 0
        self.glu_alias_bytes = self.glu_scratch_elems*4 if self.reuse_glu_scratch else 0
        if self.reuse_glu_scratch:
            self.glu_scratch_elems = 0

    def _prepare_glu(self):
        if self.warp_quant:
            identity = cute.make_identity_tensor((self.config.tile_m,self.config.tile_n))
            representative = self.tiled_mma.get_slice(0).partition_C(identity)
            keys = tuple((int(representative[2*i][0]),int(representative[2*i][1])//64)
                         for i in range(cute.size(representative)//2))
            self.quant_groups = tuple(tuple(i for i,k in enumerate(keys) if k == key)
                                      for key in dict.fromkeys(keys))

    @cute.jit
    def _quantize_down(self,source,quantized,scales,tidx,tile,total_tiles):
        cfg = self.down_quantizer.config
        warps = min(cfg.num_warps,self.config.num_threads//32)
        lane = cute.arch.lane_idx()
        threads_per_scale = 32//cfg.quant_vec
        scale_in_warp = lane//threads_per_scale
        lane_in_scale = lane%threads_per_scale
        blocks_per_row = self.down_quantizer.storage_k//32
        if tidx < warps*32:
            for group in cutlass.range(tile*warps+tidx//32,self.down_quantizer.task_groups,total_tiles*warps,unroll=1):
                block = group*cfg.quant_vec+scale_in_warp
                if block < self.down_quantizer.rows*blocks_per_row:
                    row,col = block//blocks_per_row,block%blocks_per_row
                    self.down_quantizer._quantize_task(source,quantized,scales,row,col,lane_in_scale,
                                             col*32+lane_in_scale*cfg.quant_vec,0)

    @cute.jit
    def _quantize_registers(self,values,coordinates,storage,tidx,block_m,block_n):
        quantized = cute.make_rmem_tensor(values.shape,Float8E4M3FN)
        decoded_scales = cute.make_rmem_tensor(values.shape,Float32)
        parts = self.config.atom_layout_n
        partials = self._scratch(storage,self.config.tile_n*parts//64)
        if tidx < self.config.num_mma_warps*32:
            for group in cutlass.range_constexpr(len(self.quant_groups)):
                indices = self.quant_groups[group]
                maximum = Float32(0)
                for j in cutlass.range_constexpr(len(indices)):
                    maximum = cute.arch.fmax(maximum,cute.abs(values[indices[j]]),nan=True)
                maximum = self._warp_amax(maximum,4,cute.arch.lane_idx())
                row,col = coordinates[2*indices[0]][0]-block_m*self.config.tile_m,coordinates[2*indices[0]][1]-block_n*self.config.tile_n
                if cute.arch.lane_idx()%4 == 0:
                    partials[row,(col//64)*parts+(col%64)//16] = maximum
        cute.arch.sync_threads()
        for group in cutlass.range_constexpr(len(self.quant_groups)):
            indices = self.quant_groups[group]
            row,col = coordinates[2*indices[0]][0]-block_m*self.config.tile_m,coordinates[2*indices[0]][1]-block_n*self.config.tile_n
            maximum = Float32(0)
            if tidx < self.config.num_mma_warps*32:
                for part in cutlass.range_constexpr(parts):
                    maximum = cute.arch.fmax(maximum,partials[row,(col//64)*parts+part],nan=True)
            scale,inv = self._scale_from_amax(maximum)
            for pair in cutlass.range_constexpr(len(indices)//2):
                first,second = indices[2*pair],indices[2*pair+1]
                packed = nvvm.inline_ptx_hl(
                    "cvt.rn.satfinite.e4m3x2.f32 {$w0}, {$r0}, {$r1};",
                    write_only_types=[Int16],read_only_args=[values[second]*inv,values[first]*inv])
                quantized[first] = Uint8(packed & 255).bitcast(Float8E4M3FN)
                quantized[second] = Uint8((packed >> 8) & 255).bitcast(Float8E4M3FN)
            for j in cutlass.range_constexpr(len(indices)):
                decoded_scales[indices[j]] = Float32(scale)
        return quantized,decoded_scales

    @cute.jit
    def _scratch(self, storage, cols):
        layout = cute.make_layout((self.config.tile_m, cols), stride=(cols, 1))
        if cutlass.const_expr(self.config.glu_scratch_swizzle == "xor_16b"):
            layout = cute.make_composed_layout(cute.make_swizzle(3,2,3),0,layout)
        if cutlass.const_expr(self.reuse_glu_scratch):
            # Scalar storage places the 1024-aligned byte operand arrays together.
            # All cooperative MMA readers have reached the mainloop CTA barrier.
            raw = storage.q_a.get_tensor(cute.make_layout((self.config.mxfp8_stages*self.config.tile_m*self.config.tile_k,)))
            return cute.make_tensor(cute.recast_ptr(raw.iterator, dtype=Float32), layout)
        else:
            return storage.glu_scratch.get_tensor(layout)

    @cute.jit
    def _sigmoid(self, gate):
        denominator=Float32(1)+cute.exp(-gate)
        if cutlass.const_expr(self.config.glu_silu_math == "approx_reciprocal"):
            return nvvm.inline_ptx_hl("rcp.approx.f32 {$w0}, {$r0};",
                                      write_only_types=[Float32],read_only_args=[denominator])
        else:
            return Float32(1)/denominator

    @cute.jit
    def _activate(self, gate):
        if cutlass.const_expr(self.activation == "silu"):
            if cutlass.const_expr(self.config.glu_silu_math != "division"):
                return gate * self._sigmoid(gate)
            else:
                return gate / (Float32(1) + cute.exp(-gate))
        elif cutlass.const_expr(self.activation == "gelu"):
            return Float32(0.5) * gate * (Float32(1) + cute.erf(gate * Float32(0.7071067811865476)))
        else:
            return cute.arch.fmax(gate, Float32(0), nan=True)

    @cute.jit
    def _glu_epilogue(self, storage, accumulators, coordinates, out, scales,
                      block_m, block_n, tidx, tma_out, tma_out_tensor, out_layout, work_slot):
        cfg = self.config
        if cutlass.const_expr(self.glu_tma_store):
            s_out = storage.out.get_tensor(out_layout.outer,swizzle=out_layout.inner)
            output_stage = work_slot % self.glu_store_stages
            full_output = ((block_m+1)*cfg.tile_m <= self.problem.m
                           and (block_n+1)*cfg.tile_n <= self.problem.n)
            if tidx < 32:
                cute.arch.cp_async_bulk_wait_group(self.glu_store_stages-1,read=True)
            cute.arch.sync_threads()
        if cutlass.const_expr(self.warp_quant and self.output_format == "mxfp8"):
            values = cute.make_rmem_tensor((cute.size(accumulators)//2,),Float32)
            values.fill(Float32(0))
            if tidx < cfg.num_mma_warps*32:
                for i in cutlass.range_constexpr(cute.size(values)):
                    coord = coordinates[2*i]
                    value = Float32(0)
                    if coord[0] < self.problem.m and coord[1] < self.problem.n:
                        value = self._activate(accumulators[2*i])*accumulators[2*i+1]
                    values[i] = value
            quantized,decoded_scales = self._quantize_registers(values,coordinates,storage,tidx,block_m,block_n)
            if tidx < cfg.num_mma_warps*32:
                for i in cutlass.range_constexpr(cute.size(values)):
                    row,col = coordinates[2*i][0],coordinates[2*i][1]//2
                    if row < self.problem.m and col < cute.size(out,mode=[1]):
                        out[row,col] = quantized[i]
                for group in cutlass.range_constexpr(len(self.quant_groups)):
                    i = self.quant_groups[group][0]
                    row,col = coordinates[2*i][0],coordinates[2*i][1]//2
                    if cute.arch.lane_idx()%4 == 0 and col%32 == 0 and row < self.problem.m and col < cute.size(out,mode=[1]):
                        scales[row,col//32] = Uint8((decoded_scales[i].bitcast(cutlass.Uint32) >> 23) & 255)
        elif cutlass.const_expr(self.register_epilogue and self.output_format == "bf16"):
            if tidx < cfg.num_mma_warps * 32:
                for i in range(cute.size(accumulators) // 2, unroll_full=True):
                    coord = coordinates[2*i]
                    value = self._activate(accumulators[2*i]) * accumulators[2*i+1]
                    if coord[0] < self.problem.m and coord[1] < self.problem.n:
                        if cutlass.const_expr(self.glu_tma_store):
                            if full_output:
                                s_out[coord[0]-block_m*cfg.tile_m,(coord[1]-block_n*cfg.tile_n)//2,output_stage] = BFloat16(value)
                            else:
                                out[coord[0],coord[1]//2] = BFloat16(value)
                        else:
                            out[coord[0], coord[1] // 2] = BFloat16(value)
        else:
            scratch_cols = cfg.tile_n
            if cutlass.const_expr(self.register_epilogue):
                scratch_cols = cfg.tile_n // 2
            scratch = self._scratch(storage, scratch_cols)
            if tidx < cfg.num_mma_warps * 32:
                if cutlass.const_expr(self.register_epilogue):
                    for i in range(cute.size(accumulators)//2, unroll_full=True):
                        coord = coordinates[2*i]
                        scratch[coord[0] - block_m*cfg.tile_m,
                                (coord[1] - block_n*cfg.tile_n)//2] = self._activate(accumulators[2*i]) * accumulators[2*i+1]
                else:
                    for i in range(cute.size(accumulators), unroll_full=True):
                        coord = coordinates[i]
                        scratch[coord[0] - block_m*cfg.tile_m,
                                coord[1] - block_n*cfg.tile_n] = accumulators[i]
            cute.arch.sync_threads()
            output_vec = 1
            if cutlass.const_expr(self.output_format == "mxfp8"):
                output_vec = cfg.glu_quant_vec
            for i in range(cute.ceil_div(cfg.tile_m*(cfg.tile_n//2), cfg.num_threads*output_vec)):
                index = (tidx + i*cfg.num_threads)*output_vec
                row = index // (cfg.tile_n//2)
                col = index % (cfg.tile_n//2)
                gr = block_m*cfg.tile_m + row
                gc = block_n*(cfg.tile_n//2) + col
                values = cute.make_rmem_tensor((output_vec,),Float32)
                maximum = Float32(0)
                for v in cutlass.range_constexpr(output_vec):
                    value = Float32(0)
                    if row < cfg.tile_m:
                        if cutlass.const_expr(self.register_epilogue):
                            value = scratch[row,col+v]
                        else:
                            value = self._activate(scratch[row,2*(col+v)])*scratch[row,2*(col+v)+1]
                    if gr >= self.problem.m or gc+v >= self.problem.n//2:
                        value = Float32(0)
                    values[v] = value
                    maximum = cute.arch.fmax(maximum,cute.abs(value),nan=True)
                if cutlass.const_expr(self.output_format == "mxfp8"):
                    amax = self._warp_amax(maximum, 32//output_vec, cute.arch.lane_idx())
                    scale, inv = self._scale_from_amax(amax)
                    if row < cfg.tile_m and gr < self.problem.m and gc < cute.size(out, mode=[1]):
                        if cutlass.const_expr(output_vec == 1):
                            out[gr,gc] = Float8E4M3FN(values[0]*inv)
                        else:
                            for pair in cutlass.range_constexpr(output_vec//2):
                                packed = nvvm.inline_ptx_hl(
                                    "cvt.rn.satfinite.e4m3x2.f32 {$w0}, {$r0}, {$r1};",
                                    write_only_types=[Int16],
                                    read_only_args=[values[2*pair+1]*inv,values[2*pair]*inv],
                                )
                                pointer = cute.recast_ptr(out.iterator + gr*cute.size(out,mode=[1])+gc+2*pair,dtype=Uint16)
                                pointer[0] = packed.bitcast(Uint16)
                        if cute.arch.lane_idx() % (32//output_vec) == 0:
                            scales[gr,gc//32] = scale.bitcast(Uint8)
                else:
                    if row < cfg.tile_m and gr < self.problem.m and gc < self.problem.n//2:
                        if cutlass.const_expr(self.glu_tma_store):
                            if full_output:
                                s_out[row,col,output_stage] = BFloat16(values[0])
                            else:
                                out[gr,gc] = BFloat16(values[0])
                        else:
                            out[gr,gc] = BFloat16(values[0])
        if cutlass.const_expr(self.glu_tma_store):
            cute.arch.fence_proxy("async.shared",space="cta")
            cute.arch.sync_threads()
            if tidx < 32:
                g_tiles = cute.local_tile(tma_out_tensor,(cfg.tile_m,cfg.tile_n//2),(None,None,None))
                g_tile = cute.zipped_divide(g_tiles[(None,None,block_m,block_n,0)],(cfg.tile_m,cfg.tile_n//2))
                source,dest = cpasync.tma_partition(tma_out,0,cute.make_layout(1),cute.group_modes(s_out,0,2),g_tile)
                with cute.arch.elect_one():
                    if full_output:
                        cute.copy(tma_out,source[(None,output_stage)],dest[(None,0)])
                cute.arch.cp_async_bulk_commit_group()
        cute.arch.sync_threads()


@lru_cache(maxsize=128)
def compile_mxfp8_glu(problem, config=DEFAULT_GLU_CONFIG,
                      activation="silu", output_format="bf16"):
    kernel = MXFP8GLUKernel(problem, config, activation, output_format)
    h = problem.n // 2
    width = h if output_format == "bf16" else (h + 31) // 32 * 32
    def fake(dtype, shape):
        return cute.runtime.make_fake_tensor(dtype, shape, stride=(shape[1], 1), assumed_align=16)
    x = fake(BFloat16, (problem.m, problem.k))
    w = fake(BFloat16, (problem.n, problem.k))
    out = fake(BFloat16 if output_format == "bf16" else Float8E4M3FN, (problem.m, width))
    # BF16 has no auxiliary output. Its unused scale arguments alias out;
    # CuTe layouts cannot represent a zero-size dummy tensor.
    scales = out if output_format == "bf16" else fake(Uint8, (problem.m, (h + 31) // 32))
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel, x, w, out, scales, scales, scales, scales, stream,
                        options="--enable-tvm-ffi --opt-level 3 "
                        f"--ptxas-options '-O3 --maxrregcount={config.maxrregcount}'")
