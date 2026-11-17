"""Uniform expert-batch specializations of the qualified SM120 MXFP8 core.

One launch covers every expert; no Python expert loop or BF16 transpose.
The separate gather/combine family is a baseline, not a claim of on-chip FFN
fusion. Its hidden connection is E4M3/E8M0, never a BF16 forward allocation.
"""
from functools import lru_cache

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float8E4M3FN, Float8E8M0FNU, Float32, Int32, Int64, Uint8

from .mxfp8 import MXFP8Problem
from .mxfp8_fwd import MXFP8LinearFwdKernel
from .mxfp8_glu import MXFP8GLUKernel
from .mxfp8_mlp import GLURecomputeDerivative


def grouped_rejection(config):
    if config.epilogue != 'direct' or config.cluster_reuse != 'none':
        return 'grouped core currently requires direct stores and unclustered scheduling'
    if getattr(config, 'glu_store', 'direct') != 'direct':
        return 'grouped GLU currently requires direct stores'
    return None


def _fake(dtype, rows, cols):
    return cute.runtime.make_fake_tensor(dtype, (rows, cols), stride=(cols, 1), assumed_align=16)


@lru_cache(maxsize=128)
def compile_grouped_linear(experts, problem, config, a_orientation='row', b_orientation='row', prequant=False,_byte_scales=False):
    reason = grouped_rejection(config)
    if reason:
        raise ValueError(reason)
    if prequant and (config.load_engine != 'scalar' or a_orientation != 'row'):
        raise ValueError('packed hidden grouped core currently requires scalar row transport')
    kernel = MXFP8LinearFwdKernel(problem, config, a_orientation=a_orientation, b_orientation=b_orientation)
    kernel.batch_count = experts
    kernel.a_prequantized = prequant
    ak = problem.storage_k if prequant else problem.k
    ar, ac = (problem.m, ak) if a_orientation == 'row' else (problem.k, problem.m)
    br, bc = (problem.n, problem.k) if b_orientation == 'row' else (problem.k, problem.n)
    x = _fake(Float8E4M3FN if prequant else BFloat16, experts*ar, ac)
    w = _fake(BFloat16, experts*br, bc)
    out = _fake(BFloat16, experts*problem.m, problem.n)
    aux = out
    if prequant:
        kernel.batch_aux_shape = (problem.m, ak//32)
        aux = _fake(Uint8 if _byte_scales else Float8E8M0FNU, experts*problem.m, ak//32)
        if _byte_scales:
            from .mxfp8_moe_buffers import ByteLegacyLinear
            kernel = ByteLegacyLinear(kernel)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel, x, w, out, aux, aux, aux, aux, stream,
                        options='--enable-tvm-ffi --opt-level 3 '
                        f"--ptxas-options '-O3 --maxrregcount={config.maxrregcount}'")


@lru_cache(maxsize=128)
def compile_grouped_glu(experts, problem, config, derivative=False):
    reason = grouped_rejection(config)
    if reason:
        raise ValueError(reason)
    h = problem.n//2
    if derivative:
        kernel = GLURecomputeDerivative(problem, config, 'silu', 'bf16')
        kernel.emit_hidden = False
        kernel.emit_gradient_quant = False
        kernel.emit_hidden_quant = False
        kernel.emit_gradient_transpose_quant = False
        kernel.separate_hidden = False
        kernel.glu_scratch_elems = 0
        shape = (problem.m, 2*h)
        aux_shape = (problem.m, h)
    else:
        kernel = MXFP8GLUKernel(problem, config, 'silu', 'mxfp8')
        shape = (problem.m, (h+31)//32*32)
        aux_shape = (problem.m, shape[1]//32)
    kernel.batch_count = experts
    kernel.batch_out_shape = shape
    kernel.batch_aux_shape = aux_shape
    x = _fake(BFloat16, experts*problem.m, problem.k)
    w = _fake(BFloat16, experts*problem.n, problem.k)
    out = _fake(BFloat16 if derivative else Float8E4M3FN, experts*shape[0], shape[1])
    aux = _fake(BFloat16 if derivative else Uint8, experts*aux_shape[0], aux_shape[1])
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(kernel, x, w, out, aux, aux, aux, aux, stream,
                        options='--enable-tvm-ffi --opt-level 3 '
                        f"--ptxas-options '-O3 --maxrregcount={config.maxrregcount}'")


class RoutedMovement:
    def __init__(self, n, d, e, c, k, threads, kind, gather_mode='flat',storage_c=0):
        self.n,self.d,self.e,self.c,self.k = n,d,e,c,k
        self.threads,self.kind = threads,kind
        self.gather_mode=gather_mode
        self.storage_c=storage_c or c

    @cute.jit
    def __call__(self, source, ids, inverse, weights, extra, out, auxiliary, stream):
        warp_owned=self.kind in ('slot_grad','slot_dy','route_grad') or (self.kind=='gather' and self.gather_mode=='warp')
        count = self.e*self.storage_c if warp_owned else (
            self.e*self.storage_c*self.d if self.kind=='gather' else self.n*self.d)
        grid = cute.ceil_div(count, self.threads//32 if warp_owned else self.threads)
        self.kernel(source,ids,inverse,weights,extra,out,auxiliary).launch(
            grid=(grid,1,1),block=(self.threads,1,1),stream=stream)

    @cute.jit
    def _valid(self, ids, inverse, slot):
        token = Int32(ids[slot//self.c,slot%self.c])
        valid = Int32(0)
        if token>=0 and token<self.n:
            for rank in cutlass.range_constexpr(self.k):
                if inverse[token,rank] == slot:
                    valid = Int32(1)
        return token,valid

    @cute.kernel
    def kernel(self,source,ids,inverse,weights,extra,out,auxiliary):
        thread = cute.arch.thread_idx()[0]
        block = cute.arch.block_idx()[0]
        if cutlass.const_expr(self.kind=='gather' and self.gather_mode=='warp'):
            slot=block*(self.threads//32)+thread//32
            lane=cute.arch.lane_idx()
            token,valid=Int32(0),Int32(0)
            if slot<self.e*self.storage_c and slot%self.storage_c<self.c and lane==0:
                token,valid=self._valid(ids,inverse,slot//self.storage_c*self.c+slot%self.storage_c)
            token=cute.arch.shuffle_sync(token,0)
            valid=cute.arch.shuffle_sync(valid,0)
            if slot<self.e*self.storage_c:
                for col in cutlass.range(lane,self.d,32,unroll=1):
                    value=BFloat16(0)
                    if valid!=0:value=source[token,col]
                    out[slot,col]=value
        elif cutlass.const_expr(self.kind=='gather'):
            index = block*self.threads+thread
            if index<self.e*self.storage_c*self.d:
                slot,col = index//self.d,index%self.d
                token,valid=Int32(0),Int32(0)
                if slot%self.storage_c<self.c:
                    token,valid = self._valid(ids,inverse,slot//self.storage_c*self.c+slot%self.storage_c)
                value = BFloat16(0)
                if valid!=0:
                    value = source[token,col]
                out[slot,col] = value
        elif cutlass.const_expr(self.kind=='combine' or self.kind=='dx'):
            index = block*self.threads+thread
            if index<self.n*self.d:
                token,col = index//self.d,index%self.d
                value = Float32(0)
                for rank in cutlass.range_constexpr(self.k):
                    slot = Int32(inverse[token,rank])
                    if slot>=0 and slot<self.e*self.c:
                        physical=slot//self.c*self.storage_c+slot%self.c
                        contribution = Float32(source[physical,col])
                        if cutlass.const_expr(self.kind=='combine'):
                            contribution *= weights[slot//self.c,slot%self.c]
                        value += contribution
                if cutlass.const_expr(self.kind=='dx'):
                    out[token,col] = BFloat16(value)
                else:
                    out[token,col] = value
        else:
            # One warp owns one route: weighted dY and its FP32 weight adjoint
            # share the same loads. Invalid dummy slots are zero even though
            # their safe token address aliases a real input row.
            slot = block*(self.threads//32)+thread//32
            lane = cute.arch.lane_idx()
            total = Float32(0)
            if slot<self.e*self.storage_c:
                original=slot//self.storage_c*self.c+slot%self.storage_c
                token,valid=Int32(0),Int32(0)
                weight=Float32(0)
                if slot%self.storage_c<self.c:
                    token,valid=self._valid(ids,inverse,original)
                    weight=weights[original//self.c,original%self.c]
                for col in cutlass.range(lane,self.d,32,unroll=1):
                    dy = Float32(0)
                    if valid!=0:
                        dy = Float32(source[token,col])
                    if cutlass.const_expr(self.kind!='route_grad'):
                        out[slot,col] = BFloat16(dy*weight)
                    if cutlass.const_expr(self.kind!='slot_dy'):
                        total += dy*Float32(extra[slot,col])
            for level in cutlass.range_constexpr(5):
                total += cute.arch.shuffle_sync_bfly(total,16>>level)
            if cutlass.const_expr(self.kind!='slot_dy'):
                if slot<self.e*self.storage_c and slot%self.storage_c<self.c and lane==0:
                    auxiliary[slot//self.storage_c,slot%self.storage_c] = total


@lru_cache(maxsize=128)
def compile_movement(n,d,e,c,k,threads,kind,gather_mode='flat',storage_c=0):
    if threads not in (64,128,256,512) or kind not in ('gather','combine','dx','slot_grad','slot_dy','route_grad'):
        raise ValueError('Invalid grouped movement configuration')
    if gather_mode not in ('flat','warp'):raise ValueError('Unknown gather ownership')
    cs=storage_c or c
    if cs<c:raise ValueError('Physical capacity cannot truncate expert assignments')
    source = _fake(Float32 if kind in ('slot_grad','slot_dy','route_grad') else BFloat16,
                   n if kind in ('gather','slot_grad','slot_dy','route_grad') else e*cs,d)
    ids,inverse = _fake(Int64,e,c),_fake(Int64,n,k)
    weights = _fake(Float32,e,c)
    extra = _fake(BFloat16,e*cs,d)
    out = _fake(Float32 if kind=='combine' else BFloat16,
                n if kind in ('combine','dx') else e*cs,d)
    aux = _fake(Float32,e,c)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(RoutedMovement(n,d,e,c,k,threads,kind,gather_mode,cs),
                        source,ids,inverse,weights,extra,out,aux,stream,
                        options='--enable-tvm-ffi --opt-level 3')
