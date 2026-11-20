"""Joint recompute with disjoint warp ownership of derivatives and hidden."""

import cutlass
import cutlass.cute as cute
from cutlass import BFloat16, Float8E4M3FN, Float32, Int16, Uint8, Uint16, Uint32, Uint64
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

from .mxfp8_moe_recompute import DualOperandJointRecompute


def split_recompute_scratch_elems(config):
    """Float32 allocation slots for an independent, padded BF16 H plane."""
    return (config.tile_n // 2) * (config.tile_m + 1) // 2


class SplitWarpJointRecompute(DualOperandJointRecompute):
    """Gradient warps consume retired B while hidden warps consume A + scratch."""

    def __init__(self, problem, config, activation, output_format, *, gradient_warps=3):
        super().__init__(problem, config, activation, output_format)
        if type(gradient_warps) is not int or not 1 <= gradient_warps < config.num_threads // 32:
            raise ValueError('Split joint recompute needs at least one warp in each epilogue team')
        self.gradient_threads = gradient_warps * 32
        self.hidden_threads = config.num_threads - self.gradient_threads
        self.glu_scratch_elems = split_recompute_scratch_elems(config)

    @cute.jit
    def _publish_recompute(self, products, derivatives, dh, acc, coords, bm, bn, thread):
        cfg = self.config
        m, h = self.problem.m, self.problem.n // 2
        width = cfg.tile_n // 2
        if thread < cfg.num_mma_warps * 32:
            for pair in cutlass.range_constexpr(cute.size(acc) // 2):
                r, c = coords[2 * pair][0], coords[2 * pair][1] // 2
                if r < m and c < h:
                    gate, up = acc[2 * pair], acc[2 * pair + 1]
                    sigmoid = self._sigmoid(gate)
                    if cutlass.const_expr(cfg.glu_silu_math == 'division'):
                        act = self._activate(gate)
                    else:
                        act = gate * sigmoid
                    derivative = sigmoid * (Float32(1) + gate * (Float32(1) - sigmoid))
                    grad = Float32(dh[r, c])
                    lr, lc = r - bm * cfg.tile_m, c - bn * width
                    derivatives[lr, lc] = BFloat16(grad * up * derivative)
                    derivatives[lr, lc + width] = BFloat16(grad * act)
                    products[lr, lc] = act * up

    @cute.jit
    def _gradient_team(self, derivatives, qp, sp, qpt, spt, bm, bn, tid):
        cfg = self.config
        m, h = self.problem.m, self.problem.n // 2
        width = cfg.tile_n // 2
        stride = self.gradient_threads
        for group in cutlass.range(cute.ceil_div(cfg.tile_m * cfg.tile_n, 4 * stride), unroll=1):
            idx = (tid + group * stride) * 4
            lr, lc = idx // cfg.tile_n, idx % cfg.tile_n
            gr, gc = bm * cfg.tile_m + lr, bn * width + lc % width
            target = 2 * gr + lc // width
            valid = lr < cfg.tile_m and gr < m and gc < h
            vals = cute.make_rmem_tensor((4,), Float32)
            vals.fill(Float32(0))
            if valid:
                for v in cutlass.range_constexpr(4):
                    vals[v] = Float32(derivatives[lr, lc + v])
            self._store_bf16_block(vals, qp, sp, target, gc, h, valid, self.gradient_quant_policy)
        for group in cutlass.range(cute.ceil_div(cfg.tile_m * cfg.tile_n, 4 * stride), unroll=1):
            idx = (tid + group * stride) * 4
            lc, lr = idx // cfg.tile_m, idx % cfg.tile_m
            gr, gc = bm * cfg.tile_m + lr, bn * width + lc % width
            target = gc + (lc // width) * h
            valid = lc < cfg.tile_n and gr < m and gc < h
            vals = cute.make_rmem_tensor((4,), Float32)
            vals.fill(Float32(0))
            if valid:
                for v in cutlass.range_constexpr(4):
                    vals[v] = Float32(derivatives[lr + v, lc])
            self._store_bf16_block(vals, qpt, spt, target, gr, m, valid, self.gradient_transpose_quant_policy)

    @cute.jit
    def _hidden_team(self, products, decoded, qh, sh, qht, sht, bm, bn, tid):
        cfg = self.config
        m, h = self.problem.m, self.problem.n // 2
        width, vec = cfg.tile_n // 2, cfg.glu_quant_vec
        stride = self.hidden_threads
        for group in cutlass.range(cute.ceil_div(cfg.tile_m * width, stride * vec), unroll=1):
            idx = (tid + group * stride) * vec
            lr, lc = idx // width, idx % width
            gr, gc = bm * cfg.tile_m + lr, bn * width + lc
            valid = lr < cfg.tile_m and gr < m and gc < h
            vals = cute.make_rmem_tensor((vec,), Float32)
            maximum = Float32(0)
            for v in cutlass.range_constexpr(vec):
                value = Float32(0)
                if valid:
                    value = products[lr, lc + v]
                vals[v] = value
                maximum = cute.arch.fmax(maximum, cute.abs(value), nan=True)
            amax = self._warp_amax(maximum, 32 // vec, cute.arch.lane_idx())
            scale, inv = self._scale_from_amax(amax)
            if valid:
                if cute.arch.lane_idx() % (32 // vec) == 0:
                    sh[gr, gc // 32] = scale
                word = Uint64(0)
                for pair in cutlass.range_constexpr((vec + 1) // 2):
                    lo = 2 * pair
                    hi = cutlass.min(lo + 1, vec - 1)
                    packed = nvvm.inline_ptx_hl(
                        "cvt.rn.satfinite.e4m3x2.f32 {$w0}, {$r0}, {$r1};",
                        write_only_types=[Int16], read_only_args=[vals[hi] * inv, vals[lo] * inv])
                    if cutlass.const_expr(vec > 1):
                        word |= Uint64(packed.bitcast(Uint16)) << (16 * pair)
                    for half in cutlass.range_constexpr(min(2, vec)):
                        rounded = Uint8(packed >> (8 * half)).bitcast(Float8E4M3FN)
                        if cutlass.const_expr(vec == 1):
                            qh[gr, gc] = rounded
                        # Decode the *same* FP8 byte with its feature-axis
                        # scale, then round to BF16 before finding token amax.
                        decoded[lc + lo + half, lr] = BFloat16(Float32(rounded) * Float32(scale))
                if cutlass.const_expr(vec > 1):
                    # h and width are block32 and gc is vec-aligned, even on
                    # a clipped tile, so no lane vector straddles valid data.
                    if cutlass.const_expr(vec == 2):
                        cute.recast_ptr(qh.iterator + gr * h + gc, dtype=Uint16)[0] = Uint16(word)
                    elif cutlass.const_expr(vec == 4):
                        cute.recast_ptr(qh.iterator + gr * h + gc, dtype=Uint32)[0] = Uint32(word)
                    else:
                        cute.recast_ptr(qh.iterator + gr * h + gc, dtype=Uint64)[0] = word

        # Named, explicitly team-sized barrier: every hidden warp reaches
        # this, and no dP warp touches the decoded plane or this barrier ID.
        cute.arch.barrier(barrier_id=6, number_of_threads=stride)
        for group in cutlass.range(cute.ceil_div(cfg.tile_m * width, 4 * stride), unroll=1):
            idx = (tid + group * stride) * 4
            hc, rr = idx // cfg.tile_m, idx % cfg.tile_m
            gh, gm = bn * width + hc, bm * cfg.tile_m + rr
            valid = hc < width and gh < h and gm < m
            vals = cute.make_rmem_tensor((4,), Float32)
            vals.fill(Float32(0))
            if valid:
                for v in cutlass.range_constexpr(4):
                    vals[v] = Float32(decoded[hc, rr + v])
            self._store_bf16_block(vals, qht, sht, gh, gm, m, valid, self.hidden_quant_policy)

    @cute.jit
    def _glu_epilogue(self, storage, acc, coords, out, aux, bm, bn, thread,
                      tma_out, tma_out_tensor, out_layout, work_slot):
        cfg = self.config
        width = cfg.tile_n // 2
        dh, qh, sh, qht, sht, qp, sp, qpt, spt = self._local_joint_aux(aux)
        product_layout = cute.make_layout((cfg.tile_m, width), stride=(width, 1))
        grad_layout = cute.make_layout((cfg.tile_m, cfg.tile_n), stride=(cfg.tile_n, 1))
        if cutlass.const_expr(cfg.glu_scratch_swizzle == 'xor_16b'):
            product_layout = cute.make_composed_layout(cute.make_swizzle(3, 2, 3), 0, product_layout)
            grad_layout = cute.make_composed_layout(
                cute.make_swizzle(3, 3, cfg.tile_n.bit_length() - 2), 0, grad_layout)
        products = cute.make_tensor(cute.recast_ptr(storage.a.data_ptr(), dtype=Float32), product_layout)
        derivatives = cute.make_tensor(cute.recast_ptr(storage.b.data_ptr(), dtype=BFloat16), grad_layout)
        decoded = cute.make_tensor(cute.recast_ptr(storage.glu_scratch.data_ptr(), dtype=BFloat16),
            cute.make_layout((width, cfg.tile_m + 1), stride=(cfg.tile_m + 1, 1)))

        # No role enters until all K-stage transaction waits and MMA reads
        # retire. One nonpersistent output tile means no subsequent TMA fill
        # can race the aliases. Publish products and dP before splitting.
        cute.arch.sync_threads()
        self._publish_recompute(products, derivatives, dh, acc, coords, bm, bn, thread)
        cute.arch.sync_threads()
        lane = cute.arch.lane_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        if warp < self.gradient_threads // 32:
            self._gradient_team(derivatives, qp, sp, qpt, spt, bm, bn, warp * 32 + lane)
        else:
            self._hidden_team(products, decoded, qh, sh, qht, sht, bm, bn,
                              warp * 32 + lane - self.gradient_threads)
