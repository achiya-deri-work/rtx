"""Fragment-owned forward quantization for packed expert SiLU GLU.

This is a different transport for the existing MXFP8 hidden ABI, not a
different GLU or quantizer. Product stays FP32 until E4M3 conversion. Only
cross-warp amax partials (and optionally already-quantized bytes) use SMEM.
"""
from collections import defaultdict

import cutlass
import cutlass.cute as cute
from cutlass import Float8E4M3FN, Float32, Int16, Uint8, Uint32
from cutlass.experimental.primitives import nvvm_wrapper as nvvm

from ..configs.moe_backward import fragment_glu_rejection, fragment_glu_scratch_bytes
from .mxfp8_glu_prequant import MXFP8PrequantGLUKernel


def _fragment_ownership(tiled_mma, tile_m, tile_n, atom_m, mma_warps):
    """Certify grouping and partial-slot uniqueness from partition_C.

    Each result group contains the static accumulator indices of the gate
    components. Runtime coordinates still come from the same partition_C.
    The checked slot assignment is by column-warp ordinal, not by the first
    owned hidden column's position within a scale block.
    """
    identity = cute.make_identity_tensor((tile_m, tile_n))
    values = set()
    owners = defaultdict(set)
    representative = None
    lane_group = 4

    def require(condition, reason):
        if not condition:
            raise ValueError(f'Unsupported forward GLU accumulator ownership: {reason}')

    for thread in range(32 * mma_warps):
        fragment = tiled_mma.get_slice(thread).partition_C(identity)
        groups = {}
        for gate in range(0, cute.size(fragment), 2):
            row, paired_column = map(int, fragment[gate])
            up_row, up_column = map(int, fragment[gate + 1])
            require(paired_column % 2 == 0 and (up_row, up_column) == (row, paired_column + 1),
                    'gate and up must occupy adjacent fragment indices')
            column = paired_column // 2
            require(0 <= row < tile_m and 0 <= column < tile_n // 2 and (row, column) not in values,
                    'each tile element must have one owner')
            values.add((row, column))
            groups.setdefault((row, column // 32), []).append(gate)

        signature = tuple(map(tuple, groups.values()))
        if representative is None:
            representative = signature
        require(signature == representative, 'all lanes must use one static grouping')
        keys = tuple(groups)
        if thread % lane_group == 0:
            leader_keys = keys
        require(keys == leader_keys, 'lane groups must own one shared (row, hidden32) per group')
        if thread % lane_group == 0:
            part = (thread // 32) // atom_m
            for key, indices in groups.items():
                require(len(indices) * lane_group == 16, 'a lane group must own 16 values per partial')
                require(part not in owners[key], 'each (row, hidden32, column warp) needs one writer')
                owners[key].add(part)

    require(len(values) == tile_m * tile_n // 2 and len(owners) == tile_m * tile_n // 64
            and all(parts == {0, 1} for parts in owners.values()), 'every scale block needs two complete partials')
    return representative, lane_group


class ExpertFragmentGLU(MXFP8PrequantGLUKernel):
    def __init__(self, problem, config, activation, output_format, engine):
        if reason := fragment_glu_rejection(problem, config, engine, activation, output_format):
            raise ValueError(reason)
        super().__init__(problem, config, activation, output_format)
        self.packed_output = engine == 'fragment_packed'
        self.glu_scratch_elems = fragment_glu_scratch_bytes(config, engine) // 4

    def _setup_static_layouts(self):
        super()._setup_static_layouts()
        cfg = self.config
        self.fragment_groups, self.fragment_lanes = _fragment_ownership(
            self.tiled_mma, cfg.tile_m, cfg.tile_n, cfg.atom_layout_m, cfg.num_mma_warps)

    @cute.jit
    def _byte_offset(self, row, column):
        # Rotate 4-byte words by row: four adjacent MMA lanes contribute four
        # bytes of one word, while the other lane groups touch different banks.
        # The final row-contiguous u32 reads invert the same address map.
        return row * (self.config.tile_n // 2) + (column ^ (row % 8 * 4))

    @cute.jit
    def _glu_epilogue(self, storage, accumulators, coordinates, out, scales,
                      block_m, block_n, tidx, tma_out, tma_out_tensor, out_layout, work_slot):
        cfg = self.config
        tile_h = cfg.tile_n // 2
        tile_blocks = tile_h // 32
        output_h = cute.size(out, mode=[1])
        lane = cute.arch.lane_idx()
        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        part = warp // cfg.atom_layout_m
        scratch = storage.glu_scratch.data_ptr()
        partials = cute.make_tensor(scratch, cute.make_layout((cfg.tile_m, tile_blocks * 2),
                                                              stride=(tile_blocks * 2, 1)))
        staged = cute.recast_ptr(scratch + cfg.tile_m * tile_blocks * 2, dtype=Uint8)
        products = cute.make_rmem_tensor((cute.size(accumulators) // 2,), Float32)
        maxima = cute.make_rmem_tensor((len(self.fragment_groups),), Float32)

        if warp < cfg.num_mma_warps:
            for group in cutlass.range_constexpr(len(self.fragment_groups)):
                indices = self.fragment_groups[group]
                row = coordinates[indices[0]][0]
                local_row = row - block_m * cfg.tile_m
                block = coordinates[indices[0]][1] // 64 - block_n * tile_blocks
                maximum = Float32(0)
                for j in cutlass.range_constexpr(len(indices)):
                    gate = indices[j]
                    value = Float32(0)
                    valid = True
                    if cutlass.const_expr(self.problem.m % cfg.tile_m != 0):
                        valid = row < self.problem.m
                    if cutlass.const_expr(self.problem.n % cfg.tile_n != 0):
                        valid = valid and coordinates[gate][1] < self.problem.n
                    if valid:
                        value = self._activate(accumulators[gate]) * accumulators[gate + 1]
                    products[gate // 2] = value
                    maximum = cute.arch.fmax(maximum, cute.abs(value), nan=True)
                maximum = self._warp_amax(maximum, self.fragment_lanes, lane)
                maxima[group] = maximum
                if lane % self.fragment_lanes == 0:
                    partials[local_row, 2 * block + part] = maximum
        cute.arch.sync_threads()

        if warp < cfg.num_mma_warps:
            for group in cutlass.range_constexpr(len(self.fragment_groups)):
                indices = self.fragment_groups[group]
                row = coordinates[indices[0]][0]
                local_row = row - block_m * cfg.tile_m
                block = coordinates[indices[0]][1] // 64
                local_block = block - block_n * tile_blocks
                if row < self.problem.m and block < output_h // 32:
                    other = partials[local_row, 2 * local_block + (part ^ 1)]
                    scale, inverse = self._scale_from_amax(cute.arch.fmax(maxima[group], other, nan=True))
                    for pair in cutlass.range_constexpr(len(indices) // 2):
                        first, second = indices[2 * pair], indices[2 * pair + 1]
                        packed = nvvm.inline_ptx_hl(
                            'cvt.rn.satfinite.e4m3x2.f32 {$w0}, {$r0}, {$r1};',
                            write_only_types=[Int16],
                            read_only_args=[products[second // 2] * inverse, products[first // 2] * inverse],
                        )
                        c0, c1 = coordinates[first][1] // 2, coordinates[second][1] // 2
                        lo, hi = Uint8(packed & 255), Uint8((packed >> 8) & 255)
                        if cutlass.const_expr(self.packed_output):
                            addr0 = staged + self._byte_offset(local_row, c0 - block_n * tile_h)
                            addr1 = staged + self._byte_offset(local_row, c1 - block_n * tile_h)
                            addr0[0] = lo
                            addr1[0] = hi
                        else:
                            out[row, c0] = lo.bitcast(Float8E4M3FN)
                            out[row, c1] = hi.bitcast(Float8E4M3FN)
                    if part == 0 and lane % self.fragment_lanes == 0:
                        scales[row, block] = scale.bitcast(Uint8)

        if cutlass.const_expr(self.packed_output):
            cute.arch.sync_threads()
            # Unlike an MMA fragment store (four adjacent bytes per row and
            # instruction), adjacent lanes now own successive u32 words of
            # their output row; both 32/64-wide tile rows fill entire sectors.
            for visit in cutlass.range_constexpr(cute.ceil_div(cfg.tile_m * tile_h, 4 * cfg.num_threads)):
                index = (tidx + visit * cfg.num_threads) * 4
                local_row, local_col = index // tile_h, index % tile_h
                row, col = block_m * cfg.tile_m + local_row, block_n * tile_h + local_col
                if local_row < cfg.tile_m and row < self.problem.m and col < output_h:
                    offset = cute.assume(self._byte_offset(local_row, local_col), divby=4)
                    target = cute.assume(row * output_h + col, divby=4)
                    word = cute.recast_ptr(staged + offset, dtype=Uint32)[0]
                    dest = cute.recast_ptr(out.iterator + target, dtype=Uint32)
                    dest[0] = word
