"""Host-side CuTe descriptor contracts shared by RTX kernel compilers."""
import cutlass.cute as cute


def preserve_tma_singletons(shape):
    """Keep extent-one axes in TMA descriptors without allocating padding.

    A static singleton can be folded out while the tiled SMEM destination
    still has that axis. A symbolic extent carries the real runtime value
    (still one), retaining the descriptor's dimensionality. Non-singleton
    extents remain statically specialized.
    """
    return tuple(cute.sym_int32() if extent == 1 else extent for extent in shape)
