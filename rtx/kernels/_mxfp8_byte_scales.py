"""CuTe host adapters for compiler-owned UINT8 E8M0 storage."""
import cutlass.cute as cute
from cutlass import Float8E8M0FNU


@cute.jit
def _e8(value):
    return cute.make_tensor(cute.recast_ptr(value.iterator, dtype=Float8E8M0FNU), value.layout)


class ByteQuant:
    def __init__(self, inner):
        self.inner = inner

    @cute.jit
    def __call__(self, x, out, scale, stream):
        self.inner(x, out, _e8(scale), stream)


class ByteDualQuant:
    def __init__(self, inner):
        self.inner = inner

    @cute.jit
    def __call__(self, x, w, qx, qw, sx, sw, stream):
        self.inner(x, w, qx, qw, _e8(sx), _e8(sw), stream)


class ByteGemm:
    def __init__(self, inner):
        self.inner = inner

    @cute.jit
    def __call__(self, qx, qw, sx, sw, out, aux, stream):
        self.inner(qx, qw, _e8(sx), _e8(sw), out, aux, stream)


class ByteMixedDown:
    def __init__(self, inner):
        self.inner = inner

    @cute.jit
    def __call__(self, x, w, out, s0, s1, s2, s3, stream):
        self.inner(x, w, out, _e8(s0), _e8(s1), _e8(s2), _e8(s3), stream)
