"""Exact integer construction of the producer's truncated E8M4 scale.

This is NOT general FP32 division by six. The caller supplies the producer's
nonnegative FP32 maximum: +0, positive finite values, or +infinity. Negative
values and NaNs are outside this helper's contract. Four retained mantissa bits
and flushing subnormal scales are part of the operation, not optional policies.
"""
import cutlass.cute as cute
from cutlass import Float32,Int32


@cute.jit
def producer_scale_rtz4(maximum:Float32):
    """Match flush_subnormal(trunc4(RN(maximum / 6))) on producer maxima.

    For a normal finite input M is its 24-bit integer significand. Retained
    boundaries occur at multiples of 3*2^17 or 3*2^18. One input step is 4/3
    or 2/3 quotient ULP, so RN cannot move a below-boundary value across one.
    Thus integer floor construction gives the same retained bits. E<=2 always
    flushes; at E=3 the carry selects the exact minimum-normal threshold.
    """
    bits=maximum.bitcast(Int32)
    exponent=(bits>>Int32(23))&Int32(255)
    output=Int32(0)
    if exponent==Int32(255):
        # Only positive infinity is in-domain for this exponent.
        output=Int32(0x7f800000)
    elif exponent>=Int32(3):
        # N in [64,127]; floor(N/3) == (N*171)>>9 exactly on this domain.
        n=((bits>>Int32(17))&Int32(63))|Int32(64)
        q=(n*Int32(171))>>Int32(9)
        carry=q>>Int32(5)
        normalized=q>>carry
        output_exponent=exponent-Int32(3)+carry
        if output_exponent>Int32(0):
            output=(output_exponent<<Int32(23))|((normalized-Int32(16))<<Int32(19))
    return output.bitcast(Float32)
