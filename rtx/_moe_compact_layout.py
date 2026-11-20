"""Shape-only proofs for the private compact MoE forward layout."""

from math import gcd


def compact_forward_capacity(n, pools, e, k, c, alignment):
    """Safe compact quota rounded for forward, or zero for no storage win.

    Use only for a known balanced route with the given number of pools. A
    nonempty pool of L tokens contributes at most q*ceil(L/p) scheduled rows
    to each expert, for p=E/gcd(E,K) and q=K/gcd(E,K). At most min(T,B) pools
    are nonempty, even when device boundaries change at the same addresses.
    No capacity smaller than the original storage is assumed for opaque routes.
    """
    if pools is None or pools == 0:
        return 0
    if any(type(v) is not int or v <= 0 for v in (n, pools, e, k, c, alignment)) or k > e:
        raise ValueError('Invalid compact forward geometry')
    divisor = gcd(e, k)
    period = e // divisor
    rows = (n + min(n, pools) * (period - 1)) // period * (k // divisor)
    stride = (rows + alignment - 1) // alignment * alignment
    return stride if stride < c else 0
