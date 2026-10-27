"""Independent integer Philox4x32-10 oracle for MS-EDEN scale rounding.

No CuTe, Torch, Triton, CUDA or production RNG helpers are imported. Standard
known-answer vectors are from Random123's tests/kat_vectors:
https://github.com/DEShawResearch/random123/blob/main/tests/kat_vectors

The RTX mapping is counter=(index//4, operand_stream, 0, 0), key=seed[0:2],
selected word=index%4, uniform=(word>>8)*2**-24. This is not Torch's RNG stream.
"""

MASK = (1 << 32) - 1


def _words(values, length):
    if len(values) != length or any(type(v) is not int or not 0 <= v <= MASK for v in values):
        raise ValueError(f"expected {length} unsigned 32-bit Python integers")
    return tuple(values)


def philox4x32(counter, key):
    """One four-word block, using exact Python multiplication and modulo 2**32."""
    c0, c1, c2, c3 = _words(counter, 4)
    k0, k1 = _words(key, 2)
    for _ in range(10):
        p0 = 0xD2511F53 * c0
        p1 = 0xCD9E8D57 * c2
        c0, c1, c2, c3 = ((p1 >> 32) ^ c1 ^ k0, p1 & MASK,
                          (p0 >> 32) ^ c3 ^ k1, p0 & MASK)
        k0 = (k0 + 0x9E3779B9) & MASK
        k1 = (k1 + 0xBB67AE85) & MASK
    return c0, c1, c2, c3


def uniform_at(seed, index, operand_stream):
    if type(index) is not int or not 0 <= index < (1 << 34):
        raise ValueError("index must fit the 32-bit block counter")
    if type(operand_stream) is not int or operand_stream not in (0, 1):
        raise ValueError("operand stream must be 0 or 1")
    words = philox4x32((index // 4, operand_stream, 0, 0), seed)
    return (words[index % 4] >> 8) * (2.0 ** -24)


def uniforms(seed, count, operand_stream):
    if type(count) is not int or not 0 <= count <= (1 << 34):
        raise ValueError("count must fit the 32-bit block counter")
    _words(seed, 2)
    if type(operand_stream) is not int or operand_stream not in (0, 1):
        raise ValueError("operand stream must be 0 or 1")
    result = []
    for block in range((count + 3) // 4):
        result.extend((word >> 8) * (2.0 ** -24)
                      for word in philox4x32((block, operand_stream, 0, 0), seed))
    return result[:count]
