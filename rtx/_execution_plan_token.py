"""Self-contained host metadata for runtime-selected recomputation plans.

Private forward operators can return this CPU tensor and save it for backward.
It contains the selected, revisioned configuration string, not a cache handle:
winner refinement, eviction, tensor views, and serialization cannot retarget it.
There is no registry owning activations or executable plans. This is host-only
launcher code; fake implementations allocate metadata without calling it.

The fixed extent is part of the private ABI, allowing fake/Inductor allocation
before runtime selection. Never reuse an output token across outstanding
forwards. A captured graph must retain its own token and fixed selected plan.
"""

import ctypes
import struct
import zlib

import torch


PLAN_TOKEN_BYTES = 65536
_HEADER = struct.Struct("<8sII")
_MAGIC = b"RTXPLAN1"


def _address(token):
    # Reject subclasses (including FakeTensor) before touching a data pointer.
    if type(token) is not torch.Tensor:
        raise TypeError("execution plan tokens require real, plain CPU tensors")
    if (token.device.type != "cpu" or token.dtype != torch.uint8
            or tuple(token.shape) != (PLAN_TOKEN_BYTES,)
            or not token.is_contiguous()):
        raise ValueError("execution plan token must be a contiguous CPU uint8 vector of fixed extent")
    return token.data_ptr()


def write_plan_token(token, config_key):
    """Fill a newly allocated private output; no eager tensor operations."""
    if not isinstance(config_key, str):
        raise TypeError("execution plan must be a serialized configuration string")
    payload = config_key.encode("utf-8")
    if not payload or len(payload) > PLAN_TOKEN_BYTES - _HEADER.size:
        raise ValueError("serialized execution plan is empty or exceeds token capacity")
    address = _address(token)
    header = _HEADER.pack(_MAGIC, len(payload), zlib.crc32(payload))
    # Initialize padding too: the complete output has deterministic contents.
    ctypes.memset(address, 0, PLAN_TOKEN_BYTES)
    ctypes.memmove(address, header + payload, len(header) + len(payload))


def read_plan_token(token):
    """Read host metadata without CUDA synchronization or a cache lookup."""
    address = _address(token)
    magic, size, checksum = _HEADER.unpack(ctypes.string_at(address, _HEADER.size))
    if magic != _MAGIC or not 0 < size <= PLAN_TOKEN_BYTES - _HEADER.size:
        raise ValueError("invalid execution plan token header")
    payload = ctypes.string_at(address + _HEADER.size, size)
    if zlib.crc32(payload) != checksum:
        raise ValueError("corrupted execution plan token")
    return payload.decode("utf-8")
