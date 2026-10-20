"""Persistent runner scratch must not belong to Inductor's output graph pool.

Only runner factories use this decorator: returned operator outputs still use
the caller's allocator. Pools are per-device, scratch remains owned by bounded
runner caches, and existing stream-specific runner keys govern scratch reuse.
"""
from functools import wraps
from inspect import signature
from threading import RLock, local

_pools = {}
_lock = RLock()
_thread = local()


def persistent_scratch(factory):
    parameters = signature(factory)

    @wraps(factory)
    def build(*args, **kwargs):
        import torch

        bound = parameters.bind(*args, **kwargs).arguments
        device = bound['device'] if 'device' in bound else bound['x'].device
        device = torch.device(device)
        if device.type != 'cuda':
            # CPU-side factory/configuration tests must not initialize CUDA.
            # The factory remains responsible for its actual device support.
            return factory(*args, **kwargs)
        index = device.index if device.index is not None else torch.cuda.current_device()
        # Some factories recursively resolve an autotune token before building
        # the selected runner. Do not nest allocator routing for that device.
        active = getattr(_thread, 'devices', ())
        if index in active:
            return factory(*args, **kwargs)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('RTX runner initialization requires warmup before manual CUDA graph capture')
        with _lock:
            if index not in _pools:
                with torch.cuda.device(index):
                    _pools[index] = torch.cuda.MemPool(use_on_oom=True)
            pool = _pools[index]
        from .runtime import _retain_resource
        _retain_resource(pool)
        _thread.devices = (*active, index)
        try:
            with torch.cuda.use_mem_pool(pool, device=index):
                return factory(*args, **kwargs)
        finally:
            _thread.devices = active

    return build


def clear_scratch_pools():
    """Drop pool ownership after synchronized runtime runner-cache cleanup."""
    with _lock:
        _pools.clear()


def retain_scratch_pools(owners):
    """Graph scopes pin allocator pools as well as the runner tensors."""
    with _lock:
        for pool in _pools.values():
            owners[id(pool)] = pool
