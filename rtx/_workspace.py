"""Persistent runner scratch must not belong to Inductor's output graph pool.

Only runner factories use this decorator: returned operator outputs still use
the caller's allocator. Pools are per-device/stream, scratch remains owned by bounded
runner caches, and existing stream-specific runner keys govern scratch reuse.
"""
from functools import wraps
from inspect import signature
from threading import RLock, local
from weakref import WeakValueDictionary

# Routing must not own otherwise-unused pools: bounded runner eviction must
# release the corresponding reserve without a process-wide cleanup call.
_pools = WeakValueDictionary()
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
        from ._resources import install_graph_resource_tracking
        install_graph_resource_tracking()
        # Some factories recursively resolve an autotune token before building
        # the selected runner. Do not nest allocator routing for that device.
        active = getattr(_thread, 'devices', ())
        if index in active:
            return factory(*args, **kwargs)
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('RTX runner initialization requires warmup before manual CUDA graph capture')
        key=(index,int(torch.cuda.current_stream(index).cuda_stream))
        with _lock:
            pool=_pools.get(key)
            if pool is None:
                with torch.cuda.device(index):
                    pool = torch.cuda.MemPool(use_on_oom=True)
                    _pools[key] = pool
        from .runtime import _retain_resource
        _retain_resource(pool)
        _thread.devices = (*active, index)
        try:
            with torch.cuda.use_mem_pool(pool, device=index):
                result=factory(*args, **kwargs)
                _tag_pool_owners(result,pool)
                return result
        finally:
            _thread.devices = active

    return build


def clear_scratch_pools():
    """Drop pool ownership after synchronized runtime runner-cache cleanup."""
    with _lock:
        _pools.clear()


def retain_scratch_pools(owners):
    """Legacy explicit pool pinning; normal graphs pin only accessed owners."""
    with _lock:
        for pool in _pools.values():
            owners[id(pool)] = pool


def _tag_pool_owners(value,pool):
    """Tensor lifetime owns its allocator pool, including after cache eviction."""
    from dataclasses import fields,is_dataclass
    import torch
    seen=set()
    def visit(item):
        if id(item) in seen:return
        seen.add(id(item))
        if isinstance(item,torch.Tensor) and item.is_cuda:
            root=item
            while root._base is not None:root=root._base
            if not hasattr(root,'_rtx_scratch_pool'):root._rtx_scratch_pool=pool
            item._rtx_scratch_pool=root._rtx_scratch_pool
        elif is_dataclass(item) and not isinstance(item,type):
            for f in fields(item):visit(getattr(item,f.name))
        elif isinstance(item,dict):
            for child in item.values():visit(child)
        elif isinstance(item,(tuple,list)):
            for child in item:visit(child)
        elif hasattr(item,'__rtx_resources__'):
            visit(item.__rtx_resources__)
    visit(value)


def retire_scratch_pools(streams,devices):
    """Drop only routing ownership; live runner/graph tensors still pin pools."""
    with _lock:
        for key in tuple(_pools):
            if key[0] in devices and key[1] in streams:
                del _pools[key]
