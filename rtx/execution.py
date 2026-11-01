"""Explicit ordered-execution workspace ownership, independent of kernel plans."""
from contextlib import contextmanager
from threading import RLock, local
import weakref

_local = local()
_active = {}
_lock = RLock()
_owners = weakref.WeakValueDictionary()


class ExecutionResources:
    """Optional per-stream FP32 reduction arena and graph-safe retirement.

    ``split_workspace_bytes`` is a fixed *per stream/lane* capacity, not a
    process-wide VRAM cap. Oversized requests use private storage. Serial
    consumers may share a slot; concurrent dX/dW lanes never share it. Quantized
    operands are deliberately not aliased across prepare/consume boundaries.

    Activate around uncaptured warmup and capture. Cached runners and graphs
    own their views independently of this Python scope. Retire preparation
    streams only at a boundary without concurrent host submissions.
    """
    def __init__(self, *, split_workspace_bytes=0):
        if type(split_workspace_bytes) is not int or split_workspace_bytes < 0:
            raise ValueError('split_workspace_bytes must be a nonnegative integer')
        self.capacity = (split_workspace_bytes + 255) // 256 * 256
        self._slots = {}
        self._runners = {}
        self._cache_refs = {}
        self._lock = RLock()
        with _lock:_owners[id(self)]=self

    @contextmanager
    def activate(self):
        previous = getattr(_local, 'owner', None)
        _local.owner = self
        with _lock:
            _active[id(self)] = (_active.get(id(self), (0, self))[0] + 1, self)
        try:
            yield self
        finally:
            with _lock:
                count = _active[id(self)][0] - 1
                if count:
                    _active[id(self)] = (count, self)
                else:
                    del _active[id(self)]
            _local.owner = previous

    def _allocate(self, count, device, lane):
        import torch
        size = count * 4
        if not self.capacity or size > self.capacity:
            return torch.empty(count, device=device, dtype=torch.float32)
        key = (device.index, int(torch.cuda.current_stream(device).cuda_stream), lane)
        with self._lock:
            storage = self._slots.get(key)
            if storage is None:
                storage = torch.empty(self.capacity // 4, device=device, dtype=torch.float32)
                self._slots[key] = storage
            view=storage[:count]
            view._rtx_workspace_lock=self._lock
            return view

    def retire(self, *, streams, device=None):
        from .runtime import retire_runtime_contexts
        import torch
        if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError('retire contexts outside CUDA graph capture')
        streams = tuple(streams)
        ids = {int(getattr(s, 'cuda_stream', s)) for s in streams}
        index = None if device is None else torch.device(device).index
        if device is not None and index is None:
            index = torch.cuda.current_device()
        from ._resources import _stream_context
        with self._lock:
            keys=[k for k in self._runners if (c:=_stream_context(k[1])) is not None
                  and c[1] in ids and (index is None or c[0]==index)]
        for target in {_stream_context(k[1])[0] for k in keys}:
            torch.cuda.synchronize(target)
        report = retire_runtime_contexts(streams=streams, device=device)
        with self._lock:
            for key in keys:self._runners.pop(key,None)
            for cid in tuple(self._cache_refs):
                if not any(k[0]==cid for k in self._runners):self._cache_refs.pop(cid)
            for key in tuple(self._slots):
                if key[1] in ids and (index is None or key[0] == index):
                    del self._slots[key]
        return report

    def stats(self):
        from .runtime import tensor_storage_bytes
        with self._lock:
            return dict(slots=len(self._slots), capacity_per_lane_bytes=self.capacity,
                        storage_bytes=tensor_storage_bytes(tuple(self._slots.values())),
                        retained_contexts=len(self._runners),
                        retained_storage_bytes=tensor_storage_bytes(tuple(self._runners.values())))


def current_owner():
    owner=getattr(_local,'owner',None)
    if owner is None:
        with _lock:
            if len(_active)==1:owner=next(iter(_active.values()))[1]
    return owner


def remember_resource(cache,key,value):
    owner=current_owner()
    if owner is not None:
        with owner._lock:
            owner._runners[(id(cache),key)]=value
            owner._cache_refs[id(cache)]=cache


def lookup_resource(cache,key):
    owner=current_owner()
    if owner is not None:
        with owner._lock:
            pair=(id(cache),key)
            if pair in owner._runners:return True,owner._runners[pair]
    return False,None


def execution_resources():
    with _lock:owners=tuple(_owners.values())
    result=[]
    for owner in owners:
        with owner._lock:
            result.extend(owner._runners.values())
            result.extend(owner._slots.values())
    return result


def split_workspace(count, device, *, lane='ordered'):
    """Factory-only allocation; every returned view is an explicit runner owner."""
    import torch
    owner = current_owner()
    if owner is None:
        return torch.empty(count, dtype=torch.float32, device=device)
    return owner._allocate(count, torch.device(device), lane)


def workspace_guard(tensor):
    """Serialize the complete write/reduce submission, not individual kernels."""
    from contextlib import nullcontext
    return getattr(tensor,'_rtx_workspace_lock',None) or nullcontext()
