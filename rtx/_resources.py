"""Capture ownership and cache inventory. No CUDA initialization at import."""
from functools import wraps
from threading import RLock
import weakref

_lock = RLock()
_caches = weakref.WeakValueDictionary()
_graphs = weakref.WeakKeyDictionary()
_installed = False


def register_cache(cache):
    with _lock:
        _caches[id(cache)] = cache


def caches():
    with _lock:
        return tuple(_caches.values())


def _release_graph(owners, device):
    # CUDA graph execution is asynchronous even after its Python graph dies.
    # Keep allocations pinned through completion, not merely graph destruction.
    import torch
    try:
        if owners and torch.cuda.is_initialized():
            torch.cuda.synchronize(device)
    except Exception:
        # During interpreter/driver teardown there can be no further replay.
        pass
    owners.clear()


def install_graph_resource_tracking():
    """Attach RTX scratch to public CUDAGraph capture/reset lifetime.

    Installed on the first CUDA runner warmup. This also covers Inductor Graph
    Trees, which use this same public capture API. No replay or launch hook,
    and no global graph lifetime strong reference, is introduced.
    """
    global _installed
    with _lock:
        if _installed:
            return
        import torch
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('warm RTX runners before CUDA graph capture')
        graph_type = torch.cuda.CUDAGraph
        begin, end, reset = graph_type.capture_begin, graph_type.capture_end, graph_type.reset

        @wraps(begin)
        def capture_begin(graph, *args, **kwargs):
            from .runtime import retain_runtime_resources
            # Start ownership before capture; private-pool references are CPU
            # objects and do not enqueue GPU work.
            scope = retain_runtime_resources()
            owners = scope.__enter__()
            device = torch.cuda.current_device()
            try:
                result = begin(graph, *args, **kwargs)
            except BaseException:
                scope.__exit__(*__import__('sys').exc_info())
                raise
            graph._rtx_capture_scope = scope
            # Recapture/reset is explicit; preserve any earlier pins until reset.
            pins = getattr(graph, '_rtx_resource_owners', None)
            if pins is None:
                pins = graph._rtx_resource_owners = {}
                graph._rtx_resource_finalizer = weakref.finalize(graph, _release_graph, pins, device)
            graph._rtx_current_owners = owners
            with _lock:
                _graphs[graph] = pins
            return result

        @wraps(end)
        def capture_end(graph, *args, **kwargs):
            try:
                return end(graph, *args, **kwargs)
            finally:
                scope = getattr(graph, '_rtx_capture_scope', None)
                if scope is not None:
                    graph._rtx_resource_owners.update(graph._rtx_current_owners)
                    scope.__exit__(None, None, None)
                    del graph._rtx_capture_scope, graph._rtx_current_owners

        @wraps(reset)
        def graph_reset(graph, *args, **kwargs):
            result = reset(graph, *args, **kwargs)
            finalizer = getattr(graph, '_rtx_resource_finalizer', None)
            if finalizer is not None:
                finalizer()
                del graph._rtx_resource_finalizer, graph._rtx_resource_owners
                with _lock:
                    _graphs.pop(graph, None)
            return result

        graph_type.capture_begin, graph_type.capture_end, graph_type.reset = capture_begin, capture_end, graph_reset
        _installed = True


def graph_resources():
    with _lock:
        return tuple(dict(pins) for pins in _graphs.values())


def _stream_context(key):
    """All RTX scratch caches use device/stream prefixes, including partials."""
    if not isinstance(key, tuple):
        return None
    offset = 1 if key and key[0] in ('dx', 'dw') else 0
    if len(key) > offset + 1 and type(key[offset]) is int and type(key[offset+1]) is int:
        return key[offset], key[offset+1], offset+1
    return None


def retire_runtime_contexts(*, streams, device=None, synchronize=True):
    """Retire selected quiescent stream caches, keeping live graph pins intact.

    Call at a startup/iteration boundary with no concurrent host submissions.
    Synchronization is not a concurrency lock. Compiled plans, selections, live
    graphs and unrelated streams are not cleared. Returns measured cache bytes,
    not a promised allocator/device-memory saving.
    """
    import torch
    from .runtime import tensor_storage_bytes
    if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
        raise RuntimeError('retire contexts outside CUDA graph capture')
    stream_ids = {int(getattr(s, 'cuda_stream', s)) for s in streams}
    if not stream_ids:
        raise ValueError('explicit preparation stream IDs are required')
    index = None if device is None else torch.device(device).index
    if device is not None and index is None:
        index = torch.cuda.current_device()
    removed, targets = [], set()
    for cache in caches():
        with cache._lock:
            for key, value in tuple(cache._values.items()):
                context = _stream_context(key)
                if context is not None and context[1] in stream_ids and (index is None or index == context[0]):
                    if tensor_storage_bytes(value):
                        removed.append((cache, key, value))
                        targets.add(context[0])
    if synchronize:
        for target in sorted(targets):
            torch.cuda.synchronize(target)
    before = tensor_storage_bytes([v for _, _, v in removed])
    for cache, key, value in removed:
        with cache._lock:
            if cache._values.get(key) is value:
                del cache[key]
    from ._workspace import retire_scratch_pools
    retire_scratch_pools(stream_ids,targets)
    pinned = [v for pins in graph_resources() for v in pins.values()]
    return dict(retired_entries=len(removed),retired_storage_bytes=before,
                graph_pinned_storage_bytes=tensor_storage_bytes(pinned),
                devices=sorted(targets), streams=sorted(stream_ids))


def runtime_memory_report(*, include_allocator=False):
    """Unique explicit storage across top-level, nested and graph-pinned owners."""
    from .runtime import tensor_storage_bytes, tensor_storage_inventory
    entries, names, duplicates = [], [], {}
    for cache in caches():
        with cache._lock:
            values = tuple(cache._values.items())
        if not values:
            continue
        owners = [v for _, v in values]
        entries.extend(owners)
        names.append(dict(cache_id=id(cache),entries=len(values),
                          unique_storage_bytes=tensor_storage_bytes(owners),
                          counters=cache.stats(),
                          contexts=[dict(key=repr(k),stream_context=_stream_context(k),
                                         storages=tensor_storage_inventory(v)) for k,v in values]))
        for key, value in values:
            context = _stream_context(key)
            if context is not None:
                identity = (id(cache), key[:context[2]] + key[context[2]+1:])
                duplicates.setdefault(identity, []).append(value)
    pins = [v for group in graph_resources() for v in group.values()]
    from .execution import execution_resources
    execution=execution_resources()
    result=dict(schema_version=1, caches=names,
        cached_storage_bytes=tensor_storage_bytes(entries),
        graph_pinned_storage_bytes=tensor_storage_bytes(pins),
        execution_owned_storage_bytes=tensor_storage_bytes(execution),
        unique_storage_bytes=tensor_storage_bytes([*entries,*pins,*execution]),
        duplicated_context_groups=sum(len(v)>1 for v in duplicates.values()),
        accounting='unique explicit tensor storage; excludes allocator rounding/reserve and opaque executable internals')
    if include_allocator:
        import torch
        if torch.cuda.is_initialized():
            result['allocator_segments']=torch.cuda.memory_snapshot()
    return result
