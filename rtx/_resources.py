"""Capture ownership and cache inventory. No CUDA initialization at import."""
from functools import wraps
import gc
import sys
from threading import Condition, RLock, get_ident
import weakref

_lock = RLock()
_capture_gate = Condition(_lock)
_caches = weakref.WeakValueDictionary()
_graphs = weakref.WeakKeyDictionary()
_captures = {}
_capture_leases = {}
_lease_gc_counts = None
_pending_owners = {}
_retiring_thread = None
_installed = False


def register_cache(cache):
    with _lock:
        _caches[id(cache)] = cache


def caches():
    with _lock:
        return tuple(_caches.values())


def _release_graph(owners, device):
    if owners:
        # Automatic GC can reach a weak finalizer while this very thread
        # snapshots the weak graph registry. Queue then, without running CUDA
        # operations under the outer registry lock.
        may_drain = not _lock._is_owned()
        with _lock:
            _pending_owners.setdefault(device, []).append(owners)
        if may_drain:
            _drain_graph_retirements()


def _collection_counts():
    return tuple(generation['collections'] for generation in gc.get_stats())


def _lease_graph(graph):
    global _lease_gc_counts
    if not _capture_leases:
        _lease_gc_counts = _collection_counts() if gc.isenabled() else None
    _capture_leases[id(graph)] = graph


def _drop_capture_leases(leases, previous):
    leases.clear()
    if previous is not None and gc.isenabled():
        current = _collection_counts()
        if current != previous:
            # A cycle deemed reachable only because of a lease can be
            # promoted during capture. Retry the missed collection now,
            # not inside CUDA capture and not on every graph entrance.
            generation = 1 if current[1:] == previous[1:] and sys.version_info < (3, 14) else 2
            gc.collect(generation)


def _drain_graph_retirements():
    """Claim the safe boundary under lock; execute retirement without it."""
    global _retiring_thread
    with _lock:
        if _captures or _retiring_thread is not None or not (_capture_leases or _pending_owners):
            return
        _retiring_thread = get_ident()
    _retire_graph_data()


def _retire_graph_data():
    """Retire with exclusive access to the capture boundary, not the mutex."""
    global _capture_leases, _lease_gc_counts, _retiring_thread
    import torch

    with _lock:
        leases, _capture_leases = _capture_leases, {}
        previous, _lease_gc_counts = _lease_gc_counts, None
    failed = {}

    def finish():
        global _retiring_thread
        for device, groups in failed.items():
            _pending_owners.setdefault(device, []).extend(groups)
        _retiring_thread = None
        _capture_gate.notify_all()

    try:
        # Native GraphExecDestroy can invalidate an unrelated CUDA capture.
        # Drop peer leases only after capture_end; their weak finalizers can
        # add pin groups to the same retirement batch.
        _drop_capture_leases(leases, previous)
        while True:
            with _capture_gate:
                if not _pending_owners:
                    finish()
                    return
                device, groups = _pending_owners.popitem()
            try:
                if torch.cuda.is_initialized():
                    # Replay can use a stream other than the capture stream.
                    # A device barrier here (never in replay) covers them all.
                    torch.cuda.synchronize(device)
            except BaseException as error:
                # Completion was not proved. Retain pins and retry at the
                # next safe boundary rather than swallowing and freeing them.
                failed.setdefault(device, []).extend(groups)
                if not isinstance(error, Exception):
                    raise
            else:
                for owners in groups:
                    owners.clear()
    except BaseException:
        with _capture_gate:
            finish()
        raise


def _wait_for_retirement():
    # A CUDA capture cannot start during a pending device barrier/destruction.
    # Waiting releases the mutex, including for concurrent finalizer enqueue.
    if _retiring_thread == get_ident():
        raise RuntimeError('cannot start CUDA graph capture during graph retirement')
    _capture_gate.wait_for(lambda: _retiring_thread is None)


def _capture_still_open(stream):
    """An exceptional begin/end need not have closed the native capture."""
    try:
        from cuda.bindings import driver
        error, status = driver.cuStreamIsCapturing(stream.cuda_stream)
        return bool(error) or status != driver.CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_NONE
    except Exception:
        # Do not issue a device barrier when the stream state is unknown.
        return True


def install_graph_resource_tracking():
    """Attach RTX scratch to public CUDAGraph capture/reset lifetime.

    Installed on the first CUDA runner warmup. This also covers Inductor Graph
    Trees, which use this same public capture API. Peer graph leases exist only
    during tracked capture, and no replay or launch hook is introduced.
    """
    global _installed
    with _lock:
        if _installed:
            return
        import torch
        if torch.cuda.is_initialized() and torch.cuda.is_current_stream_capturing():
            raise RuntimeError('warm RTX runners before CUDA graph capture')
        graph_type = torch.cuda.CUDAGraph
        begin, end, reset = graph_type.capture_begin, graph_type.capture_end, graph_type.reset

        def finish_capture(graph):
            scope = getattr(graph, '_rtx_capture_scope', None)
            try:
                if scope is not None:
                    graph._rtx_resource_owners.update(graph._rtx_current_owners)
                    try:
                        scope.__exit__(None, None, None)
                    finally:
                        del graph._rtx_capture_scope, graph._rtx_current_owners
            finally:
                with _lock:
                    _captures.pop(id(graph), None)
                    if _captures:
                        _lease_graph(graph)
                _drain_graph_retirements()

        @wraps(begin)
        def capture_begin(graph, *args, **kwargs):
            from .runtime import retain_runtime_resources
            stream = torch.cuda.current_stream()
            _drain_graph_retirements()
            with _capture_gate:
                _wait_for_retirement()
                if id(graph) in _captures:
                    raise RuntimeError('CUDA graph capture has already started')
                # Snapshot existing tracked graphs before the CUDA begin call.
                # Their native destruction is unsafe even if they have no RTX
                # pins. The weak registry is unchanged outside the boundary.
                for peer in tuple(_graphs):
                    if peer is not graph:
                        _lease_graph(peer)
                _captures[id(graph)] = stream
                pins = getattr(graph, '_rtx_resource_owners', None)
                if pins is None:
                    pins = graph._rtx_resource_owners = {}
                    graph._rtx_resource_finalizer = weakref.finalize(
                        graph, _release_graph, pins, stream.device.index
                    )
                _graphs[graph] = pins
            # Start ownership before capture; private-pool references are CPU
            # objects and do not enqueue GPU work.
            try:
                scope = retain_runtime_resources()
                graph._rtx_current_owners = scope.__enter__()
                graph._rtx_capture_scope = scope
                return begin(graph, *args, **kwargs)
            except BaseException:
                if not _capture_still_open(stream):
                    finish_capture(graph)
                raise

        @wraps(end)
        def capture_end(graph, *args, **kwargs):
            with _lock:
                stream = _captures.get(id(graph))
            try:
                result = end(graph, *args, **kwargs)
            except BaseException:
                if stream is not None and not _capture_still_open(stream):
                    finish_capture(graph)
                raise
            if stream is not None:
                finish_capture(graph)
            return result

        @wraps(reset)
        def graph_reset(graph, *args, **kwargs):
            global _retiring_thread
            with _capture_gate:
                already_retiring = _retiring_thread == get_ident()
                if not already_retiring:
                    _wait_for_retirement()
                if _captures:
                    raise RuntimeError('reset CUDA graphs outside active CUDA graph capture')
                if not already_retiring:
                    _retiring_thread = get_ident()
            try:
                result = reset(graph, *args, **kwargs)
                finalizer = getattr(graph, '_rtx_resource_finalizer', None)
                if finalizer is not None:
                    finalizer()
                    del graph._rtx_resource_finalizer, graph._rtx_resource_owners
                    with _lock:
                        _graphs.pop(graph, None)
                return result
            finally:
                if not already_retiring:
                    _retire_graph_data()

        graph_type.capture_begin, graph_type.capture_end, graph_type.reset = capture_begin, capture_end, graph_reset
        _installed = True


def graph_resources():
    with _lock:
        pending = (pins for groups in _pending_owners.values() for pins in groups)
        return tuple(dict(pins) for pins in (*_graphs.values(), *pending))


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
    from ._managed import stats as managed_stats
    result['automatic_execution']=managed_stats()
    if include_allocator:
        import torch
        if torch.cuda.is_initialized():
            result['allocator_segments']=torch.cuda.memory_snapshot()
    return result
