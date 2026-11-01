"""Library-owned scratch lifecycle for default-mode Inductor execution.

Version-sensitive integration is confined here. Compiled callables own their
working sets; CUDA Graph Trees additionally transfer ownership during cold
warmup/recording. Manager execution and graph replay are unchanged.
"""
from dataclasses import fields, is_dataclass
from functools import wraps
from threading import RLock, local
import weakref

_lock = RLock()
_installed = False
_submission_locks = {}
_instrumented = set()
_thread = local()
_tuning_scopes = 0
_stats = dict(compiled_functions=0, warmups=0, recordings=0,
              retired_entries=0, retired_storage_bytes=0)


def submission_lock(device):
    with _lock:
        return _submission_locks.setdefault(int(device), RLock())


def guard_submission(function):
    """Keep mutable multi-launch scratch submission atomic with retirement."""
    if getattr(function, '_rtx_submission_guard', False):
        return function

    @wraps(function)
    def launch(*args, **kwargs):
        if getattr(_thread,'submitting',False):
            return function(*args,**kwargs)
        import torch
        tensor = next((v for v in (*args, *kwargs.values())
                       if isinstance(v, torch.Tensor) and v.is_cuda), None)
        if tensor is None:
            return function(*args, **kwargs)
        with submission_lock(tensor.device.index):
            _thread.submitting=True
            try:
                return function(*args, **kwargs)
            finally:
                _thread.submitting=False

    launch._rtx_submission_guard = True
    return launch


def tuning_resources(function):
    """Do not attach cold trial allocations to the enclosing compiled model."""
    @wraps(function)
    def tune(*args,**kwargs):
        global _tuning_scopes
        if getattr(_thread,'tuning',False):
            return function(*args,**kwargs)
        _thread.tuning=True
        with _lock:_tuning_scopes+=1
        from .execution import suspend_automatic_retention
        try:
            with suspend_automatic_retention():
                return function(*args,**kwargs)
        finally:
            with _lock:_tuning_scopes-=1
            _thread.tuning=False
    return tune


def in_tuning():
    # Autograd/first-hit workers need the same conservative cold-path policy.
    return bool(_tuning_scopes)


def instrument_runner(value):
    """Preserve runner types/ABI, including prepared MLP operand consumers."""
    seen = set()

    def visit(item):
        if id(item) in seen or not is_dataclass(item) or isinstance(item, type):
            return
        seen.add(id(item))
        cls = type(item)
        if cls.__module__.startswith('rtx.'):
            with _lock:
                if cls not in _instrumented:
                    for name in ('__call__', 'matmul', 'quantize', 'quantize_quad', 'launch_raw'):
                        method = cls.__dict__.get(name)
                        if callable(method):
                            setattr(cls, name, guard_submission(method))
                    _instrumented.add(cls)
        for field in fields(item):
            visit(getattr(item, field.name))

    visit(value)


def lookup_recorded(cache, key):
    """Cold recapture may reuse live graph-owned runners evicted from the LRU.

    Called only under an explicit recording owner, never for eager execution.
    Weak cache identities prevent address/id reuse from matching another cache.
    """
    from ._resources import _graphs, _lock as graph_lock
    identity = (id(cache), key)
    with graph_lock:
        for graph, pins in tuple(_graphs.items()):
            entry = getattr(graph, '_rtx_context_keys', {}).get(identity)
            if entry is not None and entry[0]() is cache and entry[1] in pins:
                return True, pins[entry[1]]
    return False, None


def _bind_graph_contexts(graph, owner):
    pins = getattr(graph, '_rtx_resource_owners', {})
    with owner._lock:
        graph._rtx_context_keys = {
            key: (weakref.ref(owner._cache_refs[key[0]]), id(value))
            for key, value in owner._runners.items() if id(value) in pins
        }


def _retire_preparation(graph, device):
    """Discard only unpinned preparation caches at a completed capture boundary.

    Streams are discovered from exact duplicate contexts, not supplied by the
    user. Once a preparation stream is identified, its unpinned scratch may be
    rebuilt on a future eager call. Compiled code and selections are preserved.
    """
    import torch
    from ._resources import caches, graph_resources, _stream_context
    from .runtime import tensor_storage_bytes
    entries = getattr(graph, '_rtx_context_keys', {})
    if not entries:
        return
    with submission_lock(device):
        captured = {}
        for (cid, key), (cache_ref, _) in entries.items():
            context = _stream_context(key)
            if context is not None and context[0] == device and cache_ref() is not None:
                identity = (cid, key[:context[2]] + key[context[2]+1:])
                captured.setdefault(identity, set()).add(context[1])
        pinned = {identity for group in graph_resources() for identity in group}
        streams = set()
        snapshots = []
        for cache in caches():
            with cache._lock:
                values = tuple(cache._values.items())
            snapshots.append((cache, values))
            for key, value in values:
                context = _stream_context(key)
                if context is None or context[0] != device or id(value) in pinned:
                    continue
                identity = (id(cache), key[:context[2]] + key[context[2]+1:])
                if identity in captured and context[1] not in captured[identity]:
                    streams.add(context[1])
        from .execution import inactive_automatic_contexts,retire_automatic_contexts
        inactive=inactive_automatic_contexts()
        for _,cid,key,value in inactive:
            context=_stream_context(key)
            if context is None or context[0]!=device or id(value) in pinned:continue
            identity=(cid,key[:context[2]]+key[context[2]+1:])
            if identity in captured and context[1] not in captured[identity]:streams.add(context[1])
        if not streams:
            return
        # All RTX scratch submissions on this device use the same reentrant
        # guard. No user thread can race the completion/retirement boundary.
        torch.cuda.synchronize(device)
        removed = []
        for cache, values in snapshots:
            with cache._lock:
                for key, value in values:
                    context = _stream_context(key)
                    if (context is not None and context[0] == device and context[1] in streams
                            and id(value) not in pinned and tensor_storage_bytes(value)
                            and cache._values.get(key) is value):
                        removed.append(value)
                        del cache[key]
        from ._workspace import retire_scratch_pools
        retire_automatic_contexts(inactive,streams,device)
        retire_scratch_pools(streams, {device})
        with _lock:
            _stats['retired_entries'] += len(removed)
            _stats['retired_storage_bytes'] += tensor_storage_bytes(removed)


def install():
    """Install the tested cold-path integration without initializing CUDA."""
    global _installed
    with _lock:
        if _installed:
            return
        from torch._inductor import cudagraph_trees as trees
        from torch._inductor.output_code import CompiledFxGraph
        from .execution import ExecutionResources, current_owner
        manager_type = trees.CUDAGraphTreeManager
        warmup, record = manager_type.run_eager, manager_type.record_function
        post_compile = CompiledFxGraph.post_compile

        class ManagedResources(ExecutionResources):
            _rtx_automatic = True
            _rtx_recording = False

            def _allocate(self, count, device, lane):
                from .execution import automatic_split_workspace
                return automatic_split_workspace(count, device, lane=lane)

        @wraps(post_compile)
        def post_compile_with_resources(compiled, *args, **kwargs):
            result = post_compile(compiled, *args, **kwargs)
            source = getattr(compiled, 'source_code', '') or ''
            target = compiled.current_callable
            if (target is not None and ('_rtx_' in source or 'torch.ops.rtx.' in source)
                    and not getattr(target, '_rtx_compiled_resources', False)):
                owner = ManagedResources()

                @wraps(target)
                def call(inputs):
                    if in_tuning():
                        return target(inputs)
                    with owner.activate():
                        return target(inputs)

                call._rtx_compiled_resources = True
                # The owner lives only on the process-local callable. Inductor
                # strips current_callable before serializing compiled artifacts.
                compiled.current_callable = call
                with _lock:
                    _stats['compiled_functions'] += 1
            return result

        def pending(manager):
            if not hasattr(manager, '_rtx_pending_warmups'):
                manager._rtx_pending_warmups = {}
            return manager._rtx_pending_warmups

        @wraps(warmup)
        def run_eager(manager, new_inputs, function_id):
            owner = pending(manager).get(function_id)
            if owner is None:
                owner = current_owner() or ManagedResources()
                pending(manager)[function_id] = owner
            with owner.activate():
                result = warmup(manager, new_inputs, function_id)
            with _lock:
                _stats['warmups'] += 1
            return result

        @wraps(record)
        def record_function(manager, new_inputs, function_id):
            owner = pending(manager).get(function_id)
            if owner is None:
                owner = current_owner() or ManagedResources()
                pending(manager)[function_id] = owner
            previous = getattr(owner, '_rtx_recording', False)
            owner._rtx_recording = True
            try:
                with owner.activate():
                    result = record(manager, new_inputs, function_id)
            finally:
                owner._rtx_recording = previous
            graph = manager.current_node.graph
            if graph is not None:
                _bind_graph_contexts(graph, owner)
                if getattr(owner, '_rtx_automatic', False):
                    _retire_preparation(graph, manager.device_index)
            pending(manager).pop(function_id, None)
            if getattr(owner, '_rtx_automatic', False):
                # The graph now owns every address it recorded. Do not retain
                # duplicate initialization ownership for manager lifetime.
                with owner._lock:
                    owner._runners.clear()
                    owner._cache_refs.clear()
                    owner._slots.clear()
            with _lock:
                _stats['recordings'] += 1
            return result

        manager_type.run_eager = run_eager
        manager_type.record_function = record_function
        CompiledFxGraph.post_compile = post_compile_with_resources
        _installed = True


def stats():
    with _lock:
        return dict(_stats)
