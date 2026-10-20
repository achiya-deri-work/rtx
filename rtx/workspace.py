"""Explicit process-wide MXFP8 backward retained-workspace policy.

Configure during startup, before launching work or capturing graphs. This is
not an allocator limit or a bound on externally pinned graph resources.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class WorkspacePolicy:
    """Per-cache bytes with optional entry caps (None means byte-only).

    Paired runners execute both dX/dW; partial runners execute one leg. They
    have separate ownership and budgets. No hidden eight-entry cap applies
    when entries are None. Both byte limits are mandatory, in bytes.
    """
    paired_bytes: int
    partial_bytes: int
    paired_entries: int | None = None
    partial_entries: int | None = None

    def __post_init__(self):
        for name in ('paired_bytes', 'partial_bytes'):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f'{name} must be a nonnegative integer')
        for name in ('paired_entries', 'partial_entries'):
            v = getattr(self, name)
            if v is not None and (type(v) is not int or v < 0):
                raise ValueError(f'{name} must be None or a nonnegative integer')


def _caches():
    from . import fp8_bwd
    return fp8_bwd, {'paired': fp8_bwd._RUNNERS, 'partial': fp8_bwd._MATMUL_RUNNERS}


def configure_workspace(policy: WorkspacePolicy):
    """Apply startup policy to empty backward caches; return resolved report.

    Explicit policy overrides import-time environment limits for these caches.
    Never call concurrently with execution. Populated caches are rejected;
    callers must explicitly perform safe cleanup before reconfiguration.
    Durable winners, Parameters and numerical policies are untouched.
    """
    if not isinstance(policy, WorkspacePolicy):
        raise TypeError('policy must be a WorkspacePolicy')
    backend, caches = _caches()
    with backend._LOCK:
        if any(len(cache) for cache in caches.values()):
            raise RuntimeError('configure_workspace requires empty caches; configure at startup or explicitly clear runtime caches first')
        for name, cache in caches.items():
            cache.max_entries = getattr(policy, name + '_entries')
            cache.max_bytes = getattr(policy, name + '_bytes')
    return workspace_report()


def workspace_report(previous=None):
    """Resolved limits, explicit unique cache storage, and interval churn.

    Pass an earlier report to diagnose a warmed interval. Insertions count
    successful runner constructions, not lookup misses (which may double-count).
    Counters are cumulative across cache cleanup. Recent keys are bounded to 16.
    """
    from .runtime import tensor_storage_bytes
    backend, caches = _caches()
    result = {'schema_version': 1, 'caches': {}, 'diagnostics': [],
              'interval': previous is not None,
              'accounting': 'explicit cached tensor storage only; excludes allocator reserve, construction peak, outputs, opaque closures and externally retained graphs'}
    owners = []
    with backend._LOCK:
        for name, cache in caches.items():
            with cache._lock:
                stats = cache.stats()
                owners.extend(cache._values.values())
                stats['recent_evicted_keys'] = list(cache.recent_evicted_keys)
            before = {} if previous is None else previous['caches'][name]
            stats['interval_counts'] = {k: stats[k] - before.get(k, 0) for k in
                ('insertions', 'evictions', 'entry_limit_evictions', 'byte_limit_evictions', 'hits', 'misses')}
            result['caches'][name] = stats
            delta = stats['interval_counts']
            if delta['evictions']:
                limits = [label for label in ('entry', 'byte') if delta[label+'_limit_evictions']]
                result['diagnostics'].append(dict(cache=name, binding_limits=limits,
                    successful_constructions=delta['insertions'], evictions=delta['evictions'],
                    recommendation='Increase the binding limit only after measuring capacity; use entries=None with an explicit byte budget to remove the entry cap. Compare warm interval reports to identify sustained churn.'))
        result['unique_cached_storage_bytes'] = tensor_storage_bytes(owners)
    return result
