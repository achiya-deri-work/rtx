"""Lazy architecture boundaries for CuTe kernels.

Importing :mod:`rtx` is architecture neutral.  CuTe selects a target through
process-global environment variables, so kernel modules must only be imported
at the point where a concrete implementation is requested.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, MutableMapping
from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as package_version
from functools import lru_cache
import os
from threading import RLock, local
from contextlib import contextmanager
from typing import Generic, TypeVar


KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")

_resource_scope = local()
_resource_scope_lock = RLock()
_active_resource_scopes = {}


@contextmanager
def retain_runtime_resources():
    """Pin cached resources accessed while building a CUDA graph.

    Graph nodes retain addresses, not Python runner ownership. Cache eviction
    must not free the scratch allocations behind those addresses.
    """
    previous = getattr(_resource_scope, "owners", None)
    owners = {}
    _resource_scope.owners = owners
    with _resource_scope_lock:
        _active_resource_scopes[id(owners)] = owners
    # A captured allocation also depends on its private allocator pool lifetime.
    from ._workspace import retain_scratch_pools
    try:
        retain_scratch_pools(owners)
        yield owners
    finally:
        with _resource_scope_lock:
            _active_resource_scopes.pop(id(owners), None)
        _resource_scope.owners = previous
        if previous is not None:
            previous.update(owners)


def _retain_resource(value):
    owners = getattr(_resource_scope, "owners", None)
    if owners is not None:
        owners[id(value)] = value
    elif _active_resource_scopes:
        # Autograd launches can run on a different Python thread from the
        # caller-owned capture scope. Pin into every active scope rather than
        # losing those resources at the first cache eviction. Concurrent scopes
        # conservatively retain each other's worker resources until release.
        with _resource_scope_lock:
            for target in _active_resource_scopes.values():
                target[id(value)] = value


_CUTE_TARGETS = {
    # SM121 uses the same architecture-accelerated SM120a instruction family.
    "sm120": "sm_120a",
}

_SUPPORTED_TORCH = frozenset({"2.12.1", "2.13.0"})
_SUPPORTED_CUDA = "13.2"
_REQUIRED_PACKAGES = {
    "torchao": "0.18.0",
    "nvidia-cutlass-dsl": "4.7.0",
    "apache-tvm-ffi": "0.1.13.post2",
}
_REQUIRED_MAJOR_PACKAGES = {
    "cuda-python": 13,
    "numpy": 2,
}
_MINIMUM_PACKAGES = {
    "pyarrow": (16, 0),
    "einops": (0, 8),
}


def _numeric_version(value: str) -> tuple[int, ...]:
    fields: list[int] = []
    for field in value.split("."):
        digits = "".join(character for character in field if character.isdigit())
        if not digits:
            break
        fields.append(int(digits))
    return tuple(fields)


def _public_version(value: str) -> str:
    """Drop a PEP 440 local build tag without hiding the installed identity."""

    return value.split("+", 1)[0]


@lru_cache(maxsize=1)
def validate_runtime_environment() -> dict[str, str]:
    """Validate the deliberately narrow native-kernel software contract.

    Importing :mod:`rtx` remains safe on CPU-only hosts. Validation happens
    when callers explicitly request it or immediately before the first CuTe
    kernel module is loaded.
    """

    import torch

    torch_version = str(torch.__version__).split("+", 1)[0]
    cuda_version = str(torch.version.cuda)
    errors: list[str] = []
    if torch_version not in _SUPPORTED_TORCH:
        errors.append(
            f"PyTorch {torch_version} is unsupported; expected exactly "
            "2.12.1 or 2.13.0"
        )
    if cuda_version != _SUPPORTED_CUDA:
        errors.append(
            f"PyTorch CUDA {cuda_version} is unsupported; expected CUDA 13.2"
        )

    resolved = {
        "torch": str(torch.__version__),
        "cuda": cuda_version,
    }
    for distribution, expected in _REQUIRED_PACKAGES.items():
        try:
            installed = package_version(distribution)
        except PackageNotFoundError:
            errors.append(f"required distribution {distribution} is not installed")
            continue
        resolved[distribution] = installed
        if _public_version(installed) != expected:
            errors.append(
                f"{distribution} {installed} is unsupported; expected {expected}"
            )
    for distribution, expected_major in _REQUIRED_MAJOR_PACKAGES.items():
        try:
            installed = package_version(distribution)
        except PackageNotFoundError:
            errors.append(f"required distribution {distribution} is not installed")
            continue
        resolved[distribution] = installed
        parsed = _numeric_version(installed)
        if not parsed or parsed[0] != expected_major:
            errors.append(
                f"{distribution} {installed} is unsupported; expected major "
                f"version {expected_major}"
            )
    for distribution, minimum in _MINIMUM_PACKAGES.items():
        try:
            installed = package_version(distribution)
        except PackageNotFoundError:
            errors.append(f"required distribution {distribution} is not installed")
            continue
        resolved[distribution] = installed
        if _numeric_version(installed) < minimum:
            errors.append(
                f"{distribution} {installed} is unsupported; expected at least "
                + ".".join(str(value) for value in minimum)
            )
    if errors:
        raise RuntimeError(
            "rtx-blackwell native runtime contract is not satisfied:\n- "
            + "\n- ".join(errors)
        )
    return resolved


class BoundedCache(MutableMapping[KeyT, ValueT], Generic[KeyT, ValueT]):
    """Small thread-safe LRU used for launchers with resident GPU workspaces.

    Python dictionaries were previously used for shape/stream-specific runners.
    Those runners retain quantized operands and scale workspaces, so a dynamic
    workload could grow GPU memory without bound. A zero-sized cache is useful
    for debugging: the value returned by the caller remains alive for that
    invocation but is not retained afterwards.
    """

    def __init__(self, max_entries: int | None, *, max_bytes: int | None = None, sizeof=None) -> None:
        if max_entries is not None and (type(max_entries) is not int or max_entries < 0):
            raise ValueError("cache size cannot be negative")
        if max_entries is None and max_bytes is None:
            raise ValueError("a cache requires an entry or byte limit")
        self.max_entries = max_entries
        if max_bytes is not None and (type(max_bytes) is not int or max_bytes < 0):
            raise ValueError("max_bytes must be a nonnegative integer or None")
        if max_bytes is not None and sizeof is None:
            raise ValueError("a byte budget requires an explicit sizeof callback")
        self.max_bytes = max_bytes
        self._sizeof = sizeof
        self._sizes = {}
        self.resident_bytes = 0
        self.peak_resident_bytes = 0
        self.oversized = 0
        self._values: OrderedDict[KeyT, ValueT] = OrderedDict()
        self._lock = RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.insertions = 0
        self.entry_limit_evictions = 0
        self.byte_limit_evictions = 0
        self.recent_evicted_keys = []

    def __getitem__(self, key: KeyT) -> ValueT:
        with self._lock:
            try:
                value = self._values.pop(key)
            except KeyError:
                self.misses += 1
                raise
            self._values[key] = value
            self.hits += 1
            _retain_resource(value)
            return value

    def get(self, key: KeyT, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key: KeyT, value: ValueT) -> None:
        size = 0 if self._sizeof is None else self._sizeof(value)
        if type(size) is not int or size < 0:
            raise ValueError("sizeof must return a nonnegative integer")
        _retain_resource(value)
        with self._lock:
            self.insertions += 1
            self._values.pop(key, None)
            self.resident_bytes -= self._sizes.pop(key, 0)
            oversized = self.max_bytes is not None and size > self.max_bytes
            if self.max_entries == 0 or oversized:
                self.evictions += 1
                self.oversized += int(oversized)
                self.entry_limit_evictions += int(self.max_entries == 0)
                self.byte_limit_evictions += int(oversized)
                self.recent_evicted_keys = (self.recent_evicted_keys + [repr(key)])[-16:]
                return
            self._values[key] = value
            self._sizes[key] = size
            self.resident_bytes += size
            while ((self.max_entries is not None and len(self._values) > self.max_entries) or
                   (self.max_bytes is not None and self.resident_bytes > self.max_bytes)):
                self.entry_limit_evictions += int(self.max_entries is not None and len(self._values) > self.max_entries)
                self.byte_limit_evictions += int(self.max_bytes is not None and self.resident_bytes > self.max_bytes)
                old_key, _ = self._values.popitem(last=False)
                self.recent_evicted_keys = (self.recent_evicted_keys + [repr(old_key)])[-16:]
                self.resident_bytes -= self._sizes.pop(old_key)
                self.evictions += 1
            self.peak_resident_bytes = max(self.peak_resident_bytes, self.resident_bytes)

    def __delitem__(self, key: KeyT) -> None:
        with self._lock:
            del self._values[key]
            self.resident_bytes -= self._sizes.pop(key)

    def __iter__(self) -> Iterator[KeyT]:
        with self._lock:
            return iter(tuple(self._values))

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()
            self._sizes.clear()
            self.resident_bytes = 0

    def stats(self) -> dict[str, int | None]:
        with self._lock:
            return {
                "entries": len(self._values),
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
                "insertions": self.insertions,
                "entry_limit_evictions": self.entry_limit_evictions,
                "byte_limit_evictions": self.byte_limit_evictions,
                "resident_bytes": self.resident_bytes,
                "peak_resident_bytes": self.peak_resident_bytes,
                "max_bytes": self.max_bytes,
                "oversized": self.oversized,
                "bytes_accounted": self._sizeof is not None,
            }


def tensor_storage_bytes(value) -> int:
    """Count tensor storage in explicit dataclass/container scratch owners.

    Aliased views count once per owner. Opaque executables/closures are not
    traversed: callers must use explicit owners, not infer arbitrary reachability.
    This measures owned storage, not allocator reserve or graph-pool memory.
    """
    from dataclasses import fields, is_dataclass
    import torch
    seen, storages = set(), {}
    def visit(item):
        if id(item) in seen:
            return
        seen.add(id(item))
        if isinstance(item, torch.Tensor):
            storage = item.untyped_storage()
            storages[(str(item.device), storage.data_ptr())] = storage.nbytes()
        elif is_dataclass(item) and not isinstance(item, type):
            for field in fields(item):
                visit(getattr(item, field.name))
        elif isinstance(item, dict):
            for child in item.values():
                visit(child)
        elif isinstance(item, (tuple, list)):
            for child in item:
                visit(child)
    visit(value)
    return sum(storages.values())


def runner_cache_byte_limit(kind: str, *, namespace: str = "MXFP8") -> int | None:
    """Optional per-cache byte cap; unset retains the legacy entry limit."""
    name = f"RTX_{namespace}_{kind.upper()}_CACHE_BYTES"
    raw = os.getenv(name, os.getenv(f"RTX_{namespace}_RUNNER_CACHE_BYTES"))
    if raw is None:
        return None
    try:
        size = int(raw)
    except ValueError as exc:
        raise ValueError(f"{name} must be a nonnegative integer") from exc
    if size < 0:
        raise ValueError(f"{name} cannot be negative")
    return size


def runner_cache_limit(
    kind: str, default: int = 8, *, namespace: str = "MXFP8"
) -> int:
    """Resolve a per-runner cache limit with one global fallback."""

    specific = f"RTX_{namespace}_{kind.upper()}_CACHE_ENTRIES"
    raw = os.getenv(
        specific,
        os.getenv(f"RTX_{namespace}_RUNNER_CACHE_ENTRIES", str(default)),
    )
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{specific} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{specific} cannot be negative")
    return value


def load_kernel_symbol(module: str, symbol: str, *, family: str = "sm120"):
    """Load one kernel symbol after selecting its CuTe target family."""

    validate_runtime_environment()
    try:
        target = _CUTE_TARGETS[family]
    except KeyError as exc:
        raise RuntimeError(
            f"RTX has no executable CuTe kernel family {family!r}; "
            "supported kernels require the SM120/SM121 instruction family"
        ) from exc
    requested = os.environ.get("CUTE_DSL_ARCH")
    if requested not in (None, target):
        raise RuntimeError(
            f"this process already selected CuTe architecture {requested}; "
            f"the requested {family} kernels require {target}"
        )
    os.environ.setdefault("CUTE_DSL_ARCH", target)
    os.environ.setdefault("QUACK_ARCH", target)
    return getattr(import_module(f".kernels.{module}", __package__), symbol)


def clear_runtime_caches(*, synchronize: bool = True, device=None,
                         preserve_compiled: bool = False) -> dict[str, object]:
    """Release cached launch workspaces retained by the Python frontends.

    Synchronization is the safe default because CuTe/TVM-FFI launches are
    asynchronous. Advanced callers may disable it only after establishing
    their own stream/device completion boundary. Cleanup is process-wide, not
    filtered by ``device``; synchronize every visible device before eviction.
    ``preserve_compiled`` retains cheap selections and compiled MLP helpers.
    """

    if synchronize:
        import torch

        if torch.cuda.is_available():
            for index in range(torch.cuda.device_count()):
                torch.cuda.synchronize(index)
    released: dict[str, object] = {}
    for module_name in ("fp8", "fp8_bwd", "fp4", "glu", "mlp"):
        module = import_module(f".{module_name}", __package__)
        clear = getattr(module, "_clear_runtime_caches", None)
        if clear is not None:
            released[module_name] = clear(preserve_compiled=preserve_compiled)
    from ._workspace import clear_scratch_pools

    clear_scratch_pools()
    return released


def runtime_cache_stats() -> dict[str, dict[str, object]]:
    """Read loaded frontend LRU counters without importing kernels or clearing.

    Unaccounted caches report bytes_accounted=False; their zero byte counter is
    not evidence that their contents own no GPU memory. Graph pins and allocator
    pools are outside these per-cache snapshots.
    """
    import sys
    result = {}
    for name in ('fp8', 'fp8_bwd', 'fp4', 'glu', 'mlp', '_mlp_autotune'):
        module = sys.modules.get(f'rtx.{name}')
        if module is not None:
            for key, value in tuple(vars(module).items()):
                if isinstance(value, BoundedCache):
                    result[f'{name}.{key}'] = value.stats()
    return result


__all__ = [
    "BoundedCache",
    "clear_runtime_caches",
    "load_kernel_symbol",
    "runner_cache_limit",
    "runner_cache_byte_limit",
    "runtime_cache_stats",
    "validate_runtime_environment",
]
