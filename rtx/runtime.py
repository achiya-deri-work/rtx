"""Lazy architecture boundaries for CuTe kernels.

Importing :mod:`rtx` is architecture neutral.  CuTe selects a target through
process-global environment variables, so kernel modules must only be imported
at the point where a concrete implementation is requested.
"""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, MutableMapping
from importlib import import_module
import os
from threading import RLock
from typing import Generic, TypeVar


KeyT = TypeVar("KeyT")
ValueT = TypeVar("ValueT")


_CUTE_TARGETS = {
    # SM121 uses the same architecture-accelerated SM120a instruction family.
    "sm120": "sm_120a",
}


class BoundedCache(MutableMapping[KeyT, ValueT], Generic[KeyT, ValueT]):
    """Small thread-safe LRU used for launchers with resident GPU workspaces.

    Python dictionaries were previously used for shape/stream-specific runners.
    Those runners retain quantized operands and scale workspaces, so a dynamic
    workload could grow GPU memory without bound. A zero-sized cache is useful
    for debugging: the value returned by the caller remains alive for that
    invocation but is not retained afterwards.
    """

    def __init__(self, max_entries: int) -> None:
        if max_entries < 0:
            raise ValueError("cache size cannot be negative")
        self.max_entries = int(max_entries)
        self._values: OrderedDict[KeyT, ValueT] = OrderedDict()
        self._lock = RLock()
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def __getitem__(self, key: KeyT) -> ValueT:
        with self._lock:
            try:
                value = self._values.pop(key)
            except KeyError:
                self.misses += 1
                raise
            self._values[key] = value
            self.hits += 1
            return value

    def get(self, key: KeyT, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key: KeyT, value: ValueT) -> None:
        with self._lock:
            self._values.pop(key, None)
            if self.max_entries == 0:
                self.evictions += 1
                return
            self._values[key] = value
            while len(self._values) > self.max_entries:
                self._values.popitem(last=False)
                self.evictions += 1

    def __delitem__(self, key: KeyT) -> None:
        with self._lock:
            del self._values[key]

    def __iter__(self) -> Iterator[KeyT]:
        with self._lock:
            return iter(tuple(self._values))

    def __len__(self) -> int:
        with self._lock:
            return len(self._values)

    def clear(self) -> None:
        with self._lock:
            self._values.clear()

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "entries": len(self._values),
                "max_entries": self.max_entries,
                "hits": self.hits,
                "misses": self.misses,
                "evictions": self.evictions,
            }


def runner_cache_limit(kind: str, default: int = 8) -> int:
    """Resolve a per-runner cache limit with one global fallback."""

    specific = f"RTX_MXFP8_{kind.upper()}_CACHE_ENTRIES"
    raw = os.getenv(specific, os.getenv("RTX_MXFP8_RUNNER_CACHE_ENTRIES", str(default)))
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{specific} must be an integer, got {raw!r}") from exc
    if value < 0:
        raise ValueError(f"{specific} cannot be negative")
    return value


def load_kernel_symbol(module: str, symbol: str, *, family: str = "sm120"):
    """Load one kernel symbol after selecting its CuTe target family."""

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


def clear_runtime_caches(*, synchronize: bool = True, device=None) -> dict[str, object]:
    """Release cached launch workspaces retained by the Python frontends.

    Synchronization is the safe default because CuTe/TVM-FFI launches are
    asynchronous. Advanced callers may disable it only after establishing
    their own stream/device completion boundary.
    """

    if synchronize:
        import torch

        if torch.cuda.is_available():
            torch.cuda.synchronize(device)
    released: dict[str, object] = {}
    for module_name in ("fp8", "fp8_bwd"):
        module = import_module(f".{module_name}", __package__)
        clear = getattr(module, "_clear_runtime_caches", None)
        if clear is not None:
            released[module_name] = clear()
    return released


__all__ = [
    "BoundedCache",
    "clear_runtime_caches",
    "load_kernel_symbol",
    "runner_cache_limit",
]
