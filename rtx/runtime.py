"""Lazy architecture boundaries for CuTe kernels.

Importing :mod:`rtx` is architecture neutral.  CuTe selects a target through
process-global environment variables, so kernel modules must only be imported
at the point where a concrete implementation is requested.
"""

from __future__ import annotations

from importlib import import_module
import os


_CUTE_TARGETS = {
    # SM121 uses the same architecture-accelerated SM120a instruction family.
    "sm120": "sm_120a",
}


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


__all__ = ["load_kernel_symbol"]
