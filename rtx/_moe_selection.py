"""One-shot, shape-local transition from real-input tuning to a fixed plan.

The first compiled invocation can use the existing opaque selection operator.
That operator binds a configuration here after tuning its actual inputs. A
normal Dynamo guard on ``SelectionCell.config_key`` then invalidates only the
cold specialization: the next call traces the fixed, compiler-visible path.

The ensure function must NOT return the cell. ``assume_constant_result`` would
otherwise freeze the cell's attributes and suppress the transition guard. The
ordinary global-object attribute lookup in ``bound_config`` gives Dynamo a
real source for the guard, including multiple cold policies in one trace.
There are no tensors or executable/scratch owners here.
"""
from collections import OrderedDict
from dataclasses import dataclass
import hashlib
from threading import RLock

import torch


@dataclass
class SelectionCell:
    config_key: str | None = None


_MAX_CELLS = 256
_CELLS: dict[tuple, SelectionCell] = {}
_RECENCY: OrderedDict[tuple, None] = OrderedDict()
_LOCK = RLock()


class SelectionSources:
    """Per-context attribute sources, not a Dynamo-snapshotted dictionary."""
    pass


_SOURCES=SelectionSources()
_ATTRS={}


def _identity(shape, device, spec, training):
    shape = tuple(shape)
    if not shape or any(type(size) is not int or size <= 0 for size in shape):
        raise ValueError('MoE selection binding requires concrete positive dimensions')
    if not isinstance(spec, str) or type(training) is not bool:
        raise TypeError('MoE selection policy must be a string and training a bool')
    return shape, str(device), spec, training


@torch.compiler.assume_constant_result
def _ensure_cell(identity):
    # Cold/eager metadata only. No mutable dictionaries or locks survive in
    # the generated hot graph; its guard reads one real-source cell attribute.
    with _LOCK:
        if identity not in _CELLS:
            _CELLS[identity] = SelectionCell()
        attribute=_ATTRS.get(identity)
        if attribute is None:
            attribute='_cell_'+hashlib.sha256(repr(identity).encode()).hexdigest()
            _ATTRS[identity]=attribute
        setattr(_SOURCES,attribute,_CELLS[identity])
        _RECENCY[identity] = None
        _RECENCY.move_to_end(identity)
        while len(_CELLS) > _MAX_CELLS:
            expired, _ = _RECENCY.popitem(last=False)
            _CELLS.pop(expired, None)
            name=_ATTRS.pop(expired,None)
            if name is not None and hasattr(_SOURCES,name):delattr(_SOURCES,name)
    return attribute


def bound_config(shape, device, spec, training):
    """Return the bound string or None, with a shape-local Dynamo guard.

    Shapes are concrete specialization metadata, not tensor values. This
    function must remain traceable; do not decorate it as an assumed constant.
    An expired cell intentionally requires a new cold specialization. Existing
    forwards/backwards retain their own selected configuration independently.
    """
    identity = _identity(shape, device, spec, training)
    if not torch.compiler.is_compiling():
        # Eager readers must not race another thread's bounded eviction.
        # Dynamo specializes this branch away; its hot path has no lock.
        with _LOCK:
            attribute=_ensure_cell(identity)
            return getattr(_SOURCES,attribute).config_key
    attribute=_ensure_cell(identity)
    # Independent attribute sources let several cold policies coexist in one
    # trace. Reading a shared dictionary here would snapshot its keys before
    # subsequent assumed-constant initializers create the remaining cells.
    return getattr(_SOURCES,attribute).config_key


def bind_selected(shape, device, spec, training, config_key):
    """Bind once after real-input selection; never retarget an existing cell.

    The cold forward still owns its exact immutable execution-plan token.
    Another cold caller/refined winner cannot replace a previously bound hot
    plan; clearing/evicting cells only affects subsequent dispatch decisions.
    """
    if not isinstance(config_key, str) or not config_key:
        raise TypeError('Selected MoE configuration must be a nonempty string')
    identity = _identity(shape, device, spec, training)
    with _LOCK:
        _ensure_cell(identity)
        cell = _CELLS[identity]
        if cell.config_key is None:
            cell.config_key = config_key


def publish_selected(shape, device, spec, training, config_key):
    """Publish an explicitly refined policy for subsequent compiled forwards.

    Unlike first-hit binding, an explicit tuning request intentionally updates
    this context. Replace the cell so its normal Dynamo guard invalidates just
    that specialization. Existing forwards own their original static key or
    cold execution-plan token and never consult this replacement in backward.
    """
    if not isinstance(config_key, str) or not config_key:
        raise TypeError('Selected MoE configuration must be a nonempty string')
    identity = _identity(shape, device, spec, training)
    with _LOCK:
        attribute=_ensure_cell(identity)
        _CELLS[identity] = SelectionCell(config_key)
        setattr(_SOURCES,attribute,_CELLS[identity])


def clear_selection_cells(*,namespace=None):
    """Drop host metadata, optionally limited to a policy-string namespace.

    Existing default behavior clears every cell. Routing uses a distinct
    spec prefix so retiring its choices need not invalidate bound MoE plans.
    Saved forward state and GPU allocations are never owned by these cells.
    """
    if namespace is not None and not isinstance(namespace,str):
        raise TypeError('Selection namespace must be a string or None')
    with _LOCK:
        selected=list(_CELLS) if namespace is None else [
            identity for identity in _CELLS if identity[2].startswith(namespace)]
        for identity in selected:
            _CELLS.pop(identity,None);_RECENCY.pop(identity,None)
            name=_ATTRS.pop(identity,None)
            if name is not None and hasattr(_SOURCES,name):delattr(_SOURCES,name)
        return len(selected)
