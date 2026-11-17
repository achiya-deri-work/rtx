"""One-shot, shape-local transition from real-input tuning to a fixed plan.

The first compiled invocation can use the existing opaque selection operator.
That operator binds a configuration here after tuning its actual inputs. A
normal Dynamo guard on ``SelectionCell.config_key`` then invalidates only the
cold specialization: the next call traces the fixed, compiler-visible path.

The ensure function must NOT return the cell. ``assume_constant_result`` would
otherwise freeze the cell's attributes and suppress the transition guard. The
ordinary dictionary lookup in ``bound_config`` gives Dynamo a real source for
the attribute guard. There are no tensors or executable/scratch owners here.
"""
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock

import torch


@dataclass
class SelectionCell:
    config_key: str | None = None


_MAX_CELLS = 256
_CELLS: dict[tuple, SelectionCell] = {}
_RECENCY: OrderedDict[tuple, None] = OrderedDict()
_LOCK = RLock()


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
    # the generated hot graph; its guard reads _CELLS[identity].config_key.
    with _LOCK:
        if identity not in _CELLS:
            _CELLS[identity] = SelectionCell()
        _RECENCY[identity] = None
        _RECENCY.move_to_end(identity)
        while len(_CELLS) > _MAX_CELLS:
            expired, _ = _RECENCY.popitem(last=False)
            _CELLS.pop(expired, None)
    return 0


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
            _ensure_cell(identity)
            return _CELLS[identity].config_key
    _ensure_cell(identity)
    return _CELLS[identity].config_key


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
        _ensure_cell(identity)
        _CELLS[identity] = SelectionCell(config_key)


def clear_selection_cells():
    """Drop bounded host metadata; no saved forward state or GPU allocation."""
    with _LOCK:
        count = len(_CELLS)
        _CELLS.clear()
        _RECENCY.clear()
        return count
