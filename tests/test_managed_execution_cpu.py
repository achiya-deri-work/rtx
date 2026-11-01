"""Automatic retention policy without a CUDA runtime."""
from concurrent.futures import ThreadPoolExecutor
import weakref

import pytest

from rtx._managed import in_tuning, tuning_resources
from rtx.execution import (
    ExecutionResources, clear_automatic_resources, current_owner,
)
from rtx.runtime import BoundedCache


class AutomaticOwner(ExecutionResources):
    _rtx_automatic = True


@pytest.fixture(autouse=True)
def isolated_owners(monkeypatch):
    # Do not clear GPU working sets from unrelated tests in a shared process.
    import rtx.execution as execution
    monkeypatch.setattr(execution, '_owners', weakref.WeakValueDictionary())
    monkeypatch.setattr(execution, '_active', {})
    monkeypatch.setattr(execution, '_suspended', {})
    monkeypatch.setattr(execution, '_automatic_slots', weakref.WeakValueDictionary())


def test_tuning_suspends_parent_retention_and_restores_after_failure():
    cache = BoundedCache(1)
    owner = AutomaticOwner()
    first, second = object(), object()

    @tuning_resources
    def trial():
        assert in_tuning()
        assert current_owner() is None
        with ThreadPoolExecutor(1) as executor:
            assert executor.submit(current_owner).result() is None
        cache['trial'] = second
        raise ValueError('trial failed')

    with owner.activate():
        cache['model'] = first
        with pytest.raises(ValueError, match='trial failed'):
            trial()
        assert not in_tuning()
        assert current_owner() is owner
        assert cache['model'] is first
    assert len(owner._runners) == 1
    assert all(value is not second for value in owner._runners.values())


def test_cleanup_preserves_active_and_explicit_owners():
    cache = BoundedCache(1)
    automatic, explicit = AutomaticOwner(), ExecutionResources()
    with automatic.activate():
        cache['automatic'] = object()
        clear_automatic_resources()
        assert automatic._runners
    with explicit.activate():
        cache['explicit'] = object()
        clear_automatic_resources()
        assert explicit._runners
        assert not automatic._runners


def test_nested_tuning_scope_preserves_state():
    @tuning_resources
    def inner():
        assert in_tuning()

    @tuning_resources
    def outer():
        inner()
        assert in_tuning()

    outer()
    assert not in_tuning()
