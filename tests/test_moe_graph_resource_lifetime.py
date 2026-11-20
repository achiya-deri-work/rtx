"""Focused retirement tests: a real CUDA graph and its RTX owners are distinct."""

import gc
import json
from threading import Event, Thread
import weakref

import pytest
import torch

import rtx
from rtx import _resources
from rtx.runtime import retire_runtime_contexts, tensor_storage_bytes


def test_scratch_pool_tagging_does_not_need_cyclic_gc():
    from rtx._workspace import _tag_pool_owners

    class Pool:
        pass

    enabled = gc.isenabled()
    gc.disable()  # Ensure only refcounting can retire the otherwise-unused pool.
    try:
        pool = Pool()
        reference = weakref.ref(pool)
        _tag_pool_owners(None, pool)
        del pool
        assert reference() is None
    finally:
        if enabled:
            gc.enable()


def test_failed_completion_keeps_pending_pins(monkeypatch):
    class Owner:
        pass

    owner = Owner()
    reference = weakref.ref(owner)
    pins = {id(owner): owner}
    del owner
    synchronize_calls = []

    def synchronize(device):
        synchronize_calls.append(device)
        if len(synchronize_calls) == 1:
            raise RuntimeError("completion not established")

    monkeypatch.setattr(_resources, "_captures", {1: object()})
    monkeypatch.setattr(_resources, "_pending_owners", {})
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    _resources._release_graph(pins, 0)
    assert not synchronize_calls and reference() is not None
    _resources._captures.clear()
    _resources._drain_graph_retirements()
    assert synchronize_calls == [0] and reference() is not None
    _resources._drain_graph_retirements()
    assert synchronize_calls == [0, 0] and reference() is None
    assert not _resources._pending_owners


def test_barrier_does_not_hold_registry_mutex(monkeypatch):
    entered = Event()
    queued = Event()
    syncs = []

    def synchronize(device):
        syncs.append(device)
        if device == 0:
            entered.set()
            assert queued.wait(2), "another thread could not enqueue during device barrier"

    monkeypatch.setattr(_resources, "_pending_owners", {})
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.cuda, "synchronize", synchronize)
    a, b = {0: object()}, {1: object()}
    worker = Thread(target=_resources._release_graph, args=(a, 0))
    worker.start()
    try:
        assert entered.wait(2)
        _resources._release_graph(b, 1)
        assert a and b
    finally:
        queued.set()
        worker.join(2)
    assert not worker.is_alive()
    assert syncs == [0, 1]
    assert not a and not b


@pytest.fixture
def small_step():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.manual_seed(7607)
    layer = rtx.MXFP8Linear(128, 128, device="cuda", autotune=False, backend="materialized")
    x = torch.randn((4, 128), device="cuda", dtype=torch.bfloat16) * .1

    @torch.no_grad()
    def step():
        return layer(x)

    step()  # Install graph tracking before capture.
    return step, x


def _tiny_graph(step, stream):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        expected = step().clone()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            value = step()
    assert graph._rtx_resource_owners
    assert tensor_storage_bytes(graph._rtx_resource_owners) > 0
    return graph, value, expected


def _assert_settled():
    assert not _resources._captures
    assert not _resources._capture_leases
    assert not _resources._pending_owners


@pytest.mark.parametrize("drop_kind", ("graph", "pins"))
def test_two_live_tiny_graphs_wait_for_changed_stream_replay(small_step, monkeypatch, drop_kind):
    step, x = small_step
    a_stream, b_stream, c_stream, late_stream = (torch.cuda.Stream() for _ in range(4))
    first, first_value, first_expected = _tiny_graph(step, a_stream)
    second, second_value, second_expected = _tiny_graph(step, b_stream)
    first_ref = weakref.ref(first)
    pin = weakref.ref(next(iter(first._rtx_resource_owners.values())).qx)
    retire_runtime_contexts(streams=[a_stream, b_stream], device=x.device)
    holder = [first]
    del first

    complete = torch.cuda.Event()
    with torch.cuda.stream(late_stream):
        # Visible in-flight work preceding replay on a stream other than the
        # capture stream. Raw begin/end for C omit Torch's entry-time sync.
        torch.cuda._sleep(210_000_000)
        holder[0].replay()
        complete.record()
    assert not complete.query()
    if drop_kind == "pins":
        # Separate pin retirement from native graph destruction (which itself
        # may block). Move the only RTX ownership away from the live graph;
        # it will not replay again before reset.
        detached = dict(holder[0]._rtx_resource_owners)
        holder[0]._rtx_resource_owners.clear()
    real_synchronize = torch.cuda.synchronize
    observations = []

    def synchronize(device=None):
        observations.append(dict(device=str(device), pins_alive=pin() is not None,
                                 captured=torch.cuda.is_current_stream_capturing(),
                                 replay_pending=not complete.query()))
        return real_synchronize(device)

    capture = torch.cuda.CUDAGraph()
    with monkeypatch.context() as mp:
        mp.setattr(torch.cuda, "synchronize", synchronize)
        with torch.cuda.stream(c_stream):
            capture.capture_begin()
            try:
                c_value = x.square()
                if drop_kind == "graph":
                    holder.clear()
                else:
                    _resources._release_graph(detached, x.device.index)
                    assert detached
                assert first_ref() is not None and pin() is not None
                assert not observations
            finally:
                capture.capture_end()
    assert complete.query()
    assert pin() is None
    assert len(observations) == 1
    assert observations[0]["device"] == "0"
    assert observations[0]["pins_alive"] and not observations[0]["captured"]
    if drop_kind == "graph":
        assert first_ref() is None
    else:
        assert observations[0]["replay_pending"]
        assert not detached
        holder.pop().reset()
    torch.testing.assert_close(first_value, first_expected, rtol=0, atol=0)
    with torch.cuda.stream(late_stream):
        capture.replay()
        second.replay()
    late_stream.synchronize()
    torch.testing.assert_close(c_value, x.square(), rtol=0, atol=0)
    torch.testing.assert_close(second_value, second_expected, rtol=0, atol=0)
    second.reset()
    capture.reset()
    _assert_settled()
    print("ASYNC_RETIREMENT", drop_kind, json.dumps(observations))


def test_failed_capture_end_keeps_leases_until_real_boundary(small_step):
    step, x = small_step
    a_stream, b_stream, unrelated_stream = (torch.cuda.Stream() for _ in range(3))
    first, _, _ = _tiny_graph(step, a_stream)
    first_ref = weakref.ref(first)
    holder = [first]
    del first
    b_stream.wait_stream(torch.cuda.current_stream())
    second = torch.cuda.CUDAGraph()
    with torch.cuda.stream(b_stream):
        second.capture_begin()
        try:
            out = x.square()
            with torch.cuda.stream(unrelated_stream):
                holder.clear()
                with pytest.raises(RuntimeError, match="same stream"):
                    second.capture_end()
            assert first_ref() is not None
            assert torch.cuda.is_current_stream_capturing()
        finally:
            second.capture_end()
        second.replay()
    assert first_ref() is None
    b_stream.synchronize()
    torch.testing.assert_close(out, x.square(), rtol=0, atol=0)
    second.reset()
    _assert_settled()


def test_exception_in_capture_body_releases_lease(small_step):
    step, x = small_step
    a_stream, b_stream = torch.cuda.Stream(), torch.cuda.Stream()
    first, _, _ = _tiny_graph(step, a_stream)
    reference = weakref.ref(first)
    holder = [first]
    del first
    b_stream.wait_stream(torch.cuda.current_stream())
    second = torch.cuda.CUDAGraph()
    with pytest.raises(ValueError, match="intentional body exception"):
        with torch.cuda.graph(second, stream=b_stream):
            out = x.square()
            with pytest.raises(RuntimeError, match="outside active CUDA graph capture"):
                holder[0].reset()
            holder.clear()
            assert reference() is not None
            raise ValueError("intentional body exception")
    with torch.cuda.stream(b_stream):
        second.replay()
    b_stream.synchronize()
    torch.testing.assert_close(out, x.square(), rtol=0, atol=0)
    assert reference() is None
    second.reset()
    _assert_settled()


def test_default_gc_cyclic_graph_drops_plateau_and_teardown(small_step):
    step, x = small_step
    if not gc.isenabled() or not gc.get_threshold()[0]:
        pytest.skip("default-enabled cyclic GC required")
    a_stream, b_stream = torch.cuda.Stream(), torch.cuda.Stream()
    warm, value, expected = _tiny_graph(step, a_stream)
    warm.reset()
    del warm, value, expected
    b_stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.synchronize()
    baseline_live = torch.cuda.memory_allocated()
    live_registry = len(_resources._graphs)
    rows, refs = [], []

    for i in range(24):
        first, value, expected = _tiny_graph(step, a_stream)
        ref = weakref.ref(first)
        refs.append(ref)
        holder = [first, value]
        # The graph participates in the cycle; merely leasing the graph
        # doesn't allow the rest of the cycle to be collected.
        first._test_cycle = holder
        del first, value, expected
        second = torch.cuda.CUDAGraph()
        rounds_before = _resources._collection_counts()
        with torch.cuda.stream(b_stream):
            second.capture_begin()
            try:
                out = x.square()
                holder.append(holder)
                del holder
                # Trigger ordinary generation-0 collection in the capture
                # body, without changing thresholds or calling gc.collect().
                pressure = [[] for _ in range(2 * gc.get_threshold()[0] + 100)]
                del pressure
                assert ref() is not None
                assert _resources._collection_counts() != rounds_before
            finally:
                second.capture_end()
        second.reset()
        del second, out
        torch.cuda.synchronize()
        _assert_settled()
        rows.append(dict(iteration=i, live_bytes=torch.cuda.memory_allocated(),
                         live_dropped_graphs=sum(r() is not None for r in refs),
                         pending_owner_groups=sum(map(len, _resources._pending_owners.values()))))
    print("DEFAULT_GC_RETIREMENT", json.dumps(rows))
    assert not any(row["live_dropped_graphs"] for row in rows[-8:])
    assert {row["live_bytes"] for row in rows[-8:]} == {baseline_live}
    assert len(_resources._graphs) == live_registry
