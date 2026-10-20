from dataclasses import dataclass
import gc
import weakref

import pytest
import torch

from rtx.runtime import (BoundedCache, retain_runtime_resources,
                         runner_cache_byte_limit, tensor_storage_bytes)


@dataclass
class Owner:
    scratch: object
    alias: object = None


def test_storage_accounting_and_lru_budget():
    tensor = torch.empty(16)
    owner = Owner(tensor, tensor[2:])
    assert tensor_storage_bytes(owner) == 64
    cache = BoundedCache(8, max_bytes=96, sizeof=tensor_storage_bytes)
    cache['a'] = owner
    cache['b'] = Owner(torch.empty(8))
    assert cache.stats()['resident_bytes'] == 96
    assert cache['a'] is owner
    cache['c'] = Owner(torch.empty(8))
    assert list(cache) == ['a', 'c']
    cache['a'] = Owner(torch.empty(4))
    assert cache.stats()['resident_bytes'] == 48
    del cache['c']
    assert cache.stats()['resident_bytes'] == 16
    cache['huge'] = Owner(torch.empty(128))
    assert 'huge' not in cache
    assert cache.stats()['oversized'] == 1
    cache.clear()
    assert cache.stats()['resident_bytes'] == 0


def test_graph_owner_survives_byte_eviction():
    cache = BoundedCache(8, max_bytes=64, sizeof=tensor_storage_bytes)
    owner = Owner(torch.empty(16))
    ref = weakref.ref(owner)
    with retain_runtime_resources() as owners:
        cache['a'] = owner
    del owner
    cache['b'] = Owner(torch.empty(16))
    gc.collect()
    assert ref() is not None
    assert cache.stats()['resident_bytes'] == 64
    owners.clear()
    gc.collect()
    assert ref() is None


def test_budget_configuration(monkeypatch):
    monkeypatch.delenv('RTX_MXFP8_BACKWARD_CACHE_BYTES', raising=False)
    monkeypatch.setenv('RTX_MXFP8_RUNNER_CACHE_BYTES', '128')
    assert runner_cache_byte_limit('backward') == 128
    monkeypatch.setenv('RTX_MXFP8_BACKWARD_CACHE_BYTES', '0')
    assert runner_cache_byte_limit('backward') == 0
    monkeypatch.setenv('RTX_MXFP8_BACKWARD_CACHE_BYTES', '-1')
    with pytest.raises(ValueError):
        runner_cache_byte_limit('backward')
    with pytest.raises(ValueError):
        BoundedCache(2, max_bytes=100)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_live_graph_survives_scratch_cache_eviction():
    cache = BoundedCache(8, max_bytes=1024, sizeof=tensor_storage_bytes)
    x = torch.randn(256, device='cuda')
    scratch = torch.empty_like(x)
    owner = Owner(scratch)
    with retain_runtime_resources() as owners:
        cache['first'] = owner
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            torch.add(x, 1, out=scratch)
    cache['second'] = Owner(torch.empty_like(x))
    del owner
    assert cache.stats()['resident_bytes'] == 1024
    for _ in range(3):
        x.add_(.25)
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(scratch, x + 1)
    # Explicit retained graph ownership outlives eviction, not a cache leak.
    assert sum(isinstance(value, Owner) for value in owners.values()) == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_backward_byte_eviction_multishape_multistream(monkeypatch):
    import rtx.fp8_bwd as backend
    from rtx.autotune.mlp import materialized_dx_config, materialized_dw_config
    from rtx.kernels.mxfp8_bwd import DEFAULT_MXFP8_BWD_CONFIG
    config = materialized_dx_config(materialized_dw_config(DEFAULT_MXFP8_BWD_CONFIG))
    key = backend._intern_bwd_config(config)
    values = []
    for rows in (128, 256, 320):
        x = torch.randn(rows, 128, device='cuda', dtype=torch.bfloat16) * .1
        w = torch.randn(256, 128, device='cuda', dtype=torch.bfloat16) * .02
        dy = torch.randn(rows, 256, device='cuda', dtype=torch.bfloat16) * .1
        dx, dw = torch.empty_like(x), torch.empty_like(w)
        backend._launch_bwd_out(dy, x, w, key, dx, dw)
        values.append((dy, x, w, dx, dw))
    torch.cuda.synchronize()
    # This is an ownership/cache-equivalence test, not an independent numeric
    # oracle: compare to the same validated recipe with ordinary cache retention.
    cache = BoundedCache(8, max_bytes=1, sizeof=tensor_storage_bytes)
    monkeypatch.setattr(backend, '_RUNNERS', cache)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    pending = []
    for step in range(6):
        with torch.cuda.stream(streams[step % 2]):
            dy, x, w, expected_dx, expected_dw = values[step % 3]
            dx, dw = torch.empty_like(x), torch.empty_like(w)
            backend._launch_bwd_out(dy, x, w, key, dx, dw)
            pending.append((dx, dw, expected_dx, expected_dw))
        assert cache.stats()['resident_bytes'] <= 1
    torch.cuda.synchronize()
    for dx, dw, expected_dx, expected_dw in pending:
        torch.testing.assert_close(dx, expected_dx, rtol=0, atol=0)
        torch.testing.assert_close(dw, expected_dw, rtol=0, atol=0)
    assert cache.stats()['oversized'] == 6
