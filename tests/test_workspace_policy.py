import pytest
import torch

import rtx
from rtx.runtime import BoundedCache, tensor_storage_bytes


def test_byte_only_cache_and_binding_counters():
    cache = BoundedCache(None, max_bytes=128, sizeof=tensor_storage_bytes)
    for i in range(12):
        cache[i] = torch.empty(2)
    assert len(cache) == 12
    assert cache.stats()['insertions'] == 12
    cache[0]
    assert cache.stats()['insertions'] == 12
    cache[12] = torch.empty(16)
    assert cache.stats()['byte_limit_evictions'] > 0
    assert cache.stats()['entry_limit_evictions'] == 0
    assert cache.stats()['resident_bytes'] <= 128


def test_workspace_policy_resolves_limits_and_unique_storage(monkeypatch):
    import rtx.fp8_bwd as b
    paired = BoundedCache(8, sizeof=tensor_storage_bytes)
    partial = BoundedCache(8, sizeof=tensor_storage_bytes)
    monkeypatch.setattr(b, '_RUNNERS', paired)
    monkeypatch.setattr(b, '_MATMUL_RUNNERS', partial)
    report = rtx.configure_workspace(rtx.WorkspacePolicy(256, 256, partial_entries=16))
    assert report['caches']['paired']['max_entries'] is None
    assert report['caches']['partial']['max_entries'] == 16
    tensor = torch.empty(16)
    paired[1] = tensor
    partial[1] = tensor
    report = rtx.workspace_report(report)
    assert report['unique_cached_storage_bytes'] == 64
    assert report['caches']['partial']['interval_counts']['insertions'] == 1
    with pytest.raises(RuntimeError, match='empty caches'):
        rtx.configure_workspace(rtx.WorkspacePolicy(0, 0))


@pytest.mark.parametrize('kwargs', [dict(paired_bytes=-1, partial_bytes=1),
    dict(paired_bytes=1, partial_bytes=True), dict(paired_bytes=1, partial_bytes=1, partial_entries=-1)])
def test_invalid_policy(kwargs):
    with pytest.raises(ValueError):
        rtx.WorkspacePolicy(**kwargs)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('entries', [8, 16, None])
def test_five_shape_ten_partial_working_set(monkeypatch, entries, record_property):
    import rtx.fp8_bwd as b
    from rtx.kernels.mxfp8_bwd import DEFAULT_MXFP8_BWD_CONFIG
    key = b._intern_bwd_config(DEFAULT_MXFP8_BWD_CONFIG)
    paired = BoundedCache(8, sizeof=tensor_storage_bytes)
    partial = BoundedCache(8, sizeof=tensor_storage_bytes)
    monkeypatch.setattr(b, '_RUNNERS', paired)
    monkeypatch.setattr(b, '_MATMUL_RUNNERS', partial)
    rtx.configure_workspace(rtx.WorkspacePolicy(64<<20, 64<<20, partial_entries=entries))
    torch.manual_seed(721)
    values = [(torch.randn(m, 256, device='cuda', dtype=torch.bfloat16)*.1,
               torch.randn(m, 128, device='cuda', dtype=torch.bfloat16)*.1,
               torch.randn(256, 128, device='cuda', dtype=torch.bfloat16)*.02)
              for m in (128, 256, 384, 512, 640)]
    def run():
        return [(b._launch_dx(dy, x, w, key), b._launch_dw(dy, x, w, key))
                for dy, x, w in values]
    expected = run()
    torch.cuda.synchronize()
    before = rtx.workspace_report()
    for _ in range(3):
        actual = run()
        torch.cuda.synchronize()
        for pair, reference in zip(actual, expected):
            for a, ref in zip(pair, reference):
                torch.testing.assert_close(a, ref, rtol=0, atol=0)
    report = rtx.workspace_report(before)
    delta = report['caches']['partial']['interval_counts']
    record_property('workspace_report', str(report))
    assert report['caches']['partial']['resident_bytes'] <= 64<<20
    if entries == 8:
        assert delta['insertions'] == 30 and delta['entry_limit_evictions'] == 30
        assert report['diagnostics'][0]['binding_limits'] == ['entry']
    else:
        assert delta['insertions'] == delta['evictions'] == 0
        assert report['caches']['partial']['entries'] == 10
