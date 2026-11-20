"""Frozen packed recompute experiment: exact anchor and legal matched seeds."""

from dataclasses import replace

import pytest

from benchmarks.qualify_moe_iteration6_recompute import competition
from rtx._routing_varlen import workspace_bounds
from rtx.autotune.effort import AutotunePolicy
from rtx.autotune.moe import make_moe_adapter, make_moe_tuner
from rtx.autotune.store import InMemoryTuningStore


@pytest.mark.parametrize('d,h', [(384, 192), (512, 256)])
def test_populated_joint_recompute_has_exact_matched_seeds(d, h):
    n, pools, e, k = 16*1024, 16, 128, 6
    shape = (n, d, e, workspace_bounds(n, pools, e, k)[1], k, h)
    tags = {'batch_contract': {'layout': 'independent_varlen', 'sequences': pools,
                               'segment_alignment': 32}}
    policy = AutotunePolicy(max_trials=8, max_seconds=60, search_schedule='progressive',
        candidate_mix='mixed', confirmation_repeats=0, pool_size=24, structural_proposals=24)
    adapter = make_moe_adapter(shape, None, tags=tags)
    tuner = make_moe_tuner(adapter, InMemoryTuningStore(), policy)
    base, = (p.config for p in tuner.coverage_proposals if
             p.config.prepared.forward_row_layout == 'compact' and p.config.prepared.bounded_gather
             and p.config.prepared.bounded_xt and p.config.prepared.bounded_dy)
    wide, narrow = competition(shape, base), competition(shape, base, narrow_only=True)
    assert list(wide) == ['incumbent', 'dual_wide', 'reuse_wide']
    assert list(narrow) == ['incumbent', 'reuse_narrow']
    assert wide['incumbent'] == narrow['incumbent'] == base
    for label, candidate in (wide | narrow).items():
        assert adapter.rejection(candidate) is None
        p, ref = candidate.prepared, base.prepared
        assert replace(candidate, prepared=replace(p, recompute=ref.recompute,
                   joint_recompute_engine=ref.joint_recompute_engine)) == base
        assert replace(p.recompute, tile_n=ref.recompute.tile_n, stages=ref.recompute.stages) == ref.recompute
        assert p.recompute.tile_n == (128 if label.endswith('wide') else 64)
        assert p.recompute.stages == (2 if label == 'dual_wide' else 1)
    with pytest.raises(ValueError, match='exact improved packed incumbent'):
        competition(shape, replace(base, prepared=replace(base.prepared, bounded_xt=False)))
