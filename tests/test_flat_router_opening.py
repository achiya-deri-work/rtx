"""Measure the admitted flat-routing opening, not just its seed list."""
from dataclasses import replace

import pytest

from rtx.autotune.effort import AutotunePolicy
from rtx.autotune.outcomes import TrialOutcome
from rtx.autotune.routing import make_flat_effort_autotuner, make_routing_adapter
from rtx.autotune.store import ResidualTuningStore
from rtx.routing import RoutingKernelConfig


INITIAL = RoutingKernelConfig(qb_backend='native_column', metadata_backend='cute', quota_threshold=0)


def test_named_effort_presets_use_resumable_mixed_search_without_extra_flags():
    for name,budget in (('quick',8),('balanced',16),('thorough',32)):
        p=AutotunePolicy.preset(name)
        assert p.max_trials==budget and p.search_schedule=='progressive' and p.candidate_mix=='mixed'
    legacy=AutotunePolicy.preset('quick',search_schedule='legacy',candidate_mix='legacy')
    assert legacy.search_schedule=='legacy' and legacy.candidate_mix=='legacy'
EFFORT = AutotunePolicy(max_trials=8, max_seconds=15, search_schedule='progressive',
    candidate_mix='mixed', pool_size=16, structural_proposals=24, confirmation_repeats=0)


def _engine(path, *, policy=EFFORT, n=12288, e=48, k=6, sms=170, phase='train',
            initial=INITIAL, evaluate=None, reject_geometry=False, regions=False):
    def measure(c):
        value = 1.0 if c.backward != initial.backward else 2.0
        return TrialOutcome('ok', median_ms=value, timings_ms=[value])

    adapter = make_routing_adapter(n, e, k, evaluate or measure, initial=initial,
        phase=phase, device=dict(multiprocessor_count=sms))
    if reject_geometry:
        reject = adapter.rejection_fn
        adapter.rejection_fn = lambda c: ('implementation_rejected', 'test') if (
            c.backward != initial.backward) else reject(c)
    store = ResidualTuningStore(path)
    tuner = make_flat_effort_autotuner(adapter, store, policy,region_coverage=regions)
    return tuner, lambda: list(store.records(adapter.context))


def test_flat_four_and_eight_are_actual_nested_openings(tmp_path):
    prefix = []
    checkpoint = None
    for trials in (2, 4, 8):
        tuner, rows = _engine(tmp_path/'resumed', policy=replace(EFFORT, max_trials=trials))
        tuner.tune()
        actual = rows()
        ids = [row['config_id'] for row in actual]
        assert len(ids) == len(set(ids)) == trials
        assert ids[:len(prefix)] == prefix
        if checkpoint is not None:
            assert checkpoint == tuner.checkpoint_key
        checkpoint = tuner.checkpoint_key
        prefix = ids
        assert actual[0]['strategy'] == 'initial'
        assert actual[1]['strategy'] == 'initial_neighborhood'
        assert actual[1]['config']['qb_backend'] == 'torch'
        if trials >= 4:
            hard, changed, retained = actual[1:4]
            assert changed['config'] == dict(hard['config'],
                backward=dict(threads=256, row_groups=1, column_groups=1))
            assert changed['parent_config_id'] == hard['config_id']
            assert retained['strategy'] == 'initial_neighborhood'
            assert retained['config']['qb_backend'] == 'native'
    one_shot, read = _engine(tmp_path/'one-shot')
    one_shot.tune()
    assert [r['config_id'] for r in read()] == prefix
    assert any(r['strategy'] in ('structural', 'random') for r in read()[6:])


@pytest.mark.parametrize('experts,sms,groups', [(128, 170, 2), (128, 300, 1), (32, 170, 1)])
def test_geometry_depends_on_sm_capacity(tmp_path, experts, sms, groups):
    tuner, _ = _engine(tmp_path, n=16384, e=experts, k=4, sms=sms)
    assert tuner.coverage_proposals[1].config.backward.column_groups == groups


def test_inherited_torch_hard_basin_gets_the_matched_contrast(tmp_path):
    initial = replace(INITIAL, qb_backend='torch')
    tuner, rows = _engine(tmp_path, initial=initial, policy=replace(EFFORT, max_trials=4))
    tuner.tune()
    original, first, contrast, retained = rows()
    assert first['config']['qb_backend'] != 'torch'
    assert contrast['parent_config_id'] == original['config_id']
    assert contrast['config'] == dict(original['config'],
        backward=dict(threads=256, row_groups=1, column_groups=1))
    assert retained['strategy'] == 'initial_neighborhood'


@pytest.mark.parametrize('kwargs', [
    dict(n=1024), dict(phase='forward'), dict(sms=20), dict(sms=None),
    dict(reject_geometry=True),
    dict(initial=replace(INITIAL, backward=replace(INITIAL.backward, column_groups=1))),
    dict(policy=replace(EFFORT, search_schedule='legacy')),
    dict(policy=replace(EFFORT, initial_neighborhood_trials=1)),
    dict(policy=replace(EFFORT, baseline_trials=1, exploration_trials=2)),
    dict(policy=replace(EFFORT, retrieval_trials=0)),
])
def test_no_geometry_when_not_applicable_or_reservations_are_explicit(tmp_path, kwargs):
    tuner, _ = _engine(tmp_path, **kwargs)
    assert not any(p.metadata.get('opening_family') for p in tuner.coverage_proposals)


def test_prior_opening_is_preserved(tmp_path, monkeypatch):
    from rtx.autotune.prior import FrozenPrior

    class Prior:
        def bundle(self, _):
            return True

        def retrieval(self, _, limit):
            return [replace(INITIAL, qb_backend='native_warp')]

        def order(self, _, pool, policy):
            return pool

    monkeypatch.setattr(FrozenPrior, 'load', lambda *args, **kwargs: Prior())
    tuner, rows = _engine(tmp_path, policy=replace(EFFORT, max_trials=4, prior_artifact='test-prior'))
    tuner.tune()
    assert [r['strategy'] for r in rows()] == ['initial', 'initial_neighborhood',
                                               'initial_neighborhood', 'retrieval']


def test_unstable_geometry_does_not_replace_eligible_control(tmp_path):
    def measure(c):
        changed = c.backward != INITIAL.backward
        value = 1.0 if changed else 2.0
        return TrialOutcome('ok', median_ms=value, timings_ms=[value],
                            metadata=dict(sampling=dict(collection=dict(stationary=not changed))))

    tuner, rows = _engine(tmp_path, policy=replace(EFFORT, max_trials=4), evaluate=measure)
    result = tuner.tune()
    assert rows()[2]['outcome']['median_ms'] == 1.
    assert result.config == INITIAL
    assert result.median_ms == 2.


def test_deadline_does_not_force_geometry_admission(tmp_path, monkeypatch):
    from rtx.autotune import orchestrator

    now = [10.]

    def measure(_):
        now[0] += .6
        return TrialOutcome('ok', median_ms=1., timings_ms=[1.])

    monkeypatch.setattr(orchestrator.time, 'monotonic', lambda: now[0])
    tuner, rows = _engine(tmp_path, policy=replace(EFFORT, max_seconds=1.), evaluate=measure)
    tuner.tune()
    assert [r['strategy'] for r in rows()] == ['initial', 'initial_neighborhood']


def test_region_opening_reaches_both_directions_in_eight_and_resumes(tmp_path):
    prefix=[];checkpoint=None
    for budget in (8,16,32):
        engine,read=_engine(tmp_path/'resumed',regions=True,policy=replace(EFFORT,max_trials=budget))
        engine.tune();rows=read();ids=[r['config_id'] for r in rows]
        assert len(ids)==len(set(ids))==budget
        assert ids[:len(prefix)]==prefix
        if checkpoint:assert engine.checkpoint_key==checkpoint
        checkpoint=engine.checkpoint_key;prefix=ids
        if budget==8:
            assert rows[1]['config']['forward']['column_groups']==1
            assert rows[1]['config']['backward']['column_groups']==1
            assert {r['config']['qb_backend'] for r in rows[:6]} >= {'native_column','torch','native_warp'}
            assert any(r['strategy'] in ('structural','random') for r in rows[6:8])
            assert any(r['config'].get('flat_column_shards_forward',0)>0 for r in rows[:8])
            assert any(r['config'].get('flat_row_backend')=='warp' for r in rows[:8])
        if budget==32:
            assert any(r['strategy']=='flat_transport' for r in rows[8:])
    engine,read=_engine(tmp_path/'one-shot',regions=True,policy=replace(EFFORT,max_trials=32))
    engine.tune();assert [r['config_id'] for r in read()]==prefix


@pytest.mark.parametrize('sms',[None,0,16])
def test_region_opening_preserves_missing_hardware_and_filled_wave(tmp_path,sms):
    engine,_=_engine(tmp_path,regions=True,sms=sms,e=128,n=16384,k=4)
    assert not any(p.strategy=='region_coverage' for p in engine.coverage_proposals)


def test_region_opening_protects_explicit_reservations_and_small_shapes(tmp_path):
    for index,kwargs in enumerate((dict(n=512),dict(policy=replace(EFFORT,retrieval_trials=0)),
            dict(policy=replace(EFFORT,initial_neighborhood_trials=1)))):
        engine,_=_engine(tmp_path/str(index),regions=True,**kwargs)
        assert not any(p.strategy=='region_coverage' for p in engine.coverage_proposals)


def test_late_transport_anchors_only_on_eligible_completed_prefix():
    import random
    from types import SimpleNamespace
    from rtx.autotune.routing import FlatTransportSearch

    adapter=make_routing_adapter(24576,96,4,None,initial=INITIAL,
                                 device=dict(multiprocessor_count=170))
    safe=replace(INITIAL,qb_backend='torch')
    bad=replace(INITIAL,qb_backend='native_warp')
    def observation(config,latency,eligible):
        return SimpleNamespace(config=config,config_id=adapter.config_id(config),score=latency,
            outcome=TrialOutcome('ok',median_ms=latency,
                metadata=dict(sampling=dict(collection=dict(stationary=eligible)))))
    rows=[observation(safe,2.,True),*[observation(bad,.01,False) for _ in range(7)]]
    history=SimpleNamespace(current=rows,seen_ids={r.config_id for r in rows})
    search=FlatTransportSearch()
    proposals=search.propose(adapter,history,random.Random(0),8)
    assert proposals and all(p.parent_config_id==adapter.config_id(safe) for p in proposals)
    assert all(p.metadata['anchor_prefix']==8 for p in proposals)
    # A noisy latest result must not continually move the coupled neighborhood.
    rows.append(observation(bad,.02,True))
    later=search.propose(adapter,history,random.Random(0),8)
    assert [adapter.config_id(p.config) for p in later]==[adapter.config_id(p.config) for p in proposals]
    history.current=rows[:7]
    assert search.propose(adapter,history,random.Random(0),8)==[]
    history.current=[observation(bad,.01,False)]*8
    assert search.propose(adapter,history,random.Random(0),8)==[]
