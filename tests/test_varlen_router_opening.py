"""Packed opening coverage is about admitted, journaled trials, not seed order."""
from dataclasses import replace

import pytest

from rtx.autotune.effort import AutotunePolicy
from rtx.autotune.outcomes import TrialOutcome
from rtx.autotune.store import ResidualTuningStore
from rtx.autotune.varlen_routing import make_adapter, make_packed_effort_autotuner, packed_structural_seeds
from rtx.routing import RoutingKernelConfig


UNEVEN=(127,513,2001,767,4095,689)
INITIAL=RoutingKernelConfig(qb_backend='native_column',metadata_backend='cute',quota_threshold=0)
EFFORT=AutotunePolicy(max_trials=8,max_seconds=20,search_schedule='progressive',
                      candidate_mix='mixed',pool_size=12,structural_proposals=16,confirmation_repeats=0)


def _engine(path, *, lengths=UNEVEN, policy=EFFORT, initial=INITIAL, training=True, reject=None, measure=None):
    def default_measure(c):
        return TrialOutcome('ok',median_ms=1. if c.packed_column_shards_forward else 2.,timings_ms=[1.])

    adapter=make_adapter(sum(lengths),len(lengths),32,4,5,initial,measure or default_measure,
                         lengths=lengths,training=training)
    if reject is not None:
        original=adapter.rejection_fn
        adapter.rejection_fn=lambda c:reject(c) or original(c)
    store=ResidualTuningStore(path)
    tuner=make_packed_effort_autotuner(adapter,store,policy,lengths=lengths,training=training)
    # Sequence numbers restart in each continuation; the journal is append-only.
    return tuner, lambda:list(store.records(adapter.context))


def _sharded(config, cache, *, training=True):
    return (config['packed_sinkhorn_backend'],config['packed_row_backend'],
            config['packed_column_shards_forward'],config['packed_column_shards_backward'],
            config['qb_backend'],config['packed_qb_column_backend'],config['packed_column_cache_items']) == (
                'staged','warp',32,32 if training else 0,'native_warp','row_major',cache)


def test_opening_actual_prefix_and_progressive_checkpoint(tmp_path):
    path=tmp_path/'resumed'
    checkpoints=[]
    prefixes=[]
    for budget in (2,4,8,16):
        tuner,rows=_engine(path,policy=replace(EFFORT,max_trials=budget))
        tuner.tune()
        checkpoints.append(tuner.checkpoint_key)
        rows=rows()
        assert len(rows)==budget
        assert _sharded(rows[1]['config'],16) and rows[1]['outcome']['status']=='ok'
        assert rows[0]['strategy']=='initial' and rows[1]['strategy']=='structural'
        prefixes.append([r['config_id'] for r in rows])
    assert len(set(checkpoints))==1
    assert all(prefix==prefixes[-1][:len(prefix)] for prefix in prefixes)
    assert len(set(prefixes[-1]))==16
    assert [r['strategy'] for r in rows[2:4]]==['initial_neighborhood']*2
    assert rows[4]['config']['packed_sinkhorn_backend']=='persistent'
    one_shot,read=_engine(tmp_path/'one-shot',policy=replace(EFFORT,max_trials=16))
    one_shot.tune()
    assert [r['config_id'] for r in read()]==prefixes[-1]


def test_forward_only_and_legacy_keep_the_challenger_and_controls(tmp_path):
    legacy=replace(EFFORT,max_trials=4,search_schedule='legacy')
    tuner,rows=_engine(tmp_path/'legacy',policy=legacy)
    tuner.tune()
    records=rows()
    assert _sharded(records[1]['config'],16)
    assert records[2]['config']['packed_sinkhorn_backend']=='persistent'
    tuner,rows=_engine(tmp_path/'forward',training=False,policy=replace(EFFORT,max_trials=2))
    tuner.tune()
    assert _sharded(rows()[1]['config'],16,training=False)


def test_individually_short_pools_need_the_real_hybrid_bound(tmp_path):
    # The full total does not fit persistent execution; both entries exceed
    # the supported hybrid bound even though each is shorter than 512 rows.
    tuner,rows=_engine(tmp_path,lengths=(256,384),policy=replace(EFFORT,max_trials=4))
    tuner.tune()
    assert _sharded(rows()[1]['config'],4)


@pytest.mark.parametrize('lengths,settings,second',[
    ((8192,),{},'initial_neighborhood'),
    ((0,8192,0),{},'initial_neighborhood'),
    ((129,129),{},'initial_neighborhood'),
    ((128,)*9,{},'initial_neighborhood'),
    (UNEVEN,{'initial_neighborhood_trials':1},'initial_neighborhood'),
    (UNEVEN,{'baseline_trials':1,'exploration_trials':2},'initial_neighborhood'),
    (UNEVEN,{'retrieval_trials':0},'initial_neighborhood'),
])
def test_keep_short_single_pool_and_explicit_reservations(tmp_path,lengths,settings,second):
    tuner,rows=_engine(tmp_path,lengths=lengths,policy=replace(EFFORT,**settings))
    tuner.tune()
    assert rows()[1]['strategy']==second


def test_eligible_prior_retrieval_is_not_reordered(tmp_path,monkeypatch):
    from rtx.autotune.prior import FrozenPrior

    class Prior:
        def bundle(self,_):return True
        def retrieval(self,_,limit):return [replace(INITIAL,qb_backend='native_warp')]
        def order(self,_,pool,policy):return pool

    monkeypatch.setattr(FrozenPrior,'load',lambda *a,**k:Prior())
    tuner,rows=_engine(tmp_path,policy=replace(EFFORT,prior_artifact='mock-compatible',max_trials=4))
    tuner.tune()
    assert [r['strategy'] for r in rows()]==['initial','initial_neighborhood','initial_neighborhood','retrieval']


def test_rejected_or_identical_sharded_candidate_does_not_spend_a_slot(tmp_path):
    def reject(c):
        if _sharded(c.to_dict(),16):return 'implementation_rejected','test rejection'

    tuner,rows=_engine(tmp_path/'illegal',reject=reject,policy=replace(EFFORT,max_trials=4))
    tuner.tune()
    assert len(records:=rows())==len({r['config_id'] for r in records})==4
    assert not any(_sharded(r['config'],16) for r in records)
    sharded=next(packed_structural_seeds(INITIAL,max_scheduled_length=4096))
    tuner,rows=_engine(tmp_path/'already',initial=sharded,policy=replace(EFFORT,max_trials=4))
    tuner.tune()
    assert _sharded(rows()[0]['config'],16)
    assert rows()[1]['strategy']=='initial_neighborhood'
    assert sum(_sharded(row['config'],16) for row in rows())==1


def test_deadline_is_checked_at_actual_admission(tmp_path,monkeypatch):
    from rtx.autotune import orchestrator

    now=[10.]
    def measure(_):
        now[0]+=.6  # Only two trials can be admitted; the second may overrun.
        return TrialOutcome('ok',median_ms=1.,timings_ms=[1.])

    monkeypatch.setattr(orchestrator.time,'monotonic',lambda:now[0])
    tuner,rows=_engine(tmp_path,measure=measure,policy=replace(EFFORT,max_seconds=1.))
    tuner.tune()
    assert len(records:=rows())==2
    assert _sharded(records[1]['config'],16)
