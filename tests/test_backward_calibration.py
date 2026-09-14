from unittest.mock import patch

from rtx.autotune.core import TuningBudget
from rtx.autotune.store import InMemoryTuningStore
from tests.test_autotune_coverage import adapter,Config
from tools.research.backward_calibration import candidates,TRAIN,HELDOUT
from tools.research.matched_pool import MatchedPoolAdapter,make_reference_tuner


def test_calibration_pool_and_resume():
    base=adapter()
    base.axes={'a.tile':tuple(range(40)),'b.stages':tuple(range(40))}
    base.rejection_fn=lambda c:None
    chosen=candidates(base,81818)
    assert len(chosen)==len({base.config_id(c) for c in chosen})==128
    assert chosen==candidates(base,81818)
    pool=MatchedPoolAdapter(base,chosen)
    with patch('time.monotonic',return_value=100.):
        full=InMemoryTuningStore()
        make_reference_tuner(pool,full,seed=81818).tune()
        expected=[r['config_id'] for r in full.records(pool.context)]
        resumed=InMemoryTuningStore()
        for budget in (32,64,128):
            tuner=make_reference_tuner(pool,resumed,seed=81818)
            tuner.budget=TuningBudget(budget,300.)
            tuner.tune()
            assert [r['config_id'] for r in resumed.records(pool.context)]==expected[:budget]


def test_calibration_heldout_mk_are_separate():
    assert not ({(m,k) for m,n,k in TRAIN}&{(m,k) for m,n,k in HELDOUT})
