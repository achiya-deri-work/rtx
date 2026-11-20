"""Timing-invalid successes must not anchor the opt-in strict research search."""
import random
from types import SimpleNamespace

from rtx.autotune.outcomes import TrialOutcome
from rtx.autotune.strategies import CoordinateLocalSearch, RandomSearch
from tests.test_autotune_coverage import Config, adapter


def test_bandit_unstable_success_cannot_earn_speed_reward_or_set_resume_incumbent():
    from rtx.autotune.bandit import AdaptiveBanditScheduler,ArmStatistics
    policy=AdaptiveBanditScheduler(cost_weight=0)
    def row(score,eligible):
        return SimpleNamespace(outcome=TrialOutcome('ok',median_ms=score,metadata={
            'sampling':{'collection':{'stationary':eligible}}}),score=score,successful=True,
            metadata={},elapsed_s=.1,context_id='shape',strategy='random',
            features={},finished_at='0',session_id='s',sequence=0)
    unstable=row(.001,False);stable=row(1.,True)
    assert policy.reward(2.,unstable)<0
    stats={'random':ArmStatistics()}
    policy.initialize(stats,SimpleNamespace(observations=[unstable,stable],active_context_id='shape'),{})
    assert stats['random'].pulls==2 and stats['random'].successes==1
    assert stats['random'].last_reward>0


def history(a):
    def record(config, score, stationary):
        return SimpleNamespace(config=config, config_id=a.config_id(config), score=score,
            outcome=TrialOutcome("ok", median_ms=score, metadata={
                "sampling": {"collection": {"stationary": stationary}}}))
    stable = record(Config(), 1., True)
    unstable = record(Config(b=2), .01, False)
    return SimpleNamespace(successful=[unstable, stable],
        seen_ids={stable.config_id, unstable.config_id}), stable, unstable


def test_local_search_does_not_anchor_on_unstable_fast_sample():
    a = adapter()
    h, stable, unstable = history(a)
    for margin in (0., .01):
        search = CoordinateLocalSearch(beam_width=1, anchor_tolerance=margin,
                                       require_timing_eligible=True)
        proposal = search.propose(a, h, random.Random(0), 1)[0]
        assert proposal.parent_config_id == stable.config_id
    legacy = CoordinateLocalSearch(beam_width=1).propose(a, h, random.Random(0), 1)[0]
    assert legacy.parent_config_id == unstable.config_id


def test_random_search_prefers_only_stable_seeds(monkeypatch):
    a = adapter()
    h, stable, _ = history(a)
    seen = []
    def sample(self, rng, count, seeds):
        seen.extend(seeds)
        return [Config(a=1)]
    monkeypatch.setattr(type(a), "sample", sample)
    assert RandomSearch(require_timing_eligible=True).propose(a, h, random.Random(0), 1)
    assert seen and all(c == stable.config for c in seen)
