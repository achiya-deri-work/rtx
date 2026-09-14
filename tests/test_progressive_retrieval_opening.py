"""The retrieval quota counts novel legal candidates, not overlapping ranks."""
from types import SimpleNamespace
from unittest.mock import patch
import unittest

from rtx.autotune.effort import AutotunePolicy, make_effort_autotuner
from rtx.autotune.store import InMemoryTuningStore
from tests.test_autotune_coverage import Config, adapter


class RetrievalOpeningTests(unittest.TestCase):
    def test_overlap_does_not_consume_retrieval_quota(self):
        a = adapter()
        retrieved = [Config(), Config(a=1), Config(a=1), Config(a=3),
                     Config(a=2), Config(b=1), Config(b=2)]
        prior = SimpleNamespace(bundle=lambda _: object(),
                                retrieval=lambda *args, **kwargs: retrieved,
                                order=lambda adapter, pool, **kwargs: pool)
        for budget in (8, 16, 32, 64):
            with self.subTest(budget=budget), patch(
                    'rtx.autotune.prior.FrozenPrior.load', return_value=prior):
                tuner = make_effort_autotuner(a, InMemoryTuningStore(),
                    AutotunePolicy(max_trials=budget, search_schedule='progressive',
                                   prior_artifact='test-frozen-prior', allow_research_prior=True))
                coverage = tuner.coverage_proposals
                self.assertEqual([p.strategy for p in coverage],
                                 ['initial_neighborhood'] * 2 + ['retrieval'] * 2)
                self.assertEqual([p.config for p in coverage],
                                 [Config(a=1), Config(a=2), Config(b=1), Config(b=2)])

    def test_both_novel_candidates_are_evaluated_inside_eight_trials(self):
        a = adapter()
        retrieved = [Config(a=1), Config(a=2), Config(b=1), Config(b=2)]
        prior = SimpleNamespace(bundle=lambda _: object(),
                                retrieval=lambda *args, **kwargs: retrieved,
                                order=lambda adapter, pool, **kwargs: pool)
        store = InMemoryTuningStore()
        with patch('rtx.autotune.prior.FrozenPrior.load', return_value=prior):
            make_effort_autotuner(a, store, AutotunePolicy(max_trials=8,
                search_schedule='progressive', prior_artifact='test-frozen-prior',
                allow_research_prior=True, confirmation_repeats=0)).tune()
        rows = list(store.records(a.context))
        self.assertEqual(len(rows), 8)
        self.assertEqual([row['strategy'] for row in rows[:5]],
                         ['initial', 'initial_neighborhood', 'initial_neighborhood',
                          'retrieval', 'retrieval'])
        self.assertEqual([row['config_id'] for row in rows[3:5]],
                         [a.config_id(Config(b=1)), a.config_id(Config(b=2))])

    def test_progressive_evidence_does_not_match_pre_fix_contract(self):
        from dataclasses import asdict
        from rtx.autotune.core import stable_id
        policy = AutotunePolicy(search_schedule='progressive')
        values = asdict(policy)
        for name in ('max_trials', 'max_seconds', 'forward_trials', 'backward_trials',
                     'cache_behavior', 'finalist_rounds', 'finalist_margin',
                     'finalist_reserve_fraction'):
            values.pop(name)
        self.assertNotEqual(policy.evidence_contract(), stable_id(dict(schema=1, policy=values)))


if __name__ == '__main__':
    unittest.main()
