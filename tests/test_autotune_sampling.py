"""Mixed first-hit pools: legality, quota isolation, identity and resume."""
import unittest
from unittest.mock import patch
from rtx.autotune.sampling import mixed_pool
from rtx.autotune.effort import AutotunePolicy, make_effort_autotuner
from rtx.autotune.store import InMemoryTuningStore
from tests.test_autotune_coverage import adapter, Config


class MixedSamplingTests(unittest.TestCase):
    def test_full_explicit_pool_skips_only_work_that_cannot_enter_pool(self):
        a=adapter()
        seeds=[Config(2,2),Config(1,2)]
        with patch('rtx.autotune.sampling.raw_portfolio',side_effect=AssertionError('unused broad pool')):
            with patch.object(type(a),'sample',side_effect=AssertionError('unused local pool')):
                pool,audit=mixed_pool(a,limit=3,seeds=seeds,retrieved=[Config(0,1)],skip_discarded_generation=True)
        self.assertEqual(pool,[a.initial_config,*seeds])
        self.assertEqual(audit['counts'],{'seed':3})
        self.assertEqual(audit['generation_skipped'],'explicit_pool_full')

    def test_legacy_pool_audit_stays_compatible_without_explicit_opt_in(self):
        a=adapter()
        pool,audit=mixed_pool(a,limit=3,seeds=[Config(2,2),Config(1,2)])
        self.assertEqual(len(pool),3)
        self.assertNotIn('generation_skipped',audit)
        self.assertTrue(audit['structural_statuses'])

    def test_optimized_diversity_distance_is_exact_for_flattened_configs(self):
        import random
        from rtx._sampling import _signature,_signature_distance,distance,flatten
        rng=random.Random(17)
        values=[flatten({str(i):rng.choice([None,1,2,'x',True])
                         for i in range(30) if rng.random()<.7}) for _ in range(20)]
        for a in values:
            for b in values:
                self.assertEqual(distance(a,b),_signature_distance(_signature(a),_signature(b)))

    def test_explicit_seeds_are_not_lost_to_pool_fill(self):
        a = adapter()
        seeds = [Config(2, 2), Config(1, 2)]
        pool, audit = mixed_pool(a, limit=4, seeds=seeds,
                                retrieved=[Config(0, 1), Config(0, 2)])
        self.assertEqual(pool[:3], [a.initial_config, *seeds])
        self.assertEqual(audit['counts']['seed'], 3)

    def test_baseline_reservation_is_bounded_and_resumable(self):
        a = adapter()
        store = InMemoryTuningStore()
        policy = AutotunePolicy(candidate_mix='mixed', max_trials=8,
                                baseline_trials=2, confirmation_repeats=0)
        tuner = make_effort_autotuner(a, store, policy)
        self.assertEqual(tuner.scheduler.minimum_pulls,
                         {'structural': 1, 'coordinate_local': 2})
        result = tuner.tune()
        self.assertEqual(result.evaluated_trials, 8)
        self.assertGreaterEqual(result.strategy_trials['coordinate_local'], 2)
        self.assertEqual(make_effort_autotuner(a, store, policy).tune().evaluated_trials, 0)
        for value in (-1, True, 8):
            with self.assertRaises(ValueError):
                AutotunePolicy(max_trials=8, baseline_trials=value)

    def test_retrieval_cannot_starve_structural(self):
        a = adapter()
        retrieval = [Config(i, j) for i in range(4) for j in range(3)] * 30
        pool, audit = mixed_pool(a, limit=8, retrieved=retrieval)
        self.assertLessEqual(audit['counts']['retrieval'], 2)
        self.assertGreater(audit['counts']['structural'], 0)
        self.assertEqual(pool[0], a.initial_config)
        self.assertEqual(len(pool), len({a.config_id(c) for c in pool}))
        self.assertTrue(all(a.rejection(c) is None for c in pool))
        self.assertEqual((pool, audit), mixed_pool(a, limit=8, retrieved=retrieval))

    def test_opt_in_regions_survive_a_seed_saturated_pool(self):
        a=adapter();protected=[Config(2,2),Config(1,2)]
        seeds=[Config(i,j) for i in range(3) for j in range(3)]
        pool,audit=mixed_pool(a,limit=5,seeds=seeds,protected=protected,retrieved=[Config(0,1)])
        self.assertEqual(pool[:3],[a.initial_config,*protected])
        self.assertIn(Config(0,1),pool)
        self.assertEqual(audit['counts']['region'],2)

    def test_active_identity_and_small_pool(self):
        a = adapter()
        a.config_id_fn = lambda c: str(c.a)
        pool, _ = mixed_pool(a, limit=8)
        self.assertEqual(len(pool), len({c.a for c in pool}))
        self.assertEqual(mixed_pool(a, limit=1)[0], [a.initial_config])

    def test_resume_mixed_and_reserved_exploration(self):
        a = adapter(); store = InMemoryTuningStore()
        def tuner(n):
            return make_effort_autotuner(a, store, AutotunePolicy(
                candidate_mix='mixed', max_trials=n, confirmation_repeats=0))
        t = tuner(4)
        self.assertEqual(t.scheduler.minimum_pulls, {'structural': 1})
        self.assertEqual(t.tune().evaluated_trials, 4)
        self.assertEqual(tuner(4).tune().evaluated_trials, 0)
        self.assertEqual(tuner(8).tune().evaluated_trials, 4)
        self.assertIn('structural', {r['strategy'] for r in store.records(a.context)})

    def test_invalid_policy(self):
        for kwargs in ({'candidate_mix':'typo'}, {'structural_proposals':0},
                       {'structural_proposals':True}):
            with self.assertRaises(ValueError):
                AutotunePolicy(**kwargs)

    def test_concurrent_residual_refinement(self):
        from concurrent.futures import ThreadPoolExecutor
        import tempfile
        from rtx.autotune.store import ResidualTuningStore
        with tempfile.TemporaryDirectory() as root:
            def run():
                return make_effort_autotuner(adapter(), ResidualTuningStore(root),
                    AutotunePolicy(max_trials=4, confirmation_repeats=0)).tune()
            with ThreadPoolExecutor(2) as executor:
                results = list(executor.map(lambda _: run(), range(2)))
            self.assertEqual(sorted(r.evaluated_trials for r in results), [0, 4])
            self.assertEqual(len(list(ResidualTuningStore(root).records(adapter().context))), 4)


if __name__ == '__main__':
    unittest.main()
