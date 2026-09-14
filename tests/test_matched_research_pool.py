import random
import unittest
from unittest.mock import patch

from rtx.autotune.core import TuningBudget
from rtx.autotune.orchestrator import AutotuneOrchestrator, SequentialScheduler
from rtx.autotune.strategies import RandomSearch, CoordinateLocalSearch
from rtx.autotune.store import InMemoryTuningStore
from tests.test_autotune_coverage import adapter, Config
from tools.research.matched_pool import MatchedPoolAdapter, make_matched_tuner, make_reference_tuner


class MatchedResearchPoolTests(unittest.TestCase):
    def test_interleaved_prefix_is_diverse_legal_and_frozen(self):
        pool = MatchedPoolAdapter(adapter(), [Config(a=i,b=j) for i in range(3) for j in range(3)])
        ordinary = pool.protected_prefix()
        diverse = pool.protected_prefix(order='structural_interleaved')
        self.assertEqual([p.coordinate for p in ordinary], ['a.tile', 'a.tile'])
        self.assertEqual([p.coordinate for p in diverse], ['a.tile', 'b.stages'])
        self.assertTrue(all(p.strategy == 'initial_neighborhood' for p in diverse))
        self.assertTrue(all(pool.rejection(p.config) is None for p in diverse))
        self.assertEqual(pool.manifest(prefix_order='structural_interleaved')['protected_prefix'],
                         [pool.config_id(p.config) for p in diverse])
        self.assertEqual(pool.protected_prefix(0, order='structural_interleaved'), ())
        with self.assertRaises(ValueError): pool.protected_prefix(order='unknown')

    def test_interleaved_prefix_resume_and_policy_separation(self):
        base = adapter()
        base.axes = {'a.tile': tuple(range(10)), 'b.stages': tuple(range(10))}
        base.rejection_fn = lambda c: None
        pool = MatchedPoolAdapter(base, [Config(a=i,b=j) for i in range(10) for j in range(10)])
        with patch('time.monotonic', return_value=100.):
            for treatment in ('ordinary', 'portfolio', 'rules', 'portfolio_ml'):
                with self.subTest(treatment=treatment):
                    full = InMemoryTuningStore()
                    kwargs = dict(treatment=treatment, prefix_order='structural_interleaved', ranked_opening=4)
                    if treatment != 'ordinary':
                        kwargs['ranking'] = list(reversed(sorted(pool._configs)))
                    make_matched_tuner(pool, full, max_trials=64, **kwargs).tune()
                    expected = [(r['strategy'], r['config_id']) for r in full.records(pool.context)]
                    first=list(full.records(pool.context))[:8]
                    self.assertEqual(sum(r['strategy']=='random' for r in first),1)
                    self.assertEqual(sum(r['strategy']=='frozen_ranking' for r in first),4)
                    self.assertEqual(sum(r['strategy']=='initial_neighborhood' for r in first),2)
                    resumed = InMemoryTuningStore()
                    for budget in (8,16,32,64):
                        make_matched_tuner(pool, resumed, max_trials=budget, **kwargs).tune()
                        self.assertEqual([(r['strategy'], r['config_id']) for r in resumed.records(pool.context)],
                                         expected[:budget])
                    with self.assertRaisesRegex(ValueError, 'incompatible'):
                        make_matched_tuner(pool, resumed, **{**kwargs, 'prefix_order':'axis_major'}).tune()
                    with self.assertRaisesRegex(ValueError, 'incompatible'):
                        make_matched_tuner(pool, resumed, **{**kwargs, 'ranked_opening':0}).tune()

    def test_ranked_opening_deduplicates_before_counting_and_limits_budget(self):
        pool=MatchedPoolAdapter(adapter(),[Config(a=i,b=j) for i in range(3) for j in range(3)])
        covered=[pool.config_id(pool.initial_config),*[pool.config_id(p.config) for p in pool.protected_prefix()]]
        ranking=covered+[cid for cid in pool._configs if cid not in covered]
        tuner=make_matched_tuner(pool,InMemoryTuningStore(),treatment='portfolio_ml',
                                 ranking=ranking,ranked_opening=4)
        proposals=tuner.coverage_proposals
        self.assertEqual(len(proposals),6)
        self.assertEqual([pool.config_id(p.config) for p in proposals[2:]],ranking[3:7])
        for count in (-1,5,True,1.5):
            with self.assertRaises(ValueError):
                make_matched_tuner(pool,InMemoryTuningStore(),treatment='ordinary',ranked_opening=count)

    def test_gpu_builder_projects_runtime_coordinates_and_deduplicates_pool(self):
        from dataclasses import replace
        from tools.research.matched_gpu import build_adapter
        from rtx.autotune.adapters import make_mxfp8_prequant_adapter
        from rtx.kernels.mxfp8 import MXFP8Problem
        original=make_mxfp8_prequant_adapter(MXFP8Problem(128,128,128),lambda c:None,device={})
        with patch('tools.research.matched_gpu._build_adapter',return_value=(original,{'test':1})):
            actual,hardware=build_adapter({}, {}, 'unused')
        self.assertNotIn('global_l2_fetch',actual.axes)
        self.assertIn('runtime_search_space_revision',actual.context.tags)
        pool=MatchedPoolAdapter(actual,[actual.initial_config,
            replace(actual.initial_config,l2_fetch_granularity=128)])
        self.assertEqual(len(pool.pool),1)
        self.assertEqual(hardware,{'test':1})

    def test_full_reference_enumerates_once_and_resumes(self):
        pool = MatchedPoolAdapter(adapter(), [Config(a=i,b=j) for i in range(3) for j in range(3)])
        store = InMemoryTuningStore()
        tuner = make_reference_tuner(pool, store)
        result = tuner.tune()
        self.assertEqual(result.evaluated_trials, 9)
        self.assertEqual({r['config_id'] for r in store.records(pool.context)}, set(pool._configs))
        self.assertEqual(make_reference_tuner(pool, store).tune().evaluated_trials, 0)

    def test_four_treatments_resume_all_efforts_with_common_opening(self):
        import tempfile
        from rtx.autotune.store import JsonlTuningStore
        base = adapter()
        base.axes = {'a.tile': tuple(range(10)), 'b.stages': tuple(range(10))}
        base.rejection_fn = lambda c: None
        pool = MatchedPoolAdapter(base, [Config(a=i, b=j) for i in range(10) for j in range(10)])
        openings = []
        with tempfile.TemporaryDirectory() as root, patch('time.monotonic', return_value=100.):
            for treatment in ('ordinary', 'portfolio', 'rules', 'portfolio_ml'):
                kwargs = dict(treatment=treatment)
                if treatment != 'ordinary':
                    # Synthetic rankings test plumbing, not learned quality.
                    kwargs['ranking'] = list(reversed(sorted(pool._configs)))
                full = JsonlTuningStore(root + '/' + treatment + '/full')
                make_matched_tuner(pool, full, max_trials=64, **kwargs).tune()
                rows = list(full.records(pool.context))
                openings.append([r['config_id'] for r in rows[:3]])
                self.assertEqual(len(rows), 64)
                self.assertTrue(any(r['strategy'] == 'frozen_ranking' for r in rows))
                resumed = JsonlTuningStore(root + '/' + treatment + '/resume')
                for budget in (8, 16, 32, 64):
                    make_matched_tuner(pool, resumed, max_trials=budget, **kwargs).tune()
                    got = list(resumed.records(pool.context))
                    self.assertGreaterEqual(sum(r['strategy'] == 'random' for r in got),
                                            {8: 1, 16: 3, 32: 6, 64: 10}[budget])
                    self.assertEqual([(r['strategy'], r['config_id']) for r in got],
                                     [(r['strategy'], r['config_id']) for r in rows[:budget]])
        self.assertTrue(all(value == openings[0] for value in openings))

    def test_matched_rankings_fail_closed(self):
        pool = MatchedPoolAdapter(adapter(), [Config(), Config(a=1), Config(a=2)])
        store = InMemoryTuningStore()
        for ranking in (None, [], [pool.config_id(Config())] * 3):
            with self.assertRaises(ValueError):
                make_matched_tuner(pool, store, treatment='portfolio_ml', ranking=ranking)
        with self.assertRaises(ValueError):
            make_matched_tuner(pool, store, treatment='ordinary', ranking=list(pool._configs))
        with self.assertRaises(ValueError):
            make_matched_tuner(pool, store, treatment='ordinary', finalist_rounds=0)

    def test_changed_ranking_cannot_resume_existing_evidence(self):
        import tempfile
        from rtx.autotune.store import JsonlTuningStore
        pool = MatchedPoolAdapter(adapter(), [Config(a=i, b=j) for i in range(3) for j in range(3)])
        with tempfile.TemporaryDirectory() as root:
            store = JsonlTuningStore(root)
            ranking = sorted(pool._configs)
            make_matched_tuner(pool, store, treatment='rules', ranking=ranking).tune()
            with self.assertRaisesRegex(ValueError, 'incompatible'):
                make_matched_tuner(pool, store, treatment='rules',
                                  ranking=list(reversed(ranking)), max_trials=16).tune()

    def test_membership_identity_and_no_evaluator_escape(self):
        from dataclasses import replace
        base=adapter(); values=[Config(),Config(a=1),Config(b=1),Config(a=1,b=1)]
        a=MatchedPoolAdapter(base,values)
        b=MatchedPoolAdapter(base,list(reversed(values))+[Config()])
        self.assertEqual(a.manifest(),b.manifest())
        self.assertNotEqual(a.context.identifier,base.context.identifier)
        self.assertNotEqual(a.context.identifier,MatchedPoolAdapter(base,values[:-1]).context.identifier)
        changed_initial=MatchedPoolAdapter(replace(base,initial_config=Config(a=1)),values)
        self.assertNotEqual(a.pool_id,changed_initial.pool_id)
        self.assertEqual(set(a.sample(random.Random(1),100)),set(values))
        self.assertTrue(all(c in values for _,_,c in a.neighbors(Config())))
        with patch.object(type(base),'evaluate',side_effect=AssertionError('escaped evaluator')):
            with self.assertRaisesRegex(ValueError,'outside frozen'):
                a.evaluate(Config(a=2))

    def test_invalid_pool_rejected(self):
        with self.assertRaisesRegex(ValueError,'initial'):
            MatchedPoolAdapter(adapter(),[Config(a=1)])
        with self.assertRaisesRegex(ValueError,'illegal'):
            MatchedPoolAdapter(adapter(),[Config(),Config(a=3)])

    def test_two_policy_arms_share_exact_prefix_and_pool(self):
        a=MatchedPoolAdapter(adapter(),[Config(a=i,b=j) for i in range(3) for j in range(3)])
        sequences=[]
        for strategy in (RandomSearch(),CoordinateLocalSearch()):
            store=InMemoryTuningStore()
            tuner=AutotuneOrchestrator(a,store,[strategy],
                SequentialScheduler(((strategy.name,None),)),TuningBudget(6,30),
                coverage_proposals=a.protected_prefix())
            tuner.tune()
            ids=[r['config_id'] for r in store.records(a.context)]
            self.assertTrue(set(ids)<=set(a._configs))
            sequences.append(ids)
        self.assertEqual(sequences[0][:3],sequences[1][:3])
        self.assertEqual(len(sequences[0]),6)
        self.assertEqual(len(sequences[1]),6)


if __name__=='__main__':unittest.main()
