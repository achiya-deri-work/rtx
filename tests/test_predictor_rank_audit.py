import unittest
from types import SimpleNamespace
from rtx.autotune.predictors import PortablePredictorSearch
from rtx.autotune.core import SearchHistory
from tests.test_autotune_coverage import adapter, Config


class RankAuditTests(unittest.TestCase):
    def test_orchestrator_persists_the_ranking(self):
        from rtx.autotune.orchestrator import AutotuneOrchestrator, SequentialScheduler
        from rtx.autotune.core import TuningBudget
        from rtx.autotune.store import InMemoryTuningStore
        a=adapter()
        configs=[Config(1,0),Config(2,0)]
        strategy=PortablePredictorSearch(SimpleNamespace(digest='test',order=lambda *_:configs),configs,'trees')
        store=InMemoryTuningStore()
        AutotuneOrchestrator(a,store,[strategy],SequentialScheduler(((strategy.name,None),)),
                             TuningBudget(3,10)).tune()
        events=[e for e in store.events if e['kind']=='strategy_candidate_ranking']
        self.assertEqual(len(events),1)
        self.assertEqual(events[0]['ordered_config_ids'],[a.config_id(c) for c in configs])

    def test_full_order_is_available_once_and_selected_rank_is_preserved(self):
        a=adapter()
        configs=[Config(1,0),Config(2,0)]
        p=SimpleNamespace(digest='test',order=lambda *_:configs)
        strategy=PortablePredictorSearch(p,configs,'residual')
        proposals=strategy.propose(a,SearchHistory(),None,1)
        self.assertEqual(proposals[0].metadata['pool_rank'],0)
        self.assertEqual(strategy.take_audit()['ordered_config_ids'],[a.config_id(c) for c in configs])
        self.assertIsNone(strategy.take_audit())
