from dataclasses import replace
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rtx.autotune.effort import AutotunePolicy,make_effort_autotuner
from rtx.autotune.store import JsonlTuningStore
from tests.test_autotune_coverage import Config,adapter


class RankedOpeningTests(unittest.TestCase):
    def predictor(self):
        return SimpleNamespace(digest='test',models={'trees':{'coverage_test':object()}},
            bundle=lambda _:object(),retrieval=lambda *a,**kw:[Config(b=1),Config(b=2)],
            order=lambda *a,**kw:[Config(),Config(a=1),Config(b=1),Config(a=3),
                                Config(a=2,b=2),Config(a=2,b=1)])

    def policy(self,**kw):
        return AutotunePolicy(search_schedule='progressive',predictor_artifact='stub',
            predictor_heads=('trees',),allow_research_prior=True,ranked_opening_trials=2,
            confirmation_repeats=0,**kw)

    def test_eight_trial_coverage_and_reuse(self):
        a=adapter()
        with tempfile.TemporaryDirectory() as root,patch('rtx.autotune.predictors.load_predictor',return_value=self.predictor()):
            store=JsonlTuningStore(root)
            make_effort_autotuner(a,store,self.policy(max_trials=8)).tune()
            rows=list(store.records(a.context))
            self.assertEqual([r['strategy'] for r in rows],['initial']+['initial_neighborhood']*2+
                ['retrieval']*2+['portable_trees']*2+['random'])
            self.assertEqual([r['config_id'] for r in rows[5:7]],
                             [a.config_id(Config(a=2,b=2)),a.config_id(Config(a=2,b=1))])
            self.assertEqual(make_effort_autotuner(a,store,self.policy(max_trials=8)).tune().evaluated_trials,0)

    def test_resume_ladder(self):
        a=adapter();a.axes={'a.tile':tuple(range(20)),'b.stages':tuple(range(10))}
        with tempfile.TemporaryDirectory() as root,patch('rtx.autotune.predictors.load_predictor',return_value=self.predictor()),patch('time.monotonic',return_value=100.):
            full=JsonlTuningStore(root+'/full');resume=JsonlTuningStore(root+'/resume')
            make_effort_autotuner(a,full,self.policy(max_trials=64)).tune()
            expected=[(r['strategy'],r['config_id']) for r in full.records(a.context)]
            for budget in (8,16,32,64):
                make_effort_autotuner(a,resume,self.policy(max_trials=budget)).tune()
                self.assertEqual([(r['strategy'],r['config_id']) for r in resume.records(a.context)],expected[:budget])

    def test_invalid_options_and_distinct_contract(self):
        p=self.policy()
        for n in (-1,3,True):
            with self.assertRaises(ValueError):replace(p,ranked_opening_trials=n)
        for kw in ({'predictor_artifact':None},{'predictor_heads':()}, {'retrieval_trials':3}, {'search_schedule':'legacy'}):
            with self.assertRaises(ValueError):replace(p,**kw)
        self.assertNotEqual(p.evidence_contract(),replace(p,ranked_opening_trials=0).evidence_contract())


if __name__=='__main__':unittest.main()
