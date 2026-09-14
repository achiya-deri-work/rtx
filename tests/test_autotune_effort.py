from dataclasses import replace
import unittest
from unittest import mock
import rtx
from rtx.autotune.effort import AutotunePolicy,make_effort_autotuner
from rtx.autotune.store import InMemoryTuningStore
from tests.test_autotune_coverage import adapter,Config
from rtx.autotune.outcomes import TrialOutcome


class EffortTests(unittest.TestCase):
    def test_initial_neighborhood_is_fixed_and_resumable(self):
        a=adapter(); store=InMemoryTuningStore()
        p=AutotunePolicy(max_trials=4, initial_neighborhood_trials=2,
                         local_anchor_tolerance=.01, confirmation_repeats=0)
        tuner=make_effort_autotuner(a,store,p,candidates=[Config(b=2)])
        expected=[q.config for q in tuner.coverage_proposals[:2]]
        self.assertEqual(expected,[Config(a=1),Config(a=2)])
        tuner.tune()
        records=list(store.records(a.context))
        self.assertEqual([r['strategy'] for r in records[:3]],
                         ['initial','initial_neighborhood','initial_neighborhood'])
        self.assertEqual(make_effort_autotuner(a,store,p).tune().evaluated_trials,0)
        self.assertEqual(make_effort_autotuner(a,store,replace(p,max_trials=8)).tune().evaluated_trials,4)

    def test_guard_validation(self):
        for kw in ({'initial_neighborhood_trials':-1},
                   {'initial_neighborhood_trials':True},
                   {'local_anchor_tolerance':float('nan')},
                   {'local_anchor_tolerance':1},
                   {'max_trials':3,'initial_neighborhood_trials':2}):
            with self.assertRaises(ValueError): AutotunePolicy(**kw)

    def test_local_margin_keeps_initial_anchor_on_near_tie(self):
        from types import SimpleNamespace
        import random
        from rtx.autotune.strategies import CoordinateLocalSearch
        a=adapter()
        def obs(c,score):
            return SimpleNamespace(config=c,config_id=a.config_id(c),score=score)
        initial=obs(Config(),1.)
        challenger=obs(Config(b=2),.999)
        history=SimpleNamespace(successful=[initial,challenger],
                                seen_ids={initial.config_id,challenger.config_id})
        guarded=CoordinateLocalSearch(anchor_tolerance=.01)
        proposal=guarded.propose(a,history,random.Random(0),1)[0]
        self.assertEqual(proposal.parent_config_id,initial.config_id)
        challenger.score=.98
        proposal=guarded.propose(a,history,random.Random(0),1)[0]
        self.assertEqual(proposal.parent_config_id,challenger.config_id)

    def test_default_runtime_uses_shared_policy_and_environment_budget(self):
        from rtx.autotune.runtime import balanced_effort_policy
        with mock.patch.dict('os.environ', {}, clear=True):
            policy = balanced_effort_policy()
            self.assertIsInstance(policy, AutotunePolicy)
            self.assertEqual(policy.max_trials, 16)
            self.assertEqual(policy.max_seconds, 30)
        with mock.patch.dict('os.environ', {'RTX_BALANCED_AUTOTUNE_TRIALS': '8'}):
            self.assertEqual(balanced_effort_policy().max_trials, 8)

    def test_public_presets_and_validation(self):
        self.assertIs(rtx.AutotunePolicy,AutotunePolicy)
        for name,n in [('quick',8),('balanced',16),('thorough',32)]:
            self.assertEqual(AutotunePolicy.preset(name).max_trials,n)
        for kw in ({'max_trials':0},{'max_seconds':float('nan')},{'pool_size':True},
                   {'retrieval_trials':-1},{'samples':0}):
            with self.assertRaises(ValueError):AutotunePolicy(**kw)
        p=AutotunePolicy(forward_trials=8,backward_trials=32)
        self.assertEqual(p.for_pass().max_trials,8);self.assertEqual(p.for_pass(True).max_trials,32)

    def test_resume_increased_budget_and_zero_work_hit(self):
        a=adapter();store=InMemoryTuningStore()
        def run(n):return make_effort_autotuner(a,store,AutotunePolicy(max_trials=n,confirmation_repeats=0)).tune()
        self.assertEqual(run(4).evaluated_trials,4)
        self.assertEqual(run(4).evaluated_trials,0)
        self.assertEqual(run(8).evaluated_trials,4)
        rows=list(store.records(a.context));self.assertEqual(len(rows),8)
        self.assertEqual(len({r['config_id'] for r in rows}),8)

    def test_budget_not_consumed_by_warmup(self):
        a=adapter();s=InMemoryTuningStore()
        tuner=make_effort_autotuner(a,s,AutotunePolicy(max_trials=8))
        self.assertEqual(tuner.scheduler.warmup_trials,0)
        self.assertEqual(tuner.scheduler.minimum_pulls,{'random':1})
        self.assertLessEqual(len(tuner.strategies),3)

    def test_restored_illegal_winner_is_not_returned(self):
        a=adapter();store=InMemoryTuningStore()
        p=AutotunePolicy(max_trials=8,confirmation_repeats=0)
        best=make_effort_autotuner(a,store,p).tune().config
        original=a.rejection_fn
        a.rejection_fn=lambda c:('implementation_rejected','quarantine') if c==best else original(c)
        result=make_effort_autotuner(a,store,p).tune()
        self.assertNotEqual(result.config,best)

    def test_failed_attempts_consume_budget(self):
        a=adapter();store=InMemoryTuningStore()
        a.evaluator=lambda c:TrialOutcome('ok',median_ms=1.) if c==a.initial_config else TrialOutcome('compile_error',error='test compiler failure')
        result=make_effort_autotuner(a,store,AutotunePolicy(max_trials=4,confirmation_repeats=0)).tune()
        self.assertEqual(result.evaluated_trials,4)
        self.assertEqual(result.config,a.initial_config)
        self.assertEqual(sum(r['outcome']['status']=='compile_error' for r in store.records(a.context)),3)

    def test_retrieval_opening_then_ucb_and_resume(self):
        import tempfile,gzip,json
        from pathlib import Path
        from tests.test_portable_prior import PortablePriorTests
        a,rows,document=PortablePriorTests().make_prior()
        store=InMemoryTuningStore()
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'prior.json.gz'
            with gzip.open(path,'wt') as f:json.dump(document,f)
            p=AutotunePolicy(max_trials=8,prior_artifact=str(path),allow_research_prior=True,confirmation_repeats=0)
            tuner=make_effort_autotuner(a,store,p)
            self.assertEqual(tuner.scheduler.name,'adaptive_contextual_bandit')
            tuner.tune()
            records=list(store.records(a.context))
            self.assertEqual(len(records),8)
            opening=len(tuner.coverage_proposals)
            self.assertGreater(opening,0);self.assertLessEqual(opening,3)
            self.assertEqual([r['strategy'] for r in records[:opening+1]],['initial']+['retrieval']*opening)
            self.assertIn(records[opening+1]['strategy'],tuner.strategies)
            result=make_effort_autotuner(a,store,p).tune()
            self.assertEqual(result.evaluated_trials,0)


if __name__=='__main__':unittest.main()
