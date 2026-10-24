import copy
import tempfile
import unittest
from rtx.autotune.core import Proposal, SearchHistory, evaluate_proposal
from rtx.autotune.outcomes import TrialOutcome
from rtx.autotune.store import JsonlTuningStore
from rtx.autotune.paired_promotion import PROMOTION_REVISION, promote
from tests.test_autotune_coverage import adapter, Config


class WinnerPromotionTests(unittest.TestCase):
    def test_orchestrator_returns_confirmed_incumbent_and_reuses_decision(self):
        from collections import Counter
        from rtx.autotune.effort import AutotunePolicy,make_effort_autotuner
        a=adapter(); calls=Counter()
        def evaluate(c):
            calls[c]+=1
            latency=1/(1+c.a+c.b) if calls[c]==1 else (1. if c==Config() else .999)
            return TrialOutcome('ok',median_ms=latency)
        a.evaluator=evaluate
        p=AutotunePolicy(max_trials=4,search_schedule='progressive',
                         confirmation_repeats=0,finalist_rounds=4,finalist_margin=.01)
        with tempfile.TemporaryDirectory() as root:
            result=make_effort_autotuner(a,JsonlTuningStore(root),p).tune()
            self.assertEqual(result.config,Config())
            self.assertEqual(result.winner_decision['reason'],'incumbent_retained')
            records=list(JsonlTuningStore(root).records(a.context))
            self.assertLess(min(r['outcome']['median_ms'] for r in records),.5)
            a.evaluator=lambda c:self.fail('completed decision must not benchmark again')
            result=make_effort_autotuner(a,JsonlTuningStore(root),p).tune()
            self.assertTrue(result.winner_decision['reused'])
            self.assertEqual(result.evaluated_trials,0)

    def setup_history(self):
        a=adapter()
        observations=[evaluate_proposal(a,Proposal(c,'test'),session_id='test',sequence=i)
                      for i,c in enumerate((Config(),Config(a=1),Config(a=2)))]
        return a,SearchHistory(observations,a.context.identifier)

    def test_paired_decision_is_durable_and_does_not_rewrite_history(self):
        a,h=self.setup_history(); raw=copy.deepcopy([o.as_dict() for o in h.current])
        with tempfile.TemporaryDirectory() as root:
            d=promote(a,JsonlTuningStore(root),h,'session',rounds=4,margin=.01)
            self.assertEqual(d['config_id'],a.config_id(Config(a=2)))
            self.assertEqual(d['reason'],'paired_improvement')
            self.assertEqual([o.as_dict() for o in h.current],raw)
            a.evaluator=lambda c: self.fail('cache hit must not benchmark')
            restored=promote(a,JsonlTuningStore(root),h,'next',rounds=4,margin=.01)
            self.assertTrue(restored['reused'])

    def test_near_tie_retains_incumbent(self):
        a,h=self.setup_history()
        a.evaluator=lambda c:TrialOutcome('ok',median_ms=1. if c==Config() else .999)
        with tempfile.TemporaryDirectory() as root:
            d=promote(a,JsonlTuningStore(root),h,'s',rounds=4,margin=.01)
            self.assertEqual(d['config_id'],a.config_id(Config()))

    def test_unstable_search_seed_cannot_become_incumbent(self):
        a,h=self.setup_history()
        h.current[0].outcome.metadata={'sampling':{'collection':{'stationary':False}}}
        evaluated=[]
        def evaluate(c):
            evaluated.append(c)
            return TrialOutcome('ok',median_ms=1.)
        a.evaluator=evaluate
        with tempfile.TemporaryDirectory() as root:
            d=promote(a,JsonlTuningStore(root),h,'s',rounds=2,margin=.01)
        self.assertNotIn(Config(),evaluated)
        self.assertEqual(d['config_id'],a.config_id(Config(a=1)))

    def test_only_stable_search_candidate_survives_deadline(self):
        a,h=self.setup_history()
        for observation in h.current[:2]:
            observation.outcome.metadata={'sampling':{'collection':{'stationary':False}}}
        a.evaluator=lambda c:self.fail('one eligible candidate requires no race')
        with tempfile.TemporaryDirectory() as root:
            d=promote(a,JsonlTuningStore(root),h,'s',rounds=2,margin=.01,deadline=0.)
        self.assertEqual(d['config_id'],a.config_id(Config(a=2)))
        self.assertEqual(d['reason'],'only_candidate')

    def test_all_noisy_search_history_cannot_be_promoted(self):
        a,h=self.setup_history()
        for observation in h.current:
            observation.outcome.metadata={'sampling':{'collection':{'stationary':False}}}
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError,'no legal successful candidate'):
                promote(a,JsonlTuningStore(root),h,'s',rounds=2,margin=.01)

    def test_failed_fastest_candidate_cannot_win(self):
        a,h=self.setup_history()
        a.evaluator=lambda c:TrialOutcome('correctness_error',error='test') if c==Config(a=2) else TrialOutcome('ok',median_ms=1.)
        with tempfile.TemporaryDirectory() as root:
            d=promote(a,JsonlTuningStore(root),h,'s',rounds=4,margin=.01)
            self.assertEqual(d['config_id'],a.config_id(Config()))
            self.assertIn(a.config_id(Config(a=2)),d['confirmation']['invalid'])

    def test_deadline_cannot_promote_unconfirmed_challenger(self):
        a,h=self.setup_history()
        with tempfile.TemporaryDirectory() as root:
            d=promote(a,JsonlTuningStore(root),h,'s',rounds=4,margin=.01,deadline=0.)
            self.assertEqual(d['config_id'],a.config_id(Config()))
            retried=promote(a,JsonlTuningStore(root),h,'s2',rounds=4,margin=.01)
            self.assertFalse(retried['reused'])
            self.assertEqual(retried['reason'],'paired_improvement')

    def test_all_invalid_is_fatal(self):
        a,h=self.setup_history()
        a.evaluator=lambda c:TrialOutcome('correctness_error',error='test')
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaisesRegex(RuntimeError,'all winner finalists'):
                promote(a,JsonlTuningStore(root),h,'s',rounds=4,margin=.01)

    def test_nonstationary_timing_cannot_replace_incumbent(self):
        for unstable in (Config(), Config(a=2)):
            a,h=self.setup_history()
            a.evaluator=lambda c:TrialOutcome('ok',median_ms=1. if c==Config() else .5,
                metadata={'sampling':{'collection':{'stationary':c!=unstable}}})
            # Keep the other challenger slower than the incumbent.
            evaluate=a.evaluator
            a.evaluator=lambda c:TrialOutcome('ok',median_ms=2.) if c==Config(a=1) else evaluate(c)
            with tempfile.TemporaryDirectory() as root:
                d=promote(a,JsonlTuningStore(root),h,'s',rounds=2,margin=.01)
                self.assertEqual(d['config_id'],a.config_id(Config()))
                self.assertEqual(d['confirmation']['invalid'],[])
                self.assertIn(a.config_id(unstable),d['confirmation']['timing_unreliable'])

    def test_order_reversal_cannot_promote_even_if_median_improves(self):
        from collections import Counter
        a,h=self.setup_history(); calls=Counter()
        def evaluate(c):
            calls[c]+=1
            latency=1. if c==Config() else (2. if c==Config(a=1) else (.5 if calls[c]==1 else 1.1))
            return TrialOutcome('ok',median_ms=latency)
        a.evaluator=evaluate
        with tempfile.TemporaryDirectory() as root:
            d=promote(a,JsonlTuningStore(root),h,'s',rounds=2,margin=.01)
            self.assertEqual(d['config_id'],a.config_id(Config()))

    def test_old_confirmation_contract_is_not_reused(self):
        from rtx.autotune.core import stable_id
        a,h=self.setup_history()
        with tempfile.TemporaryDirectory() as root:
            store=JsonlTuningStore(root)
            d=promote(a,store,h,'s',rounds=2,margin=.01)
            old=dict(d,contract=stable_id(dict(schema=1,rounds=4,margin=.01,seed=0)))
            old.pop('promotion_revision')
            store.record_event('old','winner_decision',old)
            fresh=promote(a,store,h,'s2',rounds=4,margin=.01)
            self.assertFalse(fresh['reused'])
            self.assertEqual(fresh['promotion_revision'],PROMOTION_REVISION)


if __name__=='__main__': unittest.main()
