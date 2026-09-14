"""Deterministic search progression under identical measurement/cost evidence."""
from dataclasses import replace
import tempfile
import unittest
from unittest.mock import patch

from rtx.autotune.effort import AutotunePolicy, make_effort_autotuner
from rtx.autotune.store import JsonlTuningStore
from tests.test_autotune_coverage import adapter


class ProgressiveEffortTests(unittest.TestCase):
    def test_crash_after_durable_observation_preserves_sequence(self):
        a=adapter(); a.axes={'a.tile':tuple(range(20)), 'b.stages':tuple(range(10))}
        a.rejection_fn=lambda c: None
        policy=AutotunePolicy(max_trials=16,search_schedule='progressive',confirmation_repeats=0)
        class InterruptedStore(JsonlTuningStore):
            def record_observation(self, observation):
                super().record_observation(observation)
                if len(list(self.records(a.context)))==8:
                    raise SystemExit('simulated process exit after durable append')
        with tempfile.TemporaryDirectory() as root, patch('time.monotonic',return_value=100.):
            make_effort_autotuner(a,JsonlTuningStore(root+'/full'),policy).tune()
            with self.assertRaises(SystemExit):
                make_effort_autotuner(a,InterruptedStore(root+'/resume'),policy).tune()
            make_effort_autotuner(a,JsonlTuningStore(root+'/resume'),policy).tune()
            def sequence(path):
                return [(r['strategy'],r['config_id']) for r in JsonlTuningStore(path).records(a.context)]
            self.assertEqual(sequence(root+'/full'),sequence(root+'/resume'))

    def test_uncompleted_attempt_consumes_budget_and_is_not_reissued(self):
        a=adapter(); ordinary=a.evaluator; issued=[]
        def interrupted(config):
            issued.append(config)
            if len(issued)==4: raise SystemExit('interrupted evaluation')
            return ordinary(config)
        a.evaluator=interrupted
        policy=AutotunePolicy(max_trials=8,search_schedule='progressive',confirmation_repeats=0)
        with tempfile.TemporaryDirectory() as root:
            with self.assertRaises(SystemExit):
                make_effort_autotuner(a,JsonlTuningStore(root),policy).tune()
            lost=issued[-1]
            a.evaluator=ordinary
            result=make_effort_autotuner(a,JsonlTuningStore(root),policy).tune()
            self.assertEqual(result.evaluated_trials,4)
            rows=list(JsonlTuningStore(root).records(a.context))
            self.assertNotIn(a.config_id(lost),[r['config_id'] for r in rows])
            self.assertEqual(make_effort_autotuner(a,JsonlTuningStore(root),policy).tune().evaluated_trials,0)

    def test_prior_and_mixed_pool_resume(self):
        import gzip,json
        from pathlib import Path
        from tests.test_portable_prior import PortablePriorTests
        a,_,document=PortablePriorTests().make_prior()
        a.axes={'a.tile':tuple(range(20)), 'b.stages':tuple(range(10))}
        with tempfile.TemporaryDirectory() as root, patch('time.monotonic',return_value=100.):
            prior=Path(root)/'prior.json.gz'
            with gzip.open(prior,'wt') as stream: json.dump(document,stream)
            p=AutotunePolicy(max_trials=16,search_schedule='progressive',candidate_mix='mixed',
                prior_artifact=str(prior),allow_research_prior=True,confirmation_repeats=0)
            make_effort_autotuner(a,JsonlTuningStore(root+'/full'),p).tune()
            make_effort_autotuner(a,JsonlTuningStore(root+'/resume'),replace(p,max_trials=8)).tune()
            make_effort_autotuner(a,JsonlTuningStore(root+'/resume'),p).tune()
            def sequence(path):
                return [(r['strategy'],r['config_id']) for r in JsonlTuningStore(path).records(a.context)]
            self.assertEqual(sequence(root+'/full'),sequence(root+'/resume'))

    def test_disk_resume_matches_uninterrupted(self):
        a=adapter()
        a.axes={'a.tile':tuple(range(20)), 'b.stages':tuple(range(10))}
        a.rejection_fn=lambda c: None
        with tempfile.TemporaryDirectory() as root:
            def run(path,budget):
                # Reopen the journal and construct every strategy afresh.
                store=JsonlTuningStore(path)
                policy=AutotunePolicy(max_trials=budget,search_schedule='progressive',
                    confirmation_repeats=0,seed=1729)
                with patch('time.monotonic',return_value=100.):
                    result=make_effort_autotuner(a,store,policy).tune()
                return result,list(store.records(a.context))
            _,full=run(root+'/full',64)
            for budget in (8,16,32,64):
                _,resumed=run(root+'/resume',budget)
                self.assertEqual([(r['strategy'],r['config_id']) for r in resumed],
                                 [(r['strategy'],r['config_id']) for r in full[:budget]])
            result,_=run(root+'/resume',64)
            self.assertEqual(result.evaluated_trials,0)

    def test_policy_change_cannot_restore_foreign_scheduler_state(self):
        a=adapter()
        with tempfile.TemporaryDirectory() as root:
            store=JsonlTuningStore(root)
            policy=AutotunePolicy(max_trials=4, search_schedule='progressive',confirmation_repeats=0)
            make_effort_autotuner(a,store,policy).tune()
            with self.assertRaisesRegex(ValueError,'incompatible'):
                make_effort_autotuner(a,store,replace(policy,max_trials=8,seed=31)).tune()

    def test_failed_attempts_are_in_checkpoint_and_budget(self):
        from rtx.autotune.outcomes import TrialOutcome
        a=adapter()
        a.evaluator=lambda c: TrialOutcome('ok',median_ms=1.) if c==a.initial_config else TrialOutcome('compile_error',error='synthetic')
        with tempfile.TemporaryDirectory() as root:
            policy=AutotunePolicy(max_trials=4,search_schedule='progressive',confirmation_repeats=0)
            make_effort_autotuner(a,JsonlTuningStore(root),policy).tune()
            result=make_effort_autotuner(a,JsonlTuningStore(root),replace(policy,max_trials=8)).tune()
            self.assertEqual(result.evaluated_trials,4)
            records=list(JsonlTuningStore(root).records(a.context))
            self.assertEqual(sum(r['outcome']['status']=='compile_error' for r in records),7)
            self.assertEqual(records[-1]['metadata']['search_checkpoint']['count'],8)

    def test_initial_prefix_is_preserved_with_seed_candidates(self):
        from tests.test_autotune_coverage import Config
        a=adapter()
        with tempfile.TemporaryDirectory() as root:
            policy=AutotunePolicy(max_trials=8,search_schedule='progressive',confirmation_repeats=0)
            tuner=make_effort_autotuner(a,JsonlTuningStore(root),policy,candidates=[Config(b=2)])
            self.assertEqual([p.config for p in tuner.coverage_proposals[:2]],
                             [Config(a=1),Config(a=2)])


if __name__=='__main__': unittest.main()
