from dataclasses import replace
import gzip
import json
from pathlib import Path
import tempfile
import unittest

from tests.test_autotune_coverage import adapter,Config
from rtx.autotune.core import Proposal,evaluate_proposal,stable_id
from rtx.autotune.prior import FrozenPrior,semantic_contract,train_bundle,aggregate,prior_features
from rtx.autotune.prior_study import make_plan


class PortablePriorTests(unittest.TestCase):
    def make_prior(self):
        a=adapter();a.rejection_fn=lambda c:None
        rows=[evaluate_proposal(a,Proposal(Config(a=i),'test'),session_id='test',sequence=i) for i in range(12)]
        contract=semantic_contract(a.context.family,1,a.context.features())
        bundle=train_bundle(rows,contract)
        document=dict(schema_version=1,feature_revision=1,deployment_enabled=False,
                      bundles={stable_id(contract):bundle})
        return a,rows,document

    def test_research_gate_revision_and_permutations(self):
        a,rows,document=self.make_prior()
        with self.assertRaisesRegex(ValueError,'research-only'):FrozenPrior(document)
        prior=FrozenPrior(document,research=True);pool=[r.config for r in rows]
        for policy in ('ordered','random','retrieval','ranker','hybrid'):
            order=prior.order(a,pool,policy=policy)
            self.assertEqual(set(order),set(pool));self.assertEqual(len(order),len(pool))
            self.assertEqual(order,prior.order(a,pool,policy=policy))
        a.context=replace(a.context,kernel_revision=2);a.__post_init__()
        self.assertIsNone(prior.bundle(a))
        self.assertEqual(prior.order(a,pool),pool)

    def test_aggregation_and_feature_leakage(self):
        a,rows,_=self.make_prior()
        bad=replace(rows[0],outcome=replace(rows[0].outcome,median_ms=1e-12,
            metadata={'sampling':{'collection':{'stationary':False}}}))
        self.assertEqual(len(aggregate([*rows,*rows,bad])),len(rows))
        features=prior_features({'derived.compiled.registers':128,'context.device.uuid=x':1,
            'derived.smem_fraction':.5,'context.workload.m':128})
        self.assertNotIn('derived.compiled.registers',features)
        self.assertNotIn('context.device.uuid=x',features)
        self.assertEqual(features['derived.smem_fraction'],.5)

    def test_cache_regime_and_numerics_are_contracts(self):
        self.assertNotEqual(semantic_contract('toy',1,{'context.regime=hot':1}),
                            semantic_contract('toy',1,{'context.regime=rotate':1}))
        self.assertNotEqual(semantic_contract('toy',1,{'context.tags.scale_policy=block':1}),
                            semantic_contract('toy',1,{'context.tags.scale_policy=delayed':1}))

    def test_retrieval_does_not_initialize_models(self):
        a,rows,document=self.make_prior()
        prior=FrozenPrior(document,research=True)
        order=prior.order(a,[r.config for r in rows],policy='retrieval')
        self.assertEqual(len(order),len(rows))
        self.assertFalse(prior._models)

    def test_staged_family_policy_is_lazy_and_retains_pool(self):
        from rtx.autotune.prior_policy import PriorStage,StagedPriorPolicy,FamilyPriorPolicy
        a,rows,document=self.make_prior();prior=FrozenPrior(document,research=True)
        pool=[r.config for r in rows]
        policy=StagedPriorPolicy((PriorStage('retrieval',4),PriorStage('hybrid')))
        first=policy.order(prior,a,pool,max_candidates=4)
        self.assertFalse(prior._models)
        self.assertEqual(first,prior.order(a,pool,policy='retrieval')[:4])
        full=policy.order(prior,a,pool)
        self.assertEqual(full[:4],first);self.assertEqual(set(full),set(pool))
        routed=FamilyPriorPolicy(policy,{a.context.family:StagedPriorPolicy((PriorStage('ordered'),))})
        self.assertEqual(routed.order(prior,a,pool),pool)
        with self.assertRaisesRegex(ValueError,'duplicate'):policy.order(prior,a,[*pool,pool[0]])
        with self.assertRaises(ValueError):StagedPriorPolicy((PriorStage('retrieval',4),))
        for n in (-1,1.5,True):
            with self.assertRaises(ValueError):policy.order(prior,a,pool,max_candidates=n)
        a.context=replace(a.context,kernel_revision=999);a.__post_init__()
        self.assertEqual(policy.order(prior,a,pool,max_candidates=4),pool[:4])

    def test_prospective_summary_rejects_mismatched_pools(self):
        from rtx.autotune.prior_summary import summarize
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);a=adapter()
            for policy,pool in (('ordered','a'),('hybrid','b')):
                folder=root/'raw'/policy;folder.mkdir(parents=True)
                context=replace(a.context,tags={'prior_policy':policy}).as_dict()
                (folder/'trial.json').write_text(json.dumps(dict(context=context,
                    task={'policy':policy,'family':'coverage_test'},prior_sha256='frozen',pool_id=pool)))
            summarize([root/'raw'],root/'report')
            audit=json.loads((root/'report/audit.json').read_text())
            self.assertEqual(len(audit['pool_mismatches']),1)

    def test_plan_is_balanced_and_all_treatments_present(self):
        _,_,document=self.make_prior()
        with tempfile.TemporaryDirectory() as tmp:
            prior=Path(tmp)/'prior.json.gz'
            with gzip.open(prior,'wt') as stream:json.dump(document,stream)
            out=Path(tmp)/'plan.json';make_plan(prior,out)
            plan=json.loads(out.read_text())
            self.assertEqual(len(plan['tasks']),880)
            for start in range(0,880,5):
                tasks=plan['tasks'][start:start+5]
                self.assertEqual({t['policy'] for t in tasks},{'ordered','random','retrieval','ranker','hybrid'})
                self.assertEqual(len({t['family'] for t in tasks}),1)
            self.assertEqual(len({t['family'] for t in plan['tasks'][:55]}),11)
