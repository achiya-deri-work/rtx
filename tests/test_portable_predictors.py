"""Portable ABI validation and shared-orchestrator integration (CPU)."""
import copy
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from rtx._portable_predictor import PortablePredictor, SCHEMA, FEATURE_REVISION, contract


def document():
    return dict(schema=SCHEMA, feature_revision=FEATURE_REVISION, feature_names=['config.a'],
        bundles=[dict(contract=dict(family='coverage_test', revision=1,
                                     policy={'context.regime=hot': 1.}), winners=[])],
        models={'mlp': {'coverage_test': dict(mean=[0], scale=[1],
            layers=[dict(weight=[[-1]], bias=[0])])},
            'residual': {'coverage_test': dict(baseline=0, trees=[[
                [0, 0, .5, 1, 2, 0, 0], [0, 0, 0, 0, 0, 1, 0], [-1, 0, 0, 0, 0, 1, 0]]])}})


class PortablePredictorTests(unittest.TestCase):
    def test_plain_trees_do_not_add_a_memory_baseline(self):
        d=document()
        d['models']['trees']=copy.deepcopy(d['models']['residual'])
        p=PortablePredictor(d,research=True)
        rows=[dict(family='coverage_test',revision=1,
                   features={'config.a':v,'derived.memory_roofline_ms':100.}) for v in (0,1)]
        np.testing.assert_allclose(p.predict(rows,'trees'),[0,-1])

    def test_parameters_and_opt_in(self):
        with self.assertRaises(ValueError):
            PortablePredictor(document())
        for alter in ('cycle', 'scale', 'schema'):
            d = document()
            if alter == 'cycle':
                d['models']['residual']['coverage_test']['trees'][0][0][3] = 0
            elif alter == 'scale':
                d['models']['mlp']['coverage_test']['scale'][0] = 0
            else:
                d['schema'] = 999
            with self.assertRaises(ValueError):
                PortablePredictor(d, research=True)

    def test_numeric_direction(self):
        p = PortablePredictor(document(), research=True)
        rows = [dict(family='coverage_test', revision=1, features={'config.a': v}) for v in (0, 1)]
        np.testing.assert_allclose(p.predict(rows, 'mlp'), [0, -np.log(2)], atol=1e-7)
        self.assertLess(p.predict(rows, 'residual')[1], p.predict(rows, 'residual')[0])

    def test_ucb_retrieval_budgets_and_unknown_contract(self):
        from rtx.autotune.effort import AutotunePolicy, make_effort_autotuner
        from rtx.autotune.store import InMemoryTuningStore
        from tests.test_autotune_coverage import adapter
        a, d = adapter(), document()
        d['bundles'][0]['contract'] = contract(a.context.family, a.context.kernel_revision, a.context.features())
        d['bundles'][0]['winners'] = [dict(config={'a': 1, 'b': 0}, features={})]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'prior.json.gz'
            with gzip.open(path, 'wt') as stream:
                json.dump(d, stream)
            policy = AutotunePolicy(max_trials=8, max_seconds=30, predictor_artifact=str(path),
                                    allow_research_prior=True, confirmation_repeats=0)
            store = InMemoryTuningStore()
            tuner = make_effort_autotuner(a, store, policy)
            result = tuner.tune()
            records = list(store.records(a.context))
            self.assertEqual(result.evaluated_trials, 8)
            self.assertEqual(records[1]['strategy'], 'retrieval')
            self.assertEqual(records[2]['strategy'], 'random')
            self.assertTrue({'portable_mlp', 'portable_residual'} <= {r['strategy'] for r in records})
            self.assertEqual(make_effort_autotuner(a, store, policy).tune().evaluated_trials, 0)
            from dataclasses import replace
            retrieval = make_effort_autotuner(a, InMemoryTuningStore(), replace(policy, predictor_heads=()))
            self.assertEqual(set(retrieval.strategies), {'random', 'coordinate_local'})
            self.assertEqual(len(retrieval.coverage_proposals), 1)
            none = make_effort_autotuner(a, InMemoryTuningStore(), replace(policy, retrieval_trials=0))
            self.assertEqual(len(none.coverage_proposals), 0)
        p = PortablePredictor(d, research=True)
        d2 = copy.deepcopy(d)
        d2['bundles'][0]['contract']['revision'] += 1
        self.assertIsNone(PortablePredictor(d2, research=True).bundle(a))
        d['bundles'].append(copy.deepcopy(d['bundles'][0]))
        self.assertIsNone(PortablePredictor(d, research=True).bundle(a))


if __name__ == '__main__':
    unittest.main()
