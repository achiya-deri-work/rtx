"""Research-only rule fitting and semantic isolation."""
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec('sklearn'), 'research environment required')
class ConditionalHeuristicTests(unittest.TestCase):
    def test_export_parity_and_heldout_revision_guards(self):
        import json
        import numpy as np
        from tools.research.conditional_heuristics import export_rules, predict_rules, fit_rules
        from tools.research.data import feature_view
        from tools.research.portfolio_study import contract
        rows = [dict(family='test', revision=1, context='train', split='train',
                     ms=1.+i%4, features={'config.tile': float(i%4),
                         'context.workload.m':128., 'context.workload.k':256.})
                for i in range(160)]
        doc = json.loads(json.dumps(export_rules(rows), allow_nan=False))
        vectorizer, heads = fit_rules(rows)
        fresh = [{**r, 'context':'unseen', 'features':{**r['features'],
                  'context.workload.m':384.}} for r in rows[:8]]
        expected = heads[contract(fresh[0])].predict(vectorizer.transform(
            [feature_view(r, 'interactions') for r in fresh]))
        np.testing.assert_allclose(predict_rules(doc, fresh), expected, atol=1e-12, rtol=0)
        with self.assertRaisesRegex(ValueError, 'overlaps'):
            predict_rules(doc, [{**rows[0], 'context':'different-device-same-mk'}])
        with self.assertRaisesRegex(ValueError, 'semantic/revision'):
            predict_rules(doc, [{**fresh[0], 'revision':2}])
        with self.assertRaisesRegex(ValueError, 'exclusively training'):
            export_rules([*rows, {**fresh[0], 'split':'evaluation'}])
        self.assertFalse(doc['deployment_enabled'])

    def test_rules_do_not_merge_revisions(self):
        from tools.research.conditional_heuristics import fit_rules
        rows=[]
        for revision in (1,2):
            for i in range(80):
                rows.append(dict(family='test',revision=revision,context=str(revision),
                    ms=1.+(i%2), features={'config.tile':float(i%2),
                                          'context.workload.m':128.}))
        vectorizer,heads=fit_rules(rows)
        self.assertEqual(len(heads),2)
        self.assertTrue(all(model.get_depth()<=4 for model in heads.values()))
        self.assertFalse(any('latency' in n for n in vectorizer.get_feature_names_out()))


if __name__=='__main__':
    unittest.main()
