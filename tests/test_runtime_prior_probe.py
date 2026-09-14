import unittest
from tools.research.runtime_prior_probe import policy_for, family_spec


class RuntimePriorProbeTests(unittest.TestCase):
    def test_family_policies_keep_distinct_numerical_contracts(self):
        self.assertEqual(family_spec('mxfp8_prequant_fwd')[-1],.07)
        self.assertEqual(family_spec('nvfp4_dynamic_fwd')[-2:],({'scaling':'block'},.20))
        self.assertEqual(family_spec('nvfp4_jit_row_region_fwd')[-2:],({'scaling':'jit_row_region'},.20))
        with self.assertRaises(ValueError):family_spec('nvfp4_delayed_fwd')

    def test_policy_comparison_and_budget_contracts(self):
        ordinary = policy_for('ordinary', 8, None)
        retrieval = policy_for('retrieval', 8, 'artifact.json.gz')
        ml = policy_for('ml', 8, 'artifact.json.gz')
        self.assertIsNone(ordinary.predictor_artifact)
        self.assertEqual(retrieval.predictor_heads, ())
        self.assertEqual(ml.predictor_heads, ('trees',))
        self.assertEqual(retrieval.evidence_contract(),
                         policy_for('retrieval', 64, 'artifact.json.gz').evidence_contract())
        self.assertNotEqual(retrieval.evidence_contract(), ml.evidence_contract())
        for mode, budget, artifact in (('unknown', 8, None), ('ml', 8, None), ('ordinary', 9, None)):
            with self.assertRaises(ValueError):
                policy_for(mode, budget, artifact)


if __name__ == '__main__':
    unittest.main()
