import unittest
from tools.research.matched_compare import summarize_pairs


class MatchedComparisonTests(unittest.TestCase):
    def test_plan_provenance_digest_is_available_without_gpu(self):
        import tools.research.matched_compare as comparison
        from tools.predictor_study import source_digest
        self.assertIs(comparison.source_digest,source_digest)
        self.assertTrue(comparison.source_digest())

    def test_full_reference_coverage_and_failed_draws(self):
        from tools.research.matched_oracle import summarize_reference
        rows=[dict(config_id=str(i),outcome=dict(status='ok',median_ms=1.)) for i in range(16)]
        ids=[r['config_id'] for r in rows]
        summary=summarize_reference(rows,ids)
        self.assertEqual(summary['within_2pct'],16)
        self.assertEqual(summary['uniform_without_replacement'][0]['mean_regret_given_valid'],0.)
        with self.assertRaises(ValueError):summarize_reference(rows[:-1],ids)
        with self.assertRaises(ValueError):summarize_reference(rows+[rows[0]],ids)
        for row in rows[1:]:row['outcome']=dict(status='compile_error')
        summary=summarize_reference(rows,ids)
        self.assertGreater(summary['uniform_without_replacement'][0]['no_valid_probability'],.3)
        self.assertEqual(summary['uniform_without_replacement'][1]['no_valid_probability'],0.)

    def test_exact_tie_and_drift_invariant_ratio(self):
        rows=[dict(outcomes={key:dict(outcome=dict(median_ms=cost*drift))
                            for key,cost in (('base',2.),('fast',1.))})
              for drift in (1.,2.,3.,4.,5.,6.)]
        evidence=dict(status='complete',invalid=[],rounds=rows)
        choices={8:dict(ordinary='base',portfolio='fast',rules='base',portfolio_ml='fast')}
        summary=summarize_pairs(evidence,choices)
        by_name={r['treatment']:r for r in summary}
        self.assertEqual(by_name['portfolio']['ratio_to_ordinary'],.5)
        self.assertEqual(by_name['portfolio']['paired_block_bootstrap_95'],[.5,.5])
        self.assertTrue(by_name['rules']['same_config_as_ordinary'])
        self.assertEqual(by_name['rules']['paired_block_bootstrap_95'],[1.,1.])
        reference=summarize_pairs(evidence,choices,reference_id='fast')
        mapped={r['treatment']:r for r in reference}
        self.assertEqual(mapped['ordinary']['ratio_to_reference'],2.)
        self.assertEqual(mapped['portfolio']['ratio_to_reference'],1.)
        self.assertTrue(mapped['portfolio']['same_config_as_reference'])
        for bad in ({**evidence,'status':'deadline'}, {**evidence,'invalid':['fast']},
                    {**evidence,'rounds':rows[:3]}):
            with self.assertRaises(ValueError):
                summarize_pairs(bad,choices)


if __name__=='__main__':
    unittest.main()
