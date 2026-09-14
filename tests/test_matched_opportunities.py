import unittest
from tools.research.matched_opportunities import diagnose, ranking_diagnosis


class MatchedOpportunityTests(unittest.TestCase):
    def test_rank_quality_is_separate_from_actual_exposure(self):
        reference, observed=self.fixture()
        r=ranking_diagnosis(reference,['2','0','1'],observed,probes=(1,2))
        self.assertEqual(r['first_reference_within_5pct_rank'],2)
        self.assertIsNone(r['actual_trial_for_that_candidate'])
        self.assertAlmostEqual(r['prefix_quality'][0]['best_reference_excess'],.3)
        self.assertEqual(r['prefix_quality'][1]['best_reference_excess'],0)
        self.assertEqual(r['leading_candidates'][0]['actual_trial'],2)

    def test_failed_rank_positions_consume_budget_without_imputation(self):
        reference, observed=self.fixture()
        reference[2]['outcome']={'status':'compile_error'}
        r=ranking_diagnosis(reference,['2','0','1'],observed,probes=(1,2))
        self.assertEqual(r['prefix_quality'][0]['available_positions'],1)
        self.assertEqual(r['prefix_quality'][0]['successful_positions'],0)
        self.assertIsNone(r['prefix_quality'][0]['best_reference_excess'])
        self.assertEqual(r['prefix_quality'][1]['best_reference_excess'],0)

    def test_rank_diagnosis_rejects_partial_or_invalid_evidence(self):
        reference, observed=self.fixture()
        for ranking in (['0','1'], ['0','1','1'], ['0','1','unknown']):
            with self.assertRaises(ValueError):ranking_diagnosis(reference,ranking,observed)
        with self.assertRaises(ValueError):ranking_diagnosis(reference,['0','1','2'],observed*2)
        for count in (0,-1,True,1.5):
            with self.assertRaises(ValueError):ranking_diagnosis(reference,['0','1','2'],observed,probes=(count,))

    def fixture(self):
        reference=[dict(config_id=str(i),outcome=dict(status='ok',median_ms=t))
                   for i,t in enumerate((10.,12.,13.))]
        observed=[dict(config_id=str(i),strategy='initial_neighborhood',coordinate='same',
                       outcome=dict(status='ok')) for i in (1,2)]
        return reference,observed

    def test_coverage_and_selection_are_distinct_multiplicative_gaps(self):
        reference,observed=self.fixture()
        r=diagnose(reference,observed,'2',2)
        self.assertAlmostEqual(r['proposal_coverage_excess'],.2)
        self.assertAlmostEqual(r['selection_excess_given_proposals'],13/12-1)
        self.assertAlmostEqual((1+r['proposal_coverage_excess'])*(1+r['selection_excess_given_proposals']),1.3)
        self.assertIsNone(r['first_eligible_within_5pct_trial'])
        self.assertEqual(r['protected_coordinate_count'],1)

    def test_failed_search_cannot_supply_a_winning_proposal(self):
        reference,observed=self.fixture();observed[0]['outcome']['status']='correctness_fail'
        r=diagnose(reference,observed,'2',2)
        self.assertAlmostEqual(r['proposal_coverage_excess'],.3)
        self.assertEqual(r['unsuccessful_search_trials'],1)
        with self.assertRaises(ValueError):diagnose(reference,observed,'1',2)

    def test_missing_duplicate_and_incomplete_evidence_rejected(self):
        reference,observed=self.fixture()
        for refs,obs,budget in ((reference[:-1],observed,2),(reference+reference[:1],observed,2),
                               (reference,observed,3),(reference,[observed[0]]*2,2)):
            with self.assertRaises(ValueError):diagnose(refs,obs,'2',budget)

    def test_invalid_budgets_rejected(self):
        reference, observed = self.fixture()
        for budget in (True, False, 0, -1, 1.5, '2', None):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                diagnose(reference, observed, '2', budget)


if __name__=='__main__':unittest.main()
