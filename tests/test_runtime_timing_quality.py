import unittest
from rtx.autotune.measurement import runtime_timing_metadata
from rtx.autotune.confirmation import confirm_finalists
from rtx.autotune.outcomes import TrialOutcome


class RuntimeTimingQualityTests(unittest.TestCase):
    def test_matches_existing_limits_and_preserves_raw_samples(self):
        for values,expected in (([1.,1.,1.],True),([1.,1.,1.4],False),
                                ([1.],False),([],False),([1.,float('nan'),1.],False)):
            result=runtime_timing_metadata(values)
            self.assertIs(result['sampling']['collection']['stationary'],expected)
            self.assertEqual(result['sampling']['actual_samples'],len(values))

    def test_runtime_unstable_fast_candidate_is_not_confirmed(self):
        def evaluate(config):
            values=[.5,.5,.9] if config=='unstable' else [1.,1.,1.]
            return TrialOutcome('ok',median_ms=values[0],timings_ms=values,
                                metadata=runtime_timing_metadata(values))
        result=confirm_finalists(evaluate,{'unstable':'unstable','stable':'stable'},rounds=2)
        self.assertEqual(result['winner'],'stable')
        self.assertEqual(result['timing_unreliable'],['unstable'])
        self.assertEqual(result['invalid'],[])


if __name__=='__main__':unittest.main()
