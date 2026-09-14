import unittest
from rtx.autotune.confirmation import confirm_finalists
from rtx.autotune.outcomes import TrialOutcome, FatalDeviceContextError


class PairedConfirmationTests(unittest.TestCase):
    def test_balanced_order_drift_and_logging(self):
        calls = []
        def evaluate(config):
            drift = 10 ** (len(calls) // 2)
            calls.append(config)
            return TrialOutcome('ok', median_ms=drift * config)
        events = []
        result = confirm_finalists(evaluate, {'slow': 2., 'fast': 1.},
                                   rounds=4, record=events.append)
        self.assertEqual(result['winner'], 'fast')
        self.assertEqual(result['evaluations'], 8)
        self.assertEqual(len(events), 8)
        self.assertEqual(calls[:2], calls[2:4][::-1])

    def test_failure_cannot_win(self):
        result = confirm_finalists(lambda c: TrialOutcome('correctness_error') if c == 1 else
            TrialOutcome('ok', median_ms=2.), {'bad': 1, 'good': 2}, rounds=2)
        self.assertEqual(result['winner'], 'good')
        self.assertEqual(result['invalid'], ['bad'])

    def test_deadline_and_fatal_are_not_silent(self):
        result = confirm_finalists(lambda c: self.fail('deadline ignored'),
                                   {'a': 1, 'b': 2}, deadline=0)
        self.assertIsNone(result['winner'])
        events = []
        with self.assertRaises(FatalDeviceContextError):
            confirm_finalists(lambda c: TrialOutcome('runtime_error', error='illegal memory access was encountered'),
                              {'a': 1, 'b': 2}, record=events.append)
        self.assertEqual(len(events), 1)

    def test_invalid_budgets(self):
        for rounds in (0, 1, 3, True):
            with self.assertRaises(ValueError):
                confirm_finalists(None, {'a': 1, 'b': 2}, rounds=rounds)

    def test_unstable_is_not_a_numerical_failure_or_a_winner(self):
        result=confirm_finalists(lambda c:TrialOutcome('ok',median_ms=c,
            metadata={'sampling':{'collection':{'stationary':c!=1}}}),
            {'unstable':1.,'stable':2.},rounds=2)
        self.assertEqual(result['winner'],'stable')
        self.assertEqual(result['invalid'],[])
        self.assertEqual(result['timing_unreliable'],['unstable'])
        self.assertEqual(result['evaluations'],4)
        from tools.research.matched_compare import summarize_pairs
        with self.assertRaisesRegex(ValueError,'nonstationary'):
            summarize_pairs(result,{'8':{'ordinary':'stable'}})
        # Older artifacts lack the summary flag: inspect their raw metadata.
        result.pop('timing_unreliable')
        with self.assertRaisesRegex(ValueError,'nonstationary'):
            summarize_pairs(result,{'8':{'ordinary':'stable'}})
