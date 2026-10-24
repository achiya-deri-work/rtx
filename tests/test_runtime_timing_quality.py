import unittest
from rtx.autotune.measurement import runtime_timing_metadata
from rtx.autotune.confirmation import confirm_finalists
from rtx.autotune.outcomes import TrialOutcome


class RuntimeTimingQualityTests(unittest.TestCase):
    def test_batch_calibration_records_actual_work_and_excludes_pilots(self):
        from contextlib import nullcontext
        from unittest.mock import patch
        from rtx.autotune.measurement import measure_cuda_batches
        work=[0]
        class Event:
            def __init__(self, **kwargs): pass
            def record(self): self.position=work[0]
            def synchronize(self): pass
            def elapsed_time(self, other): return (other.position-self.position)*.01
        def call(): work[0]+=1
        with patch('torch.cuda.Event',Event), patch('torch.cuda.device',lambda _:nullcontext()):
            values,metadata=measure_cuda_batches(call,samples=3,calls_per_sample=2)
        self.assertEqual(metadata['requested_calls_per_sample'],2)
        self.assertEqual(metadata['actual_calls_per_sample'],200)
        self.assertEqual(values,[.01]*3)
        self.assertEqual(work[0],2+200+3*200)

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
