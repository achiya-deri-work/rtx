import unittest
import importlib.util
HAS_RESEARCH = importlib.util.find_spec('sklearn') is not None
if HAS_RESEARCH:
    from tools.research.evidence_dataset import timing_quality, decode


@unittest.skipUnless(HAS_RESEARCH, 'requires research environment')
class EvidenceDatasetTests(unittest.TestCase):
    def test_quality_does_not_promote_unknown_or_failed(self):
        self.assertEqual(timing_quality(dict(status='correctness_error')), 'correctness_error')
        self.assertEqual(timing_quality(dict(status='ok', median_ms=1)), 'timing_unverified')
        self.assertEqual(timing_quality(dict(status='ok', median_ms=1, timings_ms=[1, 1, 1])), 'stable')
        self.assertEqual(timing_quality(dict(status='ok', median_ms=1, timings_ms=[1, 1, 2])), 'unstable')
        self.assertEqual(timing_quality(dict(status='ok', median_ms=float('nan'))), 'invalid_latency')

    def test_contract_and_device_are_not_merged(self):
        row = dict(features={'context.workload.m':128, 'context.workload.k':640,
                   'context.workload.n':1280, 'context.device.name=one':1},
                   family='test', kernel_revision=1, config_id='a', config={})
        first = decode(row)
        second = decode({**row, 'kernel_revision':2})
        self.assertNotEqual(first['context'], second['context'])
        self.assertEqual(first['shape'], second['shape'])
