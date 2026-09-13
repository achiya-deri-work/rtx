"""CPU research invariants; run with the isolated research dependencies."""
import importlib.util
import unittest

AVAILABLE = all(importlib.util.find_spec(name) for name in ('numpy', 'sklearn', 'pyarrow'))


@unittest.skipUnless(AVAILABLE, 'requires CPU research dependencies')
class EffortResearchTests(unittest.TestCase):
    def test_measurements_and_identity_are_not_features(self):
        from tools.research_effort import features
        row = {
            'config.tile_m': 128,
            'context.workload.m': 512,
            'context.device.properties.multi_processor_count': 70,
            'derived.cta_waves': 2,
            'derived.measured_latency': .1,
            'derived.compiled_registers': 42,
            'context.device.name=example': 1,
            'outcome.median_ms': .2,
        }
        shape = features(row, sku=False)
        sku = features(row, sku=True)
        self.assertEqual(set(shape), {'config.tile_m', 'context.workload.m'})
        self.assertEqual(set(sku) - set(shape), {
            'context.device.properties.multi_processor_count', 'derived.cta_waves'})

    def test_ranking_metrics_have_known_oracle_and_deterministic_baseline(self):
        from tools.research_effort import metrics
        rows = [dict(context='c', shape='s', device='d', family='f', ms=float(i+1))
                for i in range(32)]
        perfect = metrics(rows, list(range(32)), 'test', 'perfect')
        reverse = metrics(rows, list(reversed(range(32))), 'test', 'reverse')
        self.assertEqual([r['regret'] for r in perfect], [0, 0, 0])
        self.assertEqual([r['regret'] for r in reverse], [24, 16, 0])
        self.assertEqual([r['random_regret'] for r in perfect],
                         [r['random_regret'] for r in reverse])


if __name__ == '__main__':
    unittest.main()
