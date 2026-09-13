"""Leakage/split/model semantics; uses only optional CPU research packages."""
import importlib.util
import unittest

AVAILABLE = all(importlib.util.find_spec(n) for n in ('numpy', 'sklearn', 'pyarrow'))


@unittest.skipUnless(AVAILABLE, 'requires research dependencies')
class TournamentTests(unittest.TestCase):
    def row(self, m=128, device='a'):
        from tools.research.data import shape_group
        f = {'context.workload.m': m, 'context.workload.n': 128, 'context.workload.k': 128,
             'config.tile_m': 64, 'context.device.properties.memory_bus_width_bits': 256,
             'derived.cta_waves': 1, 'derived.measured_latency': .01,
             'context.device.properties.name=secret': 1}
        return dict(features=f, family='f', revision=1, device=device, shape=shape_group(f),
                    context=f'{device}-{m}', config_id='c', ms=1., operating={'power.limit': 200})

    def test_features_exclude_identity_and_postmeasurement(self):
        from tools.research.data import feature_view
        r = self.row()
        base, hw, interaction, op = [feature_view(r, v) for v in ('shape', 'hardware', 'interactions', 'operating')]
        self.assertNotIn('context.device.properties.memory_bus_width_bits', base)
        self.assertIn('context.device.properties.memory_bus_width_bits', hw)
        self.assertNotIn('derived.cta_waves', hw)
        self.assertIn('derived.cta_waves', interaction)
        self.assertIn('operating.power.limit', op)
        self.assertFalse(any('secret' in k or 'measured' in k for k in op))

    def test_grouping_and_both_holdouts(self):
        from tools.research.data import inner_split, split_rows
        rows = [self.row(m, device) for m in range(32, 256) for device in ('a', 'b')]
        a, b = inner_split(rows)
        self.assertTrue(a and b)
        self.assertFalse({r['shape'] for r in a} & {r['shape'] for r in b})
        a, b = split_rows(rows, rows, 'heldout:b')
        self.assertEqual({r['device'] for r in a}, {'a'})
        self.assertEqual({r['device'] for r in b}, {'b'})
        a, b = split_rows(rows, rows, 'unseen_shapes')
        self.assertFalse({r['shape'] for r in a} & {r['shape'] for r in b})

    def test_ingestion_telemetry_and_failure_targets(self):
        import json
        from pathlib import Path
        import tempfile
        from tools.research.data import load_dataset
        row = self.row()
        common = dict(config={}, family='f', kernel_revision=1, features=row['features'])
        records = [dict(common, config_id='ok', observation_id='one', outcome={
            'status': 'ok', 'median_ms': 2., 'metadata': {'elapsed_s': .5,
                'telemetry_before': {'power.draw': '20'},
                'telemetry_after': {'power.draw': '999'},
                'sampling': {'collection': {'stationary': True}}}}),
            dict(common, config_id='bad', observation_id='two', outcome={
                'status': 'compile_error', 'median_ms': None, 'metadata': {'elapsed_s': 3.}})]
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, 'observations.jsonl').write_text('\n'.join(map(json.dumps, records)))
            latency, attempts, counts, hardware = load_dataset([tmp])
        self.assertEqual(len(latency), 1)
        self.assertEqual(latency[0]['operating']['power.draw'], 20)
        self.assertEqual([r['failed'] for r in attempts], [0, 1])
        self.assertEqual(attempts[1]['evaluation_s'], 3.)

    def test_model_registry_smoke(self):
        import numpy as np
        from tools.research.models import FittedCandidate, REGISTRY, Retrieval
        rows = []
        for m in (64, 128, 256, 512):
            for tile in range(1, 33):
                row = self.row(m)
                row['features'] = {**row['features'], 'config.tile_m': tile}
                row['config_id'], row['ms'] = str(tile), float(tile)
                rows.append(row)
        for name in ('ridge', 'hist_family', 'extra_trees', 'roofline_residual', 'pairwise_xgb'):
            if name == 'pairwise_xgb' and importlib.util.find_spec('xgboost') is None:
                continue
            with self.subTest(model=name):
                fitted = FittedCandidate(REGISTRY[name], 'interactions', 0, 2).fit(rows)
                pred = fitted.predict(rows)
                self.assertTrue(np.isfinite(pred).all())
                self.assertLess(pred[0], pred[31])
        self.assertTrue(np.isfinite(Retrieval().fit(rows).predict(rows)).all())
        for name in ('ridge','hist_family','small_mlp'):
            with self.subTest(weighted=name):
                fitted=FittedCandidate(REGISTRY[name],'interactions',0,2).fit(rows,balance_contexts=True)
                self.assertTrue(np.isfinite(fitted.predict(rows)).all())


if __name__ == '__main__':
    unittest.main()
