"""Independent sklearn/portable prediction equality; optional CPU research env."""
import importlib.util
import unittest

AVAILABLE = all(importlib.util.find_spec(n) for n in ('numpy', 'sklearn', 'pyarrow'))


@unittest.skipUnless(AVAILABLE, 'requires research environment')
class ExportTests(unittest.TestCase):
    def test_feature_transform_and_heads(self):
        import numpy as np
        from tools.research.models import FittedCandidate, REGISTRY
        from tools.research.data import feature_view
        from tools.export_tournament_predictors import export_model
        from rtx._portable_predictor import PortablePredictor, feature_view as portable_view, SCHEMA, FEATURE_REVISION
        rng = np.random.default_rng(100)
        rows = []
        for i in range(256):
            a = float(rng.uniform(-1, 10))
            rows.append(dict(context=str(i//32), family='test', revision=1, ms=1+abs(a),
                features={'config.a': a, 'derived.memory_roofline_ms': 1+abs(a)/10,
                          'context.device.properties.memory_bus_width_bits': 256,
                          'context.tags.precision=mxfp8': 1, 'derived.measured_latency': 100}))
        for r in rows:
            self.assertEqual(feature_view(r, 'interactions'), portable_view(r))
        for head, model_name in [('residual', 'roofline_residual'), ('mlp', 'small_mlp')]:
            fitted = FittedCandidate(REGISTRY[model_name], 'interactions', 0, 2).fit(rows)
            document = dict(schema=SCHEMA, feature_revision=FEATURE_REVISION, bundles=[],
                feature_names=fitted.vectorizer.get_feature_names_out().tolist(), models={head: export_model(fitted)})
            portable = PortablePredictor(document, research=True)
            np.testing.assert_allclose(portable.predict(rows, head), fitted.predict(rows), atol=1e-5, rtol=1e-5)


if __name__ == '__main__':
    unittest.main()
