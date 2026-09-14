import unittest
import numpy as np
import importlib.util
HAS_RESEARCH = importlib.util.find_spec('sklearn') is not None
if HAS_RESEARCH:
    from tools.research.models import FittedCandidate, REGISTRY
    from tools.research.export_portfolio import tree_models
    from rtx._portable_predictor import PortablePredictor, SCHEMA, FEATURE_REVISION


@unittest.skipUnless(HAS_RESEARCH, 'requires research environment')
class PortfolioExportTests(unittest.TestCase):
    def test_histogram_and_forest_prediction_contract(self):
        rows=[dict(family='f',revision=1,context=str(c),ms=1+(i%8),
                   features={'config.tile':i%8,'context.workload.m':128+c})
              for c in range(16) for i in range(8)]
        for name in ('hist_family','extra_trees'):
            with self.subTest(name=name):
                model=FittedCandidate(REGISTRY[name],'interactions',0,1).fit(rows,balance_contexts=True)
                doc=dict(schema=SCHEMA,feature_revision=FEATURE_REVISION,bundles=[],
                    feature_names=model.vectorizer.get_feature_names_out().tolist(),models={'trees':tree_models(model)})
                actual=PortablePredictor(doc,research=True).predict(rows,'trees')
                np.testing.assert_allclose(actual,model.predict(rows),rtol=1e-5,atol=1e-5)
