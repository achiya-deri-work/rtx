"""CPU-only documentation freshness and executable-example contracts."""
import ast
import re
from pathlib import Path
import unittest

from tools.doc_reference import ROOT,render


class DocumentationTests(unittest.TestCase):
    def test_api_reference_matches_source(self):
        self.assertEqual((ROOT/"docs/api_reference.md").read_text(),render())

    def test_manual_links_resolve(self):
        # Public manual pages; historical study notes are not rewritten here.
        names=("README","installation","quickstart","api","api_reference",
               "formats","training","inference","execution","autotuning",
               "development","model_conversion","quantization_policy",
               "nvfp4_broad_evaluation_20261023","runtime_contract",
               "mxfp8_linear","nvfp4_linear","mxfp8_glu","mxfp8_mlp",
               "mxfp8_mlp_numerics","troubleshooting","research",
               "first_hit_autotuning","mxfp8_glu_history","mxfp8_mlp_history")
        pages=[ROOT/"README.md",*(ROOT/f"docs/{name}.md" for name in names)]
        for page in pages:
            for target in re.findall(r"\]\(([^)]+)\)",page.read_text()):
                if target.startswith(("http:","https:","mailto:","#")):
                    continue
                path=target.split("#",1)[0].strip("<>")
                with self.subTest(page=page.name,target=target):
                    self.assertTrue((page.parent/path).exists())

    def test_examples_parse(self):
        examples=list((ROOT/"docs/examples").glob("*.py"))
        self.assertTrue(examples)
        for path in examples:
            ast.parse(path.read_text(),filename=str(path))

    def test_shared_policy_is_public_and_discoverable(self):
        import rtx
        from rtx.autotune import AutotunePolicy
        self.assertIn("AutotunePolicy", rtx.__all__)
        self.assertIn("AutotunePolicy", dir(rtx))
        self.assertIs(rtx.AutotunePolicy, AutotunePolicy)
        for name, trials in (("quick",8),("balanced",16),("thorough",32),("extensive",64)):
            self.assertEqual(rtx.AutotunePolicy.preset(name).max_trials,trials)

    def test_canonical_guides_do_not_reintroduce_removed_first_hit_claims(self):
        for name in ("../README", "autotuning", "quickstart", "mxfp8_glu"):
            content=(ROOT/"docs"/(name+".md")).read_text()
            for stale in ("up to 24 trials / 30 seconds", "forward` never starts a first-hit search",
                          "there is no implicit per-shape tuning dispatcher yet",
                          "GLU/MLP tuning is explicit, not automatic"):
                self.assertNotIn(stale,content)

    def test_offline_calibration_exports(self):
        import rtx
        from rtx import calibration, nvfp4_calibration, ptq
        from rtx.autotune import AccuracySelectionResult, select_accuracy_policy
        for name, provider in (
            ("NVFP4CalibrationConfig", ptq),
            ("NVFP4CalibrationCollector", calibration),
            ("NVFP4SecondMomentCollector", nvfp4_calibration),
            ("calibrate_nvfp4_weight", nvfp4_calibration),
        ):
            self.assertIn(name, rtx.__all__)
            self.assertIs(getattr(rtx, name), getattr(provider, name))
        result = select_accuracy_policy(
            ["layer"], ["max"], lambda state: 0.,
            guard_evaluate=lambda state: 0., baselines=("max",),
        )
        self.assertIsInstance(result, AccuracySelectionResult)
        self.assertEqual(result.guard_choice, "baseline:max")


if __name__=="__main__":
    unittest.main()
