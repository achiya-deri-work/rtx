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
               "development","model_conversion","runtime_contract",
               "mxfp8_linear","nvfp4_linear","mxfp8_glu","mxfp8_mlp",
               "mxfp8_mlp_numerics","troubleshooting")
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


if __name__=="__main__":
    unittest.main()
