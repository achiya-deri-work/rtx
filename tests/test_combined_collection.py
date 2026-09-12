from pathlib import Path
import tempfile
import unittest

from rtx.autotune.combined_collection import command,run


class CombinedCollectionTests(unittest.TestCase):
    def test_commands_are_foreground_and_share_device(self):
        mlp=command("mlp",Path("/tmp/study"),900,"cuda:1")
        linear=command("linears",Path("/tmp/study"),900,"cuda:1")
        self.assertIn("rtx.autotune.first_hit_study",mlp)
        self.assertIn("rtx.autotune.linear_pipeline",linear)
        self.assertEqual(mlp[mlp.index("--device")+1],"cuda:1")
        self.assertEqual(linear[linear.index("--device")+1],"cuda:1")
        self.assertNotIn("--stall-timeout",linear)
        self.assertIn("--seconds",linear)
        self.assertIn("--hours",mlp)

    def test_invalid_duration(self):
        with tempfile.TemporaryDirectory() as root:
            for hours in (0,-1,float("nan"),float("inf")):
                with self.assertRaises(ValueError):run(root,hours=hours)


if __name__=="__main__":unittest.main()
