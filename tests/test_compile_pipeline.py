import json
from pathlib import Path
import tempfile
import unittest

from rtx.autotune.compile_pipeline import GPULease,adopt_identity


class CompilePipelineTests(unittest.TestCase):
    def test_identity_migration_is_source_only(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"identity.json"
            adopt_identity(path,{"source_id":"old","task":1})
            with self.assertRaises(ValueError):adopt_identity(path,{"source_id":"new","task":1})
            adopt_identity(path,{"source_id":"new","task":1},True)
            self.assertEqual(json.loads(path.read_text())["source_history"],["old"])
            with self.assertRaises(ValueError):adopt_identity(path,{"source_id":"new","task":2},True)

    def test_lease_records_balanced_events(self):
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/"gpu.lock"
            with GPULease(path) as lease:
                lease.release();lease.acquire()
            events=[json.loads(l)["event"] for l in path.with_suffix(".jsonl").read_text().splitlines()]
            self.assertEqual(events,["gpu_acquired","gpu_released","gpu_acquired","gpu_released"])


if __name__=="__main__":unittest.main()
