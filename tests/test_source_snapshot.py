import tempfile
from pathlib import Path
import unittest
import zipfile

from tools.research.source_snapshot import write_snapshot, verify_snapshot


class SourceSnapshotTests(unittest.TestCase):
    def fixture(self,root):
        (root/'rtx').mkdir()
        (root/'rtx'/'kernel.py').write_text('VALUE = 1\n')
        (root/'requirements.txt').write_text('numpy\n')
        (root/'dataset.jsonl').write_text('private data\n')
        (root/'rtx'/'debug.log').write_text('not source\n')

    def test_deterministic_source_only_and_revision_verification(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root)
            a=write_snapshot(root,root/'a.zip')
            b=write_snapshot(root,root/'b.zip')
            self.assertEqual(a['archive_sha256'],b['archive_sha256'])
            self.assertEqual(a['files'],2)
            with self.assertRaises(FileExistsError):write_snapshot(root,root/'a.zip')
            (root/'rtx'/'kernel.py').write_text('VALUE = 2\n')
            self.assertTrue(verify_snapshot(root/'a.zip')['verified'])
            with self.assertRaisesRegex(ValueError,'workspace sources differ'):
                verify_snapshot(root/'a.zip',root=root)

    def test_symlinks_and_tampered_members_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root)
            write_snapshot(root,root/'a.zip')
            with zipfile.ZipFile(root/'a.zip','a') as archive:archive.writestr('../outside.py','bad')
            with self.assertRaisesRegex(ValueError,'unsafe'):verify_snapshot(root/'a.zip')
            (root/'rtx'/'alias.py').symlink_to(root/'dataset.jsonl')
            with self.assertRaisesRegex(ValueError,'symlink'):write_snapshot(root,root/'b.zip')

    def test_content_tampering_is_detected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);self.fixture(root)
            write_snapshot(root,root/'a.zip')
            with zipfile.ZipFile(root/'a.zip') as original, zipfile.ZipFile(root/'tampered.zip','x') as changed:
                for info in original.infolist():
                    data=original.read(info.filename)
                    changed.writestr(info,b'VALUE = 999\n' if info.filename=='rtx/kernel.py' else data)
            with self.assertRaisesRegex(ValueError,'digest mismatch'):verify_snapshot(root/'tampered.zip')

    def test_study_guard_checks_snapshot_before_execution(self):
        from unittest.mock import patch
        from tools.research.matched_gpu import verify_sources
        with patch('tools.research.matched_gpu.source_digest',return_value='runtime'):
            with self.assertRaisesRegex(ValueError,'sibling filename'):
                verify_sources(dict(source_sha256='runtime',source_snapshot=dict(path='../source.zip')),'pool.json')
            metadata=dict(path='pool.sources.zip',archive_sha256='original')
            with patch('tools.research.source_snapshot.verify_snapshot',return_value={**metadata,'archive_sha256':'changed'}):
                with self.assertRaisesRegex(ValueError,'identity mismatch'):
                    verify_sources(dict(source_sha256='runtime',source_snapshot=metadata),'pool.json')


if __name__=='__main__':unittest.main()
