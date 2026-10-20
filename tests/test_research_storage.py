import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from tools.research.storage import build, verify, archive_tree, retire_archive


class ResearchStorageTests(unittest.TestCase):
    def test_roundtrip_duplicates_conflicts_malformed_and_source_provenance(self):
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'source';source.mkdir()
            a=b'{"observation_id":"same","outcome":{"status":"ok"}}\n'
            b=b'{"observation_id":"same","outcome":{"status":"compile_error"}}\n'
            raw=a+b+b'broken\n\n'
            (source/'observations.jsonl').write_bytes(raw)
            with zipfile.ZipFile(root/'copy.zip','w') as archive:
                archive.writestr('observations.jsonl',raw)
            out=root/'merged'
            manifest=build([source,root/'copy.zip',source],out,shard_rows=2)
            self.assertEqual(manifest['unique_records'],3)
            self.assertEqual(manifest['occurrences'],6)
            self.assertEqual(manifest['duplicate_occurrences'],3)
            self.assertEqual(manifest['parse_counts']['malformed'],2)
            self.assertTrue(verify(out)['verified'])
            restored=[]
            for part in manifest['tables']['records']:
                restored.extend(pq.read_table(out/part['path'])['payload'].to_pylist())
            self.assertEqual(set(restored),{a,b,b'broken\n'})
            self.assertEqual((source/'observations.jsonl').read_bytes(),raw)
            self.assertFalse((out/'building.sqlite').exists())
            with self.assertRaises(FileExistsError):build([source],out)
            part=out/manifest['tables']['records'][0]['path']
            part.write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError,'corrupt shard'):verify(out)

    def test_reject_recursive_output_and_bad_budget(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):build([tmp],Path(tmp)/'child')
            with self.assertRaises(ValueError):build([],Path(tmp)/'other',shard_rows=0)

    def test_invalid_provenance_rejected_even_with_matching_shard_checksum(self):
        import pyarrow as pa
        import pyarrow.parquet as pq
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'source'
            source.mkdir()
            (source/'a.jsonl').write_bytes(b'{}\n')
            out = root/'merged'
            manifest = build([source], out)
            part = manifest['tables']['occurrences'][0]
            path = out/part['path']
            data = pq.read_table(path)
            data = data.set_column(1, 'source_id', pa.array([999], type=pa.int64()))
            pq.write_table(data, path)
            part['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            (out/'manifest.json').write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, 'invalid occurrence reference'):
                verify(out)

    def test_archive_verified_retirement_and_exact_restore(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'bundle';(source/'empty').mkdir(parents=True)
            (source/'observations.jsonl').write_bytes(b'original\n')
            out=root/'archives/bundle.zip'
            receipt=archive_tree(source,out,retire=True)
            self.assertTrue(receipt['retired'])
            self.assertFalse(source.exists())
            with zipfile.ZipFile(out) as archive:
                archive.extractall(root/'restore')
            self.assertEqual((root/'restore/bundle/observations.jsonl').read_bytes(),b'original\n')
            self.assertTrue((root/'restore/bundle/empty').is_dir())

    def test_archive_symlinks_and_nested_output_rejected_without_deletion(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'bundle';source.mkdir();(source/'a').write_text('data')
            with self.assertRaises(ValueError):archive_tree(source,source/'copy.zip',retire=True)
            (source/'link').symlink_to(source/'a')
            with self.assertRaises(ValueError):archive_tree(source,root/'copy.zip',retire=True)
            self.assertEqual((source/'a').read_text(),'data')

    def test_existing_archive_retirement_rejects_changes_then_restores(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'bundle';source.mkdir()
            (source/'a').write_bytes(b'original')
            out=root/'copy.zip';archive_tree(source,out)
            (source/'a').write_bytes(b'changed')
            with self.assertRaises(ValueError):retire_archive(source,out)
            self.assertEqual((source/'a').read_bytes(),b'changed')
            (source/'a').write_bytes(b'original')
            retire_archive(source,out)
            self.assertFalse(source.exists())
            with zipfile.ZipFile(out) as archive:
                self.assertEqual(archive.read('bundle/a'),b'original')


if __name__=='__main__':unittest.main()
