import json
from pathlib import Path
import tempfile
import unittest
import pyarrow.parquet as pq
from benchmarks.analyze_combined_collection import analyze


class CombinedAnalysisTests(unittest.TestCase):
    def test_latency_excludes_unstable_but_feasibility_retains_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'raw';source.mkdir()
            rows=[]
            for i,(status,stationary) in enumerate((('ok',True),('ok',False),('correctness_error',None))):
                rows.append(dict(context_id='c',observation_id=str(i),family='toy',config_id=str(i),
                    kernel_revision=1,features={},config={},outcome=dict(status=status,median_ms=1.,
                    metadata={'sampling':{'collection':{'stationary':stationary}}})))
            (source/'observations.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
            analyze([source],root/'report')
            self.assertEqual(pq.read_table(root/'report/feasibility_selection.parquet').num_rows,3)
            self.assertEqual(pq.read_table(root/'report/latency_selection.parquet').num_rows,1)

    def test_corrupt_tail_is_not_silently_ingested(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);source=root/'raw';source.mkdir()
            (source/'observations.jsonl').write_text('{broken')
            with self.assertRaisesRegex(ValueError,'Malformed JSONL'):
                analyze([source],root/'report')
