import copy
from pathlib import Path
import tempfile
import unittest
from tools.research.runtime_prior_compare import selections, verify_execution_sources
from tools.research.source_snapshot import write_snapshot


class RuntimePriorCompareTests(unittest.TestCase):
    def documents(self):
        return [dict(budget=b,mode=m,shape={'m':128,'n':128,'k':128},environment={},
                     winner=dict(family='mxfp8_prequant_fwd',device_id='device',kernel_revision=9,
                                 config={'same':1}))
                for b in (8,16,32,64) for m in ('ordinary','retrieval','ml')]

    def test_shared_configs_are_exact_selection_ties(self):
        choices, configs = selections(self.documents())
        self.assertEqual(len(configs),1)
        self.assertEqual(len({cid for modes in choices.values() for cid in modes.values()}),1)

    def test_partial_duplicate_and_mixed_contracts_rejected(self):
        docs=self.documents()
        for invalid in (docs[:-1],docs+[docs[0]]):
            with self.assertRaises(ValueError): selections(invalid)
        changed=copy.deepcopy(docs);changed[-1]['winner']['kernel_revision']=10
        with self.assertRaises(ValueError): selections(changed)
        for field,value in (('family','nvfp4_dynamic_fwd'),('variant','other-policy')):
            changed=copy.deepcopy(docs);changed[-1]['winner'][field]=value
            with self.assertRaises(ValueError):selections(changed)

    def test_nvfp4_families_supported_without_cross_family_pooling(self):
        for family in ('nvfp4_dynamic_fwd','nvfp4_jit_row_region_fwd'):
            docs=self.documents()
            for doc in docs:doc['winner']['family']=family
            choices,configs=selections(docs)
            self.assertEqual(len(choices),4)
            self.assertEqual(len(configs),1)

    def test_analysis_additions_allowed_but_runtime_changes_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'rtx').mkdir()
            (root/'tools/research').mkdir(parents=True)
            kernel=root/'rtx/kernel.py'
            kernel.write_text('original runtime\n')
            (root/'tools/research/runtime_prior_probe.py').write_text('original probe\n')
            snapshot=root/'sources.zip'
            metadata=write_snapshot(root,snapshot)
            verify_execution_sources(snapshot,metadata,root)
            (root/'tools/research/new_analysis.py').write_text('new analysis\n')
            verify_execution_sources(snapshot,metadata,root)
            kernel.write_text('changed runtime\n')
            with self.assertRaisesRegex(ValueError,'execution source changed'):
                verify_execution_sources(snapshot,metadata,root)


if __name__ == '__main__': unittest.main()
