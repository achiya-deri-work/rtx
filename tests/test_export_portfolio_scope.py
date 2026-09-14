import unittest
from tools.research.export_portfolio import supported_splits


class ExportScopeTests(unittest.TestCase):
    def test_only_supported_families_advertised(self):
        rows=[dict(family=f,split=s) for f in ('mx','nv') for s in ('train','test')]
        train,test=supported_splits(rows,{'mx'})
        self.assertEqual(train,[dict(family='mx',split='train')])
        self.assertEqual(test,[dict(family='mx',split='test')])

    def test_reject_unverified_or_untrained_family(self):
        for split in ('train','test'):
            with self.subTest(split=split),self.assertRaises(ValueError):
                supported_splits([dict(family='mx',split=split)],{'mx'})
        with self.assertRaises(ValueError):
            supported_splits([dict(family='mx',split=s) for s in ('train','test')],{'mx','nv'})


if __name__=='__main__':unittest.main()
