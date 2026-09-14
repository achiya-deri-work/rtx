import unittest
import torch
from rtx.fp8 import _prequant_source_alignment


class SourceAlignmentTests(unittest.TestCase):
    def test_oriented_partial_warp_is_rejected_before_compile(self):
        from rtx.kernels.mxfp8_quant import MXFP8OrientedDualQuantKernel
        from rtx.configs.mxfp8 import MXFP8QuantConfig
        with self.assertRaisesRegex(ValueError, 'complete warp scale groups'):
            MXFP8OrientedDualQuantKernel(9, 128, 640,
                MXFP8QuantConfig(quant_vec=8), None, 'row', 'transpose')

    def test_offset_metadata(self):
        base=torch.empty(64,dtype=torch.bfloat16)
        self.assertEqual(_prequant_source_alignment(base),16)
        for offset,alignment in ((1,2),(2,4),(4,8),(8,16)):
            self.assertEqual(_prequant_source_alignment(base,base[offset:]),alignment)


@unittest.skipUnless(torch.cuda.is_available(),'requires CUDA')
class SourceAlignmentCudaTests(unittest.TestCase):
    def test_oriented_row_tasks_cross_row_boundaries(self):
        from rtx.kernels.mxfp8_quant import compile_mxfp8_oriented_dual_quant
        from rtx.configs.mxfp8 import MXFP8QuantConfig
        from tests.test_mxfp8_quant import _reference

        torch.manual_seed(20260918)
        for rows, k in ((128, 640), (128, 384)):
            with self.subTest(rows=rows, k=k):
                a = torch.randn(rows, k, device='cuda', dtype=torch.bfloat16)
                b = torch.randn(k, 128, device='cuda', dtype=torch.bfloat16)
                expected_a, expected_sa = _reference(a)
                expected_b, expected_sb = _reference(b.T.contiguous())
                qa, qb = torch.empty_like(expected_a), torch.empty_like(expected_b)
                sa, sb = torch.empty_like(expected_sa), torch.empty_like(expected_sb)
                config = MXFP8QuantConfig(quant_vec=8, load_bits=64,
                                          quant_store_bits=32, num_warps=8)
                kernel = compile_mxfp8_oriented_dual_quant(
                    rows, 128, k, config, a_orientation='row', b_orientation='transpose')
                kernel(a, b, qa, qb, sa, sb)
                torch.cuda.synchronize()
                for actual, expected in ((qa, expected_a), (qb, expected_b),
                                         (sa, expected_sa), (sb, expected_sb)):
                    self.assertTrue(torch.equal(actual.view(torch.uint8), expected.view(torch.uint8)))

    def test_singleton_row_family_regression(self):
        from tools.predictor_study import build_adapter

        for family in (
            'mxfp8_prequant_fwd', 'nvfp4_dynamic_fwd', 'nvfp4_delayed_fwd',
            'nvfp4_jit_row_region_fwd', 'mlp_forward', 'mlp_train', 'mxfp8_bwd',
        ):
            with self.subTest(family=family):
                adapter, _ = build_adapter(
                    dict(family=family, shape=dict(m=1, n=1280, k=896), seed=20260918, regime='hot'),
                    dict(warmup=1, samples=3, calls_per_sample=2), 'cuda:0',
                )
                outcome = adapter.evaluate(adapter.initial_config)
                self.assertTrue(outcome.successful, outcome.as_dict())

    def test_odd_width_mlp_compile_forward_backward(self):
        from tools.predictor_study import build_adapter
        for shape in (dict(m=17,n=97,k=65),dict(m=513,n=1543,k=777)):
            for family in ('mlp_forward','mlp_train'):
                with self.subTest(shape=shape,family=family):
                    adapter,_=build_adapter(dict(family=family,shape=shape,seed=20260918),
                        dict(warmup=1,samples=3,calls_per_sample=2),'cuda:0')
                    outcome=adapter.evaluate(adapter.initial_config)
                    self.assertTrue(outcome.successful,outcome.as_dict())

    def test_offset_row_quantizers(self):
        from rtx.kernels.mxfp8_quant import compile_mxfp8_quant,compile_mxfp8_dual_quant
        from rtx.configs.mxfp8 import MXFP8QuantConfig
        from tests.test_mxfp8_quant import _reference
        torch.manual_seed(20260919)
        for k in (128,129,777):
            for offset in (1,2,4,8):
                with self.subTest(k=k,offset=offset):
                    rows=7
                    backing=torch.randn(rows*k+offset,device='cuda',dtype=torch.bfloat16)
                    x=backing[offset:].view(rows,k)
                    storage_k=(k+31)//32*32
                    padded=torch.nn.functional.pad(x,(0,storage_k-k))
                    expected,scales=_reference(padded)
                    config=MXFP8QuantConfig(quant_vec=8,load_bits=128)
                    alignment=_prequant_source_alignment(x)
                    q=torch.empty((rows,storage_k),device='cuda',dtype=torch.float8_e4m3fn)
                    s=torch.empty_like(scales)
                    compile_mxfp8_quant(rows,k,config,source_alignment=alignment)(x,q,s)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(q.float(),expected.float(),rtol=0,atol=0)
                    self.assertTrue(torch.equal(s.view(torch.uint8),scales.view(torch.uint8)))
                    q2=torch.empty_like(q);s2=torch.empty_like(s)
                    compile_mxfp8_dual_quant(rows,rows,k,config,source_alignment=alignment)(x,x,q,q2,s,s2)
                    torch.cuda.synchronize()
                    torch.testing.assert_close(q2.float(),expected.float(),rtol=0,atol=0)
                    self.assertTrue(torch.equal(s2.view(torch.uint8),scales.view(torch.uint8)))


if __name__=='__main__':unittest.main()
