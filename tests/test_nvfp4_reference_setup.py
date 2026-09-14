import unittest
from unittest.mock import patch
import torch
from rtx.nvfp4_inference_experiments import NVFP4JITRowRegionBenchmarkHarness,NVFP4DynamicBenchmarkHarness
from rtx.prequant_experiments import ShapeSpec,BenchmarkProtocol


class NVFP4ReferenceSetupTests(unittest.TestCase):
    def test_jit_reference_does_not_compile_a_discarded_quantized_oracle(self):
        with patch.object(NVFP4JITRowRegionBenchmarkHarness,'_make_input_ring',return_value=[]), \
             patch.object(NVFP4JITRowRegionBenchmarkHarness,'_build_runner',
                          side_effect=AssertionError('discarded kernel reference compiled')):
            harness=NVFP4JITRowRegionBenchmarkHarness(ShapeSpec(128,128,128),'hot',
                                                     BenchmarkProtocol(),device='cpu',seed=1729)
        expected=(harness.x.float()@harness.weight.float().T).bfloat16()
        torch.testing.assert_close(harness._expected,expected,rtol=0,atol=0)

    def test_block_reference_retains_its_kernel_contract(self):
        def runner(x,w,out):out.fill_(3)
        with patch.object(NVFP4DynamicBenchmarkHarness,'_make_input_ring',return_value=[]), \
             patch.object(NVFP4DynamicBenchmarkHarness,'_build_runner',return_value=runner) as build, \
             patch('torch.cuda.synchronize'):
            harness=NVFP4DynamicBenchmarkHarness(ShapeSpec(128,128,128),'hot',
                                                 BenchmarkProtocol(),device='cpu',seed=1729)
        build.assert_called_once()
        torch.testing.assert_close(harness._expected,torch.full((128,128),3,dtype=torch.bfloat16))


if __name__=='__main__':unittest.main()
