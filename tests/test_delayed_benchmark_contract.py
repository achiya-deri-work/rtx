"""Benchmark state transitions must include production history publication."""
import unittest
from unittest.mock import patch
import torch

from rtx.nvfp4_inference_experiments import _DelayedBenchmarkRunner


class DelayedBenchmarkContractTests(unittest.TestCase):
    def test_stable_addresses_and_publication_after_quant_gemm(self):
        states=[torch.zeros(4) for _ in range(4)]
        events=[]
        def execute(x,w,h,g,nh,ng,out):
            events.append('execute')
            nh.copy_(h+1); ng.copy_(g+2)
            out.zero_()
        def publish(h,g,nh,ng):
            events.append('publish')
            h.copy_(nh); g.copy_(ng)
        runner=_DelayedBenchmarkRunner(execute,*states)
        x=torch.ones(2,2); out=torch.empty_like(x)
        with patch('rtx.fp4._publish_delayed_history',side_effect=publish):
            runner(x,x,out); runner(x,x,out)
        self.assertEqual(events,['execute','publish','execute','publish'])
        self.assertIs(runner.x_history,states[0])
        self.assertIs(runner.weight_history,states[1])
        torch.testing.assert_close(states[0],torch.full((4,),2.))
        torch.testing.assert_close(states[1],torch.full((4,),4.))


if __name__=='__main__': unittest.main()
