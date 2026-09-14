import gc
import unittest
import torch
from rtx._delayed_state import DelayedHistoryBootstrap


class DelayedStateBootstrapTests(unittest.TestCase):
    def test_copy_registration_reaps_expired_storage_entries(self):
        bootstrap=DelayedHistoryBootstrap()
        source=[torch.zeros(16) for _ in range(4)]
        bootstrap.prepare(torch.ones(2,2),torch.ones(2,2),*source,stream_id=0,policy_key='test')
        for _ in range(20):
            copied=[value.clone() for value in source]
            bootstrap.inherit_copy(source,copied)
            self.assertIsNotNone(bootstrap.state_signature(*copied))
            self.assertLessEqual(len(bootstrap.entries),2)
            del copied
            gc.collect()

    def test_deepcopy_preserves_ready_history_with_independent_storage(self):
        import copy
        import rtx
        from rtx._delayed_state import DELAYED_HISTORY_BOOTSTRAP
        layer=rtx.NVFP4Linear(128,128,device='cpu',scaling='delayed',autotune='off')
        def states(module):
            return (module._x_amax_state,module._weight_amax_state,
                    module._next_x_amax_state,module._next_weight_amax_state)
        self.assertFalse(copy.deepcopy(layer)._delayed_initialized)
        DELAYED_HISTORY_BOOTSTRAP.prepare(torch.ones(128,128),layer.weight,*states(layer),
            stream_id=0,policy_key=('forward','materialized'))
        layer._x_amax_state.copy_(torch.arange(16,dtype=torch.float32))
        layer._next_x_amax_state.zero_(); layer._next_weight_amax_state.zero_()
        clone=copy.deepcopy(layer)
        self.assertTrue(clone._delayed_initialized)
        self.assertEqual(clone._delayed_problem,layer._delayed_problem)
        for source,target in zip(states(layer),states(clone)):
            self.assertNotEqual(source.data_ptr(),target.data_ptr())
            torch.testing.assert_close(source,target,rtol=0,atol=0,equal_nan=True)
        clone._x_amax_state.zero_()
        torch.testing.assert_close(layer._x_amax_state,torch.arange(16,dtype=torch.float32))
        clone.reset_delayed_scales()
        self.assertFalse(clone._delayed_initialized)
        self.assertTrue(layer._delayed_initialized)

    def test_aliased_storage_views_fail_before_initialization(self):
        bootstrap=DelayedHistoryBootstrap()
        storage=torch.full((4,16),42.)
        states=list(storage.unbind())
        self.assertEqual(len({s.data_ptr() for s in states}),4)
        with self.assertRaisesRegex(ValueError,'independent storage'):
            bootstrap.prepare(torch.ones(2,2),torch.ones(2,2),*states,stream_id=0,policy_key='test')
        torch.testing.assert_close(storage,torch.full((4,16),42.),rtol=0,atol=0)
        self.assertFalse(bootstrap.entries)

    def test_publication_uses_stream_ordered_device_copies(self):
        from unittest.mock import patch
        from cuda.bindings import runtime
        from rtx.fp4 import _publish_delayed_history
        states=[torch.empty(16) for _ in range(4)]
        with patch('torch._C._cuda_getCurrentRawStream',return_value=123), \
             patch.object(runtime,'cudaMemcpyAsync',return_value=(runtime.cudaError_t.cudaSuccess,)) as copy, \
             patch.object(torch.Tensor,'copy_',side_effect=AssertionError('eager copy forbidden')):
            _publish_delayed_history(*states)
        self.assertEqual(copy.call_count,2)
        for call,dest,source in zip(copy.call_args_list,states[:2],states[2:]):
            self.assertEqual(call.args,(dest.data_ptr(),source.data_ptr(),64,
                                       runtime.cudaMemcpyKind.cudaMemcpyDeviceToDevice,123))
        with patch('torch._C._cuda_getCurrentRawStream',return_value=123), \
             patch.object(runtime,'cudaMemcpyAsync',return_value=(runtime.cudaError_t.cudaErrorInvalidValue,)):
            with self.assertRaisesRegex(RuntimeError,'cudaMemcpyAsync failed'):
                _publish_delayed_history(*states)

    def test_ping_pong_does_not_rebootstrap_and_shape_change_does(self):
        bootstrap=DelayedHistoryBootstrap()
        x=torch.tensor([[1.,3.]])
        w=torch.tensor([[2.,4.]])
        states=[torch.empty(4) for _ in range(4)]
        def prepare(x, states):
            return bootstrap.prepare(x,w,*states,stream_id=0,policy_key='test')
        self.assertTrue(prepare(x,states))
        torch.testing.assert_close(states[0],torch.full((4,),3.))
        states[2].fill_(7.); states[3].fill_(8.)
        swapped=[states[2],states[3],states[0],states[1]]
        self.assertFalse(prepare(x*10,swapped))
        torch.testing.assert_close(swapped[0],torch.full((4,),7.))
        self.assertTrue(prepare(x.repeat(2,1),swapped))
        torch.testing.assert_close(swapped[0],torch.full((4,),3.))

    def test_registry_does_not_keep_storage_alive(self):
        bootstrap=DelayedHistoryBootstrap()
        states=[torch.empty(4) for _ in range(4)]
        bootstrap.prepare(torch.ones(1,2),torch.ones(1,2),*states,stream_id=0,policy_key='test')
        refs=next(iter(bootstrap.entries.values()))[1]
        del states; gc.collect()
        self.assertTrue(all(r.expired() for r in refs))


if __name__=='__main__': unittest.main()
