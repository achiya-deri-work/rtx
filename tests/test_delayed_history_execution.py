import gc
import threading
from concurrent.futures import ThreadPoolExecutor
import unittest

import torch
from rtx._delayed_state import DelayedHistoryExecution


class Stream:
    device = 'mock_cuda_0'
    def __init__(self, identifier, log):
        self.cuda_stream = identifier
        self.log = log
    def wait_event(self, event):
        self.log.append(('wait', self.cuda_stream))


class DelayedHistoryExecutionTests(unittest.TestCase):
    def test_eager_and_compiled_wrappers_guard_only_managed_submissions(self):
        from contextlib import contextmanager
        from unittest.mock import patch
        import rtx.fp4 as fp4
        for compiled in (False,True):
            for managed in (False,True):
                with self.subTest(compiled=compiled,managed=managed):
                    log=[];states=[torch.zeros(4) for _ in range(4)]
                    x=weight=torch.ones(2,2);out=torch.empty(2,2)
                    @contextmanager
                    def guard(actual_states,stream):
                        self.assertEqual(tuple(map(id,actual_states)),tuple(map(id,states)))
                        self.assertEqual(stream,'stream')
                        log.append('enter')
                        try:yield
                        finally:log.append('exit')
                    def submit(*args,**kwargs):
                        log.append('submit');return out
                    with patch('rtx._delayed_state.DELAYED_HISTORY_EXECUTION.submission',side_effect=guard), \
                         patch('torch.cuda.current_stream',return_value='stream'):
                        if compiled:
                            launcher=fp4._InductorNVFP4DelayedLauncher('forward','materialized',managed)
                            with patch.object(launcher,'_submit',side_effect=submit):
                                launcher(x,weight,*states,out=out)
                        else:
                            with patch('rtx.fp4._submit_nvfp4_forward_delayed_out',side_effect=submit):
                                fp4._launch_nvfp4_forward_delayed_out(x,weight,*states,'forward','materialized',managed)
                    self.assertEqual(log,['enter','submit','exit'] if managed else ['submit'])

    def setup_guard(self):
        log=[]
        class Event:
            def record(self, stream):log.append(('record',stream.cuda_stream))
        guard=DelayedHistoryExecution(event_factory=Event,capturing=lambda stream:False)
        return guard,[torch.zeros(4) for _ in range(4)],Stream(1,log),Stream(2,log),log

    def test_same_stream_has_no_events_and_migration_orders_before_body(self):
        guard,states,a,b,log=self.setup_guard()
        for _ in range(3):
            with guard.submission(states,a):pass
        self.assertEqual(log,[])
        with guard.submission(states,b):log.append(('body',2))
        self.assertEqual(log,[('record',1),('wait',2),('body',2)])

    def test_failed_submission_still_has_a_dependency(self):
        guard,states,a,b,log=self.setup_guard()
        with self.assertRaisesRegex(RuntimeError,'launch failed'):
            with guard.submission(states,a):raise RuntimeError('launch failed')
        with guard.submission(states,b):pass
        self.assertEqual(log,[('record',1),('wait',2)])

    def test_capture_migration_rejected_without_event_or_ownership_change(self):
        guard,states,a,b,log=self.setup_guard()
        with guard.submission(states,a):pass
        guard.capturing=lambda stream:True
        with guard.submission(states,a):pass
        with self.assertRaisesRegex(RuntimeError,'capture'):
            with guard.submission(states,b):self.fail('entered unsafe migration')
        self.assertEqual(log,[])
        self.assertEqual(next(iter(guard.entries.values())).stream,a)

    def test_per_history_mutex_not_global_and_no_tensor_retention(self):
        guard,states,a,b,log=self.setup_guard()
        other=[torch.zeros(4) for _ in range(4)]
        attempting=threading.Event();entered=threading.Event()
        def same():
            attempting.set()
            with guard.submission(states,b):entered.set()
        def different():
            with guard.submission(other,b):return True
        with ThreadPoolExecutor(max_workers=2) as executor:
            with guard.submission(states,a):
                future=executor.submit(same)
                self.assertTrue(attempting.wait(2))
                self.assertFalse(entered.is_set())
                self.assertTrue(executor.submit(different).result(timeout=2))
            future.result(timeout=2)
        self.assertTrue(entered.is_set())
        refs=[ref for entry in guard.entries.values() for ref in entry.refs]
        del states,other;gc.collect()
        self.assertTrue(all(ref.expired() for ref in refs))
        with guard.submission([torch.zeros(4) for _ in range(4)],a):pass
        self.assertEqual(len(guard.entries),1)


if __name__=='__main__':unittest.main()
