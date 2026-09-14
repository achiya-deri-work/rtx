"""CUDA ordering for delayed histories and their production launchers."""
import json
import unittest

import torch
from rtx._delayed_state import DelayedHistoryExecution


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
class DelayedHistoryExecutionCudaTests(unittest.TestCase):
    def test_shared_module_stream_handoffs_match_ordered_training(self):
        self._check_shared_module_training(threaded=False)

    def test_shared_module_host_threads_match_acquisition_order(self):
        self._check_shared_module_training(threaded=True)

    def _check_shared_module_training(self, *, threaded):
        import copy
        from pathlib import Path
        import tempfile
        from unittest.mock import patch
        import rtx
        from torch._inductor.compile_fx import compile_fx

        for use_compile in (False,True):
            with self.subTest(compiled=use_compile), tempfile.TemporaryDirectory() as root:
                torch.compiler.reset();torch.manual_seed(6142)
                policy=rtx.AutotunePolicy(max_trials=8,max_seconds=90,search_schedule='progressive',
                    cache_behavior='refine',confirmation_repeats=0,finalist_rounds=2,
                    warmup=1,samples=3,calls_per_sample=2)
                layer=rtx.NVFP4Linear(128,128,device='cuda',scaling='delayed',
                    tuning_policy=policy,autotune_cache_dir=root).train()
                control=copy.deepcopy(layer)
                graphs=[]
                def backend(graph,inputs):
                    graphs.append(graph);return compile_fx(graph,inputs)
                forward=torch.compile(layer,backend=backend,fullgraph=True,dynamic=False) if use_compile else layer
                inputs=[(torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*mag).requires_grad_()
                        for mag in (.08,.10,.12,.09,.11,.07)]
                upstream=[(torch.randn(128,128,device='cuda',dtype=torch.bfloat16)*.1).T for _ in inputs]
                streams=[torch.cuda.Stream(),torch.cuda.Stream()]
                sequence=(0,0,1,1,0,1)
                torch.cuda.synchronize()
                def step(fn,module,x,dy,stream):
                    with torch.cuda.stream(stream):
                        y=fn(x)
                        dx,dw=torch.autograd.grad(y,(x,module.weight),dy)
                    return y,dx,dw
                # Compile, tune and allocate stream-local runners before the
                # pending-work phase. Both states see identical warmup calls.
                for stream in streams:
                    step(forward,layer,inputs[0],upstream[0],stream)
                    stream.synchronize()
                    step(control,control,inputs[0],upstream[0],stream)
                    stream.synchronize()
                def journal():
                    return {str(p.relative_to(root)):p.read_bytes() for p in Path(root).rglob('observations.jsonl')}
                evidence=journal()
                self.assertEqual(sum(len(v.splitlines()) for v in evidence.values()),16)
                self.assertEqual(len(graphs),int(use_compile))
                with patch('rtx.nvfp4_inference_autotune.tune_nvfp4_inference_state',
                           side_effect=AssertionError('stream handoff retuned forward')), \
                     patch('rtx.bwd_autotune.tune_mxfp8_backward',
                           side_effect=AssertionError('stream handoff retuned backward')):
                    pending=torch.cuda.Event()
                    with torch.cuda.stream(streams[0]):torch.cuda._sleep(100_000_000)
                    outputs=[None]*len(inputs)
                    order=[]
                    if threaded:
                        from concurrent.futures import ThreadPoolExecutor
                        from contextlib import contextmanager
                        from threading import Barrier, local
                        from rtx._delayed_state import DELAYED_HISTORY_EXECUTION
                        thread_state=local()
                        barrier=Barrier(2,timeout=30)
                        submission=DELAYED_HISTORY_EXECUTION.submission

                        @contextmanager
                        def record_acquisition(states,stream):
                            with submission(states,stream):
                                order.append(thread_state.index)
                                yield

                        def worker(sid):
                            with torch.cuda.stream(streams[sid]):
                                barrier.wait()
                                for i,assigned in enumerate(sequence):
                                    if assigned==sid:
                                        thread_state.index=i
                                        outputs[i]=forward(inputs[i])

                        # Instrument only after acquisition of the production
                        # lock. Do not serialize entire calls in the harness.
                        with patch.object(DELAYED_HISTORY_EXECUTION,'submission',record_acquisition), \
                             ThreadPoolExecutor(max_workers=2) as executor:
                            futures=[executor.submit(worker,sid) for sid in range(2)]
                            for future in futures:future.result(timeout=60)
                        self.assertEqual(sorted(order),list(range(len(inputs))))
                    else:
                        for i,sid in enumerate(sequence):
                            with torch.cuda.stream(streams[sid]):
                                outputs[i]=forward(inputs[i])
                            order.append(i)
                            if i==0:
                                pending.record(streams[0])
                                self.assertFalse(pending.query(),'must start with pending producer work')
                    torch.cuda.synchronize()
                    # Request gradients only after the overlapping forwards
                    # finish, so autograd's cross-stream synchronization cannot
                    # mask a missing history dependency in the forward path.
                    actual=[]
                    for i,(y,sid) in enumerate(zip(outputs,sequence)):
                        with torch.cuda.stream(streams[sid]):
                            dx,dw=torch.autograd.grad(y,(inputs[i],layer.weight),upstream[i])
                        actual.append((y,dx,dw))
                    torch.cuda.synchronize()
                    expected=[None]*len(inputs)
                    for i in order:
                        sid=sequence[i]
                        expected[i]=step(control,control,inputs[i],upstream[i],streams[sid])
                        streams[sid].synchronize()
                for name in ('_x_amax_state','_weight_amax_state','_next_x_amax_state','_next_weight_amax_state'):
                    torch.testing.assert_close(getattr(layer,name),getattr(control,name),rtol=0,atol=0)
                for i,(values,ordered) in enumerate(zip(actual,expected)):
                    with torch.no_grad():
                        x,dy=inputs[i],upstream[i]
                        references=((x.float()@layer.weight.float().T).bfloat16(),
                                    (dy.float()@layer.weight.float()).bfloat16(),
                                    (dy.float().T@x.float()).bfloat16())
                        for name,value,target,reference in zip(('y','dx','dw'),values,ordered,references):
                            torch.testing.assert_close(value,target,rtol=0,atol=0)
                            self.assertTrue(bool(value.isfinite().all()))
                            difference=value.float()-reference.float()
                            rms=float(difference.square().mean().sqrt())
                            rrms=float(reference.float().square().mean().sqrt())
                            maximum=float(difference.abs().max())
                            gate=.20 if name=='y' else .07
                            metrics=dict(compiled=use_compile,threaded=threaded,step=i,output=name,finite=True,
                                rms=rms,reference_rms=rrms,max_abs=maximum,
                                normalized_rms=rms/max(rrms,1e-12),floor=1e-12,
                                normalized_max=maximum/max(float(reference.float().abs().max()),1e-12),gate=gate)
                            print('DELAYED_SHARED_MODULE '+json.dumps(metrics),flush=True)
                            self.assertLess(metrics['normalized_rms'],gate)
                self.assertEqual(journal(),evidence)
                self.assertEqual(len(graphs),int(use_compile))
                self.assertIsNone(layer.weight.grad)
                print('DELAYED_SHARED_STATE '+json.dumps(dict(compiled=use_compile,
                    stream_sequence=sequence,acquisition_order=order,threaded=threaded,
                    history_exact=True,ordered_outputs_exact=True,
                    observations=16,graphs=len(graphs),scope='overlapping forward submissions; backward after completion')),
                    flush=True)

    def test_handoff_waits_for_pending_write_including_failed_submission(self):
        for failed in (False,True):
            with self.subTest(partial_failure=failed):
                events=[]
                def factory():
                    event=torch.cuda.Event();events.append(event);return event
                guard=DelayedHistoryExecution(event_factory=factory)
                states=[torch.zeros(16,device='cuda',dtype=torch.float32) for _ in range(4)]
                output=torch.empty_like(states[0])
                a,b=torch.cuda.Stream(),torch.cuda.Stream()
                torch.cuda.synchronize()
                pending=torch.cuda.Event()
                def submit():
                    with torch.cuda.stream(a), guard.submission(states,a):
                        # Bounded artificial pending work; no performance claim.
                        torch.cuda._sleep(100_000_000)
                        states[0].fill_(7)
                        pending.record(a)
                        if failed:raise RuntimeError('after enqueue')
                if failed:
                    with self.assertRaisesRegex(RuntimeError,'after enqueue'):submit()
                else:submit()
                self.assertFalse(pending.query(),'test must observe pending producer work')
                with torch.cuda.stream(b), guard.submission(states,b):
                    output.copy_(states[0])
                b.synchronize()
                self.assertTrue(pending.query())
                torch.testing.assert_close(output,torch.full_like(output,7),rtol=0,atol=0)
                self.assertEqual(len(events),1)
                with torch.cuda.stream(b), guard.submission(states,b):
                    states[0].fill_(9)
                self.assertEqual(len(events),1,'warm same-stream call created an event')
                with torch.cuda.stream(a), guard.submission(states,a):
                    output.copy_(states[0])
                a.synchronize()
                torch.testing.assert_close(output,torch.full_like(output,9),rtol=0,atol=0)
                self.assertEqual(len(events),2)
                print('DELAYED_HANDOFF '+json.dumps(dict(partial_failure=failed,
                    pending_producer_observed=True,events=2,exact_reads=[7,9],
                    torch=str(torch.__version__),cuda=torch.version.cuda,
                    device=torch.cuda.get_device_name())),flush=True)

    def test_warm_same_stream_guard_allows_graph_capture_and_replay(self):
        def forbidden():raise AssertionError('same-stream guard must not create events')
        guard=DelayedHistoryExecution(event_factory=forbidden)
        states=[torch.zeros(16,device='cuda',dtype=torch.float32) for _ in range(4)]
        stream=torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream), guard.submission(states,stream):
            states[0].add_(0)
        stream.synchronize()
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            with guard.submission(states,stream):states[0].add_(1)
        with torch.cuda.stream(stream):
            for _ in range(5):graph.replay()
        stream.synchronize()
        torch.testing.assert_close(states[0],torch.full_like(states[0],5),rtol=0,atol=0)
        print('DELAYED_GUARD_CAPTURE '+json.dumps(dict(replays=5,exact_value=5,
            event_operations=0,scope='same-stream primitive, not shared cross-stream graph replay')),flush=True)


if __name__=='__main__':unittest.main()
