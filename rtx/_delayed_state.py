"""Cold delayed-history bootstrap without mutating compiler-visible metadata.

The registry holds only weak storage references and initialization signatures,
never numerical history or owning tensor references. Live histories cannot be
evicted: losing their initialization marker would change delayed semantics.
"""
from threading import RLock
from contextlib import contextmanager
from dataclasses import dataclass, field
import torch
from torch.multiprocessing.reductions import StorageWeakRef


class DelayedHistoryBootstrap:
    def __init__(self):
        self.entries = {}
        self.lock = RLock()

    @staticmethod
    def _key(states):
        x_history, weight_history, next_x, next_weight = states
        return (x_history.device.type, x_history.device.index,
                tuple(sorted((x_history.data_ptr(), next_x.data_ptr()))),
                tuple(sorted((weight_history.data_ptr(), next_weight.data_ptr()))))

    def state_signature(self, *states):
        """Diagnostic metadata only; never consult this from a traced forward."""
        with self.lock:
            prior = self.entries.get(self._key(states))
            if prior is None or any(ref.expired() for ref in prior[1]):
                return None
            return prior[0]

    def _remember(self, signature, states):
        # Called under self.lock on cold transitions/copying, never warm lookup.
        stale = [key for key, (_, refs) in self.entries.items()
                 if any(ref.expired() for ref in refs)]
        for key in stale:
            del self.entries[key]
        self.entries[self._key(states)] = (
            signature, tuple(StorageWeakRef(s.untyped_storage()) for s in states),
        )

    def inherit_copy(self, source_states, copied_states):
        """Transfer readiness after deepcopy has copied the numerical buffers."""
        with self.lock:
            signature = self.state_signature(*source_states)
            if signature is None:
                return
            if len(copied_states) != 4 or any(
                old.shape != new.shape or old.dtype != new.dtype or old.device != new.device
                for old, new in zip(source_states, copied_states)
            ):
                raise ValueError('copied delayed histories must preserve shape, dtype and device')
            if len({s.untyped_storage()._cdata for s in copied_states}) != 4:
                raise ValueError('copied delayed histories require independent storage')
            self._remember(signature, copied_states)

    def prepare(self, x, weight, x_history, weight_history, next_x, next_weight,
                *, stream_id, policy_key):
        states = (x_history, weight_history, next_x, next_weight)
        if any(s.dtype != torch.float32 or s.ndim != 1 or s.numel() == 0 for s in states):
            raise ValueError('delayed histories must be nonempty FP32 vectors')
        if any(s.numel() != x_history.numel() or s.device != x.device for s in states):
            raise ValueError('delayed histories must have matching lengths and devices')
        if len({s.data_ptr() for s in states}) != 4:
            raise ValueError('delayed current and next histories must not alias')
        if len({s.untyped_storage()._cdata for s in states}) != 4:
            raise ValueError('managed delayed histories require independent storage; aliased views are unsupported')
        key=self._key(states)
        signature=(tuple(x.shape),tuple(weight.shape),stream_id,policy_key,x_history.numel())
        with self.lock:
            prior=self.entries.get(key)
            if prior is not None and prior[0]==signature and not any(r.expired() for r in prior[1]):
                return False
            if x.is_cuda and torch.cuda.is_current_stream_capturing():
                raise RuntimeError('prepare delayed NVFP4 with a real warmup before CUDA graph capture')
            # Only a cold state/shape/stream transition executes tensor ops.
            # A warm call reads metadata and returns; no device flag readback.
            with torch.no_grad():
                x_history.copy_(x.detach().float().abs().amax().expand_as(x_history))
                weight_history.copy_(weight.detach().float().abs().amax().expand_as(weight_history))
            self._remember(signature, states)
            return True


DELAYED_HISTORY_BOOTSTRAP = DelayedHistoryBootstrap()


@dataclass
class _HistorySubmission:
    refs: tuple
    lock: object = field(default_factory=RLock)
    stream: object | None = None
    event: object | None = None


class DelayedHistoryExecution:
    """Serialize managed history submissions without owning history tensors.

    The caller must hold ``submission`` across prepare, launch and publication.
    Same-stream warm calls perform no event operations. A cross-stream handoff
    records completion after prior submissions and queues a wait before the
    next submission. Graph replay bypasses Python: callers must still order
    replays sharing mutable histories, and migration during capture is rejected.
    """
    def __init__(self, *, event_factory=None, capturing=None):
        self.entries = {}
        self.lock = RLock()
        self.event_factory = event_factory or torch.cuda.Event
        self.capturing = capturing or self._capturing

    @staticmethod
    def _capturing(stream):
        with torch.cuda.stream(stream):
            return torch.cuda.is_current_stream_capturing()

    @contextmanager
    def submission(self, states, stream):
        key = DelayedHistoryBootstrap._key(states)
        with self.lock:
            entry = self.entries.get(key)
            if entry is None or any(ref.expired() for ref in entry.refs):
                # Reap on cold ownership lookup, not once per layer per warm
                # call: a long model must not scan every other layer's state.
                stale = [k for k, old in self.entries.items() if any(ref.expired() for ref in old.refs)]
                for k in stale:
                    del self.entries[k]
                entry = _HistorySubmission(tuple(StorageWeakRef(s.untyped_storage()) for s in states))
                self.entries[key] = entry
        with entry.lock:
            previous = entry.stream
            if previous is not None and (previous.device, previous.cuda_stream) != (stream.device, stream.cuda_stream):
                if self.capturing(previous) or self.capturing(stream):
                    raise RuntimeError('cannot migrate shared delayed histories between streams during CUDA graph capture')
                event = self.event_factory()
                event.record(previous)
                stream.wait_event(event)
                entry.event = event
            # Even a launch raising after partial enqueue owns those pending
            # operations; the next stream must wait for that submission too.
            entry.stream = stream
            yield


DELAYED_HISTORY_EXECUTION = DelayedHistoryExecution()
