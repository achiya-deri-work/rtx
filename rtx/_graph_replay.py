"""Bounded-cache building block: CUDA launch capture, never torch.compile.

Owns static inputs/outputs and scratch resources. Each invocation stages new
values and returns independent output storage. The private stream serializes
replays; the caller stream waits for publication. No model tensors are retained.
"""
from threading import RLock

import torch

from .runtime import retain_runtime_resources


class KernelReplay:
    def __init__(self, execute, values):
        with torch.cuda.device(values[0].device):
            self._initialize(execute, values)

    def _initialize(self, execute, values):
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError("warm kernel replay before external CUDA graph capture")
        self.execute = execute
        self.lock = RLock()
        self.device = values[0].device
        self.stream = torch.cuda.Stream(device=self.device)
        caller = torch.cuda.current_stream(self.device)
        self.stream.wait_stream(caller)
        # Warm the direct caller-stream path too: an external graph must record
        # kernels directly, not attempt a nested cudaGraphLaunch.
        with torch.no_grad():
            execute(*values)
        self.stream.wait_stream(caller)
        with torch.cuda.stream(self.stream), torch.no_grad():
            self.inputs = tuple(torch.empty_like(t, memory_format=torch.contiguous_format)
                                for t in values)
            for target, value in zip(self.inputs, values):
                target.copy_(value)
                value.record_stream(self.stream)
            with retain_runtime_resources() as owners:
                execute(*self.inputs)
                self.stream.synchronize()
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph, stream=self.stream):
                    self.outputs = execute(*self.inputs)
            self.owners = tuple(owners.values())
            # Persistent scratch factories allocate outside the graph pool.
            # Keep their pool alive alongside the captured runner references.
            from ._workspace import _pools
            self.pools = tuple(pool for index, pool in _pools.items() if index == self.device.index)
        caller.wait_stream(self.stream)

    def __call__(self, *values):
        with self.lock, torch.cuda.device(self.device), torch.no_grad():
            if torch.cuda.is_current_stream_capturing():
                return self.execute(*values)
            caller = torch.cuda.current_stream(self.device)
            self.stream.wait_stream(caller)
            with torch.cuda.stream(self.stream):
                for target, value in zip(self.inputs, values):
                    target.copy_(value)
                    value.record_stream(self.stream)
                self.graph.replay()
                # Functional custom ops cannot expose mutable graph-owned
                # storage. Copies also allow several outstanding forwards and
                # reverse-order backward without a graph per invocation.
                result = tuple(None if t is None else t.clone() for t in self.outputs)
                for tensor in result:
                    if tensor is not None:
                        tensor.record_stream(caller)
            caller.wait_stream(self.stream)
            return result
