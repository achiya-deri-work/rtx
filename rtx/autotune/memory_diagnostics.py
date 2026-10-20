"""Allocator evidence for disposable CUDA workers, not a global runtime hook."""
import json
from pathlib import Path


def install_oom_recorder(output, *, history_entries=10000):
    """Install a process-lifetime observer before allocations/tuning.

    PyTorch's observer has no removal API: use only in disposable workers.
    The callback runs before the allocating Python stack unwinds. Never launch
    kernels, synchronize, collect garbage, or clear caches from this callback.
    """
    import torch
    from ..runtime import runtime_cache_stats
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    torch.cuda.memory._record_memory_history(max_entries=history_entries)
    events = []

    def observer(*args):
        index = len(events)
        event = dict(index=index, allocator_arguments=[int(x) for x in args],
                     stage='allocator_observer_before_python_unwind')
        events.append(event)
        # Failure of diagnostic IO must not replace the original allocation
        # error. Keep both snapshot and metadata failure information if possible.
        try:
            event['caches'] = runtime_cache_stats()
            event['allocated'] = torch.cuda.memory_allocated()
            event['reserved'] = torch.cuda.memory_reserved()
            path = root / f'oom-{index:04d}.pickle'
            if path.exists():
                raise FileExistsError(path)
            torch.cuda.memory._dump_snapshot(str(path))
            event['snapshot'] = str(path)
        except Exception as exc:
            event['diagnostic_error'] = repr(exc)
        try:
            with (root / f'oom-{index:04d}.json').open('x') as stream:
                json.dump(event, stream, indent=2)
        except Exception as exc:
            event['metadata_error'] = repr(exc)

    torch._C._cuda_attach_out_of_memory_observer(observer)
    return events
