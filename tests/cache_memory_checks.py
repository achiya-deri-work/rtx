"""Synchronized post-warmup live-allocation evidence, not a timing harness."""
import gc
import json
import time

import torch


def wait_for_cuda_allocation_cleanup(expected, *, observed, timeout_s=1.0, poll_s=.005):
    """Observe bounded idle cleanup without executing work or clearing caches.

    Observed gradient-sized allocations can outlive CUDA synchronization and
    disappear during idle host cleanup. Retain every sample; the caller must
    still enforce its original allocation bound on the final sample.
    """
    if timeout_s < 0 or poll_s <= 0:
        raise ValueError('cleanup timeout must be nonnegative and poll interval positive')
    samples=[observed]
    deadline=time.monotonic()+timeout_s
    while samples[-1] > expected and time.monotonic() < deadline:
        time.sleep(poll_s)
        gc.collect()
        torch.cuda.synchronize()
        samples.append(torch.cuda.memory_allocated())
    return samples


def assert_stable_cuda_memory(case, step, *, label, iterations=20):
    # Warm the same call/gradient-release lifecycle sampled below. Do not flush
    # the allocator: reserved memory and live tensor ownership are different.
    for _ in range(3):
        step()
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    def snapshot(iteration):
        torch.cuda.synchronize()
        return dict(iteration=iteration, allocated=torch.cuda.memory_allocated(),
                    reserved=torch.cuda.memory_reserved(),
                    peak_allocated=torch.cuda.max_memory_allocated())

    samples = [snapshot(0)]
    for iteration in range(1, iterations + 1):
        step()
        if iteration % 5 == 0 or iteration == iterations:
            gc.collect()
            samples.append(snapshot(iteration))
    print('MEMORY ' + json.dumps(dict(label=label, samples=samples,
                                     torch=str(torch.__version__), cuda=torch.version.cuda,
                                     device=torch.cuda.get_device_name())), flush=True)
    baseline = samples[0]['allocated']
    for sample in samples[1:]:
        case.assertLessEqual(sample['allocated'], baseline, (label, samples))
    return samples
