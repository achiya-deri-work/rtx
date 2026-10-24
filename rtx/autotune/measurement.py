"""Timing-quality annotations shared by low-overhead runtime evaluators."""
import math


def measure_cuda_batches(call, *, samples, calls_per_sample, device=None,
                         min_batch_ms=2.0, max_calls=4096):
    """Amortize event/host jitter without changing the requested sample count.

    Calibration is excluded from timings. This preserves host-launch semantics:
    it does not silently substitute a CUDA graph. Actual repetition counts are
    returned for provenance and the existing stationarity gates still apply.
    """
    import torch
    if samples < 1 or calls_per_sample < 1 or min_batch_ms <= 0:
        raise ValueError("measurement counts and batch duration must be positive")
    calls = calls_per_sample
    def batch(count):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(count):
            call()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))
    with torch.cuda.device(device):
        for _ in range(5):
            elapsed = batch(calls)
            if elapsed >= min_batch_ms or calls >= max_calls:
                break
            calls = min(max_calls, max(calls*2, math.ceil(calls*min_batch_ms/max(elapsed, 1e-6))))
        # The calibration has warmed the actual selected launch sequence.
        timings = [batch(calls)/calls for _ in range(samples)]
    return timings, dict(timing_batch_revision=1, requested_calls_per_sample=calls_per_sample,
                        actual_calls_per_sample=calls, min_batch_ms=min_batch_ms)


def runtime_timing_metadata(timings):
    """Use the calibrated harness's existing limits without extra GPU samples.

    An unstable sample set remains raw evidence, not a numerical failure.
    Promotion consumes the explicit stationarity flag; fewer than three
    samples cannot establish stability. No retries or threshold relaxation.
    """
    from ..prequant_experiments import BenchmarkProtocol
    values=list(timings)
    finite=all(math.isfinite(v) and v>0 for v in values)
    if not finite or not values:
        quality=dict(stationary=False,reason='invalid_timing_samples')
    else:
        quality=BenchmarkProtocol(samples=len(values)).timing_quality(values,requested_samples=len(values))
        if len(values)<3:
            quality=dict(quality,stationary=False,reason='insufficient_timing_samples')
    return dict(sampling=dict(actual_samples=len(values),collection=dict(
        stationary=quality['stationary'],quality=quality)),runtime_timing_quality_revision=1)
