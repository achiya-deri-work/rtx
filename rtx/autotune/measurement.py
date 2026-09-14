"""Timing-quality annotations shared by low-overhead runtime evaluators."""
import math


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
