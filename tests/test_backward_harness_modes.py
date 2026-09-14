"""Calibrated backward modes must measure and validate the same operation."""
import pytest
import torch

from rtx.bwd_experiments import BwdBenchmarkHarness
from rtx.prequant_experiments import BenchmarkProtocol, ShapeSpec


def test_invalid_mode_rejected_before_device_allocation():
    with pytest.raises(ValueError, match="measurement_mode"):
        BwdBenchmarkHarness(ShapeSpec(128, 128, 128), BenchmarkProtocol(),
                            device="cpu", measurement_mode="auto")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("regime", ["hot", "rotate"])
@pytest.mark.parametrize("shape", [(128, 128, 128), (257, 131, 129)])
def test_graph_batch_replay_reuse_and_numerics(regime, shape):
    from rtx.fp8_bwd import _default_bwd_config
    from tools.research.matched_gpu import independent_metrics
    protocol = BenchmarkProtocol(warmup_calls=2, min_calls_per_sample=3,
        max_calls_per_sample=3, target_batch_ms=1, telemetry=False,
        max_rotation_buffers=2, stabilization_target_ms=0,
        correctness_rtol=.07)
    harness = BwdBenchmarkHarness(ShapeSpec(*shape), protocol, regime=regime,
        seed=981, measurement_mode="cuda_graph")
    prepared = harness.prepare(_default_bwd_config(harness.problem))
    for offset in (0, 3, 0, 3):
        prepared.grad_x.fill_(float("nan"))
        prepared.grad_weight.fill_(float("nan"))
        assert harness._time_batch(prepared, 3, offset) > 0
        independent_metrics(prepared.grad_x, harness._reference[0], .07)
        independent_metrics(prepared.grad_weight, harness._reference[1], .07)
    assert len(prepared.graphs) == len(harness._inputs)
    capture_ms = prepared.graph_capture_ms
    harness._time_batch(prepared, 3, 0)
    assert prepared.graph_capture_ms == capture_ms
    assert capture_ms > 0
    if regime == "hot" and shape == (128, 128, 128):
        measured = harness.measure(prepared.config, samples=3, seed=981)
        assert measured["status"] == "ok"
        assert measured["measurement_mode"] == "cuda_graph"
        assert measured["graph_count"] == 1
        assert measured["graph_capture_ms"] > 0
        assert measured["calls_per_sample"] == 3
    # Host mode never silently reuses the graph even on a prepared candidate.
    harness.measurement_mode = "host"
    assert harness._time_batch(prepared, 3) > 0
    assert prepared.graph_capture_ms == capture_ms
