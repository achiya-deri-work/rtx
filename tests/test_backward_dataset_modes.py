"""Graph measurements must not collide with historical host evidence."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from rtx.autotune.dataset import DatasetJob, DatasetManifest, _bwd_adapter, _bwd_harness
from rtx.prequant_experiments import ShapeSpec


def test_mode_manifest_roundtrip_and_validation():
    job = DatasetJob('mxfp8_bwd', (ShapeSpec(128, 128, 128),))
    graph = replace(job, tags={'backward_measurement_mode': 'cuda_graph'})
    host_manifest = DatasetManifest(name='mode-test', jobs=(job,))
    graph_manifest = replace(host_manifest, jobs=(graph,))
    assert host_manifest.digest != graph_manifest.digest
    assert DatasetJob.from_dict(graph_manifest.as_dict()['jobs'][0]) == graph
    with pytest.raises(ValueError, match='only applies'):
        replace(graph, family='mxfp8_prequant_fwd')
    with pytest.raises(ValueError, match='must be'):
        replace(job, tags={'backward_measurement_mode': 'auto'})


def test_backend_carries_authoritative_mode_into_context():
    shape = ShapeSpec(128, 128, 128)
    campaign = SimpleNamespace(device='cpu', hardware_profile={},
                               manifest=SimpleNamespace(seed=41))
    for mode in ('host', 'cuda_graph'):
        job = DatasetJob('mxfp8_bwd', (shape,),
                         tags={'backward_measurement_mode': mode})
        with patch('rtx.autotune.dataset.BwdBenchmarkHarness') as factory:
            _bwd_harness(campaign, job, shape, 'hot')
            assert factory.call_args.kwargs['measurement_mode'] == mode
        harness = SimpleNamespace(measurement_mode=mode, problem=shape.problem)
        with patch('rtx.autotune.dataset.make_mxfp8_bwd_adapter') as factory:
            _bwd_adapter(campaign, job, shape, 'hot', harness, {})
            tags = factory.call_args.kwargs['tags']
            if mode == 'host':
                assert tags == {}  # Historical default identity is unchanged.
            else:
                assert tags['backward_measurement_mode'] == mode
                assert tags['backward_measurement_revision'] == 1
        with pytest.raises(ValueError, match='disagrees'):
            _bwd_adapter(campaign, job, shape, 'hot',
                         SimpleNamespace(measurement_mode='other', problem=shape.problem), {})


def test_research_rejects_mode_before_device_probe():
    from tools.predictor_study import build_adapter
    with pytest.raises(ValueError, match='requires mxfp8_bwd'):
        build_adapter(dict(family='glu', shape={}, backward_measurement_mode='cuda_graph'), {}, 'cpu')


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_research_adapter_identity_and_measured_mode_cuda():
    from tools.research.matched_gpu import build_adapter, independent_initial
    task = dict(family='mxfp8_bwd', shape=dict(m=257, n=131, k=129),
                regime='hot', seed=6120)
    timing = dict(warmup=2, samples=3, calls_per_sample=3)
    identities = []
    for mode in ('host', 'cuda_graph'):
        adapter, _ = build_adapter({**task, 'backward_measurement_mode': mode}, timing, 'cuda:0')
        identities.append(adapter.context.identifier)
        independent_initial(adapter, task, 'cuda:0')
        outcome = adapter.evaluate(adapter.initial_config)
        assert outcome.status == 'ok', outcome.error
        assert outcome.metadata['measurement_mode'] == mode
        if mode == 'cuda_graph':
            assert outcome.metadata['graph_capture_ms'] > 0
        else:
            assert outcome.metadata['graph_capture_ms'] == 0
    assert identities[0] != identities[1]
