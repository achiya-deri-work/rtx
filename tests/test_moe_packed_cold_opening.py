"""Journaled cold-opening coverage and the exact new template on device."""

from dataclasses import replace

import pytest
import torch

from rtx._routing_varlen import workspace_bounds
from rtx.autotune.effort import AutotunePolicy
from rtx.autotune.moe import make_moe_adapter, make_moe_tuner
from rtx.autotune.outcomes import TrialOutcome
from rtx.autotune.store import ResidualTuningStore
from rtx.moe import DEFAULT_MOE_CONFIG


POLICY = AutotunePolicy(max_trials=8, max_seconds=60, search_schedule='progressive',
    candidate_mix='mixed', confirmation_repeats=0, pool_size=24, structural_proposals=24)
BOUNDS = ('bounded_gather', 'bounded_dy', 'bounded_xt')


def _shape(b, length, d, e, k, h):
    n = b * length
    return (n, d, e, workspace_bounds(n, b, e, k)[1], k, h)


def _engine(path, shape, pools, *, policy=POLICY, initial=DEFAULT_MOE_CONFIG):
    tags = {'batch_contract': {'layout': 'regular' if pools == 1 else 'independent_varlen', 'sequences': pools,
                               'segment_alignment': 32}} if pools else {}

    def evaluate(config):
        p = config.prepared
        # Artificial objective to demonstrate actual admissions and selection.
        ms = 2. - .1 * p.active_prefix - .1 * sum(getattr(p, s) for s in BOUNDS)
        ms -= .1 * (p.forward_row_layout == 'compact')
        return TrialOutcome('ok', median_ms=ms, timings_ms=[ms] * 3)

    adapter = make_moe_adapter(shape, evaluate, tags=tags, initial=initial)
    store = ResidualTuningStore(path)
    return adapter, make_moe_tuner(adapter, store, policy), lambda: list(store.records(adapter.context))


def _matched_config(config):
    return replace(config, prepared=replace(config.prepared, active_prefix=False,
        bounded_gather=False, bounded_dy=False, bounded_xt=False,
        forward_row_layout='original', compact_pool_count=0))


@pytest.mark.parametrize('b,length,d,h', [(16, 1024, 384, 192), (16, 1024, 512, 256), (8, 1152, 384, 192)])
def test_eight_real_admissions_compare_identical_ownership(b, length, d, h, tmp_path):
    adapter, tuner, get_rows = _engine(tmp_path, _shape(b, length, d, 128, 6, h), b)
    result = tuner.tune()
    rows = get_rows()
    assert len(rows) == len({r['config_id'] for r in rows}) == 8
    assert [r['strategy'] for r in rows[:7]] == ['initial', *(['structural'] * 6)]
    assert len(tuner.coverage_proposals) == 6
    assert rows[7]['config_id'] not in {adapter.config_id(p.config) for p in tuner.coverage_proposals}
    assert all(r['outcome']['status'] == 'ok' for r in rows)
    legacy, static, prefix, bounded, compact, both = [adapter.deserialize(row['config']) for row in rows[1:7]]
    for config in (legacy, static, prefix, bounded, compact, both):
        p = config.prepared
        assert adapter.rejection(config) is None
        assert (p.input_preparation, p.dy_preparation, p.backward_preparation,
                p.route_gradient_epilogue) == ('indexed_transpose', 'fused', 'joint', 'fused')
        assert p.gate_dw_engine == p.down_dw_engine == 'persistent'
    old, new = legacy.prepared, static.prepared
    assert (old.forward_glu_engine, old.weight_preparation) == ('legacy', 'separate')
    assert (new.forward_glu_engine, new.weight_preparation) == ('fragment_packed', 'dual_orientation')
    assert all(getattr(old, field) == getattr(new, field)
               for field in ('down', 'down_dx', 'gate_dx', 'gate_dw', 'down_dw'))
    assert all((config.combine_reduce_engine, config.dx_reduce_engine) ==
               (static.combine_reduce_engine, static.dx_reduce_engine)
               for config in (legacy, prefix, bounded, compact, both))
    for config, flags in zip((static, prefix, bounded, compact, both), (
            (False, False, 'original'), (True, False, 'original'), (True, True, 'original'),
            (True, False, 'compact'), (True, True, 'compact')), strict=True):
        p = config.prepared
        active, bound, layout = flags
        assert p.active_prefix == active and p.forward_row_layout == layout
        assert all(getattr(p, field) == bound for field in BOUNDS)
        assert p.compact_pool_count == (b if layout == 'compact' else 0)
        assert _matched_config(config) == static
    assert adapter.config_id(result.config) == rows[6]['config_id']


def test_progressive_packed_opening_is_prefix_of_uninterrupted_search(tmp_path,monkeypatch):
    # This is a deterministic replay comparison between independent searches.
    # Hold proposal/evaluation cost fixed as well as the synthetic kernel ms;
    # otherwise unrelated host load legitimately changes the UCB rewards.
    from rtx.autotune import bandit
    monkeypatch.setattr(bandit, "observation_strategy_cost_s", lambda observation: 0.01)
    shape = _shape(8, 1152, 384, 128, 6, 192)
    _, short, rows = _engine(tmp_path/'resumed', shape, 8)
    short.tune()
    keys = [r['config_id'] for r in rows()]
    policy = replace(POLICY, max_trials=16)
    _, resumed, rows = _engine(tmp_path/'resumed', shape, 8, policy=policy)
    assert resumed.checkpoint_key == short.checkpoint_key
    resumed.tune()
    continued = [r['config_id'] for r in rows()]
    assert len(continued) == len(set(continued)) == 16 and continued[:8] == keys
    policy = replace(POLICY, max_trials=32)
    _, extended, rows = _engine(tmp_path/'resumed', shape, 8, policy=policy)
    assert extended.checkpoint_key == short.checkpoint_key
    extended.tune()
    extended_keys = [r['config_id'] for r in rows()]
    assert len(extended_keys) == len(set(extended_keys)) == 32 and extended_keys[:16] == continued
    _, full, rows = _engine(tmp_path/'full', shape, 8, policy=policy)
    full.tune()
    assert [r['config_id'] for r in rows()] == extended_keys


@pytest.mark.parametrize('shape,pools', [((16384, 384, 128, 768, 6, 192), 1),
    ((16384, 384, 128, 832, 6, 192), 16), (_shape(4, 128, 128, 8, 2, 64), 4),
    (_shape(16, 1024, 384, 32, 4, 320), 16)])
def test_flat_tight_and_small_openings_unchanged(shape, pools, tmp_path, monkeypatch):
    from rtx.autotune import moe

    _, actual, _ = _engine(tmp_path/'actual', shape, pools)
    monkeypatch.setattr(moe, 'cold_packed_padding_opening', lambda *a: ())
    _, old, _ = _engine(tmp_path/'old', shape, pools)
    assert actual.coverage_proposals == old.coverage_proposals
    assert actual.checkpoint_key == old.checkpoint_key


def test_compatible_explicit_prior_retains_its_opening(tmp_path, monkeypatch):
    from rtx.autotune.prior import FrozenPrior

    class Prior:
        def bundle(self,_):return True
        def retrieval(self,adapter,limit):
            return [adapter.update_fn(adapter.initial_config,'prepared.input_preparation','quantize_once')]
        def order(self,adapter,pool,policy):return pool

    monkeypatch.setattr(FrozenPrior,'load',lambda *a,**k:Prior())
    policy = replace(POLICY, prior_artifact='mock-compatible')
    _, tuner, rows = _engine(tmp_path, _shape(16, 1024, 384, 128, 6, 192), 16, policy=policy)
    tuner.tune()
    assert [row['strategy'] for row in rows()[:4]] == [
        'initial', 'initial_neighborhood', 'initial_neighborhood', 'retrieval']


def test_incomplete_matched_recipe_falls_back_without_new_checkpoint(tmp_path, monkeypatch):
    from rtx.autotune import moe

    shape = _shape(16, 1024, 384, 128, 6, 192)
    adapter, qualified, _ = _engine(tmp_path/'qualified', shape, 16)
    old_rejection = adapter.rejection_fn

    def rejection(config):
        p = config.prepared
        if p.forward_row_layout == 'compact' and all(getattr(p, field) for field in BOUNDS):
            return 'implementation_rejected', 'mock unsupported combination'
        return old_rejection(config)

    adapter.rejection_fn = rejection
    fallback = make_moe_tuner(adapter, ResidualTuningStore(tmp_path/'fallback'), POLICY)
    monkeypatch.setattr(moe, 'cold_packed_padding_opening', lambda *a: ())
    old = make_moe_tuner(adapter, ResidualTuningStore(tmp_path/'old'), POLICY)
    assert fallback.coverage_proposals == old.coverage_proposals
    assert fallback.checkpoint_key == old.checkpoint_key != qualified.checkpoint_key


@pytest.mark.skipif(not torch.cuda.is_available(), reason='SM120 CUDA required')
def test_exact_cold_bounded_compact_recipe_replays_changed_boundaries(tmp_path, record_property):
    from rtx._moe_reference import operational_reference
    from rtx.moe import _dispatch_experts, _packed_expert_spec, _selection_spec
    from rtx.runtime import retain_runtime_resources
    from tests.test_moe_active_prefix import _inputs, _route

    pools = 16
    shape = _shape(pools, 64, 128, 8, 2, 64)
    n, d, e, c, k, _ = shape
    _, tuner, _ = _engine(tmp_path, shape, pools)
    static = tuner.coverage_proposals[1].config
    candidate = tuner.coverage_proposals[5].config
    assert candidate.prepared.forward_row_layout == 'compact'
    assert all(getattr(candidate.prepared, name) for name in BOUNDS)

    def dispatch(config):
        spec = _packed_expert_spec(_selection_spec(config, 'off', None, None), pools)

        def call(x, a, b, ids, w, inv):
            return _dispatch_experts(x, a, b, ids, w, inv, spec, True)
        return call

    balanced = (64,) * pools
    ragged = (0, 1, 33, 127, 511, 352, *(0,) * 10)
    long = (1024, *(0,) * 15)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        x, gu, down = _inputs(n)
        ids, weights, inverse = _route(balanced, c, e, k)[0]
        operands = (x, gu, down, ids, weights, inverse)
        leaves = tuple(v.requires_grad_() for v in (x, gu, down, weights))
        dy = torch.randn((d, n), device='cuda', generator=torch.Generator(device='cuda').manual_seed(8302)).T
        native = torch.compile(dispatch(candidate), fullgraph=True, dynamic=False,
                               options={'triton.cudagraphs': False})
        control = dispatch(static)

        def step(call):
            y = call(*operands)
            return y, *torch.autograd.grad(y, leaves, dy)

        step(native)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with retain_runtime_resources(), torch.cuda.graph(graph, stream=stream):
            values = step(native)
        for idx, lengths in enumerate((balanced, ragged, long, balanced)):
            with torch.no_grad():
                current, _ = _route(lengths, c, e, k)
                for dest, src in zip((ids, weights, inverse), current, strict=True):
                    dest.copy_(src)
                zero = int(inverse[-1, -1])
                weights.flatten()[zero] = 0
                x.mul_(.8125)
                dy.mul_(-.875)
            graph.replay()
            expected = operational_reference(*operands, dy)
            reference = step(control)
            for field, a, b, ref in zip(('output','dx','dgu','ddown','route_dw'),
                                         values, reference, expected, strict=True):
                assert all(bool(v.isfinite().all()) for v in (a,b,ref))
                for label, target in (('native', b), ('oracle', ref)):
                    target = target.detach().float()
                    delta = a.detach().float() - target
                    nrms = float(delta.norm() / target.norm().clamp_min(1e-12))
                    maximum = float(delta.abs().max())
                    record_property(f'{idx}.{field}.{label}', dict(nrms=nrms, max_abs=maximum))
                    assert nrms <= (.003 if field == 'output' else .006) or maximum <= 2e-7
            assert torch.equal(ids, current[0]) and torch.equal(inverse, current[2])
            assert values[-1].flatten()[zero] != 0
            assert torch.count_nonzero(values[-1]) == n * k
            del expected, reference
        live = []
        for _ in range(5):
            graph.replay()
            stream.synchronize()
            live.append(torch.cuda.memory_allocated())
        assert len(set(live)) == 1
        record_property('live_bytes', live)
        graph.reset()
