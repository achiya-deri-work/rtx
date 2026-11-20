"""Byte-exact weighted-dY orientations, with independent BF16/scale oracle."""
from dataclasses import replace
import json

import pytest
import torch


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason='SM120 CUDA required')


def empty_pairs(e, cs, d):
    width = (d + 31) // 32 * 32
    return tuple(torch.empty(shape, device='cuda', dtype=dtype) for shape, dtype in (
        ((e * cs, width), torch.float8_e4m3fn), ((e * cs, width // 32), torch.uint8),
        ((e * d, cs), torch.float8_e4m3fn), ((e * d, cs // 32), torch.uint8),
    ))


def quantize_oracle(bf16, config):
    """Independent scale math, not the kernel's BF16 bitfield/redux path."""
    width = bf16.shape[-1]
    source = torch.nn.functional.pad(bf16, (0, (-width) % 32)).float()
    blocks = source.unflatten(-1, (-1, 32))
    amax = blocks.abs().amax(-1, keepdim=True)
    exponent = torch.ceil(torch.log2(amax / 448)).clamp(-127, 127)
    inverse = torch.exp2(-exponent.clamp_min(-126))
    if config.quant_math == 'bf16x2':
        normalized = (blocks * inverse.bfloat16().float()).bfloat16().float()
    else:
        normalized = blocks * inverse
    q = normalized.clamp(-448, 448).to(torch.float8_e4m3fn).flatten(-2)
    s = (exponent.squeeze(-1) + 127).to(torch.uint8)
    if config.scale_layout == 'mma128':
        e, rows, scale_cols = s.shape
        # [expert, 128-row tile, 4-scale tile, row16 group, row32 chunk, scale].
        s = s.reshape(e, rows // 128, 4, 32, scale_cols // 4, 4)
        s = s.permute(0, 1, 4, 3, 2, 5).contiguous().reshape(e, rows, scale_cols)
    return q.flatten(0, 1), s.flatten(0, 1)


def weighted_oracle(dy, mapping, weights, cs):
    n, d = dy.shape
    e, c = weights.shape
    token = mapping.view(e, cs)[:, :c].long()
    valid = (token >= 0) & (token < n)
    logical = (dy[token.clamp(0, n - 1)] * weights[..., None]).bfloat16()
    output = torch.zeros(e, cs, d, device=dy.device, dtype=torch.bfloat16)
    output[:, :c] = torch.where(valid[..., None], logical, 0)
    return output


@cuda
@pytest.mark.parametrize('n,d,e,c,cs,tile,vec,math,grid,layout,raster,offset', [
    (1, 1, 1, 1, 32, (32, 32), 1, 'fp32', 'operand', 'row_major', 'features', 0),
    (19, 35, 5, 29, 32, (64, 64), 2, 'bf16x2', 'expert', 'row_major', 'tokens', 1),
    (257, 63, 3, 71, 96, (32, 32), 4, 'fp32', 'portfolio', 'row_major', 'features', 1),
    (181, 384, 4, 167, 192, (128, 64), 8, 'fp32', 'operand', 'row_major', 'tokens', 0),
    (103, 128, 2, 75, 128, (256, 128), 8, 'bf16x2', 'expert', 'mma128', 'features', 0),
])
def test_routed_dy_quant_packed_values_and_scales(n, d, e, c, cs, tile, vec, math, grid, layout, raster, offset):
    from rtx.configs.moe import DEFAULT_PREPARED_MOE_CONFIG
    from rtx.kernels.mxfp8_moe_dy import compile_routed_dy_quant
    from rtx.kernels.mxfp8_moe_prepared import compile_expert_quant

    torch.manual_seed(20303)
    raw = torch.randn(n * d + offset, device='cuda', dtype=torch.float32)
    dy = raw[offset:].view(n, d)
    # This value crosses the E4M3 scaling threshold in FP32 but not after the
    # required BF16 cast. Also preserve sign of -0 on real assignments.
    dy[0].fill_(0)
    dy[0, 0] = 1.7501
    if d > 1:
        dy[0, 1] = -0.0
    weights = torch.randn(e, c, device='cuda', dtype=torch.float32)
    mapping = torch.randint(n, (e * cs, 1), device='cuda', dtype=torch.int32)
    mapping[::7] = -1
    if e > 1:
        mapping[cs:min(cs * 2, e * cs)] = -1
    mapping[0] = 0
    weights[0, 0] = 1
    cfg = replace(DEFAULT_PREPARED_MOE_CONFIG.quant, quant_vec=vec, quant_math=math,
                  load_bits=16 * vec, quant_store_bits=min(32, 8 * vec), scale_layout=layout,
                  transposed_tile_rows=tile[0], transposed_tile_k=tile[1],
                  reduction='shuffle' if vec <= 2 else 'redux')
    result = empty_pairs(e, cs, d)
    launch = compile_routed_dy_quant(n, d, e, c, cs, cfg, 8, raster, grid)
    launch(dy, mapping, weights, *result)

    materialized = weighted_oracle(dy, mapping, weights, cs)
    source = materialized.flatten(0, 1)
    baseline = empty_pairs(e, cs, d)
    compile_expert_quant(e, ((cs, d, 'row', False), (d, cs, 'transpose', False)), cfg,
                         'warp', grid, 8, raster, _byte_scales=True)(
        source, (source, source), (baseline[0], baseline[2]), (baseline[1], baseline[3]))
    oracle = (*quantize_oracle(materialized, cfg), *quantize_oracle(materialized.transpose(-1, -2), cfg))
    for output, previous, independent in zip(result, baseline, oracle):
        assert torch.equal(output.view(torch.uint8), previous.view(torch.uint8))
        assert torch.equal(output.view(torch.uint8), independent.view(torch.uint8))


@cuda
def test_routed_dy_quant_replays_caller_owned_outputs_with_updated_fp32_weights():
    from rtx.configs.moe import DEFAULT_PREPARED_MOE_CONFIG
    from rtx.kernels.mxfp8_moe_dy import compile_routed_dy_quant

    n, d, e, c, cs = 19, 64, 3, 23, 32
    torch.manual_seed(20311)
    dy = torch.randn(n, d, device='cuda', dtype=torch.float32)
    weights = torch.randn(e, c, device='cuda', dtype=torch.float32)
    mapping = torch.randint(n, (e * cs, 1), device='cuda', dtype=torch.int32)
    outputs = empty_pairs(e, cs, d)
    cfg = DEFAULT_PREPARED_MOE_CONFIG.quant
    call = compile_routed_dy_quant(n, d, e, c, cs, cfg)
    call(dy, mapping, weights, *outputs)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        call(dy, mapping, weights, *outputs)
    for factor in (-2.0, 0.5):
        weights.mul_(factor)
        graph.replay()
        materialized = weighted_oracle(dy, mapping, weights, cs)
        oracle = (*quantize_oracle(materialized, cfg), *quantize_oracle(materialized.transpose(-1, -2), cfg))
        for output, expected in zip(outputs, oracle):
            assert torch.equal(output.view(torch.uint8), expected.view(torch.uint8))


@cuda
@pytest.mark.parametrize('d,c,cs,tile,math,layout', [
    (35, 71, 96, (64, 64), 'fp32', 'row_major'),
    (128, 97, 128, (128, 32), 'bf16x2', 'mma128'),
])
def test_routed_full_backward_prepare_matches_separate_operands_on_replay(d, c, cs, tile, math, layout):
    from rtx.configs.moe import DEFAULT_PREPARED_MOE_CONFIG
    from rtx.kernels.mxfp8_moe_backward_quant import compile_routed_backward_quant
    from rtx.kernels.mxfp8_moe_dy import compile_routed_dy_quant
    from rtx.kernels.mxfp8_moe_warp_quant import compile_indexed_expert_transpose

    n, e = 379, 3
    torch.manual_seed(20319)
    # Misalign both source pointers without changing their contiguous strides.
    x = torch.randn(n*d + 1, device='cuda', dtype=torch.bfloat16)[1:].view(n, d)
    dy = torch.randn(n*d + 1, device='cuda', dtype=torch.float32)[1:].view(n, d)
    dy[0].zero_();dy[0, 0] = 1.7501
    weights = torch.randn(e, c, device='cuda', dtype=torch.float32)
    mapping = torch.randint(n, (e, cs), device='cuda', dtype=torch.int32)
    mapping[:, c:] = -1
    mapping[:, ::7] = -1
    mapping[1].fill_(-1)
    mapping[0, 0] = 0;weights[0, 0] = 1
    mapping = mapping.view(e*cs, 1)

    cfg = replace(DEFAULT_PREPARED_MOE_CONFIG.quant, scale_layout=layout, quant_math=math,
                  transposed_tile_rows=tile[0], transposed_tile_k=tile[1], reduction='shuffle')
    common = (8, 'tokens', 'expert')
    f_all = compile_routed_backward_quant(n, d, e, c, cs, cfg, *common)
    f_dy = compile_routed_dy_quant(n, d, e, c, cs, cfg, *common)
    f_xt = compile_indexed_expert_transpose(n, d, e, cs, cfg, *common)

    def buffers():
        pair = empty_pairs(e, cs, d)
        return pair + tuple(torch.empty_like(t) for t in pair[2:])

    actual, separate = buffers(), buffers()

    def step():
        f_all(dy, mapping, weights, *actual[:4], x, *actual[4:])

    step()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        step()
    for _ in range(2):
        x.neg_();weights.neg_()
        graph.replay()
        f_dy(dy, mapping, weights, *separate[:4])
        f_xt(x, mapping, *separate[4:])
        weighted = weighted_oracle(dy, mapping, weights, cs)
        indices = mapping.view(e, cs).long()
        gathered = torch.where(indices[..., None] >= 0, x[indices.clamp_min(0)], 0)
        expected = (*quantize_oracle(weighted, cfg), *quantize_oracle(weighted.transpose(1, 2), cfg),
                    *quantize_oracle(gathered.transpose(1, 2), cfg))
        for value, existing, oracle in zip(actual, separate, expected):
            assert torch.equal(value.view(torch.uint8), existing.view(torch.uint8))
            assert torch.equal(value.view(torch.uint8), oracle.view(torch.uint8))


@cuda
def test_public_packed_fused_dy_first_hit_warm_cache_and_resume(tmp_path, monkeypatch):
    from rtx import _mlp_autotune, _moe_selection
    from rtx._routing_varlen import workspace_bounds
    from rtx.autotune import AutotunePolicy
    from rtx.autotune import moe as tuning
    from rtx.moe import BalancedMoE, DEFAULT_MOE_CONFIG
    from rtx.routing import RoutingKernelConfig

    torch.manual_seed(20317)
    torch.compiler.reset()
    d, e, k, h = 384, 24, 6, 192
    bounds = ((0, 65, 65, 194), (0, 63, 193, 257))
    shape = (bounds[0][-1], d, e, workspace_bounds(bounds[0][-1], 3, e, k)[1], k, h)
    a = tuning.make_moe_adapter(shape, None, initial=DEFAULT_MOE_CONFIG)
    initial = a.update_fn(a.initial_config, 'prepared.dy_preparation', 'fused')
    assert initial.prepared.dy_preparation == 'fused' and a.rejection(initial) is None
    policy = AutotunePolicy.preset('balanced', max_trials=8, max_seconds=120, warmup=1, samples=3,
                                   calls_per_sample=1, confirmation_repeats=0,
                                   search_schedule='progressive', candidate_mix='mixed')
    cache = tmp_path / 'cache'
    model = BalancedMoE(d, e, h, top_k=k, shared_hidden_features=256, router_dim=16, device='cuda',
                        config=initial, autotune='balanced', tuning_policy=policy, autotune_cache_dir=cache,
                        router_options={'backend': 'cute', 'autotune': 'off', 'config': RoutingKernelConfig(
                            qb_backend='native_column', metadata_backend='cute', quota_threshold=0,
                            packed_row_backend='warp', packed_column_cache_items=32)})
    with torch.no_grad():
        model.router.projection_weight.normal_(std=.05)
    leaves = tuple(model.named_parameters())
    names = ('output', 'input', *(name for name, _ in leaves))
    params = tuple(p for _, p in leaves)
    assert len(params) == 79
    fixtures = []
    for boundary in bounds:
        x = (torch.randn(boundary[-1], d, device='cuda', dtype=torch.bfloat16) * .3).requires_grad_()
        cu = torch.tensor(boundary, device='cuda', dtype=torch.int32)
        # Noncontiguous upstream BF16 is an ordinary public-autograd input.
        dy = torch.randn(d, boundary[-1], device='cuda', dtype=torch.bfloat16).T
        fixtures.append((x, cu, dy))

    frames, searches = [], []
    orig_tune = tuning.tune_moe

    def tracked(*args, **kwargs):
        result = orig_tune(*args, **kwargs)
        searches.append(dict(evaluated_trials=result.evaluated_trials, identity=result.session_id,
                             evidence=result.tuning_evidence, winner=result.config.to_dict()))
        return result

    monkeypatch.setattr(tuning, 'tune_moe', tracked)

    def backend(graph, inputs):
        frames.append(dict(expert_cold='rtx.mxfp8_routed_experts.default' in graph.code,
                           shared_cold='rtx.moe_shared_select.default' in graph.code))
        (tmp_path / f'dynamo_frame_{len(frames)}.txt').write_text(graph.code)
        return torch._inductor.compile(graph, inputs, options={'triton.cudagraphs': False})

    fn = torch.compile(model, backend=backend, fullgraph=True, dynamic=False)

    def run(fixture):
        x, cu, dy = fixture
        y = fn(x, cu_seqlens=cu)
        result = (y, *torch.autograd.grad(y, (x, *params), dy))
        assert all(value.dtype == ref.dtype and value.shape == ref.shape
                   for value, ref in zip(result, (x, x, *params)))
        assert bool(torch.stack([value.isfinite().all() for value in result]).all())
        return result

    def compare(other, reference):
        metrics = []
        for value, ref in zip(other, reference):
            difference = value.float() - ref.float()
            metrics.append(torch.stack((difference.abs().max(),
                                        difference.norm() / ref.float().norm().clamp_min(1e-12))))
        values = torch.stack(metrics).cpu().tolist()
        result = {name: dict(max_abs=maximum, nrms=nrms) for name, (maximum, nrms) in zip(names, values)}
        assert all(maximum < 2e-7 or nrms < .01 for maximum, nrms in values), result
        return result

    report = dict(seed=20317, source_revision=tuning.source_revision(), shapes=bounds, frames=frames,
                  searches=searches, torch=torch.__version__, cuda=torch.version.cuda,
                  device=str(torch.cuda.get_device_properties(0)), gradients=list(names[1:]), records=[])

    def save():
        (tmp_path / 'first_hit_report.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')

    refs = []
    for index, fixture in enumerate(fixtures):
        cold = tuple(value.detach().clone() for value in run(fixture))
        refs.append(cold)
        assert len(frames) == 2 * index + 1 and len(searches) == index + 1
        warm = run(fixture)
        metrics = compare(warm, cold)
        del warm
        assert len(frames) == 2 * index + 2 and len(searches) == index + 1
        assert frames[-2]['expert_cold'] and not frames[-1]['expert_cold'] and frames[-2]['shared_cold']
        assert not frames[-1]['shared_cold']
        report['records'].append(dict(warm_vs_cold=metrics))
        # Both contexts are now hot and are revisited below without retuning.
        save()

    torch.cuda.synchronize()
    live = []
    for i in range(8):
        del_values = run(fixtures[i % 2])
        del del_values
        torch.cuda.synchronize()
        live.append(torch.cuda.memory_allocated())
    report['post_warm_live_bytes'] = live
    save()
    assert max(live) == min(live) and len(frames) == 4 and len(searches) == 2
    expert_winners = list(cache.glob('runtime_winners/mxfp8_balanced_experts/*/*.json'))
    shared_winners = list(cache.glob('runtime_winners/mxfp8_recompute_mlp/*/*.json'))
    assert len(expert_winners) == len(shared_winners) == 2
    winner_bytes = {p: p.read_bytes() for p in (*expert_winners, *shared_winners)}

    # Forget only host selections. A new cold graph must load the durable
    # winners; it must not reenter search or change the cache documents.
    tuning._choices.clear()
    _mlp_autotune.SELECTIONS.clear()
    _moe_selection.clear_selection_cells()
    import rtx.autotune.mlp as shared_tuning

    def forbidden(*args, **kwargs):
        raise AssertionError('Durable balanced cache hit reentered tuning')

    with monkeypatch.context() as patch:
        patch.setattr(tuning, 'tune_moe', forbidden)
        patch.setattr(shared_tuning, 'tune_mxfp8_mlp', forbidden)
        for i, fixture in enumerate(fixtures):
            report['records'][i]['disk_cache_vs_cold'] = compare(run(fixture), refs[i])
            del_values = run(fixture)
            del del_values
    assert len(searches) == 2 and all(p.read_bytes() == original for p, original in winner_bytes.items())
    save()

    archives = list(cache.glob('moe_effort/**/observations.jsonl'))
    before = {p: len(p.read_text().splitlines()) for p in archives}
    assert len(before) == 2 and set(before.values()) == {8}
    x, cu, _ = fixtures[0]
    refined = model.autotune_experts_(x, cu_seqlens=cu, tuning_policy=replace(policy, max_trials=10))
    after = {p: len(p.read_text().splitlines()) for p in archives}
    report['resume'] = dict(before=sorted(before.values()), after=sorted(after.values()),
                            evaluated_trials=refined.evaluated_trials)
    report['records'][0]['refined_vs_original'] = compare(run(fixtures[0]), refs[0])
    save()
    assert refined.evaluated_trials == 2 and sorted(after.values()) == [8, 10]
    assert len(list(cache.glob('moe_effort/**/observations.jsonl'))) == 2
    _moe_selection.clear_selection_cells()
    torch.compiler.reset()
