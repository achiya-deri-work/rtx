"""Matched joint recompute on populated packed backward arenas plus expert F+B."""

import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys


def competition(shape, base, *, narrow_only=False):
    from rtx.autotune.moe import joint_recompute_seeds

    p = base.prepared
    if (not (p.active_prefix and p.bounded_dy and p.bounded_xt and p.bounded_gather)
            or p.forward_row_layout != 'compact' or p.backward_preparation != 'joint'
            or (p.gate_dw_engine, p.down_dw_engine) != ('persistent', 'persistent')
            or (p.joint_recompute_engine, p.recompute.tile_m, p.recompute.tile_n,
                p.recompute.stages, p.recompute.glu_quant_vec, p.recompute.glu_scratch_swizzle)
            != ('dual_operand', 64, 64, 1, 4, 'xor_16b')):
        raise ValueError('Expected the exact improved packed incumbent')
    seeds = {(c.prepared.joint_recompute_engine, c.prepared.recompute.tile_n,
              c.prepared.recompute.stages): c for c in joint_recompute_seeds(shape, base)}
    configs = {'incumbent': seeds['dual_operand', 64, 1]}
    if narrow_only:
        configs['reuse_narrow'] = seeds['operand_reuse', 64, 1]
    else:
        configs.update(dual_wide=seeds['dual_operand', 128, 2],
                       reuse_wide=seeds['operand_reuse', 128, 1])
    assert configs['incumbent'] == base
    for config in configs.values():
        candidate = config.prepared
        assert replace(config, prepared=replace(candidate, recompute=p.recompute,
                       joint_recompute_engine=p.joint_recompute_engine)) == base
    return configs


def populated_component(operands, dy, config):
    """Populate the actual backward arena, stopping before aliases can clobber it."""
    import torch
    from rtx._moe_execution import allocate, compiled_plan, layout, plan
    from rtx.moe import _config_key

    x, gu, down, ids, weights, inverse = operands
    e, c = ids.shape
    shape = (*x.shape, e, c, inverse.shape[1], down.shape[2])
    key, dtypes, sizes, views, _ = layout(shape, _config_key(config), 'backward')
    builder, *_ = plan(key)
    buffers = allocate(x, dtypes, sizes, views)
    inputs = (x, gu.flatten(0, 1), down.flatten(0, 1), ids, weights, inverse, dy.contiguous())

    def resolve(ref):
        if isinstance(ref, int):
            return inputs[-ref-1] if ref < 0 else buffers[ref]
        return tuple(map(resolve, ref))

    functions = compiled_plan(key, x.device.index).__rtx_resources__
    prefix = []
    for step, fn in zip(builder.steps, functions, strict=True):
        arguments = resolve(step[-1])
        if step[1] == 'compile_expert_joint_recompute':
            assert len(arguments[-1]) == 10 and step[2][-2] > 0
            aux = arguments[-1]
            capacity = int(aux[-1].item())
            physical = config.prepared.physical_capacity(c)
            tile = config.prepared.recompute.tile_m
            alignment = step[2][-2]
            consumer_rows = min(physical, (capacity + alignment - 1) // alignment * alignment)
            extent = min(physical, (consumer_rows + tile - 1) // tile * tile)
            assert 0 < capacity <= extent < physical
            raw = tuple(value.view(torch.uint8).view(e, value.size(0)//e, value.size(1))
                        for value in aux[1:-1])
            spans = (extent, extent, extent, extent//32, 2*extent, 2*extent, extent, extent//32)
            live, tail = [], []
            for index, (value, width) in enumerate(zip(raw, spans, strict=True)):
                axis = 1 if index in (0, 1, 4, 5) else 2
                live.append(value.narrow(axis, 0, width))
                tail.append(value.narrow(axis, width, value.size(axis)-width))
            return (lambda: fn(*arguments)), raw, live, tail, dict(active_capacity=capacity,
                physical=physical, written_extent=extent, alignment=alignment, prefix=prefix,
                byte_shapes=[tuple(value.shape) for value in raw])
        fn(*arguments)
        prefix.append(step[1])
    raise ValueError('Expected bounded joint backward')


def _metrics(actual, reference, names):
    import torch

    result = {}
    for name, value, ref in zip(names, actual, reference, strict=True):
        assert value.shape == ref.shape
        assert bool(value.isfinite().all()) and bool(ref.isfinite().all()), name
        ref, value = ref.detach().float(), value.detach().float()
        delta = value - ref
        nrms = float(delta.norm()/ref.norm().clamp_min(1e-12))
        absolute = float(delta.abs().max())
        assert nrms <= (.003 if name == 'output' else .006) or absolute <= 2e-7, (name, nrms, absolute)
        result[name] = dict(nrms=nrms, max_abs=absolute, unequal=int(torch.count_nonzero(delta)))
    return result


def run(args):
    import torch
    import rtx
    from benchmarks.benchmark_moe_execution import _python_source_digest, _tensor_digest
    from benchmarks.benchmark_moe_varlen import compile_options, statistics_ms
    from benchmarks.profile_moe import hardware_attributes
    from benchmarks.study_moe_priors import atomic_json
    from rtx._moe_reference import operational_reference
    from rtx.autotune.measurement import measure_cuda_batches, runtime_timing_metadata
    from rtx.autotune.moe import source_revision
    from rtx.autotune.preparation_runtime import _warm_measurement_graph
    from rtx.moe import BalancedMoE, MXFP8MoEConfig, _dispatch_experts, _packed_expert_spec, _selection_spec
    from rtx.runtime import retain_runtime_resources

    source = Path(__file__).resolve().parents[1]
    assert Path(rtx.__file__).resolve().is_relative_to(source), rtx.__file__
    digest = _python_source_digest(source)
    spec = json.loads(args.spec.read_text())
    contract = spec['fixture_contract']
    assert digest == contract['baseline_runtime'], (digest, contract)
    assert hashlib.sha256(Path(contract['parent']).read_bytes()).hexdigest() == contract['parent_sha256']
    prior = json.loads(args.prior_expert_result.read_text())
    router = json.loads(args.prior_router_config.read_text())
    assert router['config'] == spec['router_config']
    assert router['affinity_layout'] == spec['router_affinity_layout']
    assert prior['configs']['winner'] == spec['expert_config']
    data = torch.load(args.fixture, map_location='cpu', weights_only=True)
    hashes = {name: _tensor_digest(data[name]) for name in ('x','dy','gu','down','ids','weights','inverse')}
    hashes['dy_bf16'] = _tensor_digest(data['dy'].bfloat16())
    assert hashes == prior['hashes'], (hashes, prior['hashes'])
    expected = spec['identities']
    assert hashes['x'] == expected['input'] and hashes['dy_bf16'] == expected['upstream']
    assert hashes['ids'] == expected['routing']['expert_ids'] and hashes['inverse'] == expected['routing']['inverse']

    base = MXFP8MoEConfig.from_dict(spec['expert_config'])
    case, lengths = spec['case'], spec['lengths']
    n, d, e, c, k, h = (sum(lengths), case['d_model'], case['num_experts'], data['ids'].shape[1],
                         case['top_k'], case['expert_hidden_dim'])
    shape = n, d, e, c, k, h
    assert tuple(prior['shape']) == shape and base.prepared.compact_pool_count == len(lengths)
    configs = competition(shape, base, narrow_only=args.narrow_only)
    stream = torch.cuda.current_stream()
    x, dy, gu, down, ids, weights, inverse = (
        data[name].cuda() for name in ('x','dy','gu','down','ids','weights','inverse'))
    x.requires_grad_(); weights.requires_grad_()
    model = BalancedMoE(d, e, h, top_k=k, num_shared_experts=0, config=base,
                        shared_hidden_features=case['shared_hidden_dim'],
                        device='cuda', dtype=torch.bfloat16, autotune='off')
    with torch.no_grad():
        model.gate_up.weight.copy_(gu)
        model.down.weight.copy_(down)
    params = tuple((name, p) for name, p in model.named_parameters() if name.startswith(('gate_up.', 'down.')))
    assert len(params) == 3*e
    assert all(_tensor_digest(p) == expected['parameters'][name] for name, p in params)
    names = ('output','input',*(name for name, _ in params),'route_weights')
    leaves = (x, *(p for _, p in params), weights)
    operands = (x, gu, down, ids, weights, inverse)
    options = compile_options(True)
    with torch.no_grad():
        y, dx, dgu, dd, dr = operational_reference(*operands, dy.float())
        oracle = (y, dx, *(g for pair in dgu for g in pair.chunk(2)), *dd, dr)
        dummy = torch.ones_like(weights, dtype=torch.bool).flatten()
        dummy[inverse.flatten()] = False
        dummy = dummy.view_as(weights)

    def capture(call):
        was_enabled = gc.isenabled()
        graph = torch.cuda.CUDAGraph()
        # Benchmark-only capture isolation: do not alter runtime GC behavior.
        gc.collect()
        gc.disable()
        try:
            with retain_runtime_resources(), torch.cuda.graph(graph, stream=stream):
                result = call()
        finally:
            if was_enabled:
                gc.enable()
        return graph, result

    def expert_step(config):
        dispatch = _packed_expert_spec(_selection_spec(config, 'off', None, None), len(lengths))

        def forward(value, route_weights):
            return _dispatch_experts(value, model.gate_up.weight, model.down.weight,
                                     ids, route_weights, inverse, dispatch, True)

        compiled = torch.compile(forward, **options)

        def step():
            out = compiled(x, weights)
            return out, *torch.autograd.grad(out, leaves, dy)
        return step

    graphs, checks, live, rounds, bindings = {}, {}, {}, {}, {}
    try:
        for label, config in configs.items():
            print('VALIDATE', label, flush=True)
            component, raw, populated, tails, bound = populated_component(operands, dy, config)
            for value in raw:
                value.fill_(255)
            component()
            if label == 'incumbent':
                original_component = tuple(value.clone() for value in populated)
            assert all(bool(value.eq(255).all()) for value in tails), label
            unequal = [int(torch.count_nonzero(a != b)) for a,b in zip(populated, original_component, strict=True)]
            populated_again = tuple(value.clone() for value in populated)
            for value in raw:
                value.fill_(119)
            component()
            assert all(bool(value.eq(119).all()) for value in tails), label
            assert all(torch.equal(a,b) for a,b in zip(populated,populated_again,strict=True)), label
            del populated_again
            graphs[label, 'component'] = capture(component)[0]
            # Graph pinning does not own these Python-created arena operands.
            bindings[label] = component
            call = expert_step(config)
            uncaptured = call()
            graph, values = capture(call)
            graphs[label, 'expert'] = graph
            graph.replay(); stream.synchronize()
            if label == 'incumbent':
                native = tuple(value.detach().clone() for value in values)
            assert all(v.dtype == torch.bfloat16 for v in values[1:-1])
            assert values[0].dtype == values[-1].dtype == torch.float32
            assert not bool(values[-1][dummy].count_nonzero())
            checks[label] = dict(oracle=_metrics(values, oracle, names), native=_metrics(values, native, names),
                replay=_metrics(values, uncaptured, names), component_unequal_bytes=unequal, component_binding=bound)
            assert not any(v['unequal'] for v in checks[label]['replay'].values())
            del uncaptured
            live[label] = []
            for _ in range(5):
                graph.replay(); stream.synchronize()
                live[label].append(torch.cuda.memory_allocated())
            assert len(set(live[label])) == 1

        for phase in ('component', 'expert'):
            rounds[phase] = {label: [] for label in configs}
            for visit in range(4):
                for label in (configs if visit % 2 == 0 else reversed(configs)):
                    graph = graphs[label, phase]
                    _warm_measurement_graph(graph)
                    values, meta = measure_cuda_batches(graph.replay, samples=7, calls_per_sample=16, device=x.device)
                    row = dict(**statistics_ms(values), **meta, **runtime_timing_metadata(values))
                    rounds[phase][label].append(row)
                    print('TIME', phase, visit, label, row['median_ms'], 'ms', flush=True)
        assert _python_source_digest(source) == digest
        atomic_json(args.output, dict(status='complete', hardware=hardware_attributes(), torch=str(torch.__version__),
            cuda=torch.version.cuda, runtime_digest=digest, cycle_gc_isolated=True, expert_source_revision=source_revision(),
            argv=sys.argv, script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), spec=str(args.spec.resolve()),
            fixture=str(args.fixture.resolve()), fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
            fixture_contract=contract, hashes=hashes, shape=shape, lengths=lengths, seed=spec['seed'], compile_options=options,
            configs={key: value.to_dict() for key,value in configs.items()}, checks=checks, rounds=rounds,
            live_allocated_bytes=live, max_allocated_bytes=torch.cuda.max_memory_allocated(),
            max_reserved_bytes=torch.cuda.max_memory_reserved(), medians_ms={phase: {
                label: statistics.median(v['median_ms'] for v in series) for label,series in rows.items()}
                for phase,rows in rounds.items()}))
        for label in configs.keys() - {'incumbent'}:
            atomic_json(args.output.with_suffix(f'.{label}.expert.json'), {'config': configs[label].to_dict(),
                        'measured_by': str(args.output.resolve()), 'baseline_runtime': digest})
    finally:
        for graph in graphs.values():
            graph.reset()


def main():
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    for name in ('spec', 'fixture', 'prior-expert-result', 'prior-router-config', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--narrow-only', action='store_true',
                   help='Use the 64×64/stage1 operand reuse as the only challenger')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    with torch.cuda.stream(torch.cuda.Stream()):
        run(args)


if __name__ == '__main__':
    main()
