"""Paired measurement for a single global routing pool (HOME GPU0 only).

Run in a fresh process from an immutable source snapshot under the shared GPU lock.
"""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, required=True, help='Archived whole-step result.json')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--geometry', action='append', default=[], help='Backward threads,row groups,column groups')
    p.add_argument('--samples', type=int, default=7)
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--fixture-mode', choices=('same', 'heldout', 'small-packed'), default='same')
    p.add_argument('--tune-trials', type=int)
    p.add_argument('--tuning-dir', type=Path)
    p.add_argument('--tune-initial', type=Path, help='Cold seed configuration for an explicit tuning run')
    p.add_argument('--max-autotune', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    import torch
    from benchmarks.benchmark_moe_varlen import initialize_fixture, compile_options
    from rtx.autotune.effort import AutotunePolicy
    from rtx.autotune.measurement import measure_cuda_batches, runtime_timing_metadata
    from rtx.autotune.preparation_runtime import _warm_measurement_graph
    from rtx.autotune.routing import RoutingEvaluator
    from rtx._routing_execution import run_forward, run_backward
    from rtx.moe import BalancedMoE
    from rtx.routing import RoutingKernelConfig, SinkhornKernelConfig
    from rtx.runtime import load_kernel_symbol

    def tensor_digest(value):
        payload = value.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    record = json.loads(args.fixture.read_text())
    case, init_case, lengths = (record[key] for key in ('case', 'initialization_case', 'lengths'))
    if args.fixture_mode == 'heldout':
        if record['layout'] != 'flat':
            raise ValueError('Heldout is a single-global-pool fixture')
        case = dict(case, id='d512_e48_k6_h256_s512', d_model=512, num_experts=48, top_k=6,
                    expert_hidden_dim=256, shared_hidden_dim=512)
        init_case = dict(case, expert_hidden_dim=128)
        lengths = [1536] * 8
    elif args.fixture_mode == 'small-packed':
        if record['layout'] != 'packed':
            raise ValueError('Small-packed control needs a packed seed')
        lengths = [127, 257]
    base = RoutingKernelConfig.from_dict(record['router_config'])
    if args.tune_initial:
        if not args.tune_trials:
            raise ValueError('--tune-initial requires --tune-trials')
        initial = json.loads(args.tune_initial.read_text())
        base = RoutingKernelConfig.from_dict(initial.get('config', initial))
    affinity = record['router_affinity_layout']
    seed = record['seed']
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)

    def construct(c):
        return BalancedMoE(c['d_model'], c['num_experts'], c['expert_hidden_dim'],
                           top_k=c['top_k'], shared_hidden_features=c['shared_hidden_dim'],
                           device='cuda', autotune='off',
                           router_options=dict(backend='cute', autotune='off', config=base,
                                               affinity_layout=affinity))

    model, x, cu, dy = initialize_fixture(construct, case, init_case, lengths, layout=record['layout'],
        seed=seed, input_profile=record['input_profile'], stable_inputs=record['stable_inputs'],
        stable_experts=record['stable_experts'])
    r = model.router
    k, steps = r.top_k, r.sinkhorn_iterations
    n, e = sum(lengths), case['num_experts']
    param_ids = {key: tensor_digest(value) for key, value in model.named_parameters()}
    identity = dict(input=tensor_digest(x), upstream=tensor_digest(dy), parameters=param_ids)
    if args.fixture_mode == 'same':
        for key, value in identity.items():
            assert value == record['identities'][key], key
    policy = AutotunePolicy.preset('balanced', max_trials=args.tune_trials or 4, max_seconds=150,
        samples=5, calls_per_sample=1, confirmation_repeats=0, search_schedule='progressive', candidate_mix='mixed')
    source = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted((source/'rtx').rglob('*.py')):
        digest.update(str(path.relative_to(source)).encode())
        digest.update(path.read_bytes())
    result = dict(fixture=str(args.fixture), fixture_mode=args.fixture_mode, case=case, lengths=lengths,
        identities=identity, torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        sms=torch.cuda.get_device_properties(0).multi_processor_count, layout=record['layout'], affinity=affinity,
        source=str(source), source_digest=digest.hexdigest())
    if args.tune_trials:
        start = time.monotonic()
        tuned = r.autotune_(x, cu_seqlens=cu, tuning_policy=policy, output_dir=args.tuning_dir,
                            progress=lambda value: print(value, flush=True))
        result['tuning'] = dict(config=tuned.config.to_dict(), evaluated_trials=tuned.evaluated_trials,
            elapsed_s=tuned.elapsed_s, invocation_s=time.monotonic()-start, evidence=tuned.tuning_evidence,
            initial=base.to_dict())
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        return

    configs = dict(baseline=base)
    for token in args.geometry:
        geometry = SinkhornKernelConfig(*map(int, token.split(',')))
        configs[token] = replace(base, backward=geometry)
    if cu is not None and list(configs) != ['baseline']:
        raise ValueError('Backward geometry intervention only applies to a flat pool')
    torch.cuda.synchronize()
    options = compile_options(args.max_autotune)
    result['compile_options'] = options
    param_names = ['dx', *(name for name, _ in r.named_parameters())]
    generator = torch.Generator(device=x.device).manual_seed(seed + 30593)
    with torch.no_grad():
        route_shape = (e, n*k//e) if cu is None else r(x, cu_seqlens=cu).route_weights.shape
    upstream = torch.randn(route_shape, device=x.device, generator=generator)
    x0 = x.detach().clone()
    # Same fixed addresses, distinct inputs and upstreams, including an
    # almost constant route-weight adjoint with naturally cancelling scalars.
    x1 = x0.roll(1, 0)
    du0 = upstream.clone()
    du1 = 1 + upstream.roll(1, 1) * 0.0001

    def capture(call):
        for _ in range(3):
            out = call()
            del out
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            out = call()
        return graph, out

    def compare(a, b):
        error = (a.double() - b.double())
        assert bool(a.isfinite().all()) and bool(b.isfinite().all())
        rms = error.square().mean().sqrt().item()
        ref = b.double().square().mean().sqrt().item()
        return dict(max_abs=error.abs().max().item(), rms=rms, ref_rms=ref, nrms=rms/max(ref, 1e-12),
                    **(dict(actual=a.item(), reference=b.item()) if a.numel() == 1 else {}))

    def compare_gradient(a, b):
        metrics = compare(a, b)
        if a.numel() == 1:
            # Same FP32 scalar gates as the existing affinity replay tests.
            torch.testing.assert_close(a, b, rtol=2e-4, atol=1e-5)
        else:
            # Existing complete-layer replay gates; independent of BF16's
            # per-element relative error at zero or tiny gradients.
            assert metrics['nrms'] < .006 or metrics['max_abs'] < 2e-7, metrics
        return metrics

    if cu is None:
        with torch.no_grad():
            scores = r.log_affinities(x).contiguous()
        spec = json.loads(r._selection_spec)
        evaluator = RoutingEvaluator(scores, spec, 'train', policy)
        scores, score_gradient, expected, expected_gradient, _, _ = evaluator.fixtures[0]
        value, history = run_forward(scores, k, steps, tuple(base.to_token()))
        compile_sinkhorn = load_kernel_symbol('routing_sinkhorn', 'compile_sinkhorn')
        oracle = {}
        for name, c in configs.items():
            token = tuple(c.to_token())
            ge = c.backward
            launch = compile_sinkhorn(n, e, k, steps, ge.threads, ge.row_groups, ge.column_groups)[-1][0]
            out = torch.empty_like(scores).view(-1)
            unused = torch.empty(1, device=x.device)
            target = score_gradient.flatten()
            component, _ = capture(lambda: launch(target, unused, unused, history, out, steps-1))
            full_backward, gradient = capture(lambda: run_backward(score_gradient, history, k, steps, token))
            oracle[name] = dict()
            for index, (src, incoming, ref_value, ref_grad, _, _) in enumerate(evaluator.fixtures):
                actual_v, hh = run_forward(src, k, steps, token)
                actual_g = run_backward(incoming, hh, k, steps, token)
                torch.testing.assert_close(actual_v.double(), ref_value, rtol=5e-5, atol=5e-6)
                torch.testing.assert_close(actual_g.double(), ref_grad, rtol=5e-4, atol=8e-6)
                oracle[name][index] = dict(transport=compare(actual_v, ref_value), score_adjoint=compare(actual_g, ref_grad))
            configs[name] = dict(config=c, column=component, backward=full_backward,
                                 captured=(target, unused, history, score_gradient, out, gradient))
        evaluator.close()
        result['oracle'] = oracle
        del scores, score_gradient, expected, expected_gradient, value, history, evaluator
    else:
        configs = {name: dict(config=c) for name, c in configs.items()}

    def check_identity(route):
        identities = dict(expert_ids=tensor_digest(route.expert_ids), inverse=tensor_digest(route.inverse),
                          route_weights=tensor_digest(route.route_weights))
        if args.fixture_mode == 'same':
            for key in ('expert_ids', 'inverse'):
                assert identities[key] == record['identities']['routing'][key], key
        return identities

    # Use different module instances in one fresh process so each arm
    # retains an immutable fullgraph specialization and its own exact graph.
    import copy
    reference = None
    result['validation'], result['timing'], result['configs'] = {}, {}, {}
    for name, arm in configs.items():
        router = copy.deepcopy(r)
        router.config = arm['config']
        router._refresh_selection_spec()
        compiled = torch.compile(router, **options)
        local_leaves = (x, *router.parameters())
        def step():
            route = compiled(x, cu_seqlens=cu)
            return route, torch.autograd.grad(route.route_weights, local_leaves, upstream)
        with torch.no_grad():
            x.copy_(x0)
            upstream.copy_(du0)
        first, _ = step()
        routing_identity = check_identity(first)
        del first
        graph, graph_output = capture(step)
        arm['router'] = graph
        arm['router_capture'] = (graph_output, router)
        result['configs'][name] = router.config.to_dict()
        snapshots = []
        comparisons, replay_comparisons = [], []
        for changed in (False, True, False):
            with torch.no_grad():
                x.copy_(x1 if changed else x0)
                upstream.copy_(du1 if changed else du0)
            graph.replay()
            stream.synchronize()
            route, grads = graph_output
            current = tuple(value.detach().clone() for value in (*route, *grads))
            fresh_route, fresh_grads = step()
            fresh = (*fresh_route, *fresh_grads)
            for a, b in zip(current[:3], fresh[:3], strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            replay_comparisons.append(dict(changed=changed, **{key: compare_gradient(a, b)
                for key, a, b in zip(param_names, current[3:], fresh[3:], strict=True)}))
            if name == 'baseline':
                snapshots.append(current)
            else:
                control = reference[len(comparisons)]
                for i in range(3):
                    torch.testing.assert_close(current[i], control[i], rtol=0, atol=0)
                comparisons.append(dict(changed=changed, **{key: compare_gradient(a, b)
                    for key, a, b in zip(param_names, current[3:], control[3:], strict=True)}))
            del fresh, fresh_route, fresh_grads, current, route, grads
        if name == 'baseline':
            reference = snapshots
        live = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(6):
            out = step()
            del out
            gc.collect()
            stream.synchronize()
            live.append(torch.cuda.memory_allocated())
        assert len(set(live)) == 1, live
        result['validation'][name] = dict(comparisons=comparisons, replay_comparisons=replay_comparisons,
            routing_identity=routing_identity,
            replay_changed_inputs=True, live_allocated_bytes=live, peak_bytes=torch.cuda.max_memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved())

    for round_index in range(args.rounds):
        order = list(configs) if round_index % 2 == 0 else list(reversed(configs))
        for name in order:
            for phase in ('column', 'backward', 'router'):
                graph = configs[name].get(phase)
                if graph is None:
                    continue
                warmup = _warm_measurement_graph(graph)
                timings, measurement = measure_cuda_batches(graph.replay, samples=args.samples,
                    calls_per_sample=1, device=x.device, min_batch_ms=10)
                item = dict(round=round_index, median_ms=statistics.median(timings), samples_ms=timings,
                            warmup=warmup, **measurement, **runtime_timing_metadata(timings))
                result['timing'].setdefault(name, {}).setdefault(phase, []).append(item)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(f'DONE {args.output}', flush=True)


if __name__ == '__main__':
    main()
