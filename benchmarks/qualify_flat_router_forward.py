"""Matched flat forward geometry qualification with fixed saved-state adjoint.

Use a fresh immutable source snapshot and the HOME GPU0 lock. Timings for
arms with nonidentical actual hard outputs are diagnostic, never promotion input.
"""
import argparse
import copy
from dataclasses import replace
import gc
import hashlib
import json
import math
from pathlib import Path
import statistics
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--forward', action='append', default=[], help='Threads,row groups,column groups')
    p.add_argument('--control-forward',help='Explicit forward geometry for a NEW control; archived inputs/parameters remain mandatory, archived route hashes do not describe this control')
    p.add_argument('--backward', help='Previously validated backward geometry, if not in the fixture')
    p.add_argument('--heldout', action='store_true')
    p.add_argument('--tune-trials', type=int)
    p.add_argument('--tuning-dir', type=Path)
    p.add_argument('--tune-initial', type=Path)
    p.add_argument('--candidate-router', type=Path)
    p.add_argument('--max-autotune', action=argparse.BooleanOptionalAction, default=True)
    args = p.parse_args()
    if args.tune_trials and not args.tuning_dir:
        p.error('--tune-trials requires an explicit --tuning-dir')
    import torch
    from benchmarks.benchmark_moe_varlen import initialize_fixture, compile_options
    from rtx._routing_execution import run_forward, run_backward
    from rtx.autotune.effort import AutotunePolicy
    from rtx.autotune.measurement import measure_cuda_batches, runtime_timing_metadata
    from rtx.autotune.preparation_runtime import _warm_measurement_graph
    from rtx.autotune.routing import RoutingEvaluator
    from rtx.moe import BalancedMoE
    from rtx.routing import RoutingKernelConfig, SinkhornKernelConfig
    from rtx.runtime import load_kernel_symbol

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(args.output)
    record = json.loads(args.fixture.read_text())
    if record['layout'] != 'flat':
        raise ValueError('Only a single global flat routing pool is supported')
    case, init, lengths = (record[key] for key in ('case', 'initialization_case', 'lengths'))
    if args.heldout:
        case = dict(case, id='d384_e56_k4_h320_s256', num_experts=56, shared_hidden_dim=256)
        init, lengths = dict(case, expert_hidden_dim=128), [1792] * 8
    n, e, k = sum(lengths), case['num_experts'], case['top_k']
    if n*k % e:
        raise ValueError('Flat qualification needs a divisible pool without dummy rows')

    def read_config(path):
        config = json.loads(path.read_text())
        return RoutingKernelConfig.from_dict(config.get('config', config))

    def geometry(text):
        return SinkhornKernelConfig(*map(int, text.split(',')))

    base = RoutingKernelConfig.from_dict(record['router_config'])
    if args.control_forward:
        base = replace(base,forward=geometry(args.control_forward))
    if args.backward:
        base = replace(base, backward=geometry(args.backward))
    if args.tune_initial:
        if args.tune_trials:
            base = read_config(args.tune_initial)
        elif not (args.candidate_router and args.tuning_dir):
            raise ValueError('--tune-initial requires tuning or a candidate-router cache proof')
    configs = {'baseline': base}
    for text in args.forward:
        configs[text] = replace(base, forward=geometry(text))
    if args.candidate_router:
        selected = read_config(args.candidate_router)
        if replace(selected, forward=base.forward) != base:
            raise ValueError('Candidate must differ from control only in forward geometry')
        configs['selected'] = selected

    affinity, seed = (record[key] for key in ('router_affinity_layout', 'seed'))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)

    def construct(c):
        return BalancedMoE(c['d_model'], c['num_experts'], c['expert_hidden_dim'], top_k=c['top_k'],
                           shared_hidden_features=c['shared_hidden_dim'], device='cuda', autotune='off',
                           router_options=dict(backend='cute', autotune='off', config=base, affinity_layout=affinity))

    def digest(t):
        payload = t.detach().contiguous().cpu().reshape(-1).view(torch.uint8).numpy().tobytes()
        return hashlib.sha256(payload).hexdigest()

    model, x, cu, dy = initialize_fixture(construct, case, init, lengths, layout='flat', seed=seed,
        input_profile=record['input_profile'], stable_inputs=record['stable_inputs'],
        stable_experts=record['stable_experts'])
    assert cu is None
    router = model.router
    steps = router.sinkhorn_iterations
    identities = dict(input=digest(x), upstream=digest(dy),
                      parameters={key: digest(t) for key, t in model.named_parameters()})
    if not args.heldout:
        for key, value in identities.items():
            assert value == record['identities'][key], key
    source = Path(__file__).resolve().parents[1]
    source_hash = hashlib.sha256()
    for path in sorted((source/'rtx').rglob('*.py')):
        source_hash.update(str(path.relative_to(source)).encode())
        source_hash.update(path.read_bytes())
    result = dict(fixture=str(args.fixture), case=case, initialization_case=init, lengths=lengths,
        layout='flat', heldout=args.heldout, seed=seed, identities=identities,
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        sms=torch.cuda.get_device_properties(0).multi_processor_count,
        affinity=affinity, source=str(source), source_digest=source_hash.hexdigest(),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        explicit_control_forward=args.control_forward,archived_router_config=record['router_config'],
        solver=dict(sinkhorn=steps, dual=router.dual_iterations, repair=router.repair_rounds))

    def save():
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
        print(f'DONE {args.output}', flush=True)

    policy = AutotunePolicy.preset('balanced', max_trials=args.tune_trials or 8, max_seconds=150,
        samples=5, calls_per_sample=1, confirmation_repeats=0, search_schedule='progressive', candidate_mix='mixed')
    if args.tune_trials:
        start = time.monotonic()
        tuned = router.autotune_(x, tuning_policy=policy, output_dir=args.tuning_dir,
                                 progress=lambda value: print(value, flush=True))
        result['tuning'] = dict(config=tuned.config.to_dict(), evaluated_trials=tuned.evaluated_trials,
            elapsed_s=tuned.elapsed_s, invocation_s=time.monotonic()-start, evidence=tuned.tuning_evidence,
            initial=base.to_dict())
        save()
        return

    cap = n*k//e
    result['compile_options'] = options = compile_options(args.max_autotune)
    upstream = torch.randn((e, cap), device=x.device,
                           generator=torch.Generator(device=x.device).manual_seed(seed + 30593))
    x0, du0 = x.detach().clone(), upstream.clone()
    # Not merely a row permutation: some features and both scalar adjoints
    # change. The nearly constant upstream stresses small scalar derivatives.
    x1, du1 = x0.roll((1, 1), (0, 1)), 1 + upstream.roll(1, 1) * 0.0001

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
        assert bool(a.isfinite().all()) and bool(b.isfinite().all())
        delta = a.double() - b.double()
        rms, ref = delta.square().mean().sqrt().item(), b.double().square().mean().sqrt().item()
        return dict(max_abs=delta.abs().max().item(), rms=rms, ref_rms=ref, nrms=rms/max(ref, 1e-12),
                    **(dict(actual=a.item(), reference=b.item()) if a.numel() == 1 else {}))

    def compare_grad(a, b):
        metrics = compare(a, b)
        if a.numel() == 1:
            torch.testing.assert_close(a.double(), b.double(), rtol=2e-4, atol=1e-5)
        else:
            assert metrics['nrms'] < .006 or metrics['max_abs'] < 2e-7, metrics
        return metrics

    def transport(source):
        value = source.double()
        for _ in range(steps):
            value = value - value.logsumexp(1, keepdim=True) + math.log(k)
            value = value - value.logsumexp(0, keepdim=True) + math.log(cap)
        return value

    def soft_weights(value, inverse):
        experts = inverse//cap
        selected = value.gather(1, experts).softmax(1)
        return torch.empty(
            n*k, device=x.device, dtype=value.dtype).scatter(0, inverse.flatten(), selected.flatten()).view(e, cap)

    def independent_router_oracle(module, expert_ids, inverse):
        # Freeze only the nondifferentiable actual hard choice. Confirm the
        # two stable metadata views are exact inverses before using it.
        slots = inverse.flatten()
        assert torch.equal(slots.sort().values, torch.arange(n*k, device=x.device))
        assert torch.equal(expert_ids.flatten()[inverse], torch.arange(n, device=x.device)[:, None].expand(n, k))
        leaves = [value.detach().double().requires_grad_() for value in (x, *module.parameters())]
        inputs, projection, experts, logit_scale, bias = leaves

        def norm(a):
            return a * (a.square().mean(-1, keepdim=True) + module.rms_epsilon).rsqrt()

        logits = norm(inputs @ projection.T) @ norm(experts).T
        value = transport(torch.nn.functional.logsigmoid(logits * logit_scale.exp() + bias))
        weights = soft_weights(value, inverse)
        return weights.detach(), torch.autograd.grad(weights, leaves, upstream.double())

    result['configs'] = {name: c.to_dict() for name, c in configs.items()}
    result['validation'], result['oracle'], result['timing'] = {}, {}, {}
    with torch.no_grad():
        scores = router.log_affinities(x).contiguous()
    evaluator = RoutingEvaluator(scores, json.loads(router._selection_spec), 'train', policy)
    compile_sinkhorn = load_kernel_symbol('routing_sinkhorn', 'compile_sinkhorn')
    arms, calibration_control, replay_control = {}, [], []
    grad_names = ['dx', *(name for name, _ in router.named_parameters())]
    for name, c in configs.items():
        print(f'QUALIFY {name}', flush=True)
        token = tuple(c.to_token())
        f = c.forward
        init_kernel, (row_kernel, col_kernel), _, _ = compile_sinkhorn(
            n, e, k, steps, f.threads, f.row_groups, f.column_groups)
        out = torch.empty_like(scores).view(-1)
        history = torch.empty(2*steps*n*e, device=x.device)
        row, column = torch.empty(n, device=x.device), torch.empty(e, device=x.device)
        flat_scores = scores.view(-1)
        init_kernel(flat_scores, column, row, history, out, 0)
        for step in range(steps):
            row_kernel(flat_scores, column, row, history, out, step)
            if step+1 != steps:
                col_kernel(flat_scores, row, column, history, out, step)

        def column_only():
            col_kernel(flat_scores, row, column, history, out, steps-1)

        arm = arms[name] = dict(column=capture(column_only)[0],
                                forward=capture(lambda: run_forward(scores, k, steps, token)),
                                column_storage=(row, column, history, out))
        _, (_, saved_history) = arm['forward']
        arm['forward'][0].replay()
        torch.testing.assert_close(history[-n*e:], saved_history[-n*e:], rtol=0, atol=0)
        result['oracle'][name] = []
        for index, (src, incoming, ref_value, ref_grad, _, _) in enumerate(evaluator.fixtures):
            actual_v, hh = run_forward(src, k, steps, token)
            actual_g = run_backward(incoming, hh, k, steps, token)
            torch.testing.assert_close(actual_v.double(), ref_value, rtol=5e-5, atol=5e-6)
            torch.testing.assert_close(actual_g.double(), ref_grad, rtol=5e-4, atol=8e-6)
            hard = evaluator.hard(actual_v.exp(), c)
            ids, weights, inverse = evaluator.metadata(actual_v, hard, c)
            expected_weights = soft_weights(ref_value, inverse)
            torch.testing.assert_close(weights.double(), expected_weights, rtol=5e-5, atol=5e-7)
            identity = dict(expert_ids=digest(ids), inverse=digest(inverse), assignments=digest(hard))
            if name == 'baseline':
                calibration_control.append(identity)
            result['oracle'][name].append(dict(identity=identity, actual_hard_matches=identity==calibration_control[index],
                transport=compare(actual_v, ref_value), score_adjoint=compare(actual_g, ref_grad),
                soft_weights=compare(weights, expected_weights)))

        module = copy.deepcopy(router)
        module.config = c
        if name == 'selected' and args.tune_initial:
            module.config, module.autotune = read_config(args.tune_initial), 'cache'
            module.tuning_policy, module.autotune_cache_dir = policy, str(args.tuning_dir)
        module._refresh_selection_spec()
        if name == 'selected' and args.tune_initial:
            from rtx._moe_selection import bound_config
            from rtx._routing_autotune import binding_identity
            from rtx.autotune.routing import cache_contract
            from rtx.autotune.winners import load_runtime_winner

            serialized = json.loads(module._selection_spec)
            serialized['workload']['real_tokens'] = n
            key, root, _ = cache_contract(scores, serialized, 'train', policy)
            persisted = load_runtime_winner(key, RoutingKernelConfig.from_dict, root=root)
            assert persisted == c, 'Cold winner missing or different in fresh process'
            bound_id = binding_identity(scores, k, steps, module._selection_spec, True, n)
            assert bound_config(*bound_id) is None
            first = module(x)
            del first
            assert RoutingKernelConfig.from_token(tuple(json.loads(bound_config(*bound_id)))) == c
            winner_file = root / key.relative_path
            result['cache_proof'] = dict(winner_file=str(winner_file), winner_sha256=hashlib.sha256(
                winner_file.read_bytes()).hexdigest(), selected=c.to_dict(), fresh_runtime_bind=True)
        compiled = torch.compile(module, **options)
        leaves = (x, *module.parameters())

        def router_step():
            route = compiled(x)
            return route, torch.autograd.grad(route.route_weights, leaves, upstream)

        arm['router'], graph_output = capture(router_step)
        arm['router_storage'] = graph_output, module
        records = []
        for index, changed in enumerate((False, True, False)):
            with torch.no_grad():
                x.copy_(x1 if changed else x0)
                upstream.copy_(du1 if changed else du0)
            arm['router'].replay()
            route, grads = graph_output
            ids, weights, inverse = route
            fresh, fresh_grads = router_step()
            for a, b in zip(route, fresh, strict=True):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            oracle_weights, oracle_grads = independent_router_oracle(module, ids, inverse)
            torch.testing.assert_close(weights.double(), oracle_weights, rtol=5e-5, atol=5e-7)
            identity = dict(expert_ids=digest(ids), inverse=digest(inverse))
            if name == 'baseline':
                replay_control.append(dict(identity=identity, weights=weights.detach().clone(),
                                           grads=tuple(g.detach().clone() for g in grads)))
                if not args.heldout and not changed and not args.control_forward:
                    assert identity == record['identities']['routing'], (name, identity)
            control = replay_control[index]
            matched = identity == control['identity']
            item = dict(changed=changed, identity=identity, actual_hard_matches=matched,
                weight_oracle=compare(weights, oracle_weights),
                oracle_grads={key: compare_grad(a, b) for key, a, b in zip(grad_names, grads, oracle_grads, strict=True)},
                replay_grads={key: compare_grad(a, b) for key, a, b in zip(grad_names, grads, fresh_grads, strict=True)})
            if matched:
                torch.testing.assert_close(weights, control['weights'], rtol=5e-5, atol=5e-7)
                item['matched_grads'] = {key: compare_grad(a, b) for key, a, b in
                                         zip(grad_names, grads, control['grads'], strict=True)}
            records.append(item)
            del route, grads, ids, weights, inverse, fresh, fresh_grads, oracle_weights, oracle_grads
        assert records[0]['identity'] == records[2]['identity'] != records[1]['identity']
        matched = all(q['actual_hard_matches'] for q in (*records, *result['oracle'][name]))
        print(f'EXACT {name} {matched}', flush=True)

        def allocation_samples(count):
            live = []
            for _ in range(count):
                value = router_step()
                del value
                gc.collect()
                stream.synchronize()
                live.append(torch.cuda.memory_allocated())
            return live

        # Fresh durable-cache binding executes one additional cold path.
        # Keep its allocator transitions visible and out of the fixed window.
        warm = allocation_samples(8) if args.tune_initial else []
        torch.cuda.reset_peak_memory_stats()
        live = allocation_samples(6)
        result['validation'][name] = dict(replays=records, eligible_for_matched_timing=matched,
            warmup_allocated_bytes=warm, live_allocated_bytes=live, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            reserved_bytes=torch.cuda.memory_reserved())
        if len(set(live)) != 1:
            result['failure'] = dict(arm=name, gate='steady_live_allocation', samples=live)
            save()
            raise AssertionError(live)
    evaluator.close()

    for visit in range(4):
        order = list(arms) if visit % 2 == 0 else list(reversed(arms))
        for name in order:
            for phase in ('column', 'forward', 'router'):
                graph = arms[name][phase]
                if phase == 'forward':
                    graph = graph[0]
                warmup = _warm_measurement_graph(graph)
                times, measurement = measure_cuda_batches(graph.replay, samples=7, calls_per_sample=1,
                                                         device=x.device, min_batch_ms=10)
                item = dict(visit=visit, median_ms=statistics.median(times), samples_ms=times,
                            warmup=warmup, **measurement, **runtime_timing_metadata(times))
                result['timing'].setdefault(name, {}).setdefault(phase, []).append(item)
    save()


if __name__ == '__main__':
    main()
