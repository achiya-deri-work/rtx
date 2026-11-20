"""Active-prefix or paired compact-M ablation with frozen routes.

Generate a real-route fixture with benchmark_moe_varlen --prepare-only and the
frozen config sidecars. This driver verifies its input, upstream, weight and
hard-route hashes against the archived result before timing. No router, shared
MLP or optimizer is included; whole-layer timing is a separate experiment.
"""

import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import statistics


def variants(cfg):
    p = cfg.prepared
    if (p.active_prefix or p.forward_glu_engine not in ("fragment_direct", "fragment_packed")
            or (p.gate_dw_engine, p.down_dw_engine) != ("persistent", "persistent")):
        raise ValueError("Expected a static incumbent with fragment GLU and both persistent dWs")
    trimmed = replace(p, active_prefix=True, forward_glu_engine="legacy",
                      gate_dw_engine="packed", down_dw_engine="packed")
    return {
        "incumbent": cfg,
        "active_only": replace(cfg, prepared=trimmed),
        "active_fragment": replace(cfg, prepared=replace(trimmed, forward_glu_engine=p.forward_glu_engine)),
        "active_both": replace(cfg, prepared=replace(p, active_prefix=True)),
    }


def compact_variants(cfg, pools):
    """Keep the active/token incumbent identical in both legs except forward M."""
    baseline = replace(cfg, combine_reduce_engine='token', dx_reduce_engine='token',
                       prepared=replace(cfg.prepared, active_prefix=True))
    candidate = replace(baseline, prepared=replace(baseline.prepared,
                        forward_row_layout='compact', compact_pool_count=pools))
    return {'active_original': baseline, 'active_compact': candidate}


def run(args):
    import torch
    import rtx
    from benchmarks.benchmark_moe_composition import compare
    from benchmarks.benchmark_moe_execution import _python_source_digest, _tensor_digest, write
    from benchmarks.benchmark_moe_joint_recompute import _capture
    from benchmarks.benchmark_moe_varlen import compile_options
    from benchmarks.profile_moe import hardware_attributes, statistics_ms
    from rtx._moe_reference import operational_reference
    from rtx.autotune.measurement import measure_cuda_batches
    from rtx.autotune.preparation_runtime import _warm_measurement_graph
    from rtx._moe_execution import layout
    from rtx.moe import (BalancedMoE, MXFP8MoEConfig, mxfp8_routed_experts,
                        _config_key, _dispatch_experts, _packed_expert_spec, _selection_spec)
    from rtx.routing import BalancedRouting

    source = Path(__file__).resolve().parents[1]
    assert Path(rtx.__file__).resolve().is_relative_to(source), rtx.__file__
    digest = _python_source_digest(source)
    script_digest = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    saved = json.loads(args.baseline_result.read_text())
    data = torch.load(args.fixture, map_location="cuda", weights_only=True)
    case = saved["case"]
    d, e, h, k = (case[name] for name in ("d_model", "num_experts", "expert_hidden_dim", "top_k"))
    cfg = MXFP8MoEConfig.from_dict(saved["expert_config"])
    # Use the original virtual packed parameters and their separate optimizer
    # leaves, not a different benchmark with only two packed leaf gradients.
    model = BalancedMoE(d, e, h, top_k=k, shared_hidden_features=case["shared_hidden_dim"],
                        config=cfg, num_shared_experts=0, device="cuda", autotune="off")
    with torch.no_grad():
        model.gate_up.weight.copy_(data["gu"])
        model.down.weight.copy_(data["down"])
    x, dy = data["x"].detach().requires_grad_(), data["dy"]
    ids, inverse = data["ids"], data["inverse"]
    weights = data["weights"].detach().requires_grad_()
    route = BalancedRouting(ids, weights, inverse)
    params = tuple(model.named_parameters())
    expert_params = [(name, value) for name, value in params if name.startswith(("gate_up.", "down."))]
    leaves = (x, *(value for _, value in expert_params), weights)
    names = ("output", "input", *(name for name, _ in expert_params), "route_weights")
    identities = saved["identities"]
    hashes = {key: _tensor_digest(value) for key, value in zip(
        ("input", "upstream", "expert_ids", "inverse", "route_weights"),
        (x, dy.bfloat16(), ids, inverse, weights), strict=True)}
    hashes["parameters"] = {name: _tensor_digest(value) for name, value in expert_params}
    assert hashes["input"] == identities["input"] and hashes["upstream"] == identities["upstream"]
    assert all(hashes[key] == identities["routing"][key] for key in ("expert_ids", "inverse"))
    assert all(value == identities["parameters"][name] for name, value in hashes["parameters"].items())
    assert len(expert_params) == 3 * e and x.shape == (sum(saved["lengths"]), d)
    c = ids.shape[1]
    active_rows = min(c, (int(inverse.remainder(c).max()) + 32) // 32 * 32)
    compact = getattr(args, 'compact_forward', False)
    pools = len(saved['lengths'])
    configs = compact_variants(cfg, pools) if compact else variants(cfg)
    options = compile_options(saved.get("inductor_max_autotune", True))
    graphs, checks, live, arena, extra_peak = {}, {}, {}, {}, {}
    stream = torch.cuda.current_stream()
    expected = changed = None
    x_original, dy_original = x.detach().clone(), dy.clone()

    def make_call(compiled, backward):
        def call():
            if backward:
                for leaf in leaves:
                    leaf.grad = None
                out = compiled(x, weights)
                out.backward(dy)
                return (out, *(leaf.grad for leaf in leaves))
            return (compiled(x, weights),)
        return call

    for label, option in configs.items():
        print("VALIDATE", label, flush=True)
        spec = _packed_expert_spec(_selection_spec(option, 'off', None, None), pools) if compact else None
        arena[label] = {}
        for phase in ('forward', 'backward'):
            _, dtypes, sizes, views, outputs = layout((x.shape[0], d, e, c, k, h), _config_key(option), phase)
            width = {dtype: torch.empty((), dtype=dtype).element_size() for dtype in dtypes}
            arena[label][phase] = sum(width[dt] * size for dt, size in zip(dtypes, sizes))

        def native(value, route_weights):
            if compact:
                return _dispatch_experts(value, model.gate_up.weight, model.down.weight,
                                         ids, route_weights, inverse, spec, True)
            return mxfp8_routed_experts(value, model.gate_up.weight, model.down.weight,
                                       BalancedRouting(ids, route_weights, inverse), config=option, autotune="off")

        # Bind this cell immediately and capture both workloads before the
        # next policy is bound. Graph replays contain no Python closure calls.
        compiled = torch.compile(native, **options)
        checks[label] = {}
        extra_peak[label] = {}
        for phase in ("forward", "step"):
            call = make_call(compiled, phase == "step")
            graph, output = _capture(call, stream)
            graphs[label, phase] = graph, output
            if compact:
                for leaf in leaves:
                    leaf.grad = None
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                start = torch.cuda.memory_allocated()
                temporary = call()
                torch.cuda.synchronize()
                extra_peak[label][phase] = torch.cuda.max_memory_allocated() - start
                del temporary
        graph, output = graphs[label, "step"]
        graph.replay()
        if expected is None:
            expected = tuple(value.clone() for value in output)
        checks[label]["incumbent"] = compare(output, expected)
        torch.cuda.synchronize()
        with torch.no_grad():
            x.copy_(x_original).mul_(-.8125)
            dy.copy_(dy_original).mul_(.6875)
        graph.replay()
        if changed is None:
            changed = tuple(value.clone() for value in output)
            # Both forward and input-gradient results must react to the two
            # changed input buffers; later cells compare every leaf to this.
            assert not torch.equal(changed[0], expected[0]) and not torch.equal(changed[1], expected[1])
        checks[label]["changed_input_upstream"] = compare(output, changed)
        with torch.no_grad():
            x.copy_(x_original)
            dy.copy_(dy_original)
        graph.replay()
        checks[label]["restored_input_upstream"] = compare(output, expected)
        if compact:
            assert not any(row['unequal'] for rows in checks[label].values() for row in rows)
        del call
        gc.collect()
        torch.cuda.synchronize()
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        live[label] = []
        for _ in range(8):
            graph.replay()
            torch.cuda.synchronize()
            live[label].append(torch.cuda.memory_allocated())
        assert len(set(live[label])) == 1, (label, live[label])

    with torch.no_grad():
        y, dx, dgu, ddown, dr = operational_reference(x, model.gate_up.weight, model.down.weight, *route, dy)
        oracle = (y, dx, *(grad for pair in dgu.unbind() for grad in pair.chunk(2, dim=0)), *ddown.unbind(), dr)
        for label in configs:
            graph, output = graphs[label, "step"]
            graph.replay()
            checks[label]["independent_quantized_fp64"] = compare(output, oracle)
        del oracle, y, dx, dgu, ddown, dr
    del expected, changed

    rounds = {label: {phase: [] for phase in ("forward", "step")} for label in configs}
    order = list(graphs)
    for index in range(args.visits):
        for label, phase in (order if index % 2 == 0 else reversed(order)):
            graph = graphs[label, phase][0]
            _warm_measurement_graph(graph)
            values, meta = measure_cuda_batches(graph.replay, samples=args.samples,
                                                  calls_per_sample=1, min_batch_ms=5., device=x.device)
            row = dict(visit=index, **statistics_ms(values), measurement=meta)
            rounds[label][phase].append(row)
            print("TIME", index, label, phase, row["median_ms"], "ms", flush=True)
    assert _python_source_digest(source) == digest
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest() == script_digest
    write(args.output, dict(status="complete", hardware=hardware_attributes(), torch=str(torch.__version__),
        runtime_digest=digest, script_digest=script_digest, baseline_result=str(args.baseline_result.resolve()),
        fixture=str(args.fixture.resolve()), fixture_sha256=hashlib.sha256(args.fixture.read_bytes()).hexdigest(),
        hashes=hashes, shape=(x.shape[0], d, e, c, k, h), active_rows=active_rows, lengths=saved["lengths"],
        seed=saved["seed"], names=names, configs={label: value.to_dict() for label, value in configs.items()},
        checks=checks, compile_options=options, live_allocated_bytes=live, rounds=rounds,
        planner_arena_bytes=arena, incremental_peak_allocated=extra_peak,
        medians_ms={label: {phase: statistics.median(row["median_ms"] for row in rows)
                            for phase, rows in phases.items()} for label, phases in rounds.items()}))


def main():
    import torch

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-result", type=Path, required=True)
    p.add_argument("--fixture", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--samples", type=int, default=7)
    p.add_argument("--visits", type=int, default=4)
    p.add_argument("--compact-forward", action="store_true",
                   help="Compare active+token original versus compact M on the same F+B recipe")
    args = p.parse_args()
    if args.visits < 2 or args.samples < 3 or args.output.exists():
        p.error("At least two visits, three samples and a new output are required")
    with torch.cuda.stream(torch.cuda.Stream()):
        run(args)


if __name__ == "__main__":
    main()
