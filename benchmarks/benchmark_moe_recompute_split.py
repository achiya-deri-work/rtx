"""Matched joint recompute and routed expert F+B on a frozen learned route."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys


def _variants(config, labels):
    p = config.prepared
    if p.backward_preparation != 'joint' or p.joint_recompute_engine != 'dual_operand' or p.active_prefix:
        raise ValueError('Expected frozen dual-operand joint recompute at static capacity')
    recompute = p.recompute
    packed = replace(recompute, glu_quant_vec=4, glu_scratch_swizzle='xor_16b')
    variants = {
        'anchor': config,
        'vec4_xor': replace(config, prepared=replace(p, recompute=packed)),
    }
    for name, cfg in (('anchor', recompute), ('vec4_xor', packed)):
        for warps in range(1, recompute.num_threads // 32):
            variants[f'split{warps}_{name}'] = replace(config, prepared=replace(p,
                recompute=cfg, joint_recompute_engine='split_warp', joint_recompute_gradient_warps=warps))
    return {name: variants[name] for name in ('anchor', *[v for v in labels if v != 'anchor'])}


def _component_base_args(data, config):
    from rtx._moe_execution import allocate, layout, plan
    from rtx.moe import _config_key
    from rtx.runtime import load_kernel_symbol

    x, gu, down, ids, weights, inverse, dy = (data[k] for k in ('x','gu','down','ids','weights','inverse','dy'))
    n, d = x.shape
    e, c = ids.shape
    h, k = down.shape[-1], inverse.shape[-1]
    key, dtypes, sizes, views, _ = layout((n,d,e,c,k,h), _config_key(config), 'backward')
    builder, *_ = plan(key)
    buffers = allocate(x, dtypes, sizes, views)
    inputs = (x, gu.flatten(0,1), down.flatten(0,1), ids, weights, inverse, dy)

    def value(ref):
        if not isinstance(ref, int):
            return tuple(value(v) for v in ref)
        return inputs[-ref-1] if ref < 0 else buffers[ref]

    for module, symbol, args, refs in builder.steps:
        values = value(refs)
        if symbol == 'compile_expert_joint_recompute':
            # Exit before later plan steps reuse component inputs or outputs.
            return args[1], values
        load_kernel_symbol(module, symbol)(*args)(*values)
    raise ValueError('Expected a joint recompute launch')


def _components(torch, data, variants, args):
    from benchmarks.benchmark_moe_route_adjoint import measure
    from rtx.kernels.mxfp8_moe_buffers import compile_expert_joint_recompute

    names = ('qh','sh','qht','sht','qp','sp','qpt','spt')
    problem, values = _component_base_args(data, variants['anchor'])
    e = data['ids'].shape[0]
    outputs = values[-1][1:]
    calls = {}
    stream = torch.cuda.current_stream()
    for label, cfg in variants.items():
        p = cfg.prepared
        fn = compile_expert_joint_recompute(e, problem, p.recompute, p.quant,
                p.joint_recompute_engine, gradient_warps=p.joint_recompute_gradient_warps)
        calls[label] = lambda fn=fn: fn(*values)
    validation, graphs = {}, {}
    for label, call in calls.items():
        print(f'Validate and capture joint recompute: {label}', flush=True)
        for out in outputs:
            out.view(torch.uint8).fill_(255)
        call()
        if label == 'anchor':
            expected = tuple(v.clone() for v in outputs)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            call()
        graph.replay()
        counts = {name: int(torch.count_nonzero(actual.view(torch.uint8) != ref.view(torch.uint8)))
                  for name, actual, ref in zip(names, outputs, expected, strict=True)}
        assert not any(counts.values()), (label, counts)
        validation[label] = counts
        graphs[label] = graph
    del expected
    timing = measure(torch, calls, graphs, args.rounds, args.samples, args.sample_iters)
    return dict(scope='Only the joint recompute launch, same packed X/GU and dH addresses',
                physical_capacity=problem.m, unequal_bytes=validation, **timing)


def _expert_steps(torch, data, variants, args):
    from benchmarks.benchmark_moe_route_adjoint import compare, measure
    from rtx.moe import mxfp8_routed_experts
    from rtx.routing import BalancedRouting

    leaves = tuple(data[name].detach().requires_grad_() for name in ('x','gu','down','weights'))
    ids, inverse, dy = (data[name] for name in ('ids','inverse','dy'))
    original_dy = dy.clone()
    names = ('output','dx','dgu','ddown','droute')
    stream = torch.cuda.current_stream()

    def make_call(cfg):
        def forward(x, gu, down, weights):
            return mxfp8_routed_experts(x, gu, down, BalancedRouting(ids, weights, inverse),
                                       config=cfg, autotune='off')
        compiled = torch.compile(forward, fullgraph=True, dynamic=False, options={'triton.cudagraphs':False})

        def step():
            y = compiled(*leaves)
            return y.detach(), *(v.detach() for v in torch.autograd.grad(y, leaves, dy))

        return step

    calls, graphs, captured = {}, {}, {}
    for label, cfg in variants.items():
        print(f'Compile and capture expert F+B: {label}', flush=True)
        call = calls[label] = make_call(cfg)
        warm = call()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            captured[label] = call()
        graph.replay()
        for a,b in zip(captured[label], warm, strict=True):
            torch.testing.assert_close(a,b,rtol=0,atol=0)
        graphs[label] = graph
    del warm

    checks = {}
    initial_dgu = captured['anchor'][2].clone()
    for upstream in ('initial', 'changed'):
        if upstream == 'changed':
            dy.mul_(-.37).add_(.21)
        for graph in graphs.values():
            graph.replay()
        if upstream == 'changed':
            assert not torch.equal(initial_dgu, captured['anchor'][2]), 'Graph ignored changed dY'
        checks[upstream] = {}
        for label in variants:
            metrics = {name: compare(torch,a,b) for name,a,b in zip(names,
                captured[label],captured['anchor'],strict=True)}
            assert all(check['equal'] for check in metrics.values()), (upstream,label,metrics)
            for a,b in zip(calls[label](), captured[label], strict=True):
                torch.testing.assert_close(a,b,rtol=0,atol=0)
            checks[upstream][label] = metrics
    dy.copy_(original_dy)
    del initial_dgu
    torch.cuda.synchronize()
    live = [torch.cuda.memory_allocated()]
    for _ in range(5):
        for call in calls.values():
            call()
        torch.cuda.synchronize()
        live.append(torch.cuda.memory_allocated())
    assert len(set(live)) == 1, live
    timing = measure(torch, calls, graphs, args.rounds, args.samples, max(1,args.sample_iters//4))
    return dict(scope='Compiled frozen-route expert F+B, every leaf, no router/shared/optimizer',
                baseline_comparisons=checks, eager_live_bytes=live, **timing)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--source-result', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--variants', nargs='+', default=['vec4_xor','split2_anchor','split3_anchor',
                                                        'split2_vec4_xor','split3_vec4_xor'])
    p.add_argument('--rounds', type=int, default=4)
    p.add_argument('--samples', type=int, default=7)
    p.add_argument('--sample-iters', type=int, default=32)
    p.add_argument('--expert-steps', action='store_true')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.rounds,args.samples,args.sample_iters) < 1:
        p.error('rounds, samples and sample-iters must be positive')

    import torch
    from benchmarks.benchmark_moe_execution import _python_source_digest, _tensor_digest, write
    from rtx.autotune.moe import rejection
    from rtx.moe import MXFP8MoEConfig

    archived = json.loads(args.source_result.read_text())
    variants = _variants(MXFP8MoEConfig.from_dict(archived['expert_config']), args.variants)
    fixture = torch.load(args.fixture, map_location='cpu', weights_only=True)
    digests = {name:_tensor_digest(fixture[name]) for name in ('x','gu','down','ids','weights','inverse','dy')}
    identity = archived['identities']
    assert digests['x'] == identity['input']
    assert _tensor_digest(fixture['dy'].bfloat16()) == identity['upstream']
    assert digests['ids'] == identity['routing']['expert_ids']
    assert digests['inverse'] == identity['routing']['inverse']
    for e,(gu, down) in enumerate(zip(fixture['gu'],fixture['down'],strict=True)):
        gate, up = gu.chunk(2)
        leaves = identity['parameters']
        assert _tensor_digest(gate) == leaves[f'gate_up.projections.e{e}_gate.weight']
        assert _tensor_digest(up) == leaves[f'gate_up.projections.e{e}_up.weight']
        assert _tensor_digest(down) == leaves[f'down.projections.e{e}.weight']
    n,d = fixture['x'].shape
    e,c = fixture['ids'].shape
    k,h = fixture['inverse'].shape[-1],fixture['down'].shape[-1]
    for label, cfg in variants.items():
        if why := rejection(cfg,(n,d,e,c,k,h)):
            raise ValueError(f'{label}: {why}')
    with torch.cuda.stream(torch.cuda.Stream()):
        data = {name:fixture[name].cuda() for name in digests}
        record = dict(command=sys.argv, source_digest=_python_source_digest(Path.cwd()),
            device=str(torch.cuda.get_device_properties(0)), torch=str(torch.__version__), cuda=torch.version.cuda,
            shape=[n,d,e,c,k,h], identities=digests, source_result=str(args.source_result), fixture=str(args.fixture),
            variants={label: cfg.to_dict() for label,cfg in variants.items()})
        record['component'] = _components(torch,data,variants,args)
        print(json.dumps(record['component']['median_us']),flush=True)
        if args.expert_steps:
            torch.cuda.reset_peak_memory_stats()
            record['expert_steps'] = _expert_steps(torch,data,variants,args)
            print(json.dumps(record['expert_steps']['median_us']),flush=True)
        write(args.output,record)


if __name__ == '__main__':
    main()
