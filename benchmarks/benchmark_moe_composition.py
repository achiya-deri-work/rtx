"""Matched factorial qualification of routing and expert-recompute schedules.

Uses the public BalancedMoE module under user-side fullgraph compilation.
Frozen settings are ablations, not a new runtime dispatch table. Every variant
receives identical inputs, parameters and upstream gradients. No phase times
or independent speedup percentages are added to claim a combined speedup.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
from pathlib import Path
import statistics


def composition_configs(shape, router, expert):
    from rtx._routing_varlen import persistent_pool_rows
    from rtx.autotune.moe import joint_recompute_seeds
    alternatives = list(joint_recompute_seeds(shape, expert))
    if not alternatives:
        raise ValueError('A legal row-major joint recompute seed is required')
    compact = alternatives[0]
    variants = {'baseline': (router, expert), 'recompute': (router, compact)}
    if persistent_pool_rows(shape[0], shape[2], shape[4]):
        persistent = replace(router, packed_sinkhorn_backend='persistent',
                             packed_persistent_threads=1024)
        variants.update(router=(persistent, expert), combined=(persistent, compact))
    return variants


def preparation_configs(router,expert,shared,candidate_expert,candidate_shared):
    """Factorial schedules, never a precision or numerical-policy fallback."""
    return dict(baseline=(router,expert,shared),
                experts=(router,candidate_expert,shared),
                shared=(router,expert,candidate_shared),
                combined=(router,candidate_expert,candidate_shared))


def component_configs(router,expert,shared,*,candidate_expert=None,candidate_shared=None,candidate_router=None):
    """Actual factorial cells; never infer the union by adding phase gains."""
    axes=[(name,value) for name,value in (('experts',candidate_expert),('shared',candidate_shared),
                                        ('router',candidate_router)) if value is not None]
    variants={'baseline':(router,expert,shared)}
    for mask in range(1,1<<len(axes)):
        selected={name:value for i,(name,value) in enumerate(axes) if mask&(1<<i)}
        label=('combined' if len(axes)>1 and len(selected)==len(axes) else
               'candidate' if len(axes)==1 and 'router' not in selected else '_'.join(selected))
        variants[label]=(selected.get('router',router),selected.get('experts',expert),selected.get('shared',shared))
    return variants


def reference_configs(router,expert,shared,*,candidate_router=None,saved=False,compact=True):
    controls={};specifications={}
    for prefix,routing in (('bf16',router),('bf16_new_router',candidate_router)):
        if routing is None:continue
        for suffix,checkpointed,layout in (('checkpoint',True,'native'),
                                          ('compact_checkpoint',True,'compact'),('saved',False,'native')):
            if layout=='compact' and not compact:continue
            if not checkpointed and not saved:continue
            name=f'{prefix}_{suffix}';controls[name]=(routing,expert,shared)
            specifications[name]=(checkpointed,layout)
    return controls,specifications


def capture_step(call,stream,*,isolate_gc=False):
    """Same replay capture, with optional benchmark-only cyclic-GC isolation."""
    import torch
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            value=call();del value
    stream.synchronize()
    if isolate_gc:gc.collect()
    enabled=gc.isenabled();graph=torch.cuda.CUDAGraph()
    if isolate_gc:gc.disable()
    try:
        with torch.cuda.graph(graph,stream=stream):
            try:
                value=call()
                if value is not None:value=tuple(v.detach() for v in value)
            except BaseException:
                # capture_end can replace the original Python/CUDA error.
                import traceback
                print('CAPTURE_BODY_EXCEPTION',flush=True);traceback.print_exc()
                raise
    finally:
        if isolate_gc and enabled:gc.enable()
    torch.cuda.current_stream().wait_stream(stream)
    return graph,value


def compare(actual, expected):
    import torch
    if len(actual) != len(expected):
        raise AssertionError('Output/gradient arity differs')
    reports = []
    for index, (value, ref) in enumerate(zip(actual, expected)):
        assert value.shape == ref.shape and value.dtype == ref.dtype, index
        assert bool(value.isfinite().all()), index
        delta = value.detach().float() - ref.detach().float()
        nrms = float(delta.norm() / ref.detach().float().norm().clamp_min(1e-12))
        maximum = float(delta.abs().max())
        assert nrms < .006 or maximum < 2e-7, (index, nrms, maximum)
        reports.append(dict(nrms=nrms, max_abs=maximum,
                            unequal=int(torch.count_nonzero(value != ref))))
    return reports


def expert_variants(router,expert,patches):
    """Benchmark-only factorial changes; never an automatic runtime policy."""
    from rtx.moe import MXFP8MoEConfig
    def merge(base,patch):
        return {key:merge(base[key],value) if isinstance(value,dict) and isinstance(base.get(key),dict) else
                merge(base['down'],value) if key=='route_gradient_gemm' and isinstance(value,dict) and base.get(key) is None else value
                for key,value in {**base,**patch}.items()}
    if (not isinstance(patches,dict) or not patches or 'baseline' in patches
            or any(name.startswith('bf16_') for name in patches)):
        raise ValueError('Nonempty uniquely named expert-variant mapping required')
    return dict(baseline=(router,expert),**{name:(router,MXFP8MoEConfig.from_dict(merge(expert.to_dict(),patch)))
        for name,patch in patches.items()})


def composition_baseline(document,variant=None):
    """Reuse a qualified cell without copying its obsolete timing claims."""
    if 'configs' not in document:
        if variant is not None:raise ValueError('A baseline variant requires a composition report')
        return document
    if variant is None or variant not in document['configs']:
        raise ValueError('Select an existing --baseline-variant from the composition report')
    cell=document['configs'][variant];hashes=document['hashes']
    return dict(case=document['case'],lengths=document['lengths'],seed=document['seed'],
        initialization_case=document.get('initialization_case',document['case']),
        router_config=cell['router'],expert_config=cell['expert'],shared_config=cell['shared'],
        router_affinity_layout=document.get('router_affinity_layout','scalar'),
        input_profile=document.get('input_profile','normal'),
        layout=document.get('layout','packed'),stable_inputs=document.get('stable_inputs',False),
        stable_experts=document.get('stable_experts',False),
        identities=dict(input=hashes['x'],upstream=hashes['dy'],parameters=hashes['parameters'],
                        routing={name:hashes[name] for name in ('expert_ids','inverse')}))


def run(args):
    import torch
    from benchmarks.benchmark_moe_execution import _python_source_digest, _tensor_digest, write
    from benchmarks.profile_moe import hardware_attributes, statistics_ms
    from rtx.autotune.measurement import measure_cuda_batches
    from rtx.autotune.moe import source_revision
    from rtx.autotune.preparation_runtime import _warm_measurement_graph
    from rtx.mlp import MXFP8MLPConfig
    from rtx.moe import BalancedMoE, MXFP8MoEConfig
    from rtx.routing import RoutingKernelConfig

    saved = composition_baseline(json.loads(args.baseline_result.read_text()),
                                 getattr(args,'baseline_variant',None))
    case, lengths = saved['case'], saved['lengths']
    d, e, h, k = (case[name] for name in
                  ('d_model', 'num_experts', 'expert_hidden_dim', 'top_k'))
    router = RoutingKernelConfig.from_dict(saved['router_config'])
    expert = MXFP8MoEConfig.from_dict(saved['expert_config'])
    shared = MXFP8MLPConfig.from_dict(saved['shared_config'])
    if args.lengths is not None:
        lengths = args.lengths
        from benchmarks.profile_moe import retarget_auxiliary
        retargeted, _ = retarget_auxiliary({'shared': {'config': shared.to_dict()}}, sum(lengths))
        shared = MXFP8MLPConfig.from_dict(retargeted['shared']['config'])
    n = sum(lengths)
    layout=saved.get('layout','packed')
    def build(routing, experts, shared_policy=shared,expert_hidden=None):
        return BalancedMoE(d, e, h if expert_hidden is None else expert_hidden, top_k=k, shared_hidden_features=case['shared_hidden_dim'],
            num_shared_experts=case['num_shared_experts'], device='cuda', autotune='off',
            config=experts, shared_config=shared_policy,
            router_options=dict(backend='cute', config=routing,
                affinity_layout=saved.get('router_affinity_layout','scalar')))
    initialization_case=saved.get('initialization_case',case)
    from benchmarks.benchmark_moe_varlen import initialize_fixture,compile_options
    original,x,cu,dy=initialize_fixture(
        lambda c:build(router,expert,expert_hidden=c['expert_hidden_dim']),case,initialization_case,lengths,
        layout=layout,seed=saved['seed'],input_profile=saved.get('input_profile','normal'),
        stable_inputs=saved.get('stable_inputs',False),stable_experts=saved.get('stable_experts',False),
        router_std=.05 if getattr(args,'match_inputs',False) else .02)
    options=compile_options(getattr(args,'max_autotune',False))
    # Match the gradient-enabled production router too: no_grad can change
    # compiler fusion/reduction order and therefore a discrete hard tie.
    route = torch.compile(original.router,**options)(x,cu_seqlens=cu)
    shape = (n, d, e, route.expert_ids.shape[1], k, h)
    candidate=None;candidate_shared=None;candidate_router=None
    if getattr(args,'candidate_config',None):
        document=json.loads(args.candidate_config.read_text())
        candidate=MXFP8MoEConfig.from_dict(document.get('config',document))
    if getattr(args,'candidate_shared_config',None):
        document=json.loads(args.candidate_shared_config.read_text())
        candidate_shared=MXFP8MLPConfig.from_dict(document.get('config',document))
    if getattr(args,'candidate_router_config',None):
        document=json.loads(args.candidate_router_config.read_text())
        candidate_router=RoutingKernelConfig.from_dict(document.get('config',document))
    if getattr(args,'expert_variants',None):
        variants=expert_variants(router,expert,json.loads(args.expert_variants))
        variants={name:(r,c,shared) for name,(r,c) in variants.items()}
    elif any(value is not None for value in (candidate,candidate_shared,candidate_router)):
        if candidate_shared is not None and candidate is None and candidate_router is None:
            raise ValueError('Shared factorial requires an expert or router candidate')
        variants=component_configs(router,expert,shared,candidate_expert=candidate,
            candidate_shared=candidate_shared,candidate_router=candidate_router)
        if getattr(args,'shared_only',False):
            variants=dict(baseline=(router,expert,shared),candidate=(router,expert,candidate_shared))
    else:
        variants = composition_configs(shape, router, expert)
        variants={name:(r,c,shared) for name,(r,c) in variants.items()}
    include_bf16=any(getattr(args,name,False) for name in ('bf16_checkpoint','bf16_compact','bf16_saved'))
    reference_specs={}
    if include_bf16:
        controls,reference_specs=reference_configs(router,expert,shared,candidate_router=candidate_router,
            saved=getattr(args,'bf16_saved',False),compact=getattr(args,'bf16_compact',False))
        variants.update(controls)
    names = ('output', 'input', *(name for name, _ in original.named_parameters()))
    hashes = dict(x=_tensor_digest(x), dy=_tensor_digest(dy),
                  parameters={name: _tensor_digest(p) for name, p in original.named_parameters()},
                  expert_ids=_tensor_digest(route.expert_ids), inverse=_tensor_digest(route.inverse))
    if getattr(args,'match_inputs',False):
        reference=saved['identities']
        assert hashes['x']==reference['input'] and hashes['dy']==reference['upstream']
        assert hashes['parameters']==reference['parameters']
        assert hashes['expert_ids']==reference['routing']['expert_ids']
        assert hashes['inverse']==reference['routing']['inverse']
    graphs, checks, live = {}, {}, {}
    stream = torch.cuda.current_stream()
    expected = None
    bf16_expected=None
    setup_seconds={}
    for label, (routing, experts, shared_policy) in variants.items():
        print('VALIDATE', label, flush=True)
        layer = build(routing, experts,shared_policy)
        layer.load_state_dict(original.state_dict())
        selected = torch.compile(layer.router,**options)(x,cu_seqlens=cu)
        for name in ('expert_ids', 'inverse'):
            assert torch.equal(getattr(route, name), getattr(selected, name)), (label, name)
        is_bf16=label in reference_specs
        if is_bf16:
            from benchmarks.benchmark_moe_checkpoint import BF16FFNReference
            checkpointed,expert_layout=reference_specs[label]
            module=BF16FFNReference(layer,checkpoint_ffns=checkpointed,expert_layout=expert_layout)
        else:module=layer
        compiled = torch.compile(module, **options)
        leaves = (x, *layer.parameters())
        def step(fn=compiled, leaves=leaves):
            if include_bf16:
                for leaf in leaves:leaf.grad=None
                y=fn(x,cu_seqlens=cu)
                y.backward(dy)
                return y,*(leaf.grad for leaf in leaves)
            y = fn(x, cu_seqlens=cu)
            return y, *torch.autograd.grad(y, leaves, dy)
        import time
        started=time.monotonic()
        from rtx.runtime import runtime_cache_stats
        capture_state=dict(label=label,allocated=torch.cuda.memory_allocated(),
            reserved=torch.cuda.memory_reserved(),free_total=torch.cuda.mem_get_info(),
            caches=runtime_cache_stats(),isolate_gc=getattr(args,'isolate_capture_gc',False))
        write(args.output.with_suffix('.capture.json'),capture_state)
        graph, output = capture_step(step,stream,isolate_gc=getattr(args,'isolate_capture_gc',False))
        for _ in range(3):
            graph.replay()
        torch.cuda.synchronize()
        if expected is None:
            expected = tuple(value.detach().clone() for value in output)
        if label=='bf16_checkpoint':bf16_expected=tuple(value.detach().clone() for value in output)
        setup_seconds[label]=time.monotonic()-started
        checks[label] = dict(hard_route_exact=True,
                             cross_schedule_comparable=not is_bf16,
                             errors=compare(output,(bf16_expected if is_bf16 else expected)),
                             replay_errors=compare(step(), output))
        if getattr(args,'changed_dy',False):
            original_dy=dy.clone()
            checks[label]['changed_dy']=[]
            for factor in (-.75,1.):
                dy.copy_(original_dy*factor)
                graph.replay();stream.synchronize()
                checks[label]['changed_dy'].append(compare(output,step()))
            assert torch.equal(dy,original_dy)
        samples = []
        # Compiler/AOT cycles can be collected after their first execution;
        # distinguish that one-time *decrease* from steady live-tensor growth.
        for _ in range(2):
            eager = step()
            del eager
        stream.synchronize()
        gc.collect()
        for _ in range(5):
            eager = step()
            del eager
            stream.synchronize()
            gc.collect()
            samples.append(torch.cuda.memory_allocated())
        assert len(set(samples))==1, (label, samples)
        live[label] = samples
        # Keep layer/leaves alive along with its graph-owned allocations.
        graphs[label] = (graph, layer, output, step)

    rounds = {label: [] for label in variants}
    order = list(variants)
    # Balanced forward/reverse visits limit order and thermal bias.
    for visit in range(args.visits):
        for label in (order if visit % 2 == 0 else reversed(order)):
            graph = graphs[label][0]
            _warm_measurement_graph(graph)
            values, metadata = measure_cuda_batches(graph.replay, samples=args.samples,
                calls_per_sample=16, device=x.device, min_batch_ms=5.)
            row = dict(visit=visit, **statistics_ms(values), **metadata)
            rounds[label].append(row)
            print('TIME', visit, label, row['median_ms'], 'ms', flush=True)
            write(args.output.with_suffix('.timings.json'),dict(status='timing',rounds=rounds,
                hashes=hashes,checks=checks,live_allocated_bytes=live))
    medians = {label: statistics.median(row['median_ms'] for row in rows)
               for label, rows in rounds.items()}
    summary = {label: dict(median_ms=ms, speedup=medians['baseline']/ms,
        max_relative_mad=max(row['relative_mad'] for row in rounds[label]),
        relative_visit_range=(max(row['median_ms'] for row in rounds[label]) -
                              min(row['median_ms'] for row in rounds[label]))/ms)
        for label, ms in medians.items()}
    interaction = None
    if 'combined' in medians:
        cells=[name for name in medians if name not in ('baseline','combined') and name not in reference_specs]
        if len(cells)==2:
            a,b=cells
            interaction = (medians['combined'] - medians[a] - medians[b] + medians['baseline'])
    optimizer_check=None
    if getattr(args,'optimizer_steps',0):
        if not getattr(args,'shared_only',False) or not args.bf16_checkpoint:
            raise ValueError('Optimizer check requires --shared-only --bf16-checkpoint')
        selected=('baseline','candidate')
        optimizers={label:torch.optim.SGD(graphs[label][1].parameters(),lr=.001) for label in selected}
        originals={label:tuple(p.detach().clone() for p in graphs[label][1].parameters()) for label in selected}
        steps=[]
        for index in range(args.optimizer_steps):
            values={}
            for label in selected:
                result=graphs[label][3]()
                assert all(bool(v.isfinite().all()) for v in result)
                values[label]=tuple(v.detach().clone() for v in result)
                optimizers[label].step()
            metrics=compare(values['candidate'],values['baseline'])
            assert all(r['unequal']==0 for r in metrics),(index,metrics)
            assert all(torch.equal(a,b) for a,b in zip(graphs['baseline'][1].parameters(),
                                                       graphs['candidate'][1].parameters()))
            steps.append(dict(step=index,errors=metrics))
        changed=sum(not torch.equal(a,b) for a,b in zip(originals['baseline'],graphs['baseline'][1].parameters()))
        assert changed>0,'Optimizer check did not update any parameter'
        optimizer_check=dict(optimizer='SGD',lr=.001,steps=steps,changed_parameter_count=changed,
            objective='Fixed FP32 linear upstream objective; qualification, not convergence')
    write(args.output, dict(shape=shape, case=case, lengths=lengths, seed=saved['seed'],
        initialization_case=initialization_case,
        layout=layout,stable_inputs=saved.get('stable_inputs',False),
        stable_experts=saved.get('stable_experts',False),
        input_profile=saved.get('input_profile','normal'),
        router_affinity_layout=saved.get('router_affinity_layout','scalar'),
        baseline_result=str(args.baseline_result.resolve()),
        source_revision=source_revision(), runtime_digest=_python_source_digest(Path(__file__).resolve().parents[1]),
        hardware=hardware_attributes(), torch=str(torch.__version__), cuda=torch.version.cuda,
        configs={label: dict(router=r.to_dict(), expert=c.to_dict(),shared=s.to_dict())
                 for label, (r,c,s) in variants.items()},
        setup_seconds=setup_seconds,compile_options=options,
        capture_gc_isolated=getattr(args,'isolate_capture_gc',False),
        shared_config=shared.to_dict(), names=names, hashes=hashes, checks=checks,
        source_inputs_matched=getattr(args,'match_inputs',False),
        live_allocated_bytes=live, rounds=rounds, summary=summary, interaction_ms=interaction,
        optimizer_check=optimizer_check))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--baseline-result', type=Path, required=True)
    p.add_argument('--baseline-variant',help='Use a named qualified cell from a previous composition report as the next baseline')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=7)
    p.add_argument('--visits', type=int, default=4)
    p.add_argument('--candidate-config',type=Path,help='Paired validation of a measured expert schedule instead of factorial recipes')
    p.add_argument('--expert-variants',help='JSON mapping of named nested expert config overrides; shares one matched baseline and all-Parameter checks')
    p.add_argument('--candidate-shared-config',type=Path,help='Cross independent expert/shared schedule changes in four native cells')
    p.add_argument('--candidate-router-config',type=Path,help='Matched router/shared/expert factorial; BF16 controls also use each router')
    p.add_argument('--shared-only',action='store_true',help='Compare only the shared-policy change, retaining identical expert/router settings')
    p.add_argument('--optimizer-steps',type=int,default=0,help='After timings, verify direct/hybrid outputs, all gradients and actual SGD parameter updates match over several steps')
    p.add_argument('--bf16-checkpoint',action='store_true',help='Add same-router checkpointed BF16 control; all cells use set-to-none backward')
    p.add_argument('--bf16-compact',action='store_true',help='Also compare shape-bound compacted checkpointed BF16 against native-capacity BF16; parity checked on every output/gradient')
    p.add_argument('--bf16-saved',action='store_true',help='Also measure saved-intermediate BF16, including a new-router control when applicable')
    p.add_argument('--changed-dy',action='store_true',help='Validate two upstream gradients on each captured graph')
    p.add_argument('--isolate-capture-gc',action='store_true',help='Collect cyclic garbage after warmup and defer cyclic collection only across each benchmark capture')
    p.add_argument('--match-inputs',action='store_true',help='Assert exact source-study input/parameter/upstream/hard-route hashes')
    p.add_argument('--max-autotune',action='store_true')
    p.add_argument('--lengths', type=int, nargs='+', help='Optional matched packing control')
    args = p.parse_args()
    if args.expert_variants and (args.candidate_config or args.candidate_shared_config or args.candidate_router_config or args.optimizer_steps):
        p.error('--expert-variants cannot be combined with candidate paths or the two-cell optimizer check')
    if args.optimizer_steps<0:p.error('--optimizer-steps must be nonnegative')
    if args.shared_only and not (args.candidate_config and args.candidate_shared_config):
        p.error('--shared-only requires both candidate configuration paths')
    if args.samples < 3 or args.visits < 2:
        p.error('At least three samples and two balanced visits required')
    if args.lengths is not None and (min(args.lengths) < 0 or sum(args.lengths) <= 0):
        p.error('Nonnegative lengths and at least one real token required')
    import torch
    # All leaves, warmup and capture use one stream (Torch 2.13 AccumulateGrad).
    # A factorial deliberately compiles many independent module/config cells
    # of the same Python forward, unlike one production model invocation.
    # Keep fullgraph strict; bound this benchmark-local specialization budget
    # rather than changing the library or falling back to eager execution.
    with torch._dynamo.config.patch(recompile_limit=64,accumulated_recompile_limit=256):
        with torch.cuda.stream(torch.cuda.Stream()):
            run(args)


if __name__ == '__main__':
    main()
