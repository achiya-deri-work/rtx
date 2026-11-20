"""Bounded shared-MLP split study using an exact named MoE initialization.

GPU0 commands must be serialized by the caller. A fresh checkout/immutable
snapshot per run is recommended. Keep the baseline result and its raw search
journal: no cached winner is changed by a frozen comparison.
"""
import argparse
from pathlib import Path
import gc
import json
import statistics
import sys
import time


def metrics(actual, expected, *, exact=()):
    import torch

    assert len(actual) == len(expected) == 4
    result = {}
    for name, a, b in zip(('output', 'dx', 'dgate_up', 'ddown'), actual, expected):
        assert a.shape == b.shape and a.dtype == b.dtype, name
        assert bool(a.isfinite().all()) and bool(b.isfinite().all()), name
        delta = a.float() - b.float()
        nrms = float(delta.norm() / b.float().norm().clamp_min(1e-12))
        maximum = float(delta.abs().max())
        unequal = int(torch.count_nonzero(a != b))
        assert nrms < (.003 if name == 'output' else .006) or maximum < 2e-7, (name, nrms, maximum)
        if name in exact:
            assert unequal == 0, (name, unequal, nrms, maximum)
        result[name] = dict(nrms=nrms, max_abs=maximum, unequal=unequal)
    return result


def oracle(x, gu, down, dy):
    """Independent quantizer and full, non-partitioned FP64 dot products."""
    import torch
    from rtx._moe_reference import quantize_decode

    def mm(a, b):
        return (quantize_decode(a).double() @ quantize_decode(b).double().T).float()

    gate, up = mm(x, gu).chunk(2, -1)
    divisor = 1 + torch.exp(-gate)
    sig, act = 1 / divisor, gate / divisor
    hidden = quantize_decode(act * up).bfloat16()
    y = (hidden.double() @ quantize_decode(down).double().T).bfloat16()
    dh = mm(dy, down.T).bfloat16().float()
    dp = torch.cat((dh * up * sig * (1 + gate * (1 - sig)), dh * act), -1).bfloat16()
    # Re-quantize independently along tokens for each of the dW operands.
    return y, mm(dp, gu.T).bfloat16(), mm(dp.T, x.T).bfloat16(), mm(dy.T, hidden.T).bfloat16()


def workspace(config, leaves):
    _, gu, down = leaves
    result={}
    for name,w in (('gate_backward',gu),('down_backward',down)):
        dw=getattr(config,name).dw
        result[name]=dw.split_reduction*w.numel()*4 if dw.reduction=='split_fp32_workspace' else 0
    return result


def run(args, record):
    import torch
    from rtx.autotune.effort import AutotunePolicy
    from rtx.autotune.measurement import measure_cuda_batches, runtime_timing_metadata
    from rtx.autotune.mlp import make_mlp_adapter, mlp_independent_dw_seeds, tune_mxfp8_mlp
    from rtx.autotune.preparation_runtime import _warm_measurement_graph
    from rtx.mlp import MXFP8MLPConfig, mxfp8_mlp
    from rtx.moe import BalancedMoE
    from benchmarks.benchmark_moe_execution import _python_source_digest, _tensor_digest, write
    from benchmarks.benchmark_moe_joint_recompute import _capture
    from benchmarks.benchmark_moe_varlen import initialize_fixture, compile_options, statistics_ms

    saved = json.loads(args.baseline_result.read_text())
    case = saved['case']
    lengths = saved['lengths'] if args.lengths is None else args.lengths
    if case['num_shared_experts'] != 1:
        raise ValueError('This study compares one ordinary shared MLP')

    def make(c):
        return BalancedMoE(c['d_model'], c['num_experts'], c['expert_hidden_dim'], top_k=c['top_k'],
            shared_hidden_features=c['shared_hidden_dim'], num_shared_experts=c['num_shared_experts'],
            device='cuda', autotune='off', router_options=dict(backend='cute', autotune='off'))

    model,x,_,dy = initialize_fixture(make, case, saved['initialization_case'], lengths,
        layout=saved['layout'], seed=saved['seed'], input_profile=saved['input_profile'],
        stable_inputs=saved.get('stable_inputs', False), stable_experts=saved.get('stable_experts', False))
    gate_layer, down_layer = model.shared_gate_up[0], model.shared_down[0]
    leaves = (x, gate_layer.weight.detach(), down_layer.weight)
    optimizer_leaves = (x, *gate_layer.parameters(), *down_layer.parameters())
    hashes = dict(input=_tensor_digest(x), upstream=_tensor_digest(dy),
                  parameters={n:_tensor_digest(p) for n,p in model.named_parameters()})
    if args.lengths is None:
        assert all(hashes[k] == saved['identities'][k] for k in hashes), 'Fixture provenance differs'
    info = torch.cuda.get_device_properties(x.device)
    record.update(lengths=lengths, case=case, seed=saved['seed'], hashes=hashes,
                  fixture_matches_archive=args.lengths is None, torch=str(torch.__version__),
                  gpu=info.name, sm_count=info.multi_processor_count, cuda=torch.version.cuda,
                  source_sha256=_python_source_digest(Path(__file__).resolve().parents[1]))

    if args.search:
        policy = AutotunePolicy.preset('balanced', max_trials=8, max_seconds=args.seconds,
            confirmation_repeats=0, samples=5, calls_per_sample=1,
            search_schedule='progressive', candidate_mix='mixed')
        started = time.monotonic()
        selected = tune_mxfp8_mlp(*leaves, initial=model.shared_config, tuning_policy=policy,
            output_dir=args.output.parent / (args.output.stem + '-tuning'),
            progress=lambda s:print(s, flush=True))
        record['search'] = dict(evidence=selected.tuning_evidence, trials=selected.evaluated_trials,
                                invocation_s=time.monotonic()-started, winner=selected.config.to_dict(),
                                decision=selected.winner_decision)
        assert selected.evaluated_trials == 8, record['search']
        if not args.timing:
            return

    anchor = MXFP8MLPConfig.from_dict(saved['shared_config'])
    if args.lengths is not None:
        from rtx._bwd_search import reduction_chunk
        from dataclasses import replace
        for name in ('gate_backward', 'down_backward'):
            b = getattr(anchor, name)
            dw = b.dw
            if dw.split_partition == 'uniform' and dw.split_reduction > 1:
                chunk = reduction_chunk(x.shape[0], dw.split_reduction, dw.gemm.tile_k)
                anchor = replace(anchor, **{name: replace(b, dw=replace(dw, reduction_tile=chunk))})
    adapter = make_mlp_adapter(*leaves, initial=anchor, _reference_cache={'values':(dy,None)})
    plans = {'baseline':anchor}
    for c in mlp_independent_dw_seeds(adapter, anchor):
        changed, = [g for g in ('gate_backward', 'down_backward') if getattr(c,g) != getattr(anchor,g)]
        plans[changed] = c
    if args.search:
        plans['opening_winner'] = selected.config
    if args.only:
        plans = {name:c for name,c in plans.items() if name in args.only}
        assert plans and set(plans) == set(args.only)
    record['variants'] = {n:dict(config=c.to_dict(),workspace_bytes=workspace(c,leaves)) for n,c in plans.items()}
    # Match the whole outer fullgraph/static compiler options used in the
    # archive. Oracle and both replay buffers use the real archived dY.
    options = compile_options(saved['inductor_max_autotune'])
    stream = torch.cuda.current_stream()
    inputs_copy = x.detach().clone(), dy.clone()
    before = None
    with torch.no_grad():
        reference = oracle(*leaves,dy)
        x.copy_(inputs_copy[0].mul(-.5)); dy.copy_(inputs_copy[1].mul(.75))
        changed_reference = oracle(*leaves,dy)
        x.copy_(inputs_copy[0]); dy.copy_(inputs_copy[1])
    graphs = {}
    for name,c in plans.items():
        print('VALIDATE', name, flush=True)
        row = record['variants'][name]
        assert adapter.rejection(c) is None, (name, adapter.rejection(c))
        def forward(value,config=c):
            return mxfp8_mlp(value,gate_layer.weight,down_layer.weight,config=config)
        fwd = torch.compile(forward, **options)
        def step(fn=fwd):
            for t in optimizer_leaves:t.grad=None
            y = fn(x)
            y.backward(dy)
            return y, *(t.grad for t in optimizer_leaves)
        def packed(values):
            y, dx, dgate, dup, ddown = values
            return y, dx, torch.cat((dgate,dup)), ddown
        graph,out = _capture(step, stream)
        graph.replay(); stream.synchronize()
        current = packed(out)
        row['oracle'] = metrics(current, reference)
        if before is None:
            before = tuple(t.detach().clone() for t in current)
        exact=() if name=='opening_winner' else ('output','dx')
        if name=='down_backward':exact+=('dgate_up',)
        if name=='gate_backward':exact+=('ddown',)
        row['native_against_first'] = metrics(current, before, exact=exact)
        del current
        live = []
        torch.cuda.reset_peak_memory_stats()
        for i in range(5):
            with torch.no_grad():
                x.copy_(inputs_copy[0].mul(-.5) if i%2 == 0 else inputs_copy[0]);
                dy.copy_(inputs_copy[1].mul(.75) if i%2 == 0 else inputs_copy[1]);
            graph.replay(); stream.synchronize()
            row[f'replay_{i}'] = metrics(packed(out), changed_reference if i%2 == 0 else reference)
            for t in optimizer_leaves:t.grad=None
            gc.collect()
            live.append(torch.cuda.memory_allocated())
        with torch.no_grad():x.copy_(inputs_copy[0]);dy.copy_(inputs_copy[1])
        assert len(set(live)) == 1, (name, live)
        row['live_allocated_bytes'] = live
        row['replay_peak_allocated_bytes'] = torch.cuda.max_memory_allocated()
        graphs[name] = graph
        row['visits'] = []
        write(args.output,record)
    for visit in range(args.visits):
        for name in (graphs if visit%2 == 0 else reversed(graphs)):
            g = graphs[name]
            _warm_measurement_graph(g)
            timings, timing_batch = measure_cuda_batches(g.replay, samples=7,
                calls_per_sample=16, device=x.device, min_batch_ms=5.)
            row = dict(visit=visit, **statistics_ms(timings), **timing_batch, **runtime_timing_metadata(timings))
            print('TIME', name, visit, row['median_ms'], flush=True)
            record['variants'][name]['visits'].append(row)
            write(args.output,record)
    for row in record['variants'].values():
        row['median_ms'] = statistics.median(r['median_ms'] for r in row['visits'])
        row['stationary'] = all(r['sampling']['collection']['stationary'] for r in row['visits'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    source = p.add_mutually_exclusive_group(required=True)
    source.add_argument('--baseline-result',type=Path)
    source.add_argument('--select-from',type=Path,help='Export a stationary paired winner from an existing result; CPU only')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lengths',nargs='+',type=int)
    p.add_argument('--search',action='store_true')
    p.add_argument('--timing',action='store_true',help='Also run the frozen paired timing when --search is given')
    p.add_argument('--only',nargs='+')
    p.add_argument('--seconds',type=float,default=180.)
    p.add_argument('--visits',type=int,default=4)
    args = p.parse_args()
    if args.output.exists():raise FileExistsError(args.output)
    from benchmarks.benchmark_moe_execution import write
    if args.select_from:
        raw = json.loads(args.select_from.read_text())
        assert raw['status']=='complete'
        base = raw['variants']['baseline']
        base_visits = [v['median_ms'] for v in base['visits']]
        eligible=[]
        for name,row in raw['variants'].items():
            if name=='baseline' or not base['stationary'] or not row['stationary']:continue
            improvements=[a/b['median_ms']-1 for a,b in zip(base_visits,row['visits'],strict=True)]
            if len(improvements)>=4 and min(improvements)>.01:
                eligible.append((row['median_ms'],name,improvements))
        if not eligible:raise ValueError('No stationary variant beat baseline by >1% in every visit')
        _,label,pairs=min(eligible)
        chosen=raw['variants'][label]
        write(args.output,dict(config=chosen['config'],label=label,paired_relative_speedups=pairs,
            median_ms=chosen['median_ms'],evidence=str(args.select_from),baseline_result=raw['baseline_result']))
        return
    import torch

    record = dict(status='running', argv=sys.argv, baseline_result=str(args.baseline_result))
    try:
        with torch.cuda.stream(torch.cuda.Stream()):
            run(args,record)
        record['status'] = 'complete'
    except BaseException as exc:
        record.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        write(args.output,record)


if __name__ == '__main__':
    main()
