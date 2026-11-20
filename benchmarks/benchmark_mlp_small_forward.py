"""Paired small-M ordinary MLP forward study; all native arms freeze backward."""
import argparse
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch


def quantized_fp64(x, gu, down, dy):
    """Independent MX series: explicit block scale, FP64 GEMMs/GLU, BF16 STE."""
    def q(t):
        width = t.shape[-1]
        blocks = torch.nn.functional.pad(t.double(), (0, -width%32)).unflatten(-1, (-1, 32))
        amax = blocks.abs().amax(-1, keepdim=True)
        scale = torch.exp2(torch.ceil(torch.log2((amax/448).clamp_min(2.**-126))))
        data = (blocks/scale).clamp(-448,448).float().to(torch.float8_e4m3fn).double()
        return (data*scale).flatten(-2)[..., :width]
    g, u = (q(x) @ q(gu).T).chunk(2, dim=-1)
    sig = torch.sigmoid(g)
    activation = g*sig
    hidden = q(activation*u)
    y = (hidden @ q(down).T).bfloat16()
    dh = (q(dy) @ q(down.T).T).bfloat16().double()
    dp = torch.cat((dh*u*sig*(1 + g*(1-sig)), dh*activation), dim=-1).bfloat16()
    return (y, (q(dp) @ q(gu.T).T).bfloat16(), (q(dp.T) @ q(x.T).T).bfloat16(),
            (q(dy.T) @ q(hidden.bfloat16().T).T).bfloat16())


def compare(actual, expected):
    rows = []
    for a, b in zip(actual, expected, strict=True):
        diff = a.detach().float() - b.detach().float()
        rows.append(dict(finite=bool(a.isfinite().all()), unequal=int((a != b).sum()),
            max_abs=float(diff.abs().max()), rms_error=float(diff.square().mean().sqrt()),
            reference_rms=float(b.float().square().mean().sqrt()),
            nrms=float(diff.norm()/b.float().norm().clamp_min(1e-12))))
    return rows


def tensors(shape, seed):
    m, k, h, o = shape
    rng = torch.Generator().manual_seed(seed)
    values = tuple((torch.randn(s, generator=rng)*.1).bfloat16().cuda()
                   for s in ((m,k), (2*h,k), (o,h), (o,m)))
    return values[:3], values[3].T  # noncontiguous upstream


def digest(t):
    data = t.detach().cpu().contiguous().view(torch.uint8).numpy()
    return hashlib.sha256(data).hexdigest()


def bf16_checkpoint(x, gu, down):
    from torch.nn.functional import linear, silu
    from torch.utils.checkpoint import checkpoint
    def core(x, gu, down):
        g, u = linear(x, gu).float().chunk(2, dim=-1)
        return linear((silu(g)*u).bfloat16(), down)
    return checkpoint(core, x, gu, down, use_reentrant=False)


def run(args):
    from benchmarks.benchmark_moe_joint_recompute import _capture
    from benchmarks.benchmark_moe_execution import _python_source_digest, write
    from rtx.autotune.measurement import measure_cuda_batches
    from rtx.autotune.preparation_runtime import _warm_measurement_graph
    from rtx.mlp import MXFP8MLPConfig, mxfp8_mlp

    shape = tuple(map(int, args.shape.split(',')))
    data = json.loads(args.config.read_text()) if args.config else None
    base = MXFP8MLPConfig.from_dict(data.get('config', data)) if data else MXFP8MLPConfig()
    policies = dict(onchip=replace(base, gate_backend='onchip', down_backend='onchip'),
                    gate_only=replace(base, gate_backend='prequant', down_backend='onchip'),
                    gate_down=replace(base, gate_backend='prequant', down_backend='prequant'))
    opts = dict(fullgraph=True, dynamic=False, options={'triton.cudagraphs':False})
    stream = torch.cuda.current_stream()
    inputs, dy = tensors(shape, args.seed)
    leaves = tuple(v.detach().requires_grad_() for v in inputs)
    expected_hashes = list(map(digest, (*leaves, dy)))
    record = dict(status='running', shape=shape, seed=args.seed, names=('output','dx','dgu','ddown'),
        torch=torch.__version__, cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(),
        source_digest=_python_source_digest(Path.cwd()),
        driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_hashes=expected_hashes, input_addresses=[v.data_ptr() for v in (*leaves, dy)],
        configs={name:c.to_dict() for name,c in policies.items()}, setup_seconds={}, checks={}, live={}, timings={})

    def save():
        write(args.output, record)

    modes = tuple(policies) + ('bf16_checkpoint',)
    calls, graphs = {}, {}
    for name in modes:
        started = time.monotonic()
        if name in policies:
            def make_forward(c):
                return lambda *values: mxfp8_mlp(*values, config=c)
            forward = make_forward(policies[name])
        else:
            forward = bf16_checkpoint
        compiled = torch.compile(forward, **opts)
        def fwd(fn=compiled):
            with torch.no_grad():
                return (fn(*leaves),)
        def step(fn=compiled):
            y = fn(*leaves)
            return (y, *torch.autograd.grad(y, leaves, dy))
        calls[name] = step
        for phase, call in (('forward', fwd), ('step', step)):
            graphs[name,phase] = _capture(call, stream)
        record['setup_seconds'][name] = time.monotonic() - started
        record['timings'][name] = {phase:[] for phase in ('forward','step')}

    for generation in range(3):
        if generation:
            changed, changed_dy = tensors(shape, args.seed + generation)
            with torch.no_grad():
                for current, value in zip(leaves, changed, strict=True):
                    current.copy_(value)
                dy.copy_(changed_dy)
        with torch.no_grad():
            oracle = quantized_fp64(*leaves,dy)
        for name in modes:
            graph, actual = graphs[name,'step']
            graph.replay()
            live = calls[name]()
            summary = dict(replay=compare(actual, live), hashes=list(map(digest, (*leaves,dy))))
            assert all(r['finite'] and r['unequal']==0 for r in summary['replay']), (name,summary)
            if name in policies:
                summary['quantized_fp64'] = compare(actual, oracle)
                summary['same_backward'] = compare(actual, graphs['onchip','step'][1])
                assert all(r['finite'] and r['nrms']<.025 for r in summary['quantized_fp64']), (name,summary)
                assert all(r['unequal']==0 for r in summary['same_backward'][1:]), (name,summary)
            record['checks'].setdefault(name, []).append(summary)
            del live
    del oracle
    with torch.no_grad():
        inputs, initial_dy = tensors(shape, args.seed)
        for current, value in zip(leaves, inputs, strict=True):
            current.copy_(value)
        dy.copy_(initial_dy)
    assert list(map(digest, (*leaves,dy))) == expected_hashes
    assert [v.data_ptr() for v in (*leaves,dy)] == record['input_addresses']
    for name in modes:
        deltas = []
        gc.collect()
        stream.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(5):
            temporary = calls[name]()
            del temporary
            stream.synchronize()
            gc.collect()
            deltas.append(torch.cuda.memory_allocated())
        assert len(set(deltas))==1, (name,deltas)
        record['live'][name] = dict(bytes=deltas, peak=torch.cuda.max_memory_allocated())

    save()
    for visit in range(args.visits):
        for name in (modes if visit%2==0 else reversed(modes)):
            for phase in ('forward','step'):
                graph = graphs[name,phase][0]
                _warm_measurement_graph(graph)
                values, metadata = measure_cuda_batches(graph.replay, samples=args.samples,
                    calls_per_sample=16, device=leaves[0].device, min_batch_ms=5.)
                row = dict(visit=visit, median_ms=statistics.median(values), samples_ms=values, **metadata)
                record['timings'][name][phase].append(row)
                print('TIME',visit,name,phase,f'{row["median_ms"]*1000:.3f} us',flush=True)
        save()
    record['median_us'] = {name:{phase:1e3*statistics.median(r['median_ms'] for r in rows)
                                for phase,rows in phases.items()}
                           for name, phases in record['timings'].items()}

    if args.admit:
        from rtx.autotune.effort import AutotunePolicy
        from rtx.autotune.mlp import tune_mxfp8_mlp
        policy = AutotunePolicy.preset('quick', max_trials=8, max_seconds=240, search_schedule='progressive',
                                      candidate_mix='mixed', confirmation_repeats=0)
        out = args.output.parent/('search-' + args.output.stem)
        started = time.monotonic()
        result = tune_mxfp8_mlp(*leaves, tuning_policy=policy, output_dir=out, qualification_rounds=0)
        journals = list(out.rglob('observations.jsonl'))
        assert len(journals)==1, journals
        rows = [json.loads(line) for line in journals[0].read_text().splitlines()]
        record['admissions'] = dict(seconds=time.monotonic()-started, journal=str(journals[0]), rows=rows,
                                    selected=result.config.to_dict())
        assert len(rows)==8
        first = rows[1:4]
        assert [(r['config']['gate_backend'],r['config']['down_backend']) for r in first]==[
            ('onchip','onchip'),('prequant','onchip'),('prequant','prequant')]
        assert all(r['outcome']['status']=='ok' for r in first), first
        assert all({k:v for k,v in first[0]['config'].items() if k not in ('gate_backend','down_backend')}
                   == {k:v for k,v in r['config'].items() if k not in ('gate_backend','down_backend')}
                   for r in first)
    record['status'] = 'complete'
    save()
    print('RESULT',record['median_us'],flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shape',default='128,384,256,384')
    p.add_argument('--config',type=Path)
    p.add_argument('--seed',type=int,default=17492)
    p.add_argument('--visits',type=int,default=4)
    p.add_argument('--samples',type=int,default=7)
    p.add_argument('--admit',action='store_true')
    p.add_argument('--output',type=Path,required=True)
    args = p.parse_args()
    with torch.cuda.stream(torch.cuda.Stream()):
        run(args)


if __name__ == '__main__':
    main()
