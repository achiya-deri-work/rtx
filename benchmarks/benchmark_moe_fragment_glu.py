"""Paired forward packed expert GLU timing with a crossed geometry control."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
from statistics import median
from time import perf_counter


def _prepare(data, config):
    from rtx._moe_execution import allocate, layout, plan
    from rtx.moe import _config_key
    from rtx.runtime import load_kernel_symbol

    x, gu, down, ids, weights, inverse = (data[name] for name in ('x', 'gu', 'down', 'ids', 'weights', 'inverse'))
    n, d = x.shape
    e, c = ids.shape
    h, k = down.shape[-1], inverse.shape[-1]
    key, dtypes, sizes, views, _ = layout((n, d, e, c, k, h), _config_key(config), 'forward')
    builder, *_ = plan(key)
    buffers = allocate(x, dtypes, sizes, views)
    inputs = x, gu.flatten(0, 1), down.flatten(0, 1), ids, weights, inverse

    def value(ref):
        if isinstance(ref, int):
            return inputs[-ref-1] if ref < 0 else buffers[ref]
        return tuple(value(v) for v in ref)

    for module, symbol, args, refs in builder.steps:
        if symbol == 'compile_expert_packed_gemm' and args[3] == 'glu':
            return value(refs)[:4], args[1]
        load_kernel_symbol(module, symbol)(*args)(*value(refs))
    raise ValueError('Expected grouped prepared packed forward GLU')


def _identity(torch, fixture, archived):
    from benchmarks.benchmark_moe_execution import _tensor_digest
    digest = {name: _tensor_digest(fixture[name]) for name in ('x', 'gu', 'down', 'ids', 'weights', 'inverse', 'dy')}
    identity = archived['identities']
    assert digest['x'] == identity['input']
    assert _tensor_digest(fixture['dy'].bfloat16()) == identity['upstream']
    for name, key in (('ids', 'expert_ids'), ('inverse', 'inverse')):
        assert digest[name] == identity['routing'][key]
    for expert, (gu, down) in enumerate(zip(fixture['gu'], fixture['down'])):
        gate, up = gu.chunk(2)
        leaves = identity['parameters']
        assert _tensor_digest(gate) == leaves[f'gate_up.projections.e{expert}_gate.weight']
        assert _tensor_digest(up) == leaves[f'gate_up.projections.e{expert}_up.weight']
        assert _tensor_digest(down) == leaves[f'down.projections.e{expert}.weight']
    return digest


def _artifact(code, directory, label):
    for suffix in ('ptx', 'cubin'):
        if contents := getattr(code, f'__{suffix}__', None):
            path = directory / f'{label}.{suffix}'
            if isinstance(contents, bytes):
                path.write_bytes(contents)
            elif '\n' in contents:
                path.write_text(contents)
            else:
                # Cached compilations may return artifact paths; take a
                # private copy before another specialization reuses the name.
                shutil.copyfile(contents, path)


def _measure(torch, calls, rounds, samples, iters):
    from rtx.autotune.preparation_runtime import _warm_measurement_graph

    torch.cuda.synchronize()
    graphs = {}
    for label, call in calls.items():
        for _ in range(3):
            call()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            call()
        graphs[label] = g
    labels = tuple(graphs)
    timings = {mode: {label: [] for label in labels} for mode in ('warmed_graph', 'ordinary_launch')}
    torch.cuda.synchronize()
    live = [torch.cuda.memory_allocated()]
    for visit in range(rounds):
        offset = visit % len(labels)
        order = labels[offset:] + labels[:offset]
        if visit % 2:
            order = order[::-1]
        for label in order:
            _warm_measurement_graph(graphs[label])
            for mode, call in (('warmed_graph', graphs[label].replay), ('ordinary_launch', calls[label])):
                values = []
                for _ in range(samples):
                    a, b = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                    a.record()
                    for _ in range(iters):
                        call()
                    b.record()
                    b.synchronize()
                    values.append(a.elapsed_time(b)*1000/iters)
                timings[mode][label].append(values)
        live.append(torch.cuda.memory_allocated())
    assert len(set(live)) == 1, live
    return dict(rounds_us=timings, median_us={mode: {label: median(map(median, visits))
                                                    for label, visits in values.items()}
                                            for mode, values in timings.items()},
                live_bytes=live, peak_allocated_bytes=torch.cuda.max_memory_allocated())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-result', type=Path, required=True)
    p.add_argument('--compact-result', type=Path, required=True)
    p.add_argument('--fixture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--subset', type=int, nargs=2, metavar=('EXPERTS', 'CAPACITY'))
    p.add_argument('--rounds', default=6, type=int)
    p.add_argument('--samples', default=7, type=int)
    p.add_argument('--iters', default=24, type=int)
    p.add_argument('--artifact-dir', type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.rounds, args.samples, args.iters) < 1:
        p.error('positive measurement counts required')
    if args.artifact_dir:
        scratch = args.artifact_dir / 'compiler'
        scratch.mkdir(parents=True, exist_ok=True)
        os.environ['CUTE_DSL_KEEP'] = 'ptx,cubin'
        os.environ['CUTE_DSL_DUMP_DIR'] = str(scratch)
        os.environ['CUTE_DSL_NO_CACHE'] = '1'

    import torch
    import cutlass
    from benchmarks.benchmark_moe_execution import _python_source_digest, _tensor_digest, write
    from benchmarks.profile_moe import hardware_attributes
    from rtx.configs.moe_backward import fragment_glu_scratch_bytes
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.kernels.mxfp8_moe_buffers import compile_expert_packed_gemm
    from rtx.moe import MXFP8MoEConfig

    archived = json.loads(args.source_result.read_text())
    base = MXFP8MoEConfig.from_dict(archived['expert_config'])
    paired = json.loads(args.compact_result.read_text())
    compact = MXFP8MoEConfig.from_dict(paired['configs']['forward_n64']['expert'])
    assert base.prepared.forward_engine == 'packed' and not base.prepared.active_prefix
    fixture = torch.load(args.fixture, map_location='cpu', weights_only=True)
    digest = _identity(torch, fixture, archived)
    data = {name: fixture[name].cuda() for name in ('x', 'gu', 'down', 'ids', 'weights', 'inverse')}
    del fixture
    inputs, problem = _prepare(data, base)
    e = data['ids'].shape[0]
    m, h, k = problem.m, problem.n // 2, problem.k
    assert m == base.prepared.physical_capacity(data['ids'].shape[1])
    del data
    if args.subset:
        es, ms = args.subset
        if not 2 <= es <= e or not 2 <= ms <= m:
            p.error('subset experts and capacity must fit the exact fixture')
        qx, qw, sx, sw = inputs
        inputs = (qx.view(e, m, k)[:es, :ms].reshape(es*ms, k).contiguous(),
                  qw[:es*2*h], sx.view(e, m, k//32)[:es, :ms].reshape(es*ms, k//32).contiguous(),
                  sw[:es*2*h])
        e, m = es, ms
        problem = MXFP8Problem(m, 2*h, k)
    w = (h+31)//32*32
    engines = 'legacy', 'fragment_direct', 'fragment_packed'
    calls, outputs, schedules = {}, {}, {}
    for prefix, config in (('source', base.prepared.gate_up), ('compact', compact.prepared.gate_up)):
        for engine in engines:
            label = f'{prefix}_{engine}'
            start = perf_counter()
            code = compile_expert_packed_gemm(e, problem, config, 'glu', glu_engine=engine)
            schedules[label] = dict(engine=engine, config=asdict(config), compile_seconds=perf_counter()-start,
                                    explicit_epilogue_smem=(config.tile_m*config.tile_n*2 if engine == 'legacy'
                                                            else fragment_glu_scratch_bytes(config, engine)))
            if args.artifact_dir:
                _artifact(code, args.artifact_dir, label)
            q = torch.empty((e*m, w), device='cuda', dtype=torch.float8_e4m3fn)
            s = torch.empty((e*m, w//32), device='cuda', dtype=torch.uint8)
            calls[label] = lambda code=code, q=q, s=s: code(*inputs, q, s)
            outputs[label] = (q.view(torch.uint8), s)
            calls[label]()
            torch.cuda.synchronize()

    reference = outputs['source_legacy']
    changes = {}
    for label, (q, s) in outputs.items():
        checks = dict(data_unequal=int(torch.count_nonzero(q != reference[0])),
                      scale_unequal=int(torch.count_nonzero(s != reference[1])))
        changes[label] = checks
        assert not any(checks.values()), (label, checks)

    result = dict(fixture=str(args.fixture), reference_source=str(args.source_result),
                  compact_source=str(args.compact_result), exact_archive_match=True, source_hashes=digest,
                  packed_hashes=[_tensor_digest(v) for v in inputs], e=e, m=m, n=2*h, k=k,
                  subset=args.subset, source_digest=_python_source_digest(Path(__file__).resolve().parents[1]),
                  schedules=schedules,
                  torch=str(torch.__version__), cuda=torch.version.cuda, cute=str(cutlass.__version__),
                  gpu=torch.cuda.get_device_name(), hardware=hardware_attributes(), checks=changes,
                  **_measure(torch, calls, args.rounds, args.samples, args.iters))
    write(args.output, result)
    print(json.dumps(result['median_us'], indent=2))


if __name__ == '__main__':
    main()
