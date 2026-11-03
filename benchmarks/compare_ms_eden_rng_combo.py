"""Interleaved complete FP4 backward comparison of compiler RNG fusion only.

This is not the MXFP8 target comparison. Numerical, current-buffer, fresh-RNG
and memory checks are inherited from the public compiled workload builder.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
import time
from pathlib import Path


def main():
    import torch
    import torch._inductor.config as config
    from benchmarks.compare_ms_eden_pipeline import options, round_orders, gradient_metrics, shape_value
    from benchmarks.compare_mixed_nvfp4_backward import run_worker
    from benchmarks.compare_ms_eden_tf32_hadamard import timing_summary
    from rtx.autotune.measurement import measure_cuda_batches, runtime_timing_metadata
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--shape', default='decoder1536')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--experiment', choices=('rng_combo','dual_finalizer'), default='rng_combo')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    selection = json.loads(args.selection.read_text())['candidate']
    for phase in ('dx', 'dw'):
        selection[phase]['producer_grouped_store_bits'] = 32
    if args.experiment=='dual_finalizer':
        selection['dx']['producer_values_per_lane']=16
    schedules = {phase: json.dumps(selection[phase], sort_keys=True) for phase in ('dx', 'dw')}
    bundles, guards = {}, {}
    shared = None
    for name, combo in (('baseline', False), ('candidate', True)):
        patch=nullcontext()
        if args.experiment=='dual_finalizer':
            from benchmarks.experiment_ms_eden_dual_finalizer import patched
            patch=patched('both' if name=='candidate' else 'control')
        with patch, config.patch(combo_kernels=combo or args.experiment=='dual_finalizer', combo_kernels_pointwise_only=True):
            worker_args = options(shape_value(args.shape), schedules, 941, 'operand_paired', policy='fp4_fp4')
            guards[name], bundles[name] = run_worker(worker_args, capture_only=True, shared_inputs=shared)
        shared = bundles[name]['inputs']
    graphs = {name: bundle['pins'][1] for name, bundle in bundles.items()}
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        for graph in graphs.values():
            graph.replay()
        torch.cuda.synchronize()
    calibration = {}
    for name, graph in graphs.items():
        _, calibration[name] = measure_cuda_batches(graph.replay, samples=1,
            calls_per_sample=16, min_batch_ms=2., device=shared[0].device)
    count = max(value['actual_calls_per_sample'] for value in calibration.values())
    def batch(graph):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(count):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end)/count
    rows = []
    before = torch.cuda.memory_allocated()
    for index, orders in round_orders(32, 941):
        windows = []
        for order in orders:
            values = {name: batch(graphs[name]) for name in order}
            windows.append(dict(order=list(order), order_kind='AB' if order[0]=='baseline' else 'BA',
                reference_ms=values['baseline'], candidate_ms=values['candidate']))
        rows.append(dict(round=index, windows=windows,
            reference_ms=math.sqrt(math.prod(w['reference_ms'] for w in windows)),
            candidate_ms=math.sqrt(math.prod(w['candidate_ms'] for w in windows))))
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    assert before == after, 'live allocations grew during replay'
    x, w, dy = shared
    references = (dy.float()@w.detach().float(), dy.float().T@x.detach().float())
    checks = {}
    for name, bundle in bundles.items():
        graphs[name].replay()
        torch.cuda.synchronize()
        checks[name] = {phase: gradient_metrics(actual, ref, gate=.25)
            for phase, actual, ref in zip(('dx', 'dw'), bundle['pins'][3], references)}
    timing = timing_summary(rows, 941)
    timing['order_quality'] = {kind: {name: runtime_timing_metadata([
        window[key] for row in rows for window in row['windows'] if window['order_kind'] == kind])
        for name, key in (('baseline', 'reference_ms'), ('candidate', 'candidate_ms'))}
        for kind in ('AB', 'BA')}
    result = dict(scope='Interleaved complete FP4 backward: '+args.experiment,
        experiment=args.experiment, shape=shape_value(args.shape), timing=timing,
        samples=rows, guards=guards, gradient_checks=checks, schedules=selection,
        memory=dict(before=before, after=after), calibration=calibration,
        torch=torch.__version__, device=torch.cuda.get_device_name(),
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    if args.experiment=='dual_finalizer':
        root=Path(__file__).resolve().parents[1]
        result['experiment_source_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
            for name in ('rtx/kernels/ms_eden_dual_finalizer.py',
                         'rtx/experimental_ms_eden.py',
                         'benchmarks/experiment_ms_eden_dual_finalizer.py')}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result['timing'], indent=2))


if __name__ == '__main__':
    main()
