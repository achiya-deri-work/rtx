"""Bounded diagnostic for alternating independent compiled backward graphs.

Use an external process timeout: a CUDA wait cannot be bounded by a Python
deadline. This is a reliability probe, never a performance measurement.
"""
import argparse
import faulthandler
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pair', choices=('stable_stable', 'stable_fp4', 'fp4_fp4'), required=True)
    parser.add_argument('--pipeline', choices=('serial', 'operand_paired'), default='operand_paired')
    parser.add_argument('--shape', default='decoder512')
    parser.add_argument('--selection', type=Path,
                        help='Optional retained producer selection matching the full comparator')
    parser.add_argument('--seconds', type=float, default=15.)
    parser.add_argument('--independent-inputs', action='store_true')
    parser.add_argument('--sync-each', action='store_true')
    parser.add_argument('--scopes', nargs='+', choices=('backward', 'total'),
                        default=['backward'],
                        help='Replay order within each policy; backward total matches comparator warmup')
    parser.add_argument('--exact-preflight', action='store_true',
                        help='Run the same compiled exactness/graph preflight as the full experiment')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not 0 < args.seconds <= 60:
        raise ValueError('diagnostic duration must be in (0,60] seconds')
    import torch
    import torch._inductor.config as compiler_config
    from benchmarks.compare_ms_eden_pipeline import options, schedule_pair, shape_value, gradient_metrics
    from benchmarks.compare_mixed_nvfp4_backward import run_worker
    policies = {'stable_stable': ('stable_mxfp8', 'stable_mxfp8'),
                'stable_fp4': ('stable_mxfp8', 'fp4_fp4'),
                'fp4_fp4': ('fp4_fp4', 'fp4_fp4')}[args.pair]
    schedules = schedule_pair()
    if args.selection is not None:
        selected = json.loads(args.selection.read_text())['candidate']
        for phase in ('dx', 'dw'):
            selected[phase]['producer_grouped_store_bits'] = 32
        selected['dx']['producer_values_per_lane'] = 16
        schedules = {phase: json.dumps(selected[phase], sort_keys=True) for phase in ('dx', 'dw')}
    bundles, metadata = [], []
    shared = None
    faulthandler.enable()
    faulthandler.dump_traceback_later(30, repeat=True)
    try:
        if args.exact_preflight:
            from benchmarks.experiment_ms_eden_operand_order import exact_smoke
            print('EXACT PREFLIGHT', flush=True)
            exact_smoke(shape_value(args.shape), schedules, 941)
            print('EXACT PREFLIGHT PASSED', flush=True)
        with compiler_config.patch(combo_kernels=True, combo_kernels_pointwise_only=True):
            for index, policy in enumerate(policies):
                print(f'BUILD {index} {policy}', flush=True)
                opts = options(shape_value(args.shape), schedules, pipeline=args.pipeline if policy == 'fp4_fp4' else 'serial', policy=policy)
                info, bundle = run_worker(opts, capture_only=True, shared_inputs=shared)
                metadata.append(info)
                bundles.append(bundle)
                if not args.independent_inputs:
                    shared = bundle['inputs']
                print(f'BUILT {index}', flush=True)
            torch.cuda.synchronize()
            before = torch.cuda.memory_allocated()
            deadline = time.monotonic() + args.seconds
            iterations = 0
            while time.monotonic() < deadline:
                for index, bundle in enumerate(bundles):
                    if iterations < 3:
                        print(f'REPLAY {iterations} {index}', flush=True)
                    for scope in args.scopes:
                        graph = bundle['pins'][1] if scope == 'backward' else bundle['graph']
                        graph.replay()
                        if args.sync_each:
                            torch.cuda.synchronize()
                torch.cuda.synchronize()
                iterations += 1
                if iterations % 1000 == 0:
                    print(f'PROGRESS {iterations}', flush=True)
            growth = torch.cuda.memory_allocated() - before
            assert growth == 0, growth
            checks = []
            for policy, bundle in zip(policies, bundles):
                x, w, dy = bundle['inputs']
                refs = (dy.float() @ w.detach().float(), dy.float().T @ x.detach().float())
                checks.append({scope: [
                    gradient_metrics(actual, ref, gate=.06 if policy == 'stable_mxfp8' else .25)
                    for actual, ref in zip(
                        bundle['pins'][3] if scope == 'backward' else bundle['grads'], refs)]
                    for scope in args.scopes})
            result = dict(pair=args.pair, pipeline=args.pipeline, shape=shape_value(args.shape),
                          shared_inputs=not args.independent_inputs, sync_each=args.sync_each,
                          scopes=args.scopes,
                          exact_preflight=args.exact_preflight,
                          iterations=iterations, seconds=args.seconds, live_growth_bytes=growth,
                          gradient_checks=checks, build_metadata=metadata,
                          scope='graph interleaving reliability only; no timing claim')
            with args.output.open('x') as handle:
                json.dump(result, handle, indent=2)
            print(f'PASS iterations={iterations}', flush=True)
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == '__main__':
    main()
