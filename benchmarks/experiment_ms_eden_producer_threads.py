"""Research-only dW/dY producer CTA sweep; complete guarded backward timing."""
import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--case', nargs=2, type=int, metavar=('THREADS', 'BLOCKS'))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.case is None:
        # Fixed CTA count and approximately fixed total launched threads.
        for threads, blocks in ((128, 256), (64, 256), (64, 512),
                                (256, 256), (256, 128), (512, 64)):
            with (args.output / f't{threads}-b{blocks}.log').open('x') as log:
                completed = subprocess.run(
                    [sys.executable, '-m', 'benchmarks.experiment_ms_eden_producer_threads',
                     '--selection', str(args.selection), '--output', str(args.output),
                     '--case', str(threads), str(blocks)],
                    stdout=log, stderr=subprocess.STDOUT,
                )
            print(f'threads={threads} blocks={blocks} status={completed.returncode}', flush=True)
            if completed.returncode:
                raise SystemExit(completed.returncode)
        return

    import torch._inductor.config as compiler_config
    from benchmarks.compare_ms_eden_pipeline import compare
    from benchmarks.experiment_ms_eden_operand_order import exact_smoke

    threads, blocks = args.case
    selection = json.loads(args.selection.read_text())['candidate']
    for phase in ('dx', 'dw'):
        selection[phase]['producer_grouped_store_bits'] = 32
    selection['dx']['producer_values_per_lane'] = 16
    geometry = dict(selection['dw'].get('a_geometry') or {})
    geometry.update(producer_threads=threads, producer_blocks=blocks)
    selection['dw']['a_geometry'] = geometry
    schedules = {phase: json.dumps(selection[phase], sort_keys=True) for phase in ('dx', 'dw')}
    shape = (4096, 1536, 512)
    checks = exact_smoke(shape, schedules, 941)
    prefix = args.output / f't{threads}-b{blocks}'
    with compiler_config.patch(combo_kernels=True, combo_kernels_pointwise_only=True):
        result = compare(
            shape, prefix.with_suffix('.raw.json'), {},
            dx_overrides=selection['dx'], dw_overrides=selection['dw'],
            pipeline='operand_paired', rounds=24, replays=16, min_batch_ms=2.,
            warmup_ms=15000., seed=941, baseline_policy='stable_mxfp8',
            scopes=('backward',),
        )
    result['producer_geometry_experiment'] = dict(
        operand='dw_a', threads=threads, blocks=blocks, exact=checks,
        compiler_options=dict(combo_kernels=True, combo_kernels_pointwise_only=True),
    )
    with prefix.with_suffix('.json').open('x') as handle:
        json.dump(result, handle, indent=2)


if __name__ == '__main__':
    main()
