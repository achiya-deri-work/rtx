"""Isolated full-model batch capacity probes; not kernel autotuning or convergence."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('nvfp4_full', 'nvfp4_mxfp8'), required=True)
    parser.add_argument('--tokens', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-batch', type=int, default=256)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    records = []

    def probe(batch):
        output = args.output / f'batch-{batch}.jsonl'
        command = [sys.executable, '-u', '-m', 'benchmarks.train_precision20m',
            '--tokens', str(args.tokens), '--output', str(output), '--mode', args.mode,
            '--layers', '16', '--width', '1024', '--intermediate', '2816', '--heads', '16',
            '--sequence', '512', '--batch', str(batch), '--corpus-tokens', '100000000',
            '--steps', '16', '--warmup-steps', '2', '--eval-every', '16', '--eval-batches', '2',
            '--log-interval', '8', '--checkpoint-interval', '16', '--mlp-activation', 'situ',
            '--cudagraphs', '--combo-kernels']
        if args.mode == 'nvfp4_full':
            command += ['--ms-eden-backward-pipeline', 'paired']
        started = time.monotonic()
        log = output.with_suffix('.log')
        print(f'PROBE {args.mode} batch={batch}', flush=True)
        with log.open('x') as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=900)
        rows = [json.loads(line) for line in output.read_text().splitlines()] if output.exists() else []
        terminal = next((r for r in reversed(rows) if r['event'] == 'result'), None)
        if result.returncode:
            if 'out of memory' not in log.read_text().lower():
                raise RuntimeError(f'non-OOM probe failure; inspect {log}')
            record = dict(batch=batch, status='oom')
        elif terminal is None or terminal['interrupted'] or not terminal['finite_parameters']:
            raise RuntimeError(f'incomplete or nonfinite probe: {output}')
        else:
            safe = terminal['peak_reserved_bytes'] <= .95 * terminal['device_total_bytes']
            record = dict(batch=batch, status='safe' if safe else 'headroom', result=terminal,
                tokens_per_s=terminal['tokens']/terminal['training_wall_s'])
        record.update(probe_wall_s=time.monotonic()-started, command=command,
                      cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
        records.append(record)
        with (args.output/'probes.jsonl').open('a') as stream:
            stream.write(json.dumps(record)+'\n')
        print(json.dumps(record), flush=True)
        return record['status']=='safe'

    low, high = 0, None
    batch = 16
    while batch <= args.max_batch:
        if not probe(batch):
            high = batch
            break
        low = batch
        batch *= 2
    # Refine the capacity boundary in increments of eight sequences.
    while high is not None and high-low > 8:
        middle = ((low+high)//16)*8
        if middle <= low:
            break
        if probe(middle): low = middle
        else: high = middle
    if not low:
        raise RuntimeError('no safe batch found')
    summary = dict(mode=args.mode, maximum_safe_tested_batch=low,
                   first_unsafe_batch=high, search_capped=high is None,
                   headroom_fraction=.05, step=8, records=records,
                   requires_fresh_tuning_and_tuned_model_requalification=True)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
