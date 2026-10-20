"""Diagnostic-only restoration of unsafe overlap; never installs winners.

Remaining arguments are forwarded to check_mlp_repeatability.py. Production
keeps conservative ordering. Use fresh outputs and processes for each control.
"""
import argparse
import json
from pathlib import Path
import runpy
import sys

import torch
import rtx.mlp as mlp
from rtx.fp8_bwd import _launch_bwd_out, _launch_dw_out, _resolve_bwd_runner


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--control', required=True,
                        choices=('concurrent', 'clone_inputs', 'sync_before', 'prequant_before'))
    args, remaining = parser.parse_known_args()
    # Optional post-join inspection never changes ordering within the fork.
    inspect_stages = '--inspect-stages' in remaining
    remaining = [arg for arg in remaining if arg != '--inspect-stages']
    first_stages = {}
    stage_iteration = 0

    def concurrent(dp, x, gu, dy, hidden, down, gate_key, down_key):
        nonlocal stage_iteration
        dx, dg, dd = torch.empty_like(x), torch.empty_like(gu), torch.empty_like(down)
        with torch.cuda.device(x.device):
            caller = torch.cuda.current_stream(x.device)
            parallel = mlp._overlap_stream(x.device.index, int(caller.cuda_stream))
            if args.control == 'clone_inputs':
                dp, hidden = dp.clone(), hidden.clone()
            if args.control == 'sync_before':
                caller.synchronize()
            if args.control == 'prequant_before':
                _, runner, *_ = _resolve_bwd_runner(dp, x, gu, gate_key)
                runner.dx.quantize(dp, gu)
                runner.dw.quantize(dp, x)
            parallel.wait_stream(caller)
            with torch.cuda.stream(parallel):
                _launch_dw_out(dy, hidden, down, down_key, dd)
            if args.control == 'prequant_before':
                runner.dx.matmul(dx)
                runner.dw.matmul(dg)
            else:
                _launch_bwd_out(dp, x, gu, gate_key, dx, dg)
            caller.wait_stream(parallel)
            if inspect_stages:
                caller.synchronize()
                _, runner, *_ = _resolve_bwd_runner(dp, x, gu, gate_key)
                stages = {'dp': dp, 'hidden': hidden, 'dx': dx, 'dgate': dg}
                for role in ('dx', 'dw'):
                    matmul = getattr(runner, role)
                    for name in ('quantized_a', 'quantized_b', 'scales_a', 'scales_b', 'workspace'):
                        value = getattr(matmul, name, None)
                        if value is not None:
                            stages[f'{role}.{name}'] = value
                changed = {}
                for name, tensor in stages.items():
                    data = tensor.detach().view(torch.uint8)
                    if name not in first_stages:
                        first_stages[name] = data.clone()
                    changed[name] = int((data != first_stages[name]).sum())
                print('STAGES ' + json.dumps(dict(iteration=stage_iteration,
                                                 changed_bytes=changed)), flush=True)
                stage_iteration += 1
        return dx, dg, dd

    mlp._parallel_backward_op = concurrent
    print(f'DIAGNOSTIC unsafe overlap control={args.control}; no timing/installation', flush=True)
    sys.argv = ['check_mlp_repeatability.py', *remaining,
                '--diagnostic-overlap-control', args.control]
    runpy.run_path(str(Path(__file__).with_name('check_mlp_repeatability.py')), run_name='__main__')


if __name__ == '__main__':
    main()
