"""Bounded, standalone reproduction of an Inductor-generated BMM candidate.

Use a fresh process and an explicitly restricted CUDA_VISIBLE_DEVICES. Direct
mode launches exactly one preserved candidate, never an autotuning sweep.
Compile mode compares the same BMM against selectable Inductor GEMM backends.
Both modes validate against a separate FP32 BMM before reporting success.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback


def geometry(path):
    """Extract literal geometry from the preserved generated source, CPU only."""
    tree = ast.parse(path.read_text())
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == 'triton_bmm')
    result = {}
    for node in function.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    result[target.id] = node.value.value
        elif isinstance(node, ast.AnnAssign) and isinstance(node.value, ast.Constant):
            result[node.target.id] = node.value.value
    for decorator in function.decorator_list:
        if isinstance(decorator, ast.Call):
            for keyword in decorator.keywords:
                if keyword.arg in ('num_stages', 'num_warps'):
                    result[keyword.arg] = ast.literal_eval(keyword.value)
    required = ('M', 'N', 'K', 'BATCH', 'stride_aq', 'stride_am', 'stride_ak',
                'stride_bq', 'stride_bk', 'stride_bn', 'BLOCK_M', 'BLOCK_N',
                'num_warps', 'num_stages')
    if any(name not in result for name in required):
        raise ValueError('Unsupported generated BMM source: missing geometry')
    return {name: result[name] for name in required}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kernel', type=Path, required=True)
    parser.add_argument('--mode', choices=('health', 'direct', 'compile'), default='direct')
    parser.add_argument('--backends', default='ATEN,TRITON')
    parser.add_argument('--subprocess', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    g = geometry(args.kernel)
    record = dict(status='running', mode=args.mode, geometry=g,
                  kernel=str(args.kernel.resolve()),
                  kernel_sha256=hashlib.sha256(args.kernel.read_bytes()).hexdigest(),
                  cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                  backends=args.backends, autotune_in_subproc=args.subprocess)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        temporary = args.output.with_suffix('.tmp')
        temporary.write_text(json.dumps(record, indent=2, allow_nan=False)+'\n')
        temporary.replace(args.output)

    save()
    try:
        import torch
        record.update(torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0))
        torch.manual_seed(314159)
        probe = torch.arange(1024, device='cuda', dtype=torch.float32)
        torch.testing.assert_close((probe+1).cpu(), torch.arange(1, 1025).float())
        torch.cuda.synchronize()
        record['health_probe'] = 'passed'
        save()
        if args.mode != 'health':
            a = torch.empty_strided((g['BATCH'], g['M'], g['K']),
                (g['stride_aq'], g['stride_am'], g['stride_ak']), device='cuda', dtype=torch.bfloat16)
            b = torch.empty_strided((g['BATCH'], g['K'], g['N']),
                (g['stride_bq'], g['stride_bk'], g['stride_bn']), device='cuda', dtype=torch.bfloat16)
            a.normal_(std=.1); b.normal_(std=.1)
            expected = torch.bmm(a.float(), b.float())
            torch.cuda.synchronize()
            record['oracle'] = 'fp32_bmm'
            save()
            if args.mode == 'direct':
                spec = importlib.util.spec_from_file_location('preserved_bmm_candidate', args.kernel)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                output = torch.full((g['BATCH'], g['M'], g['N']), float('nan'),
                                    device='cuda', dtype=torch.bfloat16)
                grid = (((g['M']+g['BLOCK_M']-1)//g['BLOCK_M']) *
                        ((g['N']+g['BLOCK_N']-1)//g['BLOCK_N']), g['BATCH'], 1)
                module.triton_bmm.fn[grid](a, b, output,
                    num_warps=g['num_warps'], num_stages=g['num_stages'])
            else:
                compiled = torch.compile(torch.bmm, fullgraph=True, dynamic=False,
                    options={'max_autotune': True, 'triton.cudagraphs': False,
                             'max_autotune_gemm_backends': args.backends,
                             'autotune_in_subproc': args.subprocess})
                output = compiled(a, b)
            torch.cuda.synchronize()
            delta = output.float()-expected
            if not bool(output.isfinite().all()):
                raise AssertionError('Nonfinite generated BMM output')
            record.update(relative_l2=float(delta.norm()/expected.norm()),
                          max_abs=float(delta.abs().max()))
            torch.testing.assert_close(output.float(), expected, rtol=.03, atol=.01)
        record['status'] = 'passed'
    except Exception as error:
        record.update(status='failed', error=f'{type(error).__name__}: {error}')
        save()
        traceback.print_exc()
        raise
    save()
    print(json.dumps(record), flush=True)


if __name__ == '__main__':
    main()
