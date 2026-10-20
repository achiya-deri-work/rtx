"""Isolate split GEMM repeatability with immutable, independently scaled operands."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch

from rtx.kernels.mxfp8 import MXFP8Problem
from rtx.kernels.mxfp8_gemm import compile_mxfp8_gemm
from rtx.mlp import MXFP8MLPConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--rounds', type=int, default=128)
    parser.add_argument('--dynamic-quant', action='store_true')
    parser.add_argument('--dynamic-competitor', action='store_true')
    parser.add_argument('--competitor-phase', choices=('all', 'quant', 'gemm'), default='all')
    args = parser.parse_args()
    if args.rounds < 2:
        parser.error('at least two repetitions are required')
    if args.competitor_phase != 'all' and not args.dynamic_competitor:
        parser.error('--competitor-phase requires --dynamic-competitor')
    if args.output.exists():
        raise FileExistsError(args.output)
    mlp_config = MXFP8MLPConfig.from_dict(json.loads(args.config.read_text())['challenger'])
    config = mlp_config.gate_backward.dw
    assert config.gemm.scale_layout == 'row_major'
    problem = MXFP8Problem(1536, 256, 20480)
    launch = compile_mxfp8_gemm(problem, config.gemm,
                               split_reduction=8, reduction_tile=2560)
    torch.manual_seed(731)
    def operands():
        q = [torch.randn(rows, problem.k, device='cuda').to(torch.float8_e4m3fn)
             for rows in (problem.m, problem.n)]
        codes = [torch.randint(120, 125, (rows, problem.k//32), device='cuda',
                               dtype=torch.uint8) for rows in (problem.m, problem.n)]
        return (*q, *(v.view(torch.float8_e8m0fnu) for v in codes))
    target, competitor = operands(), operands()
    runner = None
    if args.dynamic_quant:
        from rtx.fp8_bwd import _build_matmul_runner
        runner = _build_matmul_runner(problem=problem, config=config,
                                      device=torch.device('cuda:0'))
        sources = tuple(torch.randn(problem.k, rows, device='cuda', dtype=torch.bfloat16)*.2
                        for rows in (problem.m, problem.n))
        runner.quantize(*sources)
        target = (runner.quantized_a, runner.quantized_b, runner.scales_a, runner.scales_b)
    out = torch.empty(8*problem.m*problem.n, device='cuda', dtype=torch.float32)
    other = torch.empty_like(out)
    competitor_runner = None
    if args.dynamic_competitor:
        from rtx.fp8_bwd import _build_matmul_runner
        competitor_runner = _build_matmul_runner(problem=MXFP8Problem(256, 768, 20480),
                                                 config=mlp_config.down_backward.dw,
                                                 device=torch.device('cuda:0'))
        competitor_sources = tuple(torch.randn(20480, rows, device='cuda', dtype=torch.bfloat16)*.2
                                   for rows in (256, 768))
        competitor_result = torch.empty(256, 768, device='cuda', dtype=torch.bfloat16)
        competitor_runner.quantize(*competitor_sources)
    # Independent FP32, no quantizer/MLP/stream-cache dependency.
    a, b = [q.float()*s.float().repeat_interleave(32, dim=1)
            for q, s in ((target[0], target[2]), (target[1], target[3]))]
    expected = torch.stack([a[:,i:i+2560] @ b[:,i:i+2560].T
                            for i in range(0, problem.k, 2560)]).flatten()
    del a, b
    caller = torch.cuda.current_stream()
    parallel = torch.cuda.Stream()
    torch.cuda.synchronize()
    first = None
    rows = []
    for iteration in range(args.rounds):
        parallel.wait_stream(caller)
        with torch.cuda.stream(parallel):
            for _ in range(3):
                if competitor_runner is None:
                    launch(*competitor, other)
                else:
                    if args.competitor_phase in ('all', 'quant'):
                        competitor_runner.quantize(*competitor_sources)
                    if args.competitor_phase in ('all', 'gemm'):
                        competitor_runner.matmul(competitor_result)
        if runner is not None:
            runner.quantize(*sources)
        launch(*target, out)
        caller.wait_stream(parallel)
        caller.synchronize()
        if first is None:
            first = out.clone()
        delta = out-expected
        row = dict(iteration=iteration, equal_first=torch.equal(out, first),
                   finite=bool(out.isfinite().all()), max_abs=float(delta.abs().max()),
                   relative_l2=float(delta.norm()/expected.norm()))
        rows.append(row)
        print(json.dumps(row), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(dict(rows=rows, config=asdict(config), dynamic_quant=args.dynamic_quant,
                       dynamic_competitor=args.dynamic_competitor,
                       competitor_phase=args.competitor_phase,
                       torch=str(torch.__version__), seed=731,
                       cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
                       source_sha256={name: hashlib.sha256(
                           (Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
                           for name in ('rtx/kernels/mxfp8_gemm.py', 'rtx/fp8_bwd.py',
                                        'tools/check_split_gemm_concurrency.py')},
                       scope='fixed BF16 inputs; FP32 reference over dequantized operands; diagnostic only'), stream, indent=2)
    assert all(r['equal_first'] and r['finite'] and r['relative_l2'] < 1e-5 for r in rows)


if __name__ == '__main__':
    main()
