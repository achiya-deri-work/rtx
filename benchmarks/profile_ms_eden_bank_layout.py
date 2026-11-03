"""Single validated producer capture; run under NCU with profile-from-start off.

Output directory must be fresh. Profiled time is not ordinary backward latency.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--variant', choices=('control', 'xor', 'hoist', 'compact', 'group32', 'packed', 'tma'), required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--row', action='store_true', help='profile the native row dY producer instead of transpose')
    parser.add_argument('--threads', type=int, choices=(64,128,256,512), default=128)
    parser.add_argument('--blocks', type=int, default=256)
    parser.add_argument('--row-values', type=int, choices=(4,8,16), default=16)
    parser.add_argument('--register-cap', type=int, choices=(64, 72, 80, 96, 255), default=255)
    parser.add_argument('--min-blocks-hint', type=int, choices=(0, 7, 8), default=0)
    parser.add_argument('--butterfly-fma', action='store_true')
    parser.add_argument('--signs-mode', choices=('register', 'reload'), default='register')
    parser.add_argument('--combined-stores', action='store_true')
    parser.add_argument('--grouped-store-bits', type=int, choices=(8,16,32,64), default=32)
    parser.add_argument('--tma-consumer-order', choices=('retained','xor4','xor5','columns'), default='retained')
    args = parser.parse_args()
    if args.blocks < 1:
        parser.error('blocks must be positive')
    if args.row and args.variant != 'control':
        parser.error('row profiling requires native control variant')
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.combined_stores and args.variant != 'group32':
        raise ValueError('combined stores requires group32 research variant')
    if args.tma_consumer_order != 'retained' and args.variant != 'tma':
        raise ValueError('TMA consumer order requires TMA variant')
    if args.register_cap != 255 and args.variant != 'packed':
        raise ValueError('register ceiling is a packed-sign research control')
    if args.min_blocks_hint and args.variant not in ('compact', 'packed'):
        raise ValueError('occupancy hint requires compact or packed research variant')
    if args.butterfly_fma and args.variant != 'compact':
        raise ValueError('butterfly FMA probe requires compact variant')
    import torch
    from rtx.kernels.nvfp4_posthoc import compile_hadamard_preparation, PosthocScaleConfig
    from rtx.kernels.ms_eden_bank_layout import compile_bank_layout
    from rtx.kernels.ms_eden_packed_signs import compile_packed_rademacher
    from rtx.kernels.ms_eden_group32 import compile_group32
    from rtx.kernels.ms_eden_tma_preparation import compile_tma_preparation
    torch.manual_seed(941)
    source = torch.randn(4096, 1536, device='cuda', dtype=torch.bfloat16) * .25
    signs = (torch.randint(0, 2, (128,), device='cuda') * 2 - 1).float()
    kwargs = dict(load_mode='coalesced_x8', signs_mode=args.signs_mode, stages=1, async_copy=True,
                  correction_decode='native', producer_values_per_lane=args.row_values,
                  producer_transpose_values_per_lane=16, producer_grouped_store_bits=args.grouped_store_bits)
    compiler = compile_bank_layout if args.variant != 'control' else compile_hadamard_preparation
    if args.variant == 'packed':
        compiler = compile_packed_rademacher
    if args.variant == 'group32':
        compiler = compile_group32
    if args.variant == 'tma':
        compiler = compile_tma_preparation
    extra = {'address_hoist': args.variant in ('hoist', 'compact', 'group32', 'packed')} if args.variant not in ('control','tma') else {}
    if args.variant == 'tma':
        extra['consumer_order'] = args.tma_consumer_order
    if args.variant == 'group32':
        extra['combined_stores'] = args.combined_stores
    if args.variant == 'packed':
        extra['register_cap'] = args.register_cap
    if args.variant in ('compact', 'packed'):
        extra['min_blocks_hint'] = args.min_blocks_hint
    if args.variant == 'compact':
        extra['butterfly_fma'] = args.butterfly_fma
    if args.variant in ('compact', 'group32', 'packed', 'tma'):
        kwargs['producer_unroll_rows'] = False
    config = PosthocScaleConfig(args.threads, args.blocks)
    run, count = compiler(4096, 1536, config, 4, not args.row, **kwargs, **extra)
    out = (torch.empty(source.numel() // 2, device='cuda', dtype=torch.uint8),
           torch.empty(source.numel() // 16, device='cuda', dtype=torch.bfloat16),
           torch.empty(count, device='cuda'))
    reference_run, reference_count = compile_hadamard_preparation(4096, 1536, config, 4, not args.row, **kwargs)
    assert reference_count == count
    expected = tuple(torch.empty_like(x) for x in out)
    reference_run(source.flatten(), signs, *expected)
    for _ in range(5):
        run(source.flatten(), signs, *out)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    run(source.flatten(), signs, *out)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    for index, (actual, target) in enumerate(zip(out, expected)):
        if args.tma_consumer_order == 'columns' and index == 2:
            actual = actual.reshape(-1,4).amax(dim=1)
            target = target.reshape(-1,4).amax(dim=1)
        torch.testing.assert_close(actual.view(torch.uint8), target.view(torch.uint8), rtol=0, atol=0)
    assert torch.isfinite(out[1]).all() and torch.isfinite(out[2]).all()
    root = Path(__file__).resolve().parents[1]
    result = dict(variant=args.variant, row=args.row, threads=args.threads, blocks=args.blocks,
                  row_values=args.row_values, register_cap=args.register_cap, min_blocks_hint=args.min_blocks_hint,
                  butterfly_fma=args.butterfly_fma,
                  signs_mode=args.signs_mode,
                  combined_stores=args.combined_stores,
                  grouped_store_bits=args.grouped_store_bits,
                  tma_consumer_order=args.tma_consumer_order,
                  partial_equivalence='cta_max' if args.tma_consumer_order == 'columns' else 'warp_exact',
                  shape=[4096, 1536], replay_exact=True,
                  device=torch.cuda.get_device_name(), scope='single producer, not backward latency',
                  source_sha256={name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                                 for name in ('rtx/kernels/ms_eden_bank_layout.py', 'rtx/kernels/ms_eden_tma_preparation.py', 'rtx/kernels/ms_eden_group32.py', 'rtx/kernels/ms_eden_packed_signs.py', 'rtx/kernels/nvfp4_posthoc.py')})
    with args.output.open('x') as handle:
        json.dump(result, handle, indent=2)


if __name__ == '__main__':
    main()
