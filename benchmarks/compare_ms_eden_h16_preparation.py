"""Diagnostic row-producer timings, never a complete-backward speedup claim."""
import argparse
import hashlib
import json
import random
import statistics
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shape', choices=('4096,512','4096,1536','4096,2048'))
    parser.add_argument('--profile', choices=('retained','h16_native','h16_grouped'),
                        help='Capture one validated producer; no ordinary timing')
    parser.add_argument('--recompute-coefficients',action='store_true',
                        help='Use per-iteration coefficients in the grouped H16 candidate')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    import torch
    from rtx.kernels.nvfp4_posthoc import PosthocScaleConfig, compile_hadamard_preparation
    from rtx.kernels.ms_eden_h16_preparation import compile_h16_preparation
    from benchmarks.compare_ms_eden_tf32_hadamard import dense_fp64_oracle
    torch.manual_seed(973)
    order_rng = random.Random(973)
    results = []
    if args.profile and not args.shape:
        parser.error('--profile requires a single --shape')
    shapes=(tuple(map(int,args.shape.split(','))),) if args.shape else ((4096,512),(4096,1536),(4096,2048))
    for shape in shapes:
        print(f'START shape={shape}',flush=True)
        source = torch.randn(shape, device='cuda', dtype=torch.bfloat16) * .25
        signs = (torch.randint(2, (128,), device='cuda') * 2 - 1).float()
        config = PosthocScaleConfig(128, 256)
        variants = {
            'retained': compile_hadamard_preparation(*shape, config,
                correction_decode='native', producer_values_per_lane=16,
                producer_grouped_store_bits=64),
            'h16_native': compile_h16_preparation(*shape, config),
            'h16_grouped': compile_h16_preparation(*shape, config, grouped_output=True,
                recompute_coefficients=args.recompute_coefficients),
        }
        graphs, outputs, errors = {}, {}, {}
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        for name, (run, count) in variants.items():
            print(f'CAPTURE shape={shape} variant={name}',flush=True)
            out = (torch.empty(source.numel()//2, device='cuda', dtype=torch.uint8),
                   torch.empty(source.numel()//16, device='cuda', dtype=torch.bfloat16),
                   torch.empty(count, device='cuda'))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    run(source.flatten(), signs, *out)
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(8):
                    run(source.flatten(), signs, *out)
            graphs[name], outputs[name] = graph, out
        torch.cuda.synchronize()
        oracle = dense_fp64_oracle(source, signs)
        lut = torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],device='cuda')
        decoded = {}
        for name, (packed, scales, partials) in outputs.items():
            assert torch.isfinite(scales).all() and torch.isfinite(partials).all()
            torch.testing.assert_close(partials.max(), scales.float().max(), rtol=0, atol=0)
            codes = torch.stack((packed & 15, packed >> 4), 1).flatten().long()
            decoded[name] = (lut[codes] * scales.float().repeat_interleave(16)).double()
            errors[name] = float((decoded[name]-oracle).norm()/oracle.norm())
        for name in graphs:
            assert errors[name] <= errors['retained'] + 1e-4
            assert float((decoded[name]-decoded['retained']).norm()/oracle.norm()) <= .005
        print(f'VALIDATED shape={shape}',flush=True)
        if args.profile:
            run,_=variants[args.profile]
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()
            run(source.flatten(),signs,*outputs[args.profile])
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()
            results.append(dict(shape=shape,variant=args.profile,decoded_nrms=errors,
                                scope='single validated producer profiler capture'))
            continue
        deadline = time.monotonic()+5
        while time.monotonic() < deadline:
            for graph in graphs.values():
                graph.replay()
            torch.cuda.synchronize()
        samples = {name: [] for name in graphs}
        before = torch.cuda.memory_allocated()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        for _ in range(24):
            order = list(graphs)
            order_rng.shuffle(order)
            for name in order:
                start.record(); graphs[name].replay(); end.record(); end.synchronize()
                samples[name].append(start.elapsed_time(end)*1000/8)
        growth = torch.cuda.memory_allocated()-before
        assert growth == 0
        result = dict(shape=shape, samples_us=samples, decoded_nrms=errors,
                      median_us={k:statistics.median(v) for k,v in samples.items()},
                      live_growth_bytes=growth)
        results.append(result)
        print(json.dumps(result), flush=True)
    root = Path(__file__).resolve().parents[1]
    paths = ('rtx/kernels/ms_eden_h16_preparation.py', 'rtx/kernels/nvfp4_posthoc.py',
             'benchmarks/compare_ms_eden_h16_preparation.py')
    with args.output.open('x') as handle:
        json.dump(dict(scope='isolated row producer; diagnostic, not full backward',
                       profile=args.profile,
                       recompute_coefficients=args.recompute_coefficients,
                       device=torch.cuda.get_device_name(), torch=torch.__version__,
                       cuda=torch.version.cuda, results=results,
                       source_sha256={p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}),
                  handle, indent=2)


if __name__ == '__main__':
    main()
