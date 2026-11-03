"""Isolate compiler RNG fusion; not a complete-backward performance claim."""
import argparse
import json
import statistics
from pathlib import Path


def main():
    import torch
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    results = []
    for m, n, k in ((4096, 512, 512), (4096, 1536, 512), (4096, 2048, 2048)):
        def draw():
            ds = torch.where(torch.rand(128, device="cuda") < .5, -1., 1.)
            da = torch.rand(m*n//16, device="cuda")
            db = torch.rand(n*k//16, device="cuda")
            ws = torch.where(torch.rand(128, device="cuda") < .5, -1., 1.)
            wa = torch.rand(m*n//16, device="cuda")
            wb = torch.rand(m*k//16, device="cuda")
            return ds, da, db, ws, wa, wb
        compiled = [torch.compile(draw, fullgraph=True, dynamic=False, options={
            "triton.cudagraphs": False, "combo_kernels": combo,
            "combo_kernels_pointwise_only": True}) for combo in (False, True)]
        from torch._inductor import metrics
        kernel_counts = []
        for fn in compiled:
            before = metrics.generated_kernel_count
            fn()
            kernel_counts.append(metrics.generated_kernel_count - before)
        torch.cuda.synchronize()
        for seed in (941, 942, 943):
            torch.manual_seed(seed)
            baseline = compiled[0]()
            state = torch.cuda.get_rng_state()
            torch.manual_seed(seed)
            candidate = compiled[1]()
            assert all(torch.equal(a, b) for a, b in zip(baseline, candidate)), "RNG output mismatch"
            assert torch.equal(state, torch.cuda.get_rng_state()), "RNG consumption mismatch"
        timings = []
        for fn in compiled:
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(3):
                    output = fn()
            stream.synchronize()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=stream):
                for _ in range(32):
                    output = fn()
            graph.replay()
            first = tuple(t.clone() for t in output)
            graph.replay()
            assert any(not torch.equal(a, b) for a, b in zip(first, output)), "stale graph RNG"
            samples = []
            for _ in range(30):
                start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                start.record()
                for _ in range(128):
                    graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end)/(128*32))
            timings.append(dict(median_ms=statistics.median(samples), samples_ms=samples))
        results.append(dict(shape=[m,n,k], exact_rng_and_consumption=True,
                            generated_kernel_counts=kernel_counts, calls_per_graph=32,
                            baseline=timings[0], combo=timings[1]))
        print(json.dumps(results[-1]), flush=True)
        torch.compiler.reset()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(torch=torch.__version__, device=torch.cuda.get_device_name(),
        scope="isolated RNG diagnostic; sequential timing, not acceptance", results=results), indent=2))


if __name__ == "__main__":
    main()
