"""Matched, warmed CUDA graph probes for bounded packed Sinkhorn.

This low-level probe isolates transport only (no packing/QB or metadata).
The whole-layer comparisons use benchmark_moe_varlen.py and fixed schedules.
"""
import argparse
from dataclasses import replace
from itertools import accumulate
import json
from pathlib import Path
import statistics

import torch


def main():
    from rtx._routing_varlen import layout, _flat, sinkhorn_execute, transport_forward
    from rtx.autotune.measurement import measure_cuda_batches
    from rtx.autotune.preparation_runtime import _warm_measurement_graph
    from rtx.kernels.routing_varlen import PersistentPoolSinkhorn, build, fake
    from rtx.routing import RoutingKernelConfig
    from cutlass import Float32, Int32

    arg = argparse.ArgumentParser(description=__doc__)
    arg.add_argument("--lengths", nargs="+", type=int, required=True)
    arg.add_argument("--router-config", type=Path, required=True)
    arg.add_argument("--output", type=Path, required=True)
    arg.add_argument("--threads", nargs="+", type=int, default=[256, 512, 1024])
    arg.add_argument("--hybrid", action="store_true",
                     help="Also measure the actual conditional graph on an all-short multi-pool pack")
    args = arg.parse_args()
    config = RoutingKernelConfig.from_dict(json.loads(args.router_config.read_text())["config"])
    config = replace(config, packed_sinkhorn_backend="staged")
    t, b, e, k, steps = sum(args.lengths), len(args.lengths), 32, 4, 5
    key = layout(t, b, e, k, steps, tuple(config.to_token()))
    n = key[4]
    f = lambda size: torch.empty(size, device="cuda", dtype=torch.float32)
    generator = torch.Generator(device="cuda").manual_seed(1834)
    scores = torch.nn.functional.logsigmoid(
        torch.randn((t, e), device="cuda", generator=generator) * 1.5
    )
    cu = torch.tensor([0, *accumulate(args.lengths)], device="cuda", dtype=torch.int32)
    _, saved, scheduled, _, sequence, _, rowmap, _ = transport_forward(scores, cu, key)
    packed = f(n * e)
    from rtx._routing_varlen import prepare_execute
    prepare_execute(scores.flatten(), cu, [torch.empty_like(scheduled), torch.empty_like(scheduled),
        torch.empty_like(sequence), torch.empty_like(sequence), torch.empty_like(rowmap), packed,
        torch.empty(1, device="cuda", dtype=torch.int32)], _flat(key))
    upstream = torch.randn(n * e, device="cuda", generator=generator)
    out, probability, history = f(n * e), f(n * e), f(2 * steps * n * e)
    row, column = f(max(n, b * e)), f(max(n, b * e))
    a, z, dx = f(n * e), f(n * e), f(t * e)
    f_extra = [f(b * e * key[14])] if key[14] else []
    b_extra = [f(b * e * key[15])] if key[15] else []
    calls = {
        ("staged", "forward"): lambda: sinkhorn_execute(
            [packed, scheduled, sequence, rowmap], [out, history, row, column, probability, *f_extra], _flat(key), False),
        ("staged", "backward"): lambda: sinkhorn_execute(
            [upstream, saved, scheduled, sequence, rowmap], [a, z, dx, row, *b_extra], _flat(key), True),
    }
    calls["staged", "forward"]()
    calls["staged", "backward"]()
    expected = (out.clone(), history.clone(), dx.clone())
    hybrid_rows=0
    if args.hybrid:
        hybrid=replace(config,packed_sinkhorn_backend="persistent",packed_persistent_threads=1024)
        selected=layout(t,b,e,k,steps,tuple(hybrid.to_token()))
        hybrid_rows=selected[18]
        if not hybrid_rows or max(args.lengths)>hybrid_rows:
            raise ValueError("--hybrid diagnostic needs a total above the direct bound and all pools short")
        calls["hybrid", "forward"]=lambda:sinkhorn_execute(
            [packed,scheduled,sequence,rowmap],[out,history,row,column,probability,*f_extra],_flat(selected),False)
        calls["hybrid", "backward"]=lambda:sinkhorn_execute(
            [upstream,saved,scheduled,sequence,rowmap],[a,z,dx,row,*b_extra],_flat(selected),True)
        calls["hybrid", "forward"]()
        calls["hybrid", "backward"]()
        torch.testing.assert_close(out,expected[0],rtol=5e-5,atol=2e-6)
        torch.testing.assert_close(history,expected[1],rtol=5e-5,atol=2e-6)
        torch.testing.assert_close(dx,expected[2],rtol=5e-4,atol=8e-6)
    for threads in args.threads:
        for backward in (False, True):
            if not 64 <= threads <= 1024 or threads % 32:
                raise ValueError("CTA probe needs 64..1024 threads, divisible by 32")
            kernel = PersistentPoolSinkhorn(n, e, b, k, steps, hybrid_rows or (t+7)//8*8,
                                            threads, backward, bool(hybrid_rows))
            compiled = build(kernel, fake(Float32, n*e), fake(Float32, 2*steps*n*e),
                fake(Float32, (t if backward else n)*e), fake(Int32,b+1), fake(Int32,n), fake(Float32,n*e))
            values = (upstream, saved, dx, scheduled, rowmap, upstream) if backward else (
                packed, history, out, scheduled, rowmap, probability)
            call = lambda fn=compiled, values=values: fn(*values)
            calls[str(threads), "backward" if backward else "forward"] = call
            call()
            if backward:
                torch.testing.assert_close(dx, expected[2], rtol=5e-4, atol=8e-6)
            else:
                torch.testing.assert_close(out, expected[0], rtol=5e-5, atol=2e-6)
                torch.testing.assert_close(history, expected[1], rtol=5e-5, atol=2e-6)
    capture = torch.cuda.Stream()
    capture.wait_stream(torch.cuda.current_stream())
    graphs = {}
    for name, call in calls.items():
        with torch.cuda.stream(capture):
            call()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, stream=capture):
                call()
        torch.cuda.current_stream().wait_stream(capture)
        graphs[name] = graph
    timings = []
    for round_idx in range(4):
        for (mode, direction), graph in (list(graphs.items()) if round_idx in (0, 3) else reversed(list(graphs.items()))):
            _warm_measurement_graph(graph)
            samples, metadata = measure_cuda_batches(graph.replay, samples=7, calls_per_sample=1, min_batch_ms=5.)
            timings.append(dict(round=round_idx, mode=mode, direction=direction,
                samples_ms=samples, median_ms=statistics.median(samples), measurement=metadata))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(dict(lengths=args.lengths, timings=timings, torch=torch.__version__,
        gpu=torch.cuda.get_device_name(), cuda=torch.version.cuda, geometry=config.to_dict()), indent=2) + "\n")


if __name__ == "__main__":
    main()
