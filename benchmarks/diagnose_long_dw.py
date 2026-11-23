"""Matched long-reduction dW probe, including preparation and FP32 reduction."""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import statistics

import torch

from rtx.bwd_autotune import bwd_config_from_dict
from rtx.fp8_bwd import _build_matmul_runner
from rtx.kernels.mxfp8 import MXFP8Problem


def _quantized(x):
    # Independent tensor oracle for the reported round-to-infinity recipe.
    blocks = x.float().reshape(x.shape[0], -1, 32)
    scale = torch.exp2(torch.ceil(torch.log2(
        (blocks.abs().amax(-1, keepdim=True) / 448).clamp_min(2.**-126))))
    return ((blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).flatten(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--winner", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tokens", type=int, default=138304)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--cases", default="bf16,reported,scalar,vec2,consumer,consumer64,native_proxy")
    parser.add_argument("--counters", action="store_true")
    parser.add_argument("--tune-cache", type=Path, help="Run actual eight-trial paired-backward search first")
    args = parser.parse_args()
    torch.manual_seed(730)
    torch.backends.cuda.matmul.allow_tf32 = False
    k, d = args.tokens, args.width
    problem = MXFP8Problem(d, d, k)
    initial = bwd_config_from_dict(json.loads(args.winner.read_text()))
    base = initial.dw
    dev = dict(device="cuda", dtype=torch.bfloat16)
    x = torch.randn(k, d, **dev)
    dy = torch.randn_like(x)
    out = torch.empty(d, d, **dev)
    padded_k = (k + 127) // 128 * 128
    configs = {
        "reported": base,
        "scalar": replace(base, gemm=replace(base.gemm, scale_load_vec=1)),
        "vec2": replace(base, gemm=replace(base.gemm, scale_load_vec=2)),
    }
    consumers = replace(base.gemm, scale_role="consumers", scale_recycle="staged",
                        scale_load_vec=2, scale_smem_store="packed")
    configs["consumer"] = replace(base, gemm=consumers)
    configs["consumer64"] = replace(base, gemm=replace(consumers, tile_m=64, tile_n=64,
        atom_layout_m=2, a_swizzle="64b"), split_reduction=16)
    native = replace(base, quant_a=replace(base.quant_a, scale_layout="mma128"),
        gemm=replace(base.gemm, scale_role="tma", scale_layout="mma128"))
    configs.update(native_proxy=native, native_tail=native)
    for s in (1, 2):
        for parts in (16, 32, 64):
            configs[f"native_s{s}_{parts}"] = replace(native, split_reduction=parts,
                gemm=replace(native.gemm, stages=s))
    config = configs["native_s2_16"]
    for tk in (32, 64, 128):
        for vector in (4, 8):
            configs[f"native_q{tk}_v{vector}"] = replace(config, quant_a=replace(config.quant_a,
                transposed_tile_k=tk, quant_vec=vector, load_bits=vector*16))
    configs["native64"] = replace(config, quant_a=replace(config.quant_a, scale_layout="mma64"),
        quant_b=config.quant_a, gemm=replace(config.gemm, tile_m=64, atom_layout_m=2,
            scale_layout="mma64x128", stages=1))
    configs["native_s3"] = replace(config, gemm=replace(config.gemm, stages=3))
    if args.tune_cache:
        from rtx.autotune import AutotunePolicy
        from rtx.bwd_autotune import bwd_config_to_dict, tune_mxfp8_backward
        policy = AutotunePolicy.preset("quick", max_seconds=120, backward_measurement_mode="cuda_graph")
        result = tune_mxfp8_backward(dy, x, torch.randn_like(out), policy=policy,
                                    initial=initial, cache_dir=args.tune_cache)
        configs["tuned"] = result.config.dw
        args.tune_cache.mkdir(parents=True, exist_ok=True)
        (args.tune_cache / "selected.json").write_text(json.dumps(dict(
            config=bwd_config_to_dict(result.config), median_ms=result.median_ms,
            evaluated_trials=result.evaluated_trials, evidence=result.tuning_evidence), indent=2))
    ops = {}
    runners = {}
    choices = args.cases.split(",")
    if "bf16" in choices:
        ops["bf16"] = lambda: torch.mm(dy.T, x, out=out)
    for name in choices:
        if name == "bf16":
            continue
        config = configs[name]
        p, a, b = problem, dy, x
        if name == "native_proxy":
            # Mechanism-only control. This is not the proposed production path.
            p = MXFP8Problem(d, d, padded_k)
            a = torch.nn.functional.pad(dy, (0, 0, 0, padded_k - k))
            b = torch.nn.functional.pad(x, (0, 0, 0, padded_k - k))
        reason = config.implementation_rejection(p)
        if reason:
            print(f"SKIP {name}: {reason}", flush=True)
            continue
        runner = _build_matmul_runner(problem=p, config=config, device=x.device)
        runners[name] = runner
        ops[name] = lambda runner=runner, a=a, b=b: runner(a, b, out)
    if args.counters:
        for op in ops.values():
            for _ in range(3):
                op()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStart()
        for name, op in ops.items():
            torch.cuda.nvtx.range_push(name)
            op()
            torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        torch.cuda.cudart().cudaProfilerStop()
        return
    # Independent quantized recipe, comparing FP32 products on dequantized
    # operands. Split rounding can differ after the final BF16 conversion.
    reference = (_quantized(dy.T.contiguous()) @ _quantized(x.T.contiguous()).T).bfloat16()
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    torch.cuda.set_stream(stream)
    graphs, errors, times = {}, {}, {}
    for name, op in ops.items():
        for _ in range(3):
            op()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            op()
        graphs[name] = graph
        graph.replay()
        ref = reference.float()
        if name != "bf16" and configs[name].backend == "cublas_bf16":
            ref = (dy.float().T @ x.float()).bfloat16().float()
        delta = out.float() - ref
        ref_rms = ref.square().mean().sqrt()
        rms = delta.square().mean().sqrt()
        errors[name] = dict(nrms=float(rms/ref_rms.clamp_min(1e-12)),
                            rms_error=float(rms), reference_rms=float(ref_rms),
                            max_abs=float(delta.abs().max()), finite=bool(out.isfinite().all()))
        print(f"READY {name} {errors[name]}", flush=True)
        if name != "bf16" and (not errors[name]["finite"] or errors[name]["nrms"] >= .002):
            raise RuntimeError(f"Independent MXFP8 oracle failed: {name}")
        times[name] = []
    rng = random.Random(730)
    live_before = torch.cuda.memory_allocated()
    for _ in range(args.rounds):
        order = list(graphs)
        rng.shuffle(order)
        for name in order:
            start = torch.cuda.Event(enable_timing=True)
            stop = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(args.replays):
                graphs[name].replay()
            stop.record()
            stop.synchronize()
            times[name].append(start.elapsed_time(stop)/args.replays)
    results = {name:dict(samples_ms=samples, median_ms=statistics.median(samples),
                         min_ms=min(samples), max_ms=max(samples), **errors[name])
               for name, samples in times.items()}
    args.output.write_text(json.dumps(dict(tokens=k, width=d, seed=730,
        torch=torch.__version__, gpu=torch.cuda.get_device_name(), results=results,
        peak_bytes=torch.cuda.max_memory_allocated(),
        replay_live_delta_bytes=torch.cuda.memory_allocated()-live_before), indent=2))
    for name, row in results.items():
        print(name, row["median_ms"], row["nrms"], flush=True)


if __name__ == "__main__":
    main()
