"""Graph-event A/B of the unsplit dW panels on D768/H256 expert geometry."""

import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path
from statistics import median

import torch

from rtx.configs.moe_backward import persistent_dw_rejection
from rtx.configs.mxfp8 import MXFP8GemmConfig
from rtx.kernels.mxfp8 import MXFP8Problem
from rtx.kernels.mxfp8_moe_buffers import compile_expert_packed_gemm
from rtx.kernels.mxfp8_moe_persistent_dw import compile_persistent_dw
from rtx.kernels.mxfp8_moe_prepared import compile_expert_dw_reduce


def run(leg, seed, rounds, samples, launches, schedules, schemes, frozen):
    e, k = 48, 1600
    m, n = (512, 768) if leg == "gate" else (768, 256)
    p = MXFP8Problem(m, n, k)
    cfg = MXFP8GemmConfig(
        **{name: value for name, value in frozen[leg + "_dw"].items() if name in MXFP8GemmConfig.__dataclass_fields__}
    ) if frozen else MXFP8GemmConfig(
        tile_m=64, tile_n=64, atom_layout_m=2, atom_layout_n=2,
        scale_load_vec=1, epilogue="direct", store_vec=1,
    )
    rng = torch.Generator(device="cuda").manual_seed(seed)

    def operand(rows):
        data = torch.randn(e * rows, k, generator=rng, device="cuda").mul_(64).to(torch.float8_e4m3fn)
        scale = torch.randint(113, 122, (e * rows, k // 32), generator=rng, device="cuda", dtype=torch.uint8)
        return data, scale

    a, sa = operand(m)
    b, sb = operand(n)
    out = torch.empty((e * m, n), dtype=torch.bfloat16, device="cuda")
    variants, configs = {}, {}

    def packed(config, split=1):
        kernel = compile_expert_packed_gemm(e, p, config, "linear", "tma", split)
        if split == 1:
            return lambda: kernel(a, b, sa, sb, out, out)
        scratch = torch.empty((e * split * m, n), dtype=torch.float32, device="cuda")
        reduce = compile_expert_dw_reduce(e, m, n, split, 128, 4)

        def invoke():
            kernel(a, b, sa, sb, scratch, scratch)
            reduce(scratch, out)

        return invoke

    variants["packed"] = packed(cfg)
    configs["packed"] = asdict(cfg)
    if "wide" in schemes:
        wide = replace(cfg, tile_n=128)
        variants["wide"] = packed(wide)
        configs["wide"] = asdict(wide)
    if "split2" in schemes:
        variants["split2"] = packed(cfg, 2)
        configs["split2"] = {**asdict(cfg), "split": 2}
    for axis in ("m", "n"):
        for tiles in (2, 3):
            for stages, scale_schedule in schedules:
                candidate = replace(cfg, stages=stages, scale_schedule=scale_schedule)
                name = f"{axis}{tiles}_{stages}s_{scale_schedule}"
                if persistent_dw_rejection(p, candidate, axis, tiles):
                    continue
                kernel = compile_persistent_dw(e, p, candidate, axis, tiles)
                variants[name] = lambda kernel=kernel: kernel(a, b, sa, sb, out)
                configs[name] = {**asdict(candidate), "axis": axis, "group_tiles": tiles}

    # Compare the entire tensor to matched unsplit and three representative
    # experts to an independent dequantized FP64 matmul before capture.
    indices = torch.tensor((0, e // 2, e - 1), device="cuda")

    def expanded(data, scale, rows):
        data = data.view(e, rows, k).float().index_select(0, indices).double()
        scale = scale.view(e, rows, k // 32).index_select(0, indices).double()
        return (data.view(-1, rows, k // 32, 32) * torch.exp2(scale[..., None] - 127)).flatten(-2)

    oracle = (expanded(a, sa, m) @ expanded(b, sb, n).mT).bfloat16()
    variants["packed"]()
    expected = out.clone()
    graphs = {}
    accuracy = {}
    accuracy_fp64 = {}
    for name, invoke in variants.items():
        out.fill_(torch.nan)
        invoke()
        assert torch.isfinite(out).all(), name
        delta = out.float() - expected.float()
        nrms = float(delta.norm() / expected.float().norm().clamp_min(1e-8))
        accuracy[name] = {"nrms": nrms, "max_abs": float(delta.abs().max())}
        assert nrms < 0.002, (name, accuracy[name])
        oracle_delta = out.view(e, m, n).index_select(0, indices).float() - oracle.float()
        nrms = float(oracle_delta.norm() / oracle.float().norm().clamp_min(1e-8))
        accuracy_fp64[name] = {"nrms": nrms, "max_abs": float(oracle_delta.abs().max())}
        assert nrms < 0.002, (name, accuracy_fp64[name])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(launches):
                invoke()
        graphs[name] = graph
    del delta, expected, oracle, oracle_delta, indices
    event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    results = {name: [] for name in graphs}
    order = tuple(graphs)
    for graph in graphs.values():
        for _ in range(4):
            graph.replay()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    live = [torch.cuda.memory_allocated()]
    for visit in range(rounds):
        offset = visit % len(order)
        rotation = order[offset:] + order[:offset]
        if visit % 2:
            rotation = rotation[::-1]
        for name in rotation:
            graph = graphs[name]
            graph.replay()
            trial = []
            for _ in range(samples):
                event_start.record()
                graph.replay()
                event_end.record()
                event_end.synchronize()
                trial.append(1000 * event_start.elapsed_time(event_end) / launches)
            results[name].append(trial)
        torch.cuda.synchronize()
        live.append(torch.cuda.memory_allocated())
    peak = torch.cuda.max_memory_allocated()
    assert all(x == live[0] for x in live)
    return {
        "shape": [e, m, n, k], "config": configs, "accuracy_packed": accuracy,
        "fp64_experts": [0, e // 2, e - 1], "accuracy_fp64": accuracy_fp64,
        "microseconds": results, "median_visit_microseconds": {name: median(map(median, times)) for name, times in results.items()},
        "live_bytes": live, "peak_measured_bytes": peak,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--leg", choices=("gate", "down", "both"), default="both")
    parser.add_argument("--seed", type=int, default=271263)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--samples", type=int, default=7)
    parser.add_argument("--launches", type=int, default=8)
    parser.add_argument("--frozen-expert", type=Path)
    parser.add_argument("--schedule", nargs="+", default=["2:after_wait", "2:before_wait", "3:after_wait"])
    parser.add_argument("--control", nargs="*", choices=("wide", "split2"), default=["wide"])
    args = parser.parse_args()
    schedule = []
    for spec in args.schedule:
        stages, name = spec.split(":")
        if name not in ("before_wait", "after_wait"):
            parser.error(f"unknown scale schedule: {name}")
        schedule.append((int(stages), name))
    frozen = json.loads(args.frozen_expert.read_text())["config"]["prepared"] if args.frozen_expert else None
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability() != (12, 0):
        parser.error("benchmark expects one CUDA-visible SM120 device")
    props = torch.cuda.get_device_properties(0)
    output = {
        "gpu": props.name, "sm_count": props.multi_processor_count, "uuid": str(props.uuid),
        "torch": str(torch.__version__), "cuda": torch.version.cuda,
        "seed": args.seed, "rounds": args.rounds, "samples": args.samples, "graph_launches": args.launches,
        "legs": {},
    }
    for leg in ("gate", "down") if args.leg == "both" else (args.leg,):
        result = run(leg, args.seed, args.rounds, args.samples, args.launches, schedule, args.control, frozen)
        output["legs"][leg] = result
        print(leg, json.dumps(result["median_visit_microseconds"], sort_keys=True), flush=True)
    with args.output.open("x") as stream:
        json.dump(output, stream, indent=2)


if __name__ == "__main__":
    main()
