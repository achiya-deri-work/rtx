"""Same-device CuTe versus BiMoE Torch Sinkhorn5/QB5/fixed-repair64.

Load the original standalone router.py with --baseline-source; no BiMoE
installation, expert kernels, or changes to that checkout are needed. Both
paths use fullgraph Inductor and CUDA graph replay. Cold tuning is excluded.
"""
import argparse
from contextlib import nullcontext
import hashlib
import importlib.util
import json
from pathlib import Path
import statistics
import sys
import time

import torch

from benchmarks.benchmark_balanced_router import measure
from rtx import AutotunePolicy
from rtx.autotune.deployment_qualification import qualify_pair
from rtx.autotune.outcomes import TrialOutcome
from rtx.routing import BalancedRouter


def load_reference(path):
    spec = importlib.util.spec_from_file_location("bimoe_fixed64_reference", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def error(actual, expected):
    delta = actual.float() - expected.float()
    return dict(max_abs=delta.abs().max().item(),
                relative_l2=(delta.norm() / expected.float().norm().clamp_min(1e-12)).item())


def benchmark(shape, reference, args):
    n, d, e, k = map(int, shape.split(":"))
    torch.manual_seed(args.seed)
    candidate = BalancedRouter(d, e, top_k=k, router_dim=64, device="cuda", autotune="off")
    with torch.no_grad():
        candidate.projection_weight.normal_(std=d ** -.5)
    baseline = reference.SinkhornRouter(
        d, e, top_k=k, router_dim=64, sinkhorn_iterations=5,
        dual_iterations=5, repair_rounds=64, scheduler_backend="dual",
        repair_execution="fixed", device="cuda", dtype=torch.bfloat16)
    baseline.load_state_dict(candidate.state_dict(), strict=True)
    x = torch.randn(n, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    started = time.monotonic()
    tuned = {}
    for phase in ("train", "forward"):
        result = candidate.autotune_(
            x, phase=phase, tuning_policy=AutotunePolicy.preset(args.effort),
            output_dir=Path(args.output).parent / "tuning", progress=print)
        tuned[phase] = result.config.to_dict()
    cold_seconds = time.monotonic() - started
    models = (baseline, candidate)
    functions = [torch.compile(model, fullgraph=True, dynamic=False,
                               options={"triton.cudagraphs": False}) for model in models]
    coefficient = torch.randn(e, reference.padded_token_count(n, e, k) * k // e,
                              device="cuda", dtype=torch.float32)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    # Compare whole-router metadata, normalized weights, and every gradient
    # before collecting any performance result. Hard decisions must match.
    with torch.cuda.stream(stream):
        routes = [fn(x) for fn in functions]
        for index in (0, 2):
            torch.testing.assert_close(routes[1][index], routes[0][index], rtol=0, atol=0)
        torch.testing.assert_close(routes[1][1], routes[0][1], rtol=5e-5, atol=5e-6)
        gradients = [torch.autograd.grad((route[1] * coefficient).sum(), (x, *model.parameters()))
                     for route, model in zip(routes, models)]
        names = ["input", *dict(baseline.named_parameters())]
        gradient_errors = {}
        for name, actual, expected in zip(names, gradients[1], gradients[0]):
            assert bool(actual.isfinite().all()) and bool(expected.isfinite().all())
            torch.testing.assert_close(actual, expected, rtol=.015, atol=.001)
            gradient_errors[name] = error(actual, expected)
        correctness = dict(exact_metadata=True, weights=error(routes[1][1], routes[0][1]),
                           gradients=gradient_errors)
        del routes, gradients
    stream.synchronize()
    print(f"CORRECTNESS {shape} passed", flush=True)
    phases = {}
    for phase in ("forward", "forward_backward"):
        observations = []

        def evaluate(index):
            model, fn = models[index], functions[index]

            def call():
                route = fn(x)
                if phase == "forward":
                    return route
                return torch.autograd.grad((route[1] * coefficient).sum(), (x, *model.parameters()))

            with torch.no_grad() if phase == "forward" else nullcontext():
                result = measure(call, samples=args.samples, stream=stream,
                                 warmup_ms=args.warmup_ms)
            eligible = result["sampling"]["collection"]["stationary"]
            observations.append(dict(candidate=index, **result))
            print(f"MEASURE {shape} {phase} {'torch_fixed64' if index == 0 else 'cute_tuned'} "
                  f"{result['median_ms']:.6f}ms eligible={eligible}", flush=True)
            return TrialOutcome("ok" if eligible else "runtime_error",
                                median_ms=result["median_ms"] if eligible else None,
                                timings_ms=result["timings_ms"],
                                error=None if eligible else "nonstationary timing window")

        try:
            _, decision = qualify_pair(0, 1, evaluate, rounds=args.rounds, margin=.01)
        except RuntimeError as exc:
            decision = dict(promoted=False, status="inconclusive", error=str(exc))
        medians = [statistics.median(row["median_ms"] for row in observations if row["candidate"] == i)
                   for i in range(2)]
        phases[phase] = dict(torch_fixed64_ms=medians[0], cute_tuned_ms=medians[1],
                             observed_speedup=medians[0] / medians[1],
                             qualification=decision, observations=observations)
        print(f"RESULT {shape} {phase} speedup={medians[0] / medians[1]:.3f}x "
              f"qualified={decision['promoted']}", flush=True)
    return dict(shape=dict(tokens=n, d_model=d, experts=e, top_k=k, router_dim=64),
                correctness=correctness, tuning=dict(configs=tuned,
                cold_seconds=cold_seconds, effort=args.effort), phases=phases)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--baseline-source", type=Path, required=True)
    workloads = p.add_mutually_exclusive_group()
    workloads.add_argument("--shapes", nargs="+")
    workloads.add_argument("--cases", type=Path, help="BalancedMoE workload JSON; measures routing only")
    p.add_argument("--effort", choices=("quick", "balanced", "thorough", "extensive"), default="balanced")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--samples", type=int, default=21)
    p.add_argument("--warmup-ms", type=float, default=750.)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--output", required=True)
    args = p.parse_args()
    suite = None
    if args.cases is not None:
        suite = json.loads(args.cases.read_text())
        if suite["schema_version"] != 1:
            p.error("Unsupported BalancedMoE workload schema")
        args.shapes = [f"{n}:{case['d_model']}:{case['num_experts']}:{case['top_k']}"
                       for case in suite["cases"] for n in suite["token_counts"]]
    elif args.shapes is None:
        args.shapes = ["1024:768:128:8", "4096:768:128:8"]
    torch.set_float32_matmul_precision("highest")
    baseline = load_reference(args.baseline_source)
    output = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  configuration={key: str(value) if isinstance(value, Path) else value
                                 for key, value in vars(args).items()},
                  workload_suite=suite,
                  baseline_sha256=hashlib.sha256(args.baseline_source.read_bytes()).hexdigest(),
                  benchmark_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  measurement="alternating fullgraph CUDA graph replay; cold work excluded", results=[])
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    for shape in args.shapes:
        output["results"].append(benchmark(shape, baseline, args))
        destination.write_text(json.dumps(output, indent=2) + "\n")


if __name__ == "__main__":
    main()
