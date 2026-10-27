"""Controlled public fullgraph MS-EDEN schedule/pipeline A/B.

Gaussian FP32 gates detect stale/corrupt gradients, not identical-draw kernel
equivalence. Use --validate-exact separately before performance acceptance.
Use --baseline-policy stable_mxfp8 for the public stable MXFP8 dX+dW target;
the default remains the original serial FP4 schedule comparison.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
import time

SHAPES = {
    "tiny": (512, 512, 512),
    "decoder512": (4096, 512, 512),
    "decoder1536": (4096, 1536, 512),
    "decoder2048": (4096, 2048, 512),
    "decoder1024": (4096, 512, 1024),
    "long": (32768, 512, 512),
    "wide": (4096, 2048, 2048),
}


def shape_value(value):
    shape = SHAPES.get(value)
    if shape is None:
        shape = tuple(map(int, value.split(",")))
    if len(shape) != 3 or any(type(v) is not int or v <= 0 or v % 128 for v in shape):
        raise ValueError("shape requires three positive multiples of 128")
    return shape


def schedule_pair(overrides=None, dx_overrides=None, dw_overrides=None):
    from rtx.autotune.posthoc import PosthocPreparationSchedule
    base = dict(producer_threads=128, producer_blocks=256,
                producer_load="coalesced_wide", producer_signs="register",
                rng_mode="explicit", amax_reduction="inline",
                finalizer_threads=128, finalizer_blocks=224)
    base.update(overrides or {})
    result = {}
    for phase, order, changes in (("dx", "row_major", dx_overrides),
                                   ("dw", "reduction_major", dw_overrides)):
        values = dict(base)
        values.setdefault("producer_order", order)
        values.update(changes or {})
        schedule = PosthocPreparationSchedule(**values)
        if schedule.reader_mode != "direct":
            raise ValueError("pipeline comparison preserves original BF16 direct reader")
        result[phase] = json.dumps(asdict(schedule), sort_keys=True)
    return result


def options(shape, schedules, seed=941, pipeline="serial", *, policy="fp4_fp4",
            mxfp8_config_json=None):
    from benchmarks.compare_ms_eden_amortization import options as public_options
    args = public_options(shape, schedules, seed=seed, policy=policy)
    args.backward_pipeline = pipeline
    args.mxfp8_config_json = mxfp8_config_json
    return args


def performance_target(timings, *, baseline_policy, target=1.20):
    """Conservative timing-only gate; never substitutes for exact qualification.

    Require stationary round aggregates AND each AB/BA order's raw windows for
    both paths, and a clustered confidence lower bound meeting the target.
    Total-graph speedup cannot substitute for the complete-backward target.
    """
    backward = timings.get("backward")
    reasons = []
    if baseline_policy != "stable_mxfp8":
        reasons.append("baseline_is_not_stable_mxfp8")
    if backward is None:
        reasons.append("complete_backward_not_measured")
    else:
        for name in ("baseline", "candidate"):
            if not backward["quality"][name]["sampling"]["collection"]["stationary"]:
                reasons.append(f"nonstationary_rounds_{name}")
        for kind in ("AB", "BA"):
            for name in ("baseline", "candidate"):
                quality = backward["order_quality"][kind][name]
                if not quality["sampling"]["collection"]["stationary"]:
                    reasons.append(f"nonstationary_{kind}_{name}")
        for label, paired in (("rounds", backward["paired"]),
                              *((kind, backward["by_order"][kind]) for kind in ("AB", "BA"))):
            if paired["round_cluster_bootstrap95"][0] < target:
                reasons.append(f"speedup_lower_bound_below_target_{label}")
    return dict(target_reference_over_candidate=target, met=not reasons,
                reasons=reasons, scope="Timing only; independent exact kernel qualification still required")


def round_orders(rounds, seed):
    rng = random.Random(seed)
    for index in range(rounds):
        orders = [("baseline", "candidate"), ("candidate", "baseline")]
        rng.shuffle(orders)
        yield index, orders


def gradient_metrics(actual, reference, gate=.25):
    import torch
    if not torch.isfinite(actual).all() or not torch.isfinite(reference).all():
        raise RuntimeError("nonfinite current-buffer gradient/reference")
    error = (actual.float() - reference).double()
    ref = reference.double()
    nrms = float(error.norm() / ref.norm().clamp_min(1e-12))
    if nrms > gate:
        raise RuntimeError(f"Gaussian FP32 gradient NRMS {nrms} exceeds {gate}")
    return dict(nrms=nrms, max_abs=float(error.abs().max()),
                rms=float(error.square().mean().sqrt()),
                reference_rms=float(ref.square().mean().sqrt()),
                normalized_max=float(error.abs().max() / ref.abs().max().clamp_min(1e-12)),
                denominator_floor=1e-12, gate=gate, finite=True)


def validate_exact(candidate_overrides=None, *, seed=941):
    """Fixed-draw byte equivalence; orthogonal to public stochastic FP32 gates.

    Two partial-count regimes and zero/sparse/outlier inputs. Changed uniforms,
    signs and source buffers are replayed through a captured candidate graph.
    No Cartesian sweep of scheduling parameters is performed.
    """
    import torch
    from rtx.autotune.posthoc import PosthocPreparationSchedule
    from benchmarks.study_ms_eden_backward import prepare
    from rtx.kernels.nvfp4_posthoc import compile_hadamard_preparation, PosthocScaleConfig
    torch.manual_seed(seed)
    baseline = PosthocPreparationSchedule(**json.loads(schedule_pair()["dx"]))
    candidate = PosthocPreparationSchedule(**json.loads(schedule_pair(candidate_overrides)["dx"]))
    if candidate.rng_mode != "explicit":
        raise ValueError("fixed-uniform preparation probe requires explicit RNG; use seeded independent-reference tests")
    cases = []
    for shape, transpose in (((128, 128), False), ((128, 128), True),
                             ((4096, 512), False), ((4096, 512), True)):
        source = torch.empty(shape, device="cuda", dtype=torch.bfloat16)
        a = prepare(source, transpose, "mma128", baseline)
        b = prepare(source, transpose, "mma128", candidate)
        signs = torch.ones(128, device="cuda")
        source.zero_()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                b["run"](signs)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            b["run"](signs)
        torch.cuda.current_stream().wait_stream(stream)
        _, partial_count = compile_hadamard_preparation(*shape,
            PosthocScaleConfig(baseline.producer_threads, baseline.producer_blocks), 4,
            transpose, baseline.producer_order, baseline.producer_load, baseline.producer_signs)
        for distribution in ("zero", "sparse", "outliers"):
            source.normal_(0, .2)
            if distribution == "zero":
                source.zero_()
            elif distribution == "sparse":
                source.mul_((torch.rand(shape, device="cuda") < .05).to(source.dtype))
            else:
                source.flatten()[::97].mul_(32)
            signs.random_(0, 2).mul_(2).sub_(1)
            a["uniforms"].uniform_()
            b["uniforms"].copy_(a["uniforms"])
            a["run"](signs)
            graph.replay()
            torch.cuda.synchronize()
            for field in ("packed", "scales", "factor"):
                torch.testing.assert_close(a[field].view(torch.uint8), b[field].view(torch.uint8), atol=0, rtol=0)
            if not torch.isfinite(b["factor"]).all() or not torch.isfinite(b["scales"].float()).all():
                raise RuntimeError("nonfinite exact preparation outputs")
            cases.append(dict(shape=shape, transpose=transpose, distribution=distribution,
                              baseline_partial_count=partial_count, byte_identical=True, finite=True))
    root = Path(__file__).resolve().parents[1]
    paths = ("benchmarks/compare_ms_eden_pipeline.py", "benchmarks/study_ms_eden_backward.py",
             "rtx/kernels/nvfp4_posthoc.py")
    return dict(seed=seed, torch=torch.__version__, cuda=torch.version.cuda,
                device=str(torch.cuda.get_device_properties(0)), command=sys.argv,
                source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths},
                cases=cases, baseline=asdict(baseline), candidate=asdict(candidate),
                scope="Identical signs and per-scale uniforms; packed bytes, scales and factor exact; candidate graph reads changed source/sign/uniform buffers")


def compare(shape, output, candidate_overrides=None, *, dx_overrides=None,
            dw_overrides=None, pipeline="serial", rounds=10, replays=16,
            min_batch_ms=2., warmup_ms=200., seed=941, scopes=("backward", "total"),
            baseline_policy="fp4_fp4", mxfp8_config_json=None, warmup_sync_each=False):
    import torch
    from benchmarks.compare_mixed_nvfp4_backward import run_worker, paired_statistics
    from rtx.autotune.measurement import measure_cuda_batches, runtime_timing_metadata
    shape = shape_value(",".join(map(str, shape)))
    if not 4 <= rounds <= 64 or replays < 1 or min_batch_ms < 2 or warmup_ms < 0:
        raise ValueError("require 4..64 rounds, positive replays, batch>=2ms, nonnegative warmup")
    if pipeline not in ("serial", "paired", "operand_paired") or not scopes or set(scopes) - {"backward", "total"}:
        raise ValueError("invalid pipeline or timing scopes")
    if baseline_policy not in ("fp4_fp4", "stable_mxfp8"):
        raise ValueError("invalid baseline policy")
    if mxfp8_config_json is not None:
        from rtx.bwd_autotune import bwd_config_from_dict
        bwd_config_from_dict(json.loads(mxfp8_config_json))
        if baseline_policy != "stable_mxfp8":
            raise ValueError("MXFP8 config requires stable_mxfp8 baseline")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    schedules = dict(baseline=schedule_pair(), candidate=schedule_pair(
        candidate_overrides, dx_overrides, dw_overrides))
    bundles, metadata = {}, {}
    shared = None
    for name in ("baseline", "candidate"):
        args = options(shape, schedules[name], seed, pipeline if name == "candidate" else "serial",
            policy=baseline_policy if name == "baseline" else "fp4_fp4",
            mxfp8_config_json=mxfp8_config_json if name == "baseline" else None)
        metadata[name], bundles[name] = run_worker(args, capture_only=True, shared_inputs=shared)
        shared = bundles[name]["inputs"]
    if metadata["baseline"]["forward_config"] != metadata["candidate"]["forward_config"]:
        raise RuntimeError("baseline and candidate forward configurations differ")
    graphs = {name: dict(backward=item["pins"][1], total=item["graph"])
              for name, item in bundles.items()}
    deadline = time.monotonic() + warmup_ms / 1000
    warmup_iteration = 0
    while time.monotonic() < deadline:
        for name in bundles:
            for scope in scopes:
                if warmup_sync_each and warmup_iteration < 3:
                    print(f'WARMUP begin iteration={warmup_iteration} graph={name}/{scope}', flush=True)
                graphs[name][scope].replay()
                if warmup_sync_each:
                    torch.cuda.synchronize()
                    if warmup_iteration < 3:
                        print(f'WARMUP done iteration={warmup_iteration} graph={name}/{scope}', flush=True)
        torch.cuda.synchronize()
        warmup_iteration += 1
    # Calibrate each graph outside sampling, then use a common repetition count
    # within a scope so A and B windows charge identical host/event overhead.
    calibration, counts = {}, {}
    for scope in scopes:
        calibration[scope] = {}
        for name in bundles:
            _, info = measure_cuda_batches(graphs[name][scope].replay, samples=1,
                calls_per_sample=replays, min_batch_ms=min_batch_ms, device=shared[0].device)
            calibration[scope][name] = info
        counts[scope] = max(info["actual_calls_per_sample"] for info in calibration[scope].values())
    def batch(graph, count):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(count):
            graph.replay()
        end.record()
        end.synchronize()
        return start.elapsed_time(end) / count
    torch.cuda.synchronize()
    before = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for index, orders in round_orders(rounds, seed):
        for scope in scopes:
            windows = []
            for order in orders:
                values = {name: batch(graphs[name][scope], counts[scope]) for name in order}
                windows.append(dict(order=list(order), order_kind="AB" if order[0] == "baseline" else "BA",
                    reference_ms=values["baseline"], candidate_ms=values["candidate"]))
            rows.append(dict(round=index, scope=scope, windows=windows,
                reference_ms=math.sqrt(math.prod(w["reference_ms"] for w in windows)),
                candidate_ms=math.sqrt(math.prod(w["candidate_ms"] for w in windows))))
    torch.cuda.synchronize()
    after = torch.cuda.memory_allocated()
    replay_peak = torch.cuda.max_memory_allocated()
    if after != before:
        raise RuntimeError("live allocation changed during paired graph replay")
    x, w, dy = shared
    references = (dy.float() @ w.detach().float(), dy.float().T @ x.detach().float())
    checks = {}
    # Validate the exact standalone backward graph being timed, as well as total.
    for name, bundle in bundles.items():
        checks[name] = {}
        for scope, grads in (("backward", bundle["pins"][3]), ("total", bundle["grads"])):
            graphs[name][scope].replay()
            checks[name][scope] = {phase: gradient_metrics(actual, ref,
                gate=.06 if name == "baseline" and baseline_policy == "stable_mxfp8" else .25)
                for phase, actual, ref in zip(("dx", "dw"), grads, references)}
    timings = {}
    for scope in scopes:
        samples = [r for r in rows if r["scope"] == scope]
        timings[scope] = dict(paired=paired_statistics(samples, seed=seed),
            median_ms={name: statistics.median(r[key] for r in samples)
                       for name, key in (("baseline", "reference_ms"), ("candidate", "candidate_ms"))},
            quality={name: runtime_timing_metadata([r[key] for r in samples])
                     for name, key in (("baseline", "reference_ms"), ("candidate", "candidate_ms"))},
            by_order={kind: paired_statistics([window for row in samples for window in row["windows"]
                      if window["order_kind"] == kind], seed=seed) for kind in ("AB", "BA")},
            order_quality={kind: {name: runtime_timing_metadata([
                window[key] for row in samples for window in row["windows"] if window["order_kind"] == kind])
                for name, key in (("baseline", "reference_ms"), ("candidate", "candidate_ms"))}
                for kind in ("AB", "BA")})
    root = Path(__file__).resolve().parents[1]
    paths = ("benchmarks/compare_ms_eden_pipeline.py", "benchmarks/compare_mixed_nvfp4_backward.py",
             "rtx/experimental_ms_eden.py", "rtx/kernels/nvfp4_posthoc.py", "rtx/autotune/posthoc.py",
             "rtx/fp8_bwd.py", "rtx/fp4.py", "rtx/kernels/mxfp8_bwd.py",
             "rtx/kernels/nvfp4_gemm.py", "rtx/kernels/mxfp8_gemm.py",
             "rtx/kernels/nvfp4_rng.py", "rtx/autotune/measurement.py")
    result = dict(shape=shape, seed=seed, command=sys.argv, rounds=rounds,
        torch=torch.__version__, cuda=torch.version.cuda, device=str(torch.cuda.get_device_properties(0)),
        schedules=schedules, candidate_pipeline=pipeline,
        baseline_pipeline="stable_paired" if baseline_policy == "stable_mxfp8" else "serial",
        baseline_policy=baseline_policy, candidate_policy="fp4_fp4",
        baseline_mxfp8_config_source=("explicit" if mxfp8_config_json else "repository_default")
            if baseline_policy == "stable_mxfp8" else None,
        baseline_mxfp8_config=metadata["baseline"]["mxfp8_backward_config"] if baseline_policy == "stable_mxfp8" else None,
        performance_target=performance_target(timings, baseline_policy=baseline_policy),
        fairness=dict(shared_current_bf16_buffers=True, matched_exact_nvfp4_forward=True,
            baseline_public_api="rtx.nvfp4_linear" if baseline_policy == "stable_mxfp8" else "ms_eden_linear",
            candidate_public_api="ms_eden_linear", backward_includes="Both gradients, all preparation and caller RNG where required; no prequantized substitution",
            capture_state="Both warmed/captured bundles coexist before calibrated randomized AB/BA timing"),
        build_guards=metadata, gradient_checks=checks, timings=timings, samples=rows,
        calibration=calibration, calls_per_window=counts, warmup_ms=warmup_ms,
        warmup_sync_each=warmup_sync_each,
        memory=dict(replay_live_growth_bytes=after-before, replay_peak_allocated_bytes=replay_peak),
        source_sha256={name: hashlib.sha256((root/name).read_bytes()).hexdigest() for name in paths},
        scope="Paired complete public fullgraph backward and independently captured forward+backward CUDA graph latency, all caller and fused RNG included, identical current BF16 buffers; compilation/capture/optimizer excluded",
        numerical_scope="FP32 gates are Gaussian algorithm/stale-buffer checks, not identical-draw kernel qualification; independent forward bitwise, current-input/current-weight/upstream and fresh RNG guards inherited from public builder")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape", default="decoder2048", help="M,N,K or " + ", ".join(SHAPES))
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--candidate-json", default="{}", help="explicit schedule changes; default is identity")
    p.add_argument("--candidate-dx-json", default="{}")
    p.add_argument("--candidate-dw-json", default="{}")
    p.add_argument("--candidate-pipeline", choices=("serial", "paired", "operand_paired"), default="serial")
    p.add_argument("--baseline-policy", choices=("fp4_fp4", "stable_mxfp8"), default="fp4_fp4")
    p.add_argument("--mxfp8-config-json", help="stable baseline MXFP8BwdConfig; omitted means repository default (recorded in output)")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--replays", type=int, default=16)
    p.add_argument("--min-batch-ms", type=float, default=2.)
    p.add_argument("--warmup-ms", type=float, default=200.)
    p.add_argument("--seed", type=int, default=941)
    p.add_argument("--scope", choices=("backward", "total", "both"), default="both")
    p.add_argument("--validate-exact", action="store_true", help="fixed-draw preparation matrix only; no timing")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.validate_exact:
        if args.output.exists():
            raise FileExistsError(args.output)
        result = validate_exact(json.loads(args.candidate_json), seed=args.seed)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as handle:
            json.dump(result, handle, indent=2)
        print(json.dumps(result, indent=2))
        return
    result = compare(shape_value(args.shape), args.output, json.loads(args.candidate_json),
        dx_overrides=json.loads(args.candidate_dx_json), dw_overrides=json.loads(args.candidate_dw_json),
        pipeline=args.candidate_pipeline, rounds=args.rounds, replays=args.replays,
        min_batch_ms=args.min_batch_ms, warmup_ms=args.warmup_ms, seed=args.seed,
        baseline_policy=args.baseline_policy, mxfp8_config_json=args.mxfp8_config_json,
        scopes=("backward", "total") if args.scope == "both" else (args.scope,))
    print(json.dumps(result["timings"], indent=2))


if __name__ == "__main__":
    main()
