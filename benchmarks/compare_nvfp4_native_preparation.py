"""Matched-policy delayed FP4 preparation: row-major versus native scale emission.

History is fixed at input amax: this is a steady-state layout diagnostic, not
a stateful training qualification. No scale-layout conversion is timed.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch

from benchmarks.compare_nvfp4_packed_gemm import decode, native_scales
from benchmarks.profile_preparation_phases import measure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape", default="2048,768,1536")
    parser.add_argument("--output", required=True)
    parser.add_argument("--stages", type=int, default=2)
    parser.add_argument("--outer-mode", choices=("exact","power2"), default="power2")
    args = parser.parse_args()
    from rtx.configs.nvfp4 import NVFP4Problem, NVFP4ScaleConfig
    from rtx.fp4 import NVFP4ForwardConfig, _make_delayed_dynamic_runner, _delayed_amax_state
    from rtx.fp8_bwd import _zero_tensor_async
    from rtx.autotune.measurement import runtime_timing_metadata

    torch.manual_seed(20261023)
    m, n, k = map(int, args.shape.split(","))
    device = torch.device("cuda:0")
    x = torch.randn(m, k, device=device, dtype=torch.bfloat16) * .25
    w = torch.randn(n, k, device=device, dtype=torch.bfloat16) * .02
    policy = NVFP4ScaleConfig(tensor_scale_mode=args.outer_mode)
    hx, hw = (_delayed_amax_state(t, policy.amax_history_len) for t in (x, w))
    base = NVFP4ForwardConfig()
    base = replace(base, gemm=replace(base.gemm, stages=args.stages, atom_layout_m=2, atom_layout_n=2))
    problem = NVFP4Problem(m, n, k)
    runners, outputs, histories, variants = {}, {}, {}, {}
    original_reference = x.float() @ w.float().T
    reference = None
    for layout in ("row_major", "mma128"):
        config = replace(base, quant=replace(base.quant, scale_layout=layout),
                         gemm=replace(base.gemm, scale_layout=layout,
                                      scale_role="tma" if layout == "mma128" else base.gemm.scale_role))
        rejection = config.rejection(problem)
        if rejection:
            raise ValueError(rejection)
        runner = _make_delayed_dynamic_runner(problem, config, policy, device)
        nx, nw = torch.empty_like(hx), torch.empty_like(hw)
        out = torch.empty(m, n, device=device, dtype=torch.bfloat16)

        def prep():
            _zero_tensor_async(nx)
            _zero_tensor_async(nw)
            runner.quant(x, w, runner.qx, runner.qw, runner.sx, runner.sw,
                         hx, hw, nx, nw, runner.output_scale)

        def gemm():
            runner.gemm(runner.qx_packed, runner.qw_packed, runner.sx, runner.sw,
                        out, runner.output_scale)

        def full():
            runner(x, w, hx, hw, nx, nw, out)

        full()
        torch.cuda.synchronize()
        if layout == "row_major":
            reference = (decode(runner.qx, runner.sx, 1.) @
                         decode(runner.qw, runner.sw, 1.).T) * runner.output_scale
        else:
            previous = runners["row_major"]
            for field in ("qx", "qw", "output_scale"):
                torch.testing.assert_close(getattr(runner, field), getattr(previous, field), rtol=0, atol=0)
            for field in ("sx", "sw"):
                assert torch.equal(getattr(runner, field).view(torch.uint8),
                                   native_scales(getattr(previous, field)).view(torch.uint8))
            for actual, expected in zip((nx, nw), histories["row_major"]):
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        nrms = ((out.float() - reference).square().mean() / reference.square().mean()).sqrt().item()
        if not torch.isfinite(out).all() or nrms > .005:
            raise RuntimeError(f"packed oracle failed: {layout} {nrms}")
        timings = {name: measure(fn,time_based_warmup=True) for name,fn in
                   (("preparation",prep),("gemm",gemm),("full",full))}
        original_nrms = ((out.float() - original_reference).square().mean() /
                         original_reference.square().mean()).sqrt().item()
        variants[layout] = dict(config=asdict(config), packed_reference_nrms=nrms,
                                original_bf16_reference_nrms=original_nrms,
                                timings=timings, timing_quality={
                                    name: runtime_timing_metadata(value["samples_ms"])
                                    for name, value in timings.items()})
        runners[layout], outputs[layout] = runner, out
        histories[layout] = (nx, nw)
    torch.testing.assert_close(outputs["row_major"], outputs["mma128"], rtol=0, atol=0)
    result = dict(shape=[m, n, k], policy=asdict(policy), variants=variants,
                  measurement_contract="graph8x10_samples15_warm200ms_revision2",
                  seed=20261023, torch=torch.__version__, cuda=torch.version.cuda,
                  device=str(torch.cuda.get_device_properties(device)),
                  contract="Fixed-history delayed; identical packed values, scales and outputs; no layout conversion; phase timings are not additive")
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({layout: {name: value["median_ms"] * 1000
                              for name, value in variant["timings"].items()}
                      for layout, variant in variants.items()}))


if __name__ == "__main__":
    main()
