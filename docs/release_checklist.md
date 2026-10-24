# 1.0 release checklist

This is the blocking release gate for `rtx-blackwell`. A historical benchmark
or a test report from a different source revision does not satisfy the gate.

## Source and package

- tracked sources and built distributions contain no generated datasets,
  checkpoints, local DINOv3 checkout, logs, or virtual environments (ignored
  local evidence may remain in the worktree);
- every retained autotuning manifest validates and has never been edited after
  producing data;
- `python -m pytest -q` passes (unittest classes and pytest functions);
- `python -m build` and `python -m twine check dist/*` pass;
- the built wheel imports as `rtx` in a clean environment;
- both `requirements.txt` and `requirements-torch212-cu132.txt` satisfy
  `rtx.validate_runtime_environment()`.

## GPU matrix

Run from the exact commit intended for the tag on both RTX 5070 variants and,
when available, RTX 5090:

```bash
python -m pytest -q
python benchmarks/validate_production.py \
  --output autotune_reports/release_1_0_production.json
```

For one logged, bounded invocation with a fresh output directory:

```bash
bash benchmarks/run_release_qualification.sh .venv/bin/python
```

The same script powers the manual self-hosted GPU release-qualification CI.
Run it separately for each supported environment; it does not modify installed
dependencies or update the checkout.

Install the optional `model-test` dependencies before this script. It now also
runs pinned Ettin/WikiText compiled inference with prospective quality guards
and the 12-step BF16-state training matrix. Asset downloads use the external
Hugging Face cache. This short training gate proves finite execution and
gradient/state coverage, not convergence.

By the model-target change on 2026-10-24, default pytest excludes the explicitly
marked `dinov3_research` model studies. Their original assertions and known
failures are retained; run `python -m pytest -q -m dinov3_research` to reproduce
them with separately obtained assets. DINO format/oracle unit tests remain in
the default suite. A green core/Ettin run is not a claim that DINO passes.

By the release-policy decision on 2026-10-25, default pytest also excludes
`timing_environment`: the strict singleton host-clock diagnostic is not a
release gate on noisy laptop/WSL systems. Run it explicitly with
`python -m pytest -q -m timing_environment`. Its timing thresholds remain
unchanged. Forced-noise fallback safety, correctness, and persistent cache
reuse remain in the default qualification suite.

The matrix must cover eager and `torch.compile(fullgraph=True, dynamic=False)`,
dynamic training, dynamic-X/prequantized-W and fully packed inference, ragged
M/N/K, dX-only, dW-only, combined backward, long-sequence dW accumulation,
multiple streams, runtime-cache bounds, PTQ conversion, and every retained
NVFP4 scaling policy.

Run the matrix once with PyTorch 2.12.1+cu132 and once with 2.13.0+cu132. A
native release is blocked if either stack fails runtime validation or produces
a CUDA-gated test failure.

## Autotuning and winners

Collect new data only from immutable retained-family manifests:

```bash
rtx-autotune run autotune_manifests/blackwell_all_kernels_scaling_v2.json \
  --device cuda:0 --output-dir autotune_datasets --format both
rtx-autotune audit autotune_datasets/blackwell_all_kernels_scaling_v2
rtx-autotune verify-winners \
  autotune_manifests/blackwell_all_kernels_scaling_v2.json \
  --device cuda:0 --output-dir autotune_datasets
```

Only audited, remeasured schema-v2 winners whose kernel revision matches the
release may be installed. Verify an online cache miss, persistent cache write,
same-shape hit, cross-process load, and the correctness-checked process-local
fallback when no candidate has eligible timing evidence. No unmeasured
fallback may be installed as a persistent winner. Candidate compilation
failures may be rejected during search; compilation or numerical failure
alone never authorizes execution of an unchecked fallback.

## Numerical and performance evidence

- MXFP8 and all NVFP4 policies remain finite and within their documented BF16
  reference envelopes;
- the default `jit_row_region` NVFP4 policy has a matched convergence run, not
  only microbenchmarks;
- peak memory is recorded for BF16, MXFP8, NVFP4 delayed, JIT row-region, and
  block scaling;
- release notes distinguish kernel microbenchmarks from complete-model
  throughput and do not promise a universal speedup.

After every gate passes, move `Unreleased` entries into `1.0.0`, change the
package version and development-status classifier, rebuild artifacts from a
clean checkout, commit, tag, and publish.
