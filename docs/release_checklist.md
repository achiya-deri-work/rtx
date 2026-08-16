# 1.0 release checklist

This is the blocking release gate for `rtx-blackwell`. A historical benchmark
or a test report from a different source revision does not satisfy the gate.

## Source and package

- the worktree contains no generated datasets, checkpoints, local DINOv3
  checkout, logs, or virtual environments;
- every retained autotuning manifest validates and has never been edited after
  producing data;
- `python -m unittest discover -s tests -v` passes;
- `python -m build` and `python -m twine check dist/*` pass;
- the built wheel imports as `rtx` in a clean environment;
- both `requirements.txt` and `requirements-torch212-cu132.txt` satisfy
  `rtx.validate_runtime_environment()`.

## GPU matrix

Run from the exact commit intended for the tag on both RTX 5070 variants and,
when available, RTX 5090:

```bash
python -m unittest discover -s tests -v
python benchmarks/validate_production.py \
  --output autotune_reports/release_1_0_production.json
```

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
same-shape hit, cross-process load, and safe fallback after compilation error.

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
