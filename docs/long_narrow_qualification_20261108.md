# Long/narrow MXFP8 qualification, 2026-11-08

The added implementation is an **autotuner choice**, not a hardcoded schedule:
`gradient_quantization="recompute_both"` with recomputed hidden. It computes
independent feature-axis and token-axis FP8 derivative packs from shared,
BF16-rounded values. It removes the BF16 derivative allocation, global write,
subsequent read and standalone derivative quantizer. FP32 activation/gating,
FP32 GEMM/reduction accumulation and original scale policies are preserved.
No new user-managed streams, buffers or graphs are required.

## RTX 5090: previous commit versus current work

Previous RTX sources: `dd665e6`. Current work: MLP search revision 44.
Both used Torch 2.13.0+cu132, CuTe 4.7.0, torchao 0.18.0+cu132, the same seed,
initial schedule and **eight fresh tuning trials per shape per revision**.
M=40000, H=2K and O=K; all inputs, weights and output gradients are BF16.
These are synthetic complete MLP forward + all-three-gradient workloads,
not BF16-relative speedups or real detector-step measurements.

| K / width | Previous ms | Current ms | Speedup | Paired decision |
|---|---:|---:|---:|---|
| 128 | 0.49773 | 0.49597 | 1.0035× | Keep previous; below margin |
| 256 | 0.97366 | 0.92309 | **1.0548×** | Current qualifies |
| 384 | 1.57516 | 1.49775 | **1.0517×** | Current qualifies |

Each revision runs in its own source process. A common evaluator protocol
alternates AB/BA for 12 rounds, with at least 200 measured GPU milliseconds
of warmup before nine timing batches of three complete graph workloads.
Compilation, tuning, capture, warmup and correctness checks are excluded.
The benchmark-only warmup shim makes both source evaluators replay once before
timing; its outer loop supplies the same measured warmup to both. No previous
RTX source file is changed. Source hashes, selected policies and every sample
are retained. All outputs/gradients pass the independent quantized-recipe
oracle; promotion uses the unchanged 1% aggregate/paired-lower-quartile gate.
This decision rule is not a confidence interval.

The idle detector server on port 8765 was stopped with user authorization
before these benchmarks. No other service or project was changed. The server
was not automatically restarted.

## Laptop study and mechanism

On the 5070 laptop, 16-trial searches plus 12 paired rounds qualified gains
of 17.4% at M40000/K128, 9.5% at M40000/K256 and 12.9% at M40960/K256,
against inherited already-fused preparation schedules. Other tested cases,
including width 384, failed the consistency gate and kept their incumbents.
These are separate studies, not matched 5090 cross-revision results.

The width-128 Nsight Systems capture removes one kernel per training pair:
the approximately 220 us derivative transpose-axis quantizer disappears,
while the recompute kernel grows from approximately 554 to 609 us. Those
direct, instrumented timings diagnose the mechanism; they are not added to
claim an exact graph-replay speedup. Compute counters were unavailable, so
no measured DRAM/L2 hit-rate, spill or tensor-core-saturation claim is made.

Fresh-process warm compiled-direct PyTorch peak allocation at M40000/K128
fell from 178.6 MB to 146.7 MB; before/after live allocation stayed constant
in each process. This is not total VRAM or graph-pool accounting. The removed
BF16 derivative is 40.96 MB, with a modeled 81.92 MB write/read round trip;
the peak difference is smaller because lifetimes overlap differently.

Standalone linear backward also retained quantize-once staged GEMMs in the
tested winners. Its calibrated 12-round races qualified median latency
reductions around 14.0% at width 128 and 9.2% at width 256 against inherited
schedules; width 384 tied. These measure **backward only**, not a complete
linear training pair. One-load/shared-G preparation was explored, not forced.

## What reuse means here

For a 40000×128 input, BF16 X is 10.24 MB while a 128×128 BF16 weight is only
32 KiB. Weight packing once avoids repeated conversion across many CTAs even
when those BF16 weight reads hit cache. But packing X globally adds writes
and reads and is not universally best when it has only one consumer tile.
The tuner must compare complete fused and staged paths.

The useful hybrid is to prepare reusable operands once in scratch **per
quantization orientation**, while quantizing newly computed intermediates in
their producer. Row-wise and token-wise block groupings cannot share scales.
Packed weights are not cached across optimizer updates. More CTA persistence,
larger tiles or one fused launch are not automatically better.

## Validation and reproduction

The CUDA-enabled laptop regression passed **194 tests** (32 intentionally
deselected), covering exact derivative bytes/scales, SiLU/GELU, both supported
quantization arithmetic pairs, widths 128/256/384 at 40000 rows, registered
operator checks, compiled/selected APIs, partial gradients, changed-buffer
graphs, cache cleanup and bounded live allocations. CPU compatibility checks
also passed. This is kernel/integration qualification, not model convergence.

Tools are listed in [benchmarks](../benchmarks/README.md). For example:

```bash
python -m benchmarks.qualify_mxfp8_long_mlp \
  --initial prior-result.json --shape 40000,256,512,256 \
  --trials 16 --rounds 12 --output autotune_reports/new-long-study
```

Retained local artifacts: `autotune_reports/long_narrow_20261108/`, including
the `5090/` source-comparison journals and worker manifests. Output directories
are ignored and reside on Storage 4. No exploratory winner was installed in
an application's runtime cache. Both layouts and existing alternatives remain
available to the shared UCB/local/random tuner; defaults are unchanged.
