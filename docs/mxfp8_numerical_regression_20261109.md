# MXFP8 numerical regression check — 2026-11-09

## Result

No numerical regression was found in the recent dual-orientation MLP fusion
on the RTX 5070 laptop. With the inherited schedule fixed, `dd665e6` and
`f23aad5` produced **bitwise-identical output, dX, gate/up dW and down dW**
in all 18 cases. Both new `recompute_both` layouts (`none` and `xor_16b`)
also matched that previous-source result bitwise in all 18 cases.

The actual selected configurations are a separate comparison. They match
bitwise in 14 of 18 cases; four width-128 cases have tiny differences in the
weight gradients after changing split counts, partitioning and final reducers.
Output and dX match bitwise in every case. The largest weight-gradient
relative-L2 difference from the previous result is **0.008705%**. The largest
increase in relative-L2 error versus BF16 is **0.0000465 percentage points**.
There is no evidence here of a meaningful accuracy trade for the speedup.

All **90 source/policy cases** (18 previous plus 72 current) passed the
independent explicitly quantized recipe oracle. Across them, the worst
relative-L2 implementation errors are:

| Tensor | Maximum recipe error |
| --- | ---: |
| Output | 0.012045% |
| dX | 0.007145% |
| Gate/up dW | 0.013689% |
| Down dW | 0.019274% |

These are not the error versus an unquantized MLP. On the ordinary SiLU
40000-token input, that BF16-relative difference is approximately **6.5%**
for output and gradients in **both** source versions. Multiple forward and
backward quantizations contribute; the new fusion did not introduce this gap.
This is tensor-level numerical qualification, not a training-convergence claim.

A further **21 targeted CUDA tests passed** (24 unrelated cases deselected):
16 byte-exact derivative-pack/scale cases across SiLU/GELU, both SMEM layouts,
FP32 versus BF16x2 quantization and 128/192 token extents; four independent
long-dW comparisons up to 208192 tokens with changed-input CUDA-graph replay;
and a compiled, fully recomputed balanced-backward MLP comparison. The pack
tests also mutate inputs during replay and check output guard bytes. No kernel
or numerical policy was changed to obtain these results.

## Controlled comparison

- GPU: RTX 5070 Laptop, 36 SMs; driver 610.88.
- Torch 2.13.0+cu132, CUDA 13.2, CuTe 4.7.0.
- Isolated archived RTX sources from `dd665e6` and `f23aad5`; the installed
  laptop checkout was not modified. The identical benchmark script and
  shape-retargeting helper were copied into the old benchmark directory,
  without changing old RTX package sources.
- CPU-generated FP32 random values rounded to BF16 before transfer. SHA-256
  checks verify identical input, weights and upstream gradient across sources.
- Widths K=128/256/384, hidden H=2K, output O=K; gate/up weight `[2H,K]`.
- Per width: M192 SiLU normal; M40000 SiLU normal with two seeds; M40000
  GELU uniform X in `[-0.5,0.5)`; M40960 GELU normal; M512 SiLU outliers.
- Normal X standard deviation 0.2, weights 0.02, upstream gradient 0.1.
  Outliers multiply every 32nd X feature by 32 and weight feature by 8.
- Upstream gradients are noncontiguous. One M40000 normal case per width
  runs default-mode `torch.compile(fullgraph=True, dynamic=False)` with
  Inductor CUDA graphs disabled; the others run the functional API eagerly.
- The current source runs the inherited policy, the two new fusion layouts,
  and the actual current selected policy. The 5090 study's saved policies are
  reused without tuning or installing runtime winners on the laptop.
- Tiny oracle shapes cannot use long-sequence split counts. Both sources use
  the same unsplit dW reduction there; long shapes keep their saved split
  policies and only retarget shape-dependent uniform reduction ranges.

## References and numerical gates

The independent operational oracle uses PyTorch E4M3 conversion, integer-bit
E8M0 exponent selection, FP32 matmuls/activation/derivatives, and explicit
BF16 rounding at the production boundaries. No production quantizer, GEMM
or derivative kernel is used. Separate BF16 and FP32 autograd MLPs measure
the total precision-policy difference. TF32 and BF16 reduced-precision
reductions are disabled.

Every output/gradient records finiteness, maximum absolute error, error RMS,
reference RMS, relative L2, normalized maximum error, cosine and unequal
element count. Relative denominators have floor `1e-30`. The existing
implementation gate remains finite and (relative L2 < 2.5% or maximum
absolute error <= `1e-12`); no threshold was relaxed.

This same-device experiment isolates the source change. It does **not**
independently measure a hardware/driver migration: that requires running the
same fixtures and schedules on the other assigned device. No 5090 was used
for this check. Application-specific regressions still require the actual
affected fixture, selected configurations and precision/stack metadata.

## Artifacts and reproduction

Developer driver: `benchmarks/compare_mlp_numerics_revisions.py`. Run the
identical script from each archived source root, previous first:

```bash
python -m benchmarks.compare_mlp_numerics_revisions \
  --version previous --configs /absolute/configs --output /absolute/previous-results
python -m benchmarks.compare_mlp_numerics_revisions \
  --version current --configs /absolute/configs --output /absolute/current-results \
  --previous-output /absolute/previous-results
```

`configs` contains `previous-w128.json` / `current-w128.json` and equivalents
for widths 256/384, from the retained 2026-11-08 5090 qualification results.
Both source roots need the same driver and `qualify_mxfp8_long_mlp.py` helper.

Local residuals: `autotune_reports/numerics_revision_20261109/` (ignored,
Storage 4), with all case metrics and per-source environment/source hashes.
Full previous outputs and isolated sources remain on `laptop_5070` under
`/home/achiya/rtx-numeric-20261109.IPuy1v/`; deterministic seeds allow fixture
reconstruction without copying hundreds of megabytes of output tensors.
