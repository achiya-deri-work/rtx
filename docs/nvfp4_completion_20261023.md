# NVFP4 forward and experimental backward — 2026-10-23

The stable contract remains **NVFP4 forward, MXFP8 backward**. Experimental
MS-EDEN FP4 backward works in eager/fullgraph execution, but has not earned a
default change: its gradient error is higher and the tested training paths
are slower. This report is about NVFP4, not MXFP8 forward.

## Four-over-Six policy-fixed search and training

Follow-up completed overnight October 23–24. **Keep Four-over-Six opt-in:**
it lowers forward error and is now considerably faster than its old fixed
schedule, but does not consistently improve short decoder training.
MXFP8 backward and runtime defaults are unchanged.

The research tuner now covers tensor-current outer scaling, including compiled
amax/outer-factor calculation in preparation/full timings. Both current and
JIT-region use native `mma128` scale emission and search quantizer/GEMM schedules
with FP32 outer factors and the Four-over-Six numerical policy pinned.
192 trials passed numerical checks; one unstable timing was ineligible.
An explicit packed-conversion reuse probe is also available. No new quantizer
math was substituted: this pass improves schedules/layout use and evaluation.

Fresh public-module confirmations on RTX 5090 GPU 0, fullgraph/static compiled
grad-enabled forward with graph replay, two processes per cell, reverse mode
order on repeat two; mean process medians in microseconds:

| M × N × K | JIT amax control | JIT Four-over-Six | Current Four-over-Six | Quartet Four-over-Six |
|---|---:|---:|---:|---:|
| 2048 × 768 × 1536 | 14.09 | 19.32 | 18.21 | 19.56 |
| 8192 × 768 × 1536 | 37.72 | 47.17 | 49.96 | 41.06 |
| 32768 × 768 × 1536 | 166.92 | 186.58 | 223.01 | 204.17 |
| 8192 × 3072 × 768 | 61.21 | 64.10 | 64.10 | 56.72 |

All 32 timing windows were stationary and passed exact changed-X/W replay
against eager. Both operands are dynamic BF16; quantization is included.
Current's earlier fixed-default times were 34.07/99.78/406.75/170.29 us:
the tuned confirmations are roughly 1.8–2.7× faster. Current wins the small
case against Quartet; regional Four-over-Six wins the longest case. Quartet
still wins the middle and wide cases. These are bounded searches, not global
optima. Research phase timings are diagnostic; this public-API table is the
authoritative complete-forward comparison, especially for tensor-current.

JIT Four-over-Six costs about 5–37% more than the amax control, depending on
shape, for about 8.6% lower Gaussian forward NRMS (13.44–13.45% → 12.29–12.30%).
Tensor-current and Quartet are 12.28–12.30%, numerically close but not asserted
bit-identical. JIT geometry remains 8×8 rows; Four-over-Six uses explicit
`exact_four_over_six` headroom, while the amax control uses `exact`.

Held-out seed 29, two shapes (2048×768×1536 and 8192×3072×768), 16 additional
public-API cases, all passing numerical/replay gates:

| Activation distribution | JIT amax NRMS | JIT Four-over-Six NRMS | Current / Quartet NRMS |
|---|---:|---:|---:|
| 10% nonzero, variance-rescaled | 11.07–11.08% | 9.73–9.74% | 9.73–9.75% |
| 1% 16× outliers | 13.27% | 12.52–12.54% | 12.53–12.54% |

### Matched short decoder training

16 runs: widths 256/512, seeds 17/29, four policies, 250 updates each;
4,000 total updates / 8.192M byte tokens. Two-layer TinyStories decoder,
batch 16, sequence 128, BF16 boundaries/AdamW, QK/pre/post norms and the existing
0.02/depth-scaled initialization. Fullgraph compilation, MXFP8 backward for
all NVFP4 policies. Initial-weight and corpus hashes matched within each
width/seed. All runs had finite parameters and gradient norms.

| Policy | Width 256 mean validation CE | Width 512 mean validation CE |
|---|---:|---:|
| BF16 | 1.485440 | 1.321631 |
| JIT amax exact | 1.511987 | 1.330364 |
| JIT Four-over-Six | 1.508198 | 1.332543 |
| Current Four-over-Six | 1.511988 | 1.332770 |

JIT Four-over-Six improves one seed and worsens the other at each width.
The favorable width-256 mean does not establish superiority: width-512's mean
reverses. These short two-seed, one-corpus studies are not long-run convergence
qualification. Training schedules are fixed research configurations, not a
training-throughput comparison. There is no new-default recommendation.

Validation additionally passed 21 tests and 67 subtests covering independent
Four-over-Six references, policies, and graph replay. Source changes are in
`tune_nvfp4_adaptive_scale`, `compare_nvfp4_forward`, and the existing matched
decoder harness. Artifacts live under `autotune_reports/nvfp4_four_six_20261023/`:
`jit_*/winner.json`, `current_*/winner.json`, per-trial residuals,
`confirm_*.json`, `holdout_*.json`, and `train_*.jsonl`. The 8192×768×1536
current confirmation uses `current_pack_on_*/confirmation.json` after the
explicit reuse probe. No research winner is silently installed in runtime caches.

Next performance work: target the middle/wide shapes and preparation cost;
do not assume lower local error guarantees better training or make numerical
policy a hidden performance-autotuning choice.

## Completed Quartet II forward-only comparison

Measured on the same RTX 5090 GPU 0, with two fresh processes per mode/shape
and reversed mode order in the second repeat. Mean of the two process medians,
in microseconds; **dynamic BF16 X/W → quantization → GEMM → BF16 output**:

| M × N × K | RTX tuned JIT-region | Quartet II Four-over-Six | RTX Four-over-Six, fixed defaults |
|---|---:|---:|---:|
| 2048 × 768 × 1536 | 14.04 | 19.54 | 34.07 |
| 8192 × 768 × 1536 | 37.66 | 40.94 | 99.78 |
| 32768 × 768 × 1536 | 166.22 | 204.06 | 406.75 |
| 8192 × 3072 × 768 | 60.93 | 56.48 | 170.29 |

RTX JIT is 1.39×, 1.09× and 1.23× faster on the first three; Quartet is
1.08× faster on the wide projection. This **is not equal numerical policy**:
JIT uses 8×8-row regional exact factors with amax block scales, while Quartet
uses tensor-current Four-over-Six. RTX Four-over-Six uses explicit
`exact_four_over_six` tensor-current headroom and Four-over-Six block selection.
That third column is a fixed-default implementation, not a tuned-winner bound.
Quartet uses its FlashInfer/cuDNN backend and vendor plan selection.

Against FP32 GEMM on the identical original BF16 inputs, JIT forward NRMS is
13.44–13.45%; both Four-over-Six implementations yield 12.28–12.30%. Their
aggregate errors nearly coincide, not a claim of bit-identical outputs.
Quartet therefore gives about 8.6% less forward NRMS than JIT on these Gaussian
inputs. Our explicit Four-over-Six path matches that local accuracy but needs
performance work; the JIT speed advantage does not establish equal-accuracy
superiority or a model-level quality ranking.

All 24 measurements passed timing stationarity and exact eager/compiled/
changed-input graph checks. Both X and W were mutated after capture, compared
to fresh eager execution, then restored and rechecked. Between-process median
spread was below 0.62% in every cell. Each process uses fullgraph/static
`torch.compile`, automatic compiler CUDA graphs disabled, and manual CUDA
graphs (8 calls × 10 replays × 15 samples, 200 ms bounded warmup). Forward is
grad-enabled; no backward is run. Inputs use seed 17, X std 1 and W std 0.02;
TF32 is off. Compilation, validation and graph capture are excluded; all
forward quantization and outer-scale calculation are included. No weights
are prequantized. This is hot replay, not a cold-cache or full-model benchmark.

Quartet is the previously qualified isolated stream/GPU-seed patched build,
upstream revision `5f2a47e8f464b17ef4b1cac78db8d14edd1f3f7d`. The source path and
SHA256 are recorded in its artifacts; this is not unmodified-upstream
qualification. RTX JIT uses the previous completed forward-search winners;
no new search was performed for this comparison.

Harness: `benchmarks/compare_nvfp4_forward.py`. Four harness tests passed,
including live CUDA replay and shape/policy rejection. Raw artifacts:
`autotune_reports/nvfp4_forward_quartet_20261023/repeat{1,2}_{mode}_{M}_{N}_{K}.json`.
The shorter-named `repeat1_quartet_2048.json` is a pilot and is excluded.
No runtime defaults were changed. The remaining forward target is efficient
Four-over-Six preparation and wide-projection scheduling, not an assumption
that Quartet wins all large forwards.

## Implemented scope

- Full quantizer/GEMM schedule search for JIT-region and tensor-delayed
  forward, keeping the numerical policy fixed during performance tuning.
- Original-BF16-source experimental FP4 backward with independently selected
  dX/dW schedules and explicit current/JIT-region/block forward configs.
- Fused Hadamard rotation, quantization, correction statistics and layout
  construction; scale-only post-hoc range alignment; optional staged amax.
- Explicit packed-forward-source ablation, restricted to tensor-current
  forward. This changes the backward source values; it is not a free
  scheduling optimization.
- Policy/revision/shape-validated selection loading, persistent residuals,
  focused preparation/GEMM refinement, and separate phase measurements.

The experimental API and limitations are in [NVFP4Linear](nvfp4_linear.md#experimental-fp4-backward).
Stateful delayed forward is **not** supported by the stateless experimental
FP4-backward function. Stable `NVFP4Linear` retains its existing delayed path.

## Forward results

RTX 5090, Torch 2.13.0+cu132, CuTe DSL 4.7.0; BF16 inputs/outputs, FP32
outer factors. 168 candidate evaluations across four shapes and two policies,
followed by fresh-process confirmation of all finalists on the same GPU.
Microseconds, CUDA-graph replay:

| M × N × K | JIT-region | Tensor-delayed |
|---|---:|---:|
| 2048 × 768 × 1536 | 13.83 | 16.69 |
| 8192 × 768 × 1536 | 37.86 | 41.06 |
| 32768 × 768 × 1536 | 165.61 | 170.14 |
| 8192 × 3072 × 768 | 61.81 | 63.55 |

JIT uses exact FP32 factors and 8×8-row geometry. Delayed uses fixed history
initialized from input amax: this is a steady-state schedule comparison, not
changing-distribution delayed-training qualification. Earlier cross-GPU
rankings did not reproduce; use this same-device confirmation instead.

## Matched compiled forward + backward

Both paths use the **same selected NVFP4 JIT forward**, checked bit-exact,
original BF16 operands, and identical upstream gradients. FP4 random-sign
and stochastic-rounding RNG generation is included. MXFP8 backward uses the
explicit default backward config, not a newly exhaustive MXFP8 search.
Initial backward search used 12 trials per shape/gradient (96 total).

| M × N × K | MXFP8 backward | Experimental FP4 backward |
|---|---:|---:|
| 2048 × 768 × 1536 | 46.49 µs | 112.14 µs |
| 8192 × 768 × 1536 | 160.62 µs | 264.81 µs |
| 32768 × 768 × 1536 | 716.33 µs | 838.46 µs |
| 8192 × 3072 × 768 | 314.47 µs | 427.85 µs |

All eight full-path timing windows passed stationarity checks. Each uses 15
samples, eight calls per captured graph, ten replays per sample, and bounded
time-based warmup; compilation is excluded. These are bounded searches, not
global optima. Cross-run timing changes must not be attributed solely to tuning.

On the largest shape, isolated FP4 dX preparation/GEMM were 109/113 µs;
dW preparation/GEMM were 298/128 µs. Separate RNG cost was 6/10 µs.
Isolated phases are not additive. Preparation is the main remaining target,
especially dW; merely replacing MXFP8 multiplication with FP4 is insufficient.
Staged amax refinement was approximately tied with inline reduction, not a
demonstrated material win.

An additional 40 focused trials (16 preparation, 12 dX GEMM, 12 dW GEMM)
on the largest shape produced a fresh matched confirmation of **716.26 µs
versus 836.32 µs**. Both windows were stationary; the ~0.3% FP4 change from
the table is not a material speedup. The baseline's forward is still NVFP4.

## Numerical and integration evidence

- Identical forward NRMS: 13.44–13.45% against original BF16-input FP32 GEMM.
- MXFP8 gradient NRMS: 3.76–3.77%; experimental FP4: 13.70–13.76%.
- Independent decoded-packed GEMM gate: 0.5% NRMS. Quantization error is not
  hidden inside this accumulation/layout correctness gate.
- 158 CUDA/integration tests passed, with 67 subtests; the final CPU/reference/
  documentation recheck passed 57 tests and 250 subtests. Coverage includes fake/autograd/fullgraph, checkpoint RNG, packed
  state ownership, independent preparation references and scale policies.
- A width-256, 200-step compiled teacher-regression smoke with BF16 AdamW
  moments completed. Held-out NRMS: BF16 3.92%, NVFP4/MXFP8 18.76%,
  NVFP4/MS-EDEN 18.74%. This is one synthetic dataset/seed, not transformer
  qualification or evidence that FP4 backward is generally as accurate.

MS-EDEN is **not advertised as unbiased**. Earlier independent-oracle sparse
tests show residual bias; scale stochastic rounding does not remove every
source of estimator bias. Packed-source reuse is also a distinct numerical
policy. Neither is selected silently by the performance autotuner.

## Reproduction and artifacts

Raw samples, configs, device/toolchain identities, numerical metrics and
per-trial residuals live under
`autotune_reports/nvfp4_completion_20261023/` (local, not runtime defaults).
`matched_tuned*.json` contains the four matched comparisons;
`matched_refined_32768.json` contains the final largest-shape confirmation;
`confirmed_gpu0_*/confirmation.json` contains forward confirmations;
`training_jit_256.json` contains the training smoke.

```bash
python -m benchmarks.tune_nvfp4_adaptive_scale --help
python -m benchmarks.tune_ms_eden_preparation --help
python -m benchmarks.compare_nvfp4_backward_policies --help
python -m benchmarks.train_ms_eden_probe --help
```

Quartet/FlashInfer packed-GEMM diagnostic results and numerical-policy
differences remain recorded in [the preparation study](operand_preparation_plan.md).
Those fixed-schedule diagnostics establish backend headroom, not an
end-to-end paper reproduction. Next kernel work should target the measured
backward preparation cost and independently confirm any winner; further
transformer/cross-device qualification remains necessary before promotion.
