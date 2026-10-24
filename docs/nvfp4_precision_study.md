# NVFP4 numerical-policy experiments

For the combined two-GPU correctness, training, PTQ, performance and external
comparison report, see [the broad evaluation](nvfp4_broad_evaluation_20261023.md).

These are opt-in experiments, not a new default or a claim of universal training
qualification. The production design remains BF16 public inputs/master weights,
NVFP4 forward, FP32 outer scales, and MXFP8 backward.

## Policies and contracts

`NVFP4QuantConfig.block_scale_selection` selects `amax`, `lsq1`, `mse5`, or
`four_over_six`. These are numerical policies, not interchangeable tuning
schedules. `lsq1` is one algebraic least-squares update, not trainable-scale QAT.
Adaptive policies compare actual E4M3/E2M1 reconstructions and retain a candidate
only for lower block squared error. This does not guarantee lower GEMM error.

`four_over_six` tests block maxima mapped to six or four. With the existing
`exact` outer scale, the second candidate can clip at the E4M3 ceiling. Explicit
`tensor_scale_mode="exact_four_over_six"` instead computes the FP32 outer factor
as `amax * FP32(1/1536)` (one for zero regions), reserving room for raw block
scales 256 and 384. Supply the mode in both the public scale config and explicit
dynamic config. Pair it with `amax` for a same-outer control.

All existing defaults remain unchanged. Experimental policies cannot be
promoted into ordinary runtime-winner caches. The headroom mode is pinned by
the tuning adapter. Schedule choices, including retaining packed candidate
bytes, remain independently tunable.

The upstream [Four Over Six implementation](https://github.com/mit-han-lab/fouroversix)
motivates this headroom. RTX retains its minimum-normal E4M3 scale floor and
native nearest-even rounding; this is not a claim of bitwise identity with every
upstream implementation. CPU format tests compare against TorchAO's independent
encoder/decoder and quantizer.

## Reproducible GPU study

Start with correctness, then tune within one numerical policy:

```bash
python -m unittest tests.test_nvfp4_adaptive_scale tests.test_nvfp4_headroom -v
python -m benchmarks.tune_nvfp4_adaptive_scale \
  --selection four_over_six --outer-mode exact_four_over_six \
  --shape 8192,768,1536 --trials 32 --output autotune_reports/4over6-tune
python -m benchmarks.benchmark_nvfp4_precision_frontier \
  --shape 8192,768,1536 --config autotune_reports/4over6-tune/winner.json \
  --mx-trials 32 --variants jit-8x8-exact explicit-four_over_six \
  --output autotune_reports/4over6-frontier.jsonl
```

The tuner fixes GEMM and region geometry and searches quantizer schedules. This
is not an exhaustive joint search. CUDA-graph replay timings exclude compilation
and tuning; preserve all samples, not only the minimum. Use fresh output paths.

## Pretrained-model evidence

The offline evaluator requires separately supplied official DINOv3 source and
checkpoint plus CIFAR-10. These assets are not bundled. It preserves biases
outside RTX's intentionally bias-free linears. The experiment uses a fixed BF16
gallery from the training split and held-out test queries; cosine 1-NN accuracy
is a representation-regression task, not ImageNet accuracy or training loss.

```bash
python -m benchmarks.evaluate_nvfp4_dinov3_cifar \
  --dataset /path/to/cifar-parent --checkpoint /path/to/dinov3-vits16.pth \
  --gallery-count 1000 --test-count 500 --batch-size 8 --compile \
  --outer-mode exact_four_over_six \
  --variants bf16 native:amax native:four_over_six \
  --output autotune_reports/dino-headroom.jsonl
```

Outputs contain checkpoint/input/harness hashes, sample indices, exact configs,
feature RMS/cosine metrics, paired classification changes, and an eager-versus-
compiled first-batch diagnostic. Native backbone timing is not a tuned
component benchmark. Optional `emulated:POLICY:w`, `:x`, and `:both` interventions
isolate operand format error using decoded BF16 GEMMs; these are not bit-exact
native references or kernel performance measurements.

Do not turn finite outputs into an eager/compiled equivalence claim. Full-model
compilation can change BF16 intermediates and low-precision quantization can
amplify those changes. Investigate observed drift separately from isolated
registered-operator correctness.

## Offline activation-aware weight calibration

The research-only CPU study implements max, bounded Four Over Six, exhaustive
weight MSE and block-diagonal Local-Hessian scale selection. The latter minimizes
`delta @ E[x_block x_block.T] @ delta.T` per 16-weight block, inspired by
[NVIDIA's Local-Hessian method](https://nvidia.github.io/Model-Optimizer/announcements/20260909-local-hessian.html).
An opt-in offline API now emits the native packed weight consumed by existing
RTX inference kernels. This is experimental PTQ functionality, not a new
default, dynamic-training policy, or guarantee of model-quality improvement.

```python
import rtx

# Collect representative TRAIN/calibration inputs to this particular linear.
# Never include the evaluation split or select scales using held-out labels.
moments = rtx.NVFP4SecondMomentCollector(linear.in_features)
for x_calibration in calibration_inputs:
    moments.update(x_calibration)

packed_w = rtx.calibrate_nvfp4_weight(
    linear.weight, policy="local_hessian",
    second_moments=moments.second_moments(),
)
replacement = rtx.NVFP4Linear(
    linear.in_features, linear.out_features,
    packed_weight=packed_w, scaling="current", autotune="balanced",
).eval()
```

RTX linears remain bias-free; the example assumes `linear.bias is None`.
Collection and scale search are detached CPU-side offline work. Calibration
returns packed storage on the original weight device; CUDA inference still
requires a supported GPU. No calibration hooks, moment tensors, or scale search
execute in the replacement's forward. BF16 dynamic activations use the existing
tensor-current quantizer; packed weights do **not** enable regional activation
scaling. Hold this activation policy fixed when comparing weight calibration.

`max`, `four_over_six`, `mse`, and `local_hessian` are available controls. All
use one fixed FP32 tensor outer scale (default `amax/2688`, or an explicit
`outer_scale`) and normal E4M3 block scales. MSE/Local-Hessian exhaustively
search 119 scales with deterministic ties and bounded row-chunk scratch space.
The collector accumulates uncentered moments in FP64; scale objectives are
evaluated in FP64, while the stored decode factor remains FP32. This offline
arithmetic is not promised bit-identical to the runtime FP32 quantizer.

Experimental regularization is available as
`calibrate_nvfp4_weight(..., policy="local_hessian", hessian_shrinkage=alpha)`.
For `alpha` in `[0,1]`, the objective uses
`(1-alpha)*H + alpha*mean(diag(H))*I`. Zero preserves the original objective;
one produces exactly the MSE packing. Positive shrinkage falls back to MSE for
zero-energy blocks. This changes only offline scale selection, never the FP32
outer factor or runtime launches. It is not enabled by default.

The evaluator's `ptq:selected` research policy compares max, MSE, and
Local-Hessian with shrinkage 0, 0.5, and 0.9 **per layer**, using a separate
training-validation split. Selection minimizes full-layer output SSE with
quantized validation activations; it does not optimize test accuracy. All
layers remain W4A4: selection does not silently restore BF16 layers.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m benchmarks.evaluate_nvfp4_dinov3_cifar \
  --dataset /path/to/cifar-parent --checkpoint /path/to/dinov3-vits16.pth \
  --gallery-count 1000 --test-count 1000 --test-offset 500 --compile \
  --calibration-count 64 --calibration-max-tokens 4096 \
  --calibration-validation-count 32 --calibration-validation-max-tokens 1024 \
  --variants bf16 ptq:max ptq:mse ptq:selected \
  --output autotune_reports/dino-selected-ptq.jsonl
```

All choices are frozen before any held-out forward, including compile checks.
Validation X is quantized using its original full BF16 activation batch before
token sampling; this preserves the batch's tensor outer scale. Selection SSE
is computed on the complete layer output, so it includes cross-block error
interactions even though candidate fitting uses a block-local objective.
`--test-offset 500` excludes the first 500 deterministic test-query indices,
not the first 500 raw dataset records. Keep the same seed for this exclusion.

The first fixed selection experiment did **not** improve whole-model quality.
On 1,000 fresh queries disjoint from the earlier 500, compiled BF16 scored
931 correct, packed max 902, MSE 912, and selected 889. Feature NRMS was
0.4181 / 0.3973 / 0.4855 for max / MSE / selected. Selection chose unregularized
Local-Hessian in 44 layers and shrinkage 0.5 in the last four attention-output
projections. Median independent validation layer-output SSE decreased 22.1%
versus max, yet backbone feature error increased. Thus independent local
validation alone does not establish downstream quality. Do not promote this
selector as a recommended model conversion recipe.

Paired 20,000-sample bootstrap intervals for accuracy change versus max were
[-0.4, 2.4] points for MSE and [-3.0, 0.4] for selected (seed 20261023).
Neither interval establishes a classification improvement. These intervals
condition on a fixed checkpoint, gallery, and calibration set. Future research
should validate block-level or end-to-end feature reconstruction on calibration
data, rather than assuming local SSE is a sufficient proxy.

The first execution exposed a TorchAO CPU-to-CUDA packed-wrapper `.to()` error.
The rerun copied underlying buffers explicitly; all selection scores and BF16
predictions matched exactly. The packed `NVFP4Linear` constructor now also
uses explicit buffer transfer, preserving ragged logical shape and scale layout.
Frozen selections are saved exclusively as a tensor-buffer `.selection.pt`
artifact before held-out forwards, with checkpoint/split provenance and a hash.

### Whole-model selection with a compiled guard

`benchmarks.select_nvfp4_dinov3_model` replaces independent-linear scoring with
blockwise coordinate choices scored on the **actual complete W4A4 backbone**.
Upstream quantization, attention, nonlinearities, residuals and normalization
therefore participate in the objective. Each transformer block selects one
weight policy for its four linears; the activation policy stays fixed.

The prespecified objective equally weights normalized CLS and patch-feature
squared errors versus a BF16 teacher. One coordinate pass starts from the better
global max/MSE baseline on search images. An independent training-image guard
then compares global max, global MSE and the searched model under fullgraph
compilation. The guard can retain a baseline. No test labels or images are
accessed until this decision and the packed weights are frozen.

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m benchmarks.select_nvfp4_dinov3_model \
  --dataset /path/to/cifar-parent --checkpoint /path/to/dinov3-vits16.pth \
  --gallery-count 1000 --fit-count 64 --fit-max-tokens 4096 \
  --search-count 32 --guard-count 32 --test-count 1000 --test-offset 1500 \
  --output autotune_reports/dino-model-selection.jsonl
```

Fit, search, guard and gallery images are mutually disjoint CIFAR training
subsets. The offset excludes all 1,500 earlier test queries at the same seed.
Packed buffers are copied in place only between candidates; there is no search
inside inference and all 48 linears remain W4A4. Compilation explicitly disables
Inductor CUDA graphs for this experiment. The audit artifact contains every
candidate's packed buffers, all search/guard scores, and the frozen state.

On the 2026-10-23 RTX 5070 laptop run, the compiled guard retained the searched
model. On 1,000 fresh test queries:

| Policy | Correct | CLS NRMS | Patch NRMS | Mean CLS cosine |
| --- | ---: | ---: | ---: | ---: |
| BF16 | 932 | 0 | 0 | 1 |
| Max | 907 | 0.4162 | 0.3361 | 0.9119 |
| MSE | 900 | 0.4000 | 0.3196 | 0.9186 |
| Whole-model selected | 917 | 0.3599 | 0.2897 | 0.9345 |

CLS normalized RMS decreased 13.5% versus max and 10.0% versus MSE. The compiled
guard objective decreased from 0.12613 (MSE) to 0.10513 (selected), consistent
with the held-out feature improvement. Paired 20,000-resample accuracy intervals
for selected-minus-max and selected-minus-MSE were [-0.5,2.5] and [0.3,3.1]
percentage points respectively (seed 20261023). These are conditional on one
checkpoint/calibration split; accuracy superiority over max is not established.
Selected still trails BF16 by 1.5 points on this sample. This is an experimental
offline PTQ selector, not a new library default or training recipe.

Capture distinct calibration and held-out activations from the BF16 model:

```bash
python -m benchmarks.evaluate_nvfp4_dinov3_cifar \
  --dataset /path/to/cifar-parent --checkpoint /path/to/dinov3-vits16.pth \
  --gallery-count 16 --test-count 16 --variants bf16 \
  --capture-linear blocks.0.mlp.fc1 --capture-output /tmp/dino-fc1.pt \
  --output autotune_reports/dino-capture.jsonl
python -m benchmarks.study_nvfp4_calibration \
  --input-tensors /tmp/dino-fc1.pt --output-rows 64 \
  --output autotune_reports/dino-calibration.jsonl
```

The full weight tensor determines the FP32 outer scale even when only a prefix
of output rows is evaluated. Default `production-normal` uses representable
E4M3 scales at least `2**-6`; `full-e4m3` is an explicitly different research
domain including subnormals. Calibration ignores cross-block error interactions
and does not prove improvement on unseen models. Evaluate held-out outputs and
model quality rather than selecting solely on weight MSE.

For a matched native packed-inference comparison (all 48 DINO block linears):

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 python -m benchmarks.evaluate_nvfp4_dinov3_cifar \
  --dataset /path/to/cifar-parent --checkpoint /path/to/dinov3-vits16.pth \
  --gallery-count 1000 --test-count 500 --batch-size 8 --compile \
  --calibration-count 64 --calibration-max-tokens 4096 \
  --variants bf16 ptq:max ptq:four_over_six ptq:mse ptq:local_hessian \
  --output autotune_reports/dino-calibrated-ptq.jsonl
```

Calibration uses image-balanced samples from separate training images, excludes
gallery images, and never reads test activations or labels. Each layer records
activation/moment hashes; outputs include labels for paired statistical analysis.
All PTQ candidates use the same tensor-current activation policy. Remove
`--compile` and choose a fresh output path for the independent eager control.

### 2026-10-23 matched-checkpoint result: no Local-Hessian promotion

On the RTX 5070 laptop / Torch 2.13, the above compiled 500-query experiment
produced the following. Errors are CLS-feature normalized RMS versus the matched
BF16 backbone, not GEMM error. Accuracy is CIFAR cosine 1-NN, not ImageNet.

| Policy | Correct / 500 | Feature NRMS | Mean cosine |
| --- | ---: | ---: | ---: |
| BF16 | 480 | 0 | 1 |
| Packed max | 464 | 0.4152 | 0.9123 |
| Packed four-over-six | 466 | 0.4087 | 0.9150 |
| Packed weight-MSE | 470 | 0.3947 | 0.9207 |
| Packed Local-Hessian | 449 | 0.4809 | 0.8822 |

With 20,000 paired query bootstrap samples (seed 20261023), the 95% intervals
for accuracy change versus max are [-1.4, 2.2], [-0.6, 3.2], and [-5.2, -0.8]
percentage points for four-over-six, MSE, and Local-Hessian respectively. These
intervals condition on this checkpoint/gallery/calibration set; they do not
represent training-seed or model-family variability. The earlier single-layer
Local-Hessian gain does **not** generalize to this whole-model experiment.
No default changes or model-quality superiority claim follows from these data.
The separately executed eager control also regressed: BF16 480/500, max
463/500, Local-Hessian 445/500 (feature NRMS 0.4169 versus 0.4849).
Thus the observed Local-Hessian regression is not exclusively a compiler effect.
Possible causes include ignored cross-block error interactions, changing
activation distributions through the quantized model, and objectives that do
not preserve downstream-sensitive directions. These are hypotheses, not yet
isolated causes. Regularized or selective calibration needs a calibration-only
validation split before any further held-out comparison.

Full-model eager/compiled feature drift remains substantial, despite bitwise
equality in isolated packed operators. Diagnostic full-backbone times here
were 4.04 ms BF16 and 4.88–6.69 ms PTQ; this workload does not demonstrate an
NVFP4 speedup, nor are the untuned timings a ranking of calibration algorithms.
Calibration changes stored weights/scales, not the inference launch sequence.

## Validation commands

```bash
CUDA_VISIBLE_DEVICES='' python -m pytest \
  tests/test_nvfp4_external_reference.py tests/test_nvfp4_calibration_study.py \
  tests/test_nvfp4_calibration.py tests/test_nvfp4_dinov3_cifar.py \
  tests/test_nvfp4_headroom.py tests/test_nvfp4_convergence.py
python -m unittest tests.test_nvfp4 tests.test_nvfp4_adaptive_scale \
  tests.test_nvfp4_headroom -v
```

CPU skips do not establish GPU support. Repeat numerical, compiled, and timing
checks on each target GPU/Torch stack before adopting a winner.

## Public API integration

The reusable calibration and guarded accuracy-search APIs are documented in
[quantization policy](quantization_policy.md). The DINO adapter described here remains
research-specific; its results do not establish universal PTQ or training
improvements. Runtime defaults and kernel qualification boundaries are unchanged.
