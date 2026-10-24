# NVFP4 broad evaluation — 2026-10-23

## Bottom line

RTX now has useful, tested NVFP4 options, but no universally best recipe.
The strongest supported conclusions from this study are:

1. Retained production paths passed the selected GPU correctness/API suites on
   both laptop GPUs. This includes the previously problematic delayed
   distribution-jump test on the present Torch 2.12.1 snapshot.
2. All four outer-scaling modes trained a small compiled decoder successfully.
   None matched BF16 validation loss exactly; none was faster on that tiny,
   untuned model. Larger GEMMs did show useful NVFP4 speedups.
3. Four-over-six improves Gaussian GEMM fidelity with less cost than MSE5 in
   the measured schedules. It is a candidate, not a universal default: its
   advantage changes with width/distribution and does not guarantee task gains.
4. PTQ must be evaluated end-to-end. Raw Local-Hessian can hurt a model despite
   improving the local objective. Guarded whole-model selection is more credible
   than a universal per-layer reconstruction rule.
5. We cannot claim NVIDIA-level training convergence, or superiority to
   Transformer Engine, ModelOpt or Quartet II. The external comparisons below
   compare recipes and evidence, not matched end-to-end benchmark scores.

No runtime defaults, numerical gates, quarantines or production kernels were
changed by this study. The decoder test CLI gained `nvfp4_current`, and a new
checkpoint/allocator lifecycle test was added.

## Coverage and provenance

Fresh execution used isolated RTX snapshots on `personal_laptop` (RTX 5060
Laptop, Torch 2.12.1/CUDA 13.2) and `laptop_5070` (RTX 5070 Laptop,
Torch 2.13.0/CUDA 13.2). TorchAO 0.18 and CuTe DSL 4.7 are used. No shared
training/service was stopped. Pytest was installed into the 5070 snapshot's
`test_deps`, not its shared environment. GPU work was serialized per device.

Artifacts: `autotune_reports/nvfp4_broad_20261023/{5060,5070}/`.
JSON/JSONL records contain seeds, configurations, environments and metrics;
pytest JUnit files retain individual tests/skips. The checkout was dirty, so
HEAD alone is not a complete source identity; benchmark hashes and decoder
source snapshots are retained in manifests. This is not a released build.

| Fresh coverage | Result / scope |
|---|---|
| 5060 GPU suite | 107 passed, 1 skipped, 156 subtests |
| 5070 GPU suite | 97 passed, 1 skipped, 150 subtests |
| 5060 lifecycle | 1 passed, 4 mode subtests |
| Focused local CPU suite | 53 passed, 1 CUDA skip, 296 subtests |
| Compiled TinyStories decoder | 6 modes, 120 updates each, one seed |
| CUDA-graph linear comparison | 18 variants × 3 shapes = 54 results |
| Independent distribution study | 1,800 numerical-simulation records |
| Fresh DINOv3 evaluation | 10 policies, 300 shared held-out queries |

GPU suites cover current/delayed/block/JIT-region, exact/power-of-two outer
scales, four-over-six and headroom, adaptive scale candidates, native packed
inference, TorchAO format parity, ragged shapes, compiled training and packed
conversion, dX/dW, delayed history, stream changes and state loading.
The intentionally quarantined warp-specialized regional epilogue stayed
skipped. This is risk-based coverage, not all autotuner configurations.

The lifecycle addition compares ordinary execution against reentrant
checkpointing exactly for output, dX and dW, with noncontiguous upstream
gradients. After warmup, live allocated bytes stayed constant over three
10-step windows in every outer-scaling mode. This small fixed-shape result
does not establish variable-shape cache bounds or long-run memory safety.

## Numerical policies are separate axes

| Option | Strength | Weakness / appropriate use |
|---|---|---|
| JIT row-region | Current local FP32 scales; immediately follows range changes; no history state | Region reduction/scale traffic/epilogue cost; region size is a numerical choice as well as a scheduling choice |
| Tensor delayed | Avoids current tensorwide scale synchronization after bootstrap | Stateful; history can be stale; shape/stream/reset contracts matter; not always fastest |
| Tensor current | Straightforward current-tensor baseline, easy numerical interpretation | Global reduction overhead was conspicuous in measured latency |
| Block-only | No nontrivial FP32 outer scale; cheapest measured scale path | E4M3 scale range can lose tiny values or saturate large ones; distribution-dependent |
| Exact FP32 outer | Uses actual amax-derived scale | Not uniformly more accurate than a power-of-two outer after E4M3 rounding |
| Power-of-two outer | Simple exponent adjustment and broad range protection | Often changes range coverage rather than the effective quantization grid; regions can give identical results on benign inputs |
| Amax block scale | Cheap, simple control | Does not explicitly minimize reconstruction error |
| LSQ1 | One accepted discrete least-squares refinement | Smaller fidelity gains; extra conversion work; not learned-scale QAT |
| MSE5 | Searches five nearby block-scale candidates | Lowest local error need not give lowest output/task error; often more expensive |
| Four-over-six | Two candidates exploit FP4's nonuniform spacing | Different distributions/widths can prefer amax; not free |
| Four-over-six headroom | Explicit amax/1536 outer leaves room for the /4 candidate | Changes outer-scale policy separately from block selection; RTX still uses normal E4M3 scale floor, unlike some references |
| Offline weight MSE | Better weight reconstruction without runtime scale-search overhead | Conversion cost; unchanged activation error; not a downstream guarantee |
| Offline Local-Hessian | Uses observed activation directions within block16 | Ignores cross-block/whole-model effects; can overfit/misalign with task fidelity |
| Hessian shrinkage | Interpolates toward MSE; regularizes noisy directions | Alpha remains calibration/model dependent; alpha=1 is weight-MSE, not extra information |
| Whole-model policy selection | Captures accumulated model effects; independent guard may fall back | Requires representative data and costly offline evaluation; one checkpoint is not a general recipe |

Cached/reused packed quantization, direct versus fragment-register epilogues,
scale layout, TMA, stages and CTA tiling are implementation options, not new
accuracy recipes. Their correctness coverage does not imply every legal
schedule was benchmarked here. Removed families were not resurrected.

## Short real-model training

Byte-level TinyStories decoder: two layers, hidden256, FFN512, four heads,
batch16 × sequence128, 120 updates = 245,760 training tokens per mode.
Same source initialization/data seed, BF16 parameters and AdamW state, QK norm,
existing decoder normalization and initialization recipe. Fullgraph compilation,
autotune off, separate processes. Validation uses four held-out batches.
Model initialization and learning-rate schedule are matched across modes.
An independent CPU read of every final checkpoint found all model/optimizer
tensors finite (89 tensors per checkpoint, all six modes at step 120).

| Mode | Final validation cross-entropy | Delta vs BF16 | Median logged warm tokens/s | Peak allocated MiB |
|---|---:|---:|---:|---:|
| BF16 | 2.188936 | — | 550,678 | 143.2 |
| MXFP8 | 2.199128 | +0.010192 | 372,379 | 154.7 |
| NVFP4 current | 2.215512 | +0.026576 | 354,377 | 133.5 |
| NVFP4 delayed | 2.215512 | +0.026576 | 316,093 | 147.2 |
| NVFP4 JIT-region | 2.215512 | +0.026576 | 381,119 | 133.5 |
| NVFP4 block | 2.217547 | +0.028611 | 415,904 | 133.5 |

Throughput is the median of trainer interval records at steps40–120, not
isolated GEMM timing. Validation/checkpoint interval accounting and laptop
power variation make it diagnostic. The result does not meet a claim that
NVFP4 always beats BF16 or MXFP8 end-to-end. All runs completed and recorded
finite gradient norms; no universal training tolerance was invented afterward.

Identical current/delayed/JIT losses are not evidence of identical kernels:
power-of-two outer scaling can preserve the same effective E4M3×FP4 grid
when scales stay away from range boundaries. This benign workload does not
adequately distinguish range-shift behavior; separate GPU tests do that.
This is one short seed, not converged language modeling or a statistical
ranking of training recipes. Adaptive block-policy training evidence remains
the separate synthetic sweep described next, not this decoder experiment.

### Existing width study, not newly rerun here

The preceding [width sweep](../autotune_reports/width_sweep_5060_20261023/REPORT.md)
contains 84 compiled synthetic regression trajectories, widths128/256/384/512,
three seeds and seven policies. Every trajectory stayed finite. Four-over-six
plus headroom was best among NVFP4 at128/256; amax slightly won at384/512.
Relative gaps shrink with width partly because the BF16 residual increases
and wider models have slower updates under mean loss. Do not turn this into
evidence that width alone fixes FP4 training.

## Native forward/backward accuracy and speed

5070 laptop; fixed schedules, TF32 disabled, independent FP32 GEMM/gradient
reference. CUDA graphs, four calls per capture, nine timing rounds of20 replays,
warmup outside timing. Both dX and dW included; no autotuning was performed.
Variant order is deterministically shuffled. Values below are median microseconds
for forward / forward+backward, not best-ever configurations.

| Mode | 128×256×256 | 512×768×768 | 2048×768×1536 |
|---|---:|---:|---:|
| BF16 | 3.10 / 7.74 | 17.03 / 47.46 | 131.17 / 391.97 |
| MXFP8 | 5.23 / 11.58 | 11.92 / 32.74 | 65.71 / 193.64 |
| NVFP4 current | 55.96 / 56.54 | 72.08 / 84.31 | 198.02 / 321.10 |
| NVFP4 delayed | 26.16 / 37.69 | 26.05 / 50.63 | 53.99 / 198.58 |
| NVFP4 block | 9.83 / 16.20 | 8.87 / 29.79 | 40.47 / 167.67 |
| JIT8×8 exact amax | 20.55 / 27.09 | 10.99 / 32.49 | 51.14 / 176.19 |
| JIT8×8 exact four-over-six | 27.83 / 34.12 | 14.45 / 35.83 | 69.03 / 193.93 |
| JIT8×8 exact MSE5 | 37.56 / 43.75 | 20.24 / 42.41 | 95.72 / 220.55 |
| JIT8×8 exact LSQ1 | 30.76 / 38.05 | 14.64 / 35.93 | 71.57 / 198.56 |

For normal inputs, forward NRMS against FP32 was approximately BF16 0.17%,
MXFP8 3.75–3.76%, NVFP4 amax13.4–13.5%, four-over-six12.3–12.6%,
MSE5 12.6–12.7%, LSQ1 12.9–13.0%, block16.1%. Quantized dX/dW use
MXFP8, not NVFP4, so their errors remain roughly3.8%; this is not full-FP4
training parity with external stacks.

There is a clear size crossover: on the largest shape JIT amax is ~2.22x
BF16 forward+backward and ~1.10x MXFP8; four-over-six is ~2.02x BF16 but
approximately tied with MXFP8. Small-shape BF16 wins. This does not establish
an architecture-independent crossover, tuned optimum or sustained-power speedup.
Some timing samples drift noticeably (e.g. large-shape LSQ1); retain distributions
and repeat balanced sustained-power timing before promoting a performance winner.
No Nsight counters were collected, so no bandwidth/occupancy diagnosis is claimed.

## Distribution and geometry study

Independent quantize/dequantize simulation followed by FP32 GEMM: three seeds,
three shapes including129×257×385 tails, normal/uniform/row-varying/outlier
inputs, five region pairs, ten scale treatments =1,800 finite records.
These are numerical simulations, not timings or bitwise production equivalence.

Mean output NRMS (%) for region8×8 across shapes/seeds:

| Distribution | Power2 amax | Exact amax | Exact MSE5 | Exact LSQ1 |
|---|---:|---:|---:|---:|
| Normal | 13.446 | 13.444 | 12.595 | 12.928 |
| Uniform | 13.872 | 13.948 | 12.832 | 13.261 |
| Row-varying range | 13.548 | 13.422 | 12.721 | 12.927 |
| Injected outliers | 5.143 | 4.895 | 4.872 | 4.963 |

The lower outlier NRMS is not proof that outliers are easier: large reference
energy changes the denominator. LSQ1 can improve local SSE while worsening
GEMM NRMS, as seen in the outlier aggregate. Exact amax is not uniformly better
than power2. Averaging all distributions gives exact-amax NRMS11.25% for1×1,
11.41% for5×4,11.43% for8×8 and11.47% for32×32: modest numerical differences,
not enough to choose a universal region size without latency/model evidence.

## Fresh pretrained-model PTQ / dynamic evaluation

DINOv3 ViT-S/16 checkpoint hash begins `08c60483bc63c04f`; all48 transformer
block linears use the selected policy. DINO's existing biases are preserved by
the research adapter; the core RTX API remains bias-free. CIFAR-10 cosine1-NN
uses a fixed1000-image BF16 train gallery and300 test queries at offset3000.
Same queries, seed20261023, fullgraph compiled, batch8. No classifier training.
PTQ fits64 distinct train images with4096 sampled tokens per linear. No policy
was selected on these test queries. Dynamic policies use exact8×8 outer scaling;
PTQ policies use identical current-amax activation quantization, differing in W.

| Policy | Correct /300 | CLS NRMS | Mean CLS cosine |
|---|---:|---:|---:|
| BF16 | 282 | 0 | 1 |
| MXFP8 | 281 | .1479 | .9890 |
| Dynamic amax | 273 | .4066 | .9160 |
| Dynamic LSQ1 | 271 | .4030 | .9174 |
| Dynamic MSE5 | 268 | .3964 | .9203 |
| Dynamic four-over-six | 273 | .3924 | .9219 |
| Packed max | 270 | .4100 | .9145 |
| Packed four-over-six | 275 | .4097 | .9144 |
| Packed MSE | 270 | .3938 | .9213 |
| Packed Local-Hessian | 266 | .4814 | .8819 |

Paired20,000-resample95% intervals, conditional on this checkpoint/gallery:
dynamic4/6 minus amax0pp [-2.67,+2.67]; packed4/6 minus max+1.67pp
[-1.33,+4.67]; packed MSE minus max0pp [-3.33,+3.33]; Local-Hessian
minus max-1.33pp [-5,+2]. None establishes superiority over its max baseline.
Dynamic amax minus BF16 is-3pp [-5.67,-0.33]. Feature-fidelity changes and
accuracy changes must be reported separately. This is not ImageNet accuracy.

Untuned full-backbone timings are recorded but not used for a winner claim.
In particular MXFP8 was21.3ms versus BF16 3.60ms in this run: this path requires
separate performance investigation, not an assumption of low-precision speedup.
Full-backbone eager/compiled drift is recorded; references are matched compiled
BF16, and isolated packed-operator compile parity was tested independently.

### Earlier guarded selection evidence, separate query set

The earlier [model-level selection study](../autotune_reports/nvfp4_model_selection_20261023/REPORT.md)
used1000 different queries: BF16932, max907, MSE900, selected917. Selected CLS
NRMS .3599 versus max .4162. The selected-minus-max accuracy interval included
zero, while selected-minus-MSE did not. This is supportive evidence for a
whole-model objective and guard, not a result reproduced on the new300 queries
and not a generic per-block heuristic. Independent-layer selection previously
failed; raw Local-Hessian also regressed on the older500-query experiment.

## Outside comparisons: what is and is not comparable

### NVIDIA Transformer Engine and NVFP4 pretraining

NVIDIA's paper reports a 12B model trained on 10T tokens with FP8-comparable
loss/task accuracy using RHT, 2D weight quantization, stochastic gradient
rounding and selective high precision. That is vastly stronger training evidence
than our short runs, and a different backward recipe. We use MXFP8 backward;
we did not run their matched checkpoint/data/hardware protocol.
[Paper](https://arxiv.org/abs/2509.25149).

TE documentation describes FP32 tensor outer scales, E4M3 block-16 scales,
2D weights, RHT for wgrad operands and stochastic rounding for gradients.
Our regional FP32 outer scales and hybrid backward are deliberate differences,
not reproductions of the complete TE recipe.
[NVFP4 guide](https://docs.nvidia.com/deeplearning/transformer-engine/features/low_precision_training/nvfp4/nvfp4.html).

Important update: the TE 2.22 development support matrix describes partial SM12.x
NVFP4 training with RHT/SR disabled, while its 2.20.2 NVFP4 guide lists training
SM10.0/10.3. MXFP8 remains unsupported on 12.x in the development matrix.
Support is version-specific; do not claim nobody else supports RTX NVFP4.
We did not install/benchmark TE here.
[Development matrix](https://nvidia.github.io/TransformerEngine/support_matrix.html).

### Four Over Six

The paper reports improved NVFP4 training quality on Nemotron 3 Nano 30B-A3B.
Our two-candidate method is motivated by that work, not independently invented.
Explicit headroom matches the amax/1536 design choice; our normal E4M3 floor
still differs from the reference's treatment of tiny/subnormal scales. Thus
RTX's implementation is not claimed bitwise identical to the official recipe.
[Paper](https://arxiv.org/abs/2512.02010),
[pinned reference](https://github.com/mit-han-lab/fouroversix/blob/dadfad6901d473a734fe71e0b082e70ee993e23a/src/fouroversix/quantize/pytorch/reference.py).

### Quartet II

Quartet II uses MS-EDEN unbiased gradient estimation, reports training up to
1.9B parameters/38B tokens, and publishes RTX 5090 as well as B200 kernels.
Its reported up-to-4.2x BF16 speedup is not a matched comparison to our laptops.
It is a serious RTX training comparator; it also means lack of native stochastic
rounding does not by itself prove useful FP4 backward is impossible. We have
not reproduced its full recipe or replaced our MXFP8 backward.
[Paper](https://arxiv.org/abs/2601.22813),
[official kernels](https://github.com/IST-DASLab/Quartet-II).

### TorchAO

TorchAO 0.18 encoding/packing/dequantization routines are an actually executed
external reference in our tests, including midpoint rounding, saturation,
normal scale floor and zero/ragged handling. Format parity is stronger than
just comparing two RTX paths, but it is not a comparison of complete training
recipes or end-to-end throughput.
[Versioned source](https://github.com/pytorch/ao/blob/v0.18.0/torchao/prototype/mx_formats/nvfp4_tensor.py).

### NVIDIA ModelOpt and PTQ alternatives

ModelOpt exposes max/MSE/Local-Hessian, AWQ, smoothing, SVDQuant and layerwise
calibration. Our block-local Hessian search is in this family, not a replacement
for its whole deployment ecosystem or full error-propagation methods.
[Calibration API](https://nvidia.github.io/Model-Optimizer/reference/generated/modelopt.torch.quantization.model_calib.html).

Selective high-precision layers and QAD are sensible external baselines for
recovering task accuracy; lower precision everywhere is not the only valid
comparison. SVDQuant adds a high-precision low-rank correction branch, so it
changes runtime work/storage and cannot be compared as a free quantizer swap.
None of AWQ/SVDQuant/QAD was executed here.
[ModelOpt concepts](https://nvidia.github.io/Model-Optimizer/guides/_basic_quantization.html),
[SVDQuant paper](https://arxiv.org/abs/2411.05007).

## Recommendations and unresolved work

- Keep JIT-region as the general dynamic policy. Tune schedules under a fixed
  numerical contract; geometry/policy changes need model-quality checks.
- Retain delayed as an explicit option, not a guaranteed throughput winner.
  The current jump test passes, superseding that specific historical failure
  on these snapshots, but does not clear unrelated quarantines or all schedules.
- Retain block as an explicitly range-sensitive fast path; expose risks clearly.
- Keep current as the straightforward reference/debug path.
- Offer four-over-six/headroom as opt-in candidates. Do not promote them from
  Gaussian GEMM error or the narrow-regression result alone.
- For PTQ, compare max/MSE/4-over-6 and regularized Hessian candidates with
  whole-model search plus independent guard; preserve fallback and test isolation.
- Next high-value evidence: multi-seed longer decoder runs; independent model/
  calibration seeds for PTQ; selective attention/MLP quantization; matched
  same-device external implementations; sustained-power autotuned performance.
- W4A16/W4A8 research emulation, RHT, full-FP4 backward,2D weight training,
  low-rank correction, distributed training and all legal autotuner schedules
  are not qualified by this study. `WeightOnlyConfig` still means packed W
  residency with dynamically quantized X, not a production W4A16 GEMM.

## Reproduction commands

Use the appropriate isolated snapshot interpreter and `PYTHONPATH=.`; on5070
pytest additionally uses `PYTHONPATH=.:test_deps`. All output paths must be new.

```bash
# 5060 GPU suite (5070 omits mutation and legacy PTQ files in this study)
python -m pytest -q tests/test_nvfp4.py tests/test_nvfp4_adaptive_scale.py \
  tests/test_nvfp4_external_reference.py tests/test_nvfp4_calibration.py \
  tests/test_nvfp4_calibration_mutation.py tests/test_ptq_calibration.py \
  tests/test_ptq.py --junitxml=pytest.xml
python -m pytest -q tests/test_nvfp4_lifecycle.py --junitxml=lifecycle.xml

# Fresh process per mode: bf16,mxfp8,nvfp4_delayed,nvfp4_jit_row_region,
# nvfp4_block,nvfp4_current. Original corpus uses the TinyStories train shard.
python benchmarks/train_decoder.py --precision MODE --steps 120 \
  --batch-size 16 --sequence-length 128 --layers 2 --hidden-size 256 \
  --intermediate-size 512 --heads 4 --warmup-steps 10 --validation-batches 4 \
  --validation-interval 40 --checkpoint-interval 120 --log-interval 20 \
  --optimizer bf16 --autotune off --no-resume --output OUTPUT

# Fresh process per shape:128,256,256;512,768,768;2048,768,1536
python benchmarks/benchmark_nvfp4_precision_frontier.py --shape M,N,K \
  --rounds 9 --calls 20 --output frontier.jsonl
python benchmarks/study_nvfp4_precision.py --shape 128,256,256 \
  --shape 129,257,385 --shape 512,768,768 --seeds 3 --output distributions.jsonl

python benchmarks/evaluate_nvfp4_dinov3_cifar.py --dataset research_data \
  --checkpoint fp8dinov3/weights/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --gallery-count 1000 --test-count 300 --test-offset 3000 --batch-size 8 \
  --compile --calibration-count 64 --calibration-max-tokens 4096 \
  --variants bf16 mxfp8 native:amax native:lsq1 native:mse5 \
  native:four_over_six ptq:max ptq:four_over_six ptq:mse ptq:local_hessian \
  --output dino-300.jsonl
```
