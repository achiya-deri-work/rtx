# Ettin application qualification

Status: prospective application regression qualification passed on RTX 5090
with Torch 2.12.1 and 2.13.0. Not universal release certification or training
convergence evidence.
Ettin replaces DINO as the primary application target by user decision on
2026-10-24. DINO's numerical findings and unchanged tests remain available;
rowwise-FP8 training as the cause of its sensitivity is a hypothesis, not a
causal conclusion from the experiments.

## Model and numerical scope

The [official checkpoint](https://huggingface.co/jhu-clsp/ettin-encoder-32m)
is pinned to revision `1b8ba06455dd44f80fc9c1ca9e22806157a57379`.
It contains 32,081,344 parameters, ten ModernBERT layers, hidden width 384,
six attention heads, and gated GELU MLP intermediate width 576. We convert
all forty bias-free backbone attention/MLP linears. Embeddings, normalization,
the masked-LM head, vocabulary decoder, and tied weights remain unchanged in
BF16. The checkpoint's float32 storage declaration does not prove its original
training precision.

The BF16 reference uses identical checkpoint weights, SDPA attention, input
tokens, padding masks, and masked targets. Model-internal `reference_compile`
is disabled; optional compilation wraps the complete model with
`fullgraph=True, dynamic=False`. Autotuning is off for this numerical study.
The scripted dynamic modes exercise BF16-input inference, not backward.

## Reproduce

Install optional research dependencies in the project virtual environment:

```bash
python -m pip install 'transformers==4.57.6' safetensors
python -m benchmarks.ettin_regression --output autotune_reports/ettin-eager.jsonl
python -m benchmarks.ettin_regression --compile --seeds 4700 \
  --modes mxfp8_dynamic mxfp8_ptq nvfp4_jit nvfp4_block_four_over_six \
  --output autotune_reports/ettin-compiled.jsonl
```

Output paths must be new. Checkpoints/tokenizers live in the external Hugging
Face cache and are not committed or packaged. JSONL records include checkpoint
revision, source identity, fixture hash, shape, mask seed, selected-token count,
loss, KL divergence, prediction agreement, and hidden-state errors. Delayed
scaling state persists through the recorded batch sequence, so its order and
first-call bootstrap are part of that experiment.

## Initial RTX 5090 / Torch 2.13 results

Sixteen authored sentences with three deterministic masking seeds produce
twelve batches and 107 masked targets. This is a small fidelity fixture, not
a representative held-out corpus. Loss/KL/agreement are weighted by masked
target count; hidden cosine below is the mean of batch means.

| Mode | Masked NLL | KL from BF16 | Top-1 agreement with BF16 | Hidden cosine |
|---|---:|---:|---:|---:|
| BF16 | 2.6312 | 0 | 100% | 1.0000 |
| MXFP8 dynamic | 2.6266 | 0.0313 | 89.72% | 0.9946 |
| MXFP8 PTQ | 2.6266 | 0.0313 | 89.72% | 0.9946 |
| NVFP4 JIT-region | 2.9846 | 0.4776 | 70.09% | 0.9268 |
| NVFP4 delayed | 2.9776 | 0.4594 | 69.16% | 0.9252 |
| NVFP4 current | 2.9826 | 0.4558 | 70.09% | 0.9252 |
| NVFP4 block | 3.0068 | 0.4480 | 64.49% | 0.9275 |
| NVFP4 block Four Over Six | 2.9213 | 0.3982 | 75.70% | 0.9364 |
| NVFP4 PTQ current | 2.9658 | 0.4010 | 72.90% | 0.9334 |

All observed outputs are finite. The slightly lower MXFP8 loss is not evidence
that quantization improves language modeling. Four Over Six improves this
NVFP4 fixture but is not selected as a universal default.

Whole-model fullgraph/static compilation completed for BF16, MXFP8 dynamic,
MXFP8 PTQ, NVFP4 JIT, and NVFP4 block Four Over Six on seed 4700 (four batches
each), with finite outputs and no graph-break fallback. This is a compiler
execution smoke, not eager/compiled numerical equivalence. Weighted NLLs on
that same seed were:

| Mode | Eager | Compiled |
|---|---:|---:|
| BF16 | 2.9397 | 2.9438 |
| MXFP8 dynamic / PTQ | 2.9492 | 2.9862 |
| NVFP4 JIT | 3.6198 | 3.4557 |
| NVFP4 block Four Over Six | 3.4177 | 3.5363 |

Compilation changes surrounding floating-point computations even for BF16;
the larger low-precision differences need broader comparison and localization,
not a bitwise-parity assertion or a hidden tolerance adjustment. Four offline
tests also pass for deterministic masking, special-token preservation,
backbone-only conversion, preserved parameter identity, and untouched heads.

## Prospective qualification

Before the next, disjoint paragraph tranche (offset 128, mask seed 4801),
the coarse prospective regression guards are fixed as follows: all outputs
finite, at least 1,000 masked targets, weighted NLL increase versus matched
BF16 at most 0.1 for MXFP8 (including recomputed MLP) or 0.5 for NVFP4, and
mean hidden cosine at least 0.97 or 0.85 respectively. These deliberately coarse
guards detect severe application regressions; passing is not an accuracy or
convergence guarantee. They are enforced by `--require-quality`, recorded in
the result, and must not be revised to bless a failed run. The earlier 128
paragraphs are exploratory evidence, not the prospective test.

The disjoint tranche passed these unchanged guards for all ten policies on
both Torch stacks. It uses 128 paragraphs from the
[WikiText test split](https://huggingface.co/datasets/Salesforce/wikitext), pinned
to `b08601e04326c79dfdd32d625aee71d232d685c3`, offset 128, seed 4801,
batch size 4 and padded sequence length 511 (ragged flattened M=2044).
There are 2,898 masked targets in 32 batches. This tranche is held out from our
exploratory selection, not necessarily from the checkpoint's original training.

| Policy | NLL increase, Torch 2.13 | NLL increase, Torch 2.12.1 | Hidden cosine, Torch 2.13 |
|---|---:|---:|---:|
| MXFP8 dynamic / PTQ | 0.00522 | 0.00555 | 0.99234 |
| MXFP8 recomputed MLP | 0.00664 | 0.00594 | 0.99207 |
| NVFP4 JIT-region | 0.32073 | 0.32045 | 0.91613 |
| NVFP4 delayed | 0.32367 | 0.32243 | 0.91684 |
| NVFP4 current | 0.32481 | 0.32357 | 0.91684 |
| NVFP4 block | 0.30175 | 0.30147 | 0.92255 |
| NVFP4 block Four Over Six | 0.29210 | 0.29019 | 0.93414 |
| NVFP4 PTQ current | 0.32021 | 0.32267 | 0.91315 |

Paired batch bootstrap (5,000 replicates, seed 4802) gives Torch 2.13
95% intervals of [-0.00661, 0.01649] for MXFP8 dynamic NLL increase,
[-0.00554, 0.01848] for recomputed MLP, [0.28047, 0.36015] for NVFP4
JIT-region, and [0.26172, 0.32248] for Four Over Six. These quantify this
sample, not general model accuracy. They are reporting metrics, not new gates.
NVFP4's degradation is real despite passing the coarse regression guard.

Whole-model fullgraph runs passed on both versions. The preceding exploratory
128-paragraph tranche also completed eager on 2.13 and compiled on both stacks.
The recomputed MLP preserves gate ordering and exact GELU activation choice,
but its high-precision gating and packed hidden output have different rounding
boundaries from the ordinary BF16 model.

## Short training lifecycle, not convergence

Eight dynamic policies completed twelve AdamW steps in eager Torch 2.13 and
compiled Torch 2.13/2.12.1, with BF16 parameters and optimizer moments. Every
step checked all 65 trainable parameter gradients and optimizer states for
finiteness. Sixteen training paragraphs and sixteen separate validation
paragraphs use length 128; validation copies avoid modifying delayed state.

Whole-model initial gradient relative L2 versus BF16 was about 0.393 for MXFP8
and 0.832–1.074 for NVFP4 in the eager run. These are not gradient-parity
results; quantized forward computations also change the loss and its gradient.
Validation changes were mixed across policies and stacks. Twelve steps establish
an executable training lifecycle, not convergence or a quality recommendation.

```bash
python -m benchmarks.ettin_regression --compile --corpus wikitext \
  --paragraph-offset 128 --paragraphs 128 --max-length 511 --seeds 4801 \
  --require-quality --output autotune_reports/ettin-prospective.jsonl
python -m benchmarks.train_ettin_qualification --compile --steps 12 \
  --output autotune_reports/ettin-training.jsonl
```

The release qualification script runs these checks after pytest and the kernel
production matrix. Seven offline helper tests pass on both Torch stacks.
Historical DINO model-quality tests remain opt-in via `-m dinov3_research`,
with their original assertions intact. Longer matched convergence runs and
cross-SKU qualification remain outside this local merge qualification.
