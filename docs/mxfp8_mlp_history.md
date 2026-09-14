# MXFP8 MLP historical measurements and qualification

Historical, revision-specific evidence extracted from the user guide. Commands
and pending-work statements describe the recorded experiment, not current
API defaults. Use [the current guide](mxfp8_mlp.md) and
[shared frontend controls](api.md#shared-frontend-controls) for integration.

## Reproduce the target-shape result

An explicitly scoped RTX 5070 Ti preset is supplied, not installed as a default:

```bash
python -m benchmarks.benchmark_mxfp8_mlp \
  --shape 512,768,1536,768 \
  --config-json benchmarks/configs/mxfp8_mlp_5070ti_m512_k768_h1536_o768.json \
  --output autotune_reports/mlp/reproduce.json
```

The benchmark checks the preset's device/shape/activation, compiles the entire
MLP, validates forward and all gradients against the quantized reference,
audits saved storage, and measures warmed CUDA graph replay. The matched BF16
baseline checkpoints the whole MLP with `use_reentrant=True`. Ordinary BF16
without recomputation is reported separately. See the optimization ledger for
measurements and qualification; do not generalize one shape's speedup.

```bash
python -m rtx.autotune.mlp --shape 128,128,256,128 \
  --phase train --trials 32 --seconds 120 \
  --output-dir autotune_results/mlp
```

Shape order is M,K,H,O. `--phase forward` tunes the two-kernel forward;
`--phase train` tunes total forward plus recomputed backward, not independent
component winners. The shared bandit/model/local-search portfolio persists
trial residuals and confirmations. Timing is warmed CUDA graph replay, with
warmup on the capture stream. Graphs belong to the surrounding model/step.

All arithmetic helpers are registered CUDA operators with fake implementations;
the MLP owns custom autograd. Direct Inductor lowerings avoid opaque dispatcher
calls in the compiled forward and recompute helpers. The existing linear
backward operators retain their compiler-visible lowerings. No eager tensor
arithmetic is hidden in the registered forward body.

## Qualification and remaining release gates

For stage-wise error against BF16, including bounded inputs and selective
precision ablations, see [the divergence study](mxfp8_mlp_divergence.md).
Low implementation error against a quantized reference does not mean the
complete all-MXFP8 MLP is numerically near BF16.

**Known blocker from the expanded numerical stress matrix:** the optional
producer-quantized packed-down path can issue an illegal GPU instruction for
M=K=H=O=1. The conservative default passes that case. Do not use the packed
policy on singleton projections pending a kernel/legality fix. See the
[numerical stress report](mxfp8_mlp_numerics.md) for full results and reproduction.

Run `python -m unittest tests.test_mxfp8_mlp -v` on the target GPU. The matrix
covers independent forward/all-gradient references, all activations, ragged
shapes, saved-tensor identity, partial gradients, noncontiguous dY, opcheck,
fullgraph compilation, launch counts, long reductions, accumulation, BF16-state
optimizer training, graph replay, live-memory stability, independent streams,
weight version checks, schedule parity, and vectorized weight loads.

Release requires passing this matrix on both supported Torch stacks, device
memory/race validation, meaningful end-to-end performance comparisons, and
larger-model convergence evidence. Current scope is local SM120 qualification;
do not infer 5090/laptop qualification or a speedup from successful compilation.

### Local evidence (2026-09-12)

The latest target-shape result supersedes the early development measurements.
On RTX 5070 Ti, M512/K768/H1536/O768, SiLU, Torch 2.13.0, the explicitly
tuned preset measured 112.896–113.056 µs forward + backward over three
independent runs. Compiled reentrant-checkpointed BF16 measured
170.731–170.795 µs: **1.510–1.512× speedup**. Against ordinary compiled
BF16 without recomputation, the speedup is approximately **1.27×**.
Torch 2.12.1 measured 112.384 µs versus 171.680 µs checkpointed BF16.
These are warmed CUDA graph replay measurements, not compilation latency or
full optimizer-step timings, and require the supplied shape/device preset.

The preset explicitly selects approximate FP32 reciprocal SiLU, producer-side
down-weight quantization, packed recomputation, and overlapping backward
branches. It is not the conservative constructor default. Numerical errors
against the independent quantized reference were below 0.017% relative L2
for output and all three gradients in the target benchmark; this is **not**
an error bound against BF16 or evidence of transformer convergence.

Both supported Torch stacks passed the consolidated 28-test frontend/kernel
matrix (conversion, policy serialization, all activations, independent
forward/gradient references, opcheck, compilation, graph training, streams,
and memory stability). Run it with:

```bash
python -m unittest tests.test_mxfp8_mlp_frontend tests.test_mxfp8_mlp \
  tests.test_mxfp8_mlp_decode -v
```

The suite also includes a subsequently added real `autotune_` smoke test,
checking persistent trial records, policy roundtrip, and unchanged weights.
Selected device memcheck and racecheck suites reported zero errors/hazards.
The forward trace verified two actual compute launches for the winning path.
See [the optimization ledger](mxfp8_mlp_optimization.md) for exact artifacts,
historical measurements, and remaining cross-device qualification work.

Explicit tuning belongs before compilation and capture. Changing policy afterward requires
recompiling/recapturing the surrounding workload. `rtx.clear_runtime_caches()`
releases compiled helper references; it does not rewrite model parameters.
`fullgraph=True` removes graph breaks, not the GMEM connection between kernels.
Opt-in first-hit selection can use internal replay plans following the
surrounding compiler graph setting; explicit-config execution is unchanged.
See [first-hit plan ownership](mlp_first_hit_plan_contract.md) for graph nesting
and capture boundaries. Warm required paths before manual application capture.
