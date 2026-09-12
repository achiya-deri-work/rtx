# Selective precision, shape schedules, and small-workload qualification

This follow-up turns the BF16-gate experiment into an explicit
`rtx.MXFP8MLP(..., precision="bf16_gate")` / `rtx.mxfp8_mlp(..., precision=...)`
choice. Native `precision="mxfp8"` remains the default. `precision="bf16"` is an
explicit fallback, not an automatically substituted numerical policy.

## Ownership and contracts

- The native operator retains its registered CUDA/fake/autograd implementation.
- The hybrid composes existing registered MXFP8 linear operators with a
  traceable BF16 gate GEMM, FP32 activation/product and BF16 hidden boundary.
  A traceable autograd function combines gate/up gradients before MXFP8 dX/dW.
- Configuration normalization is outside the traced tensor computation, using
  constant keys. No opaque custom-op launcher contains eager Torch gating.
- Reentrant checkpointing retains only input/parameter storage. Hybrid/BF16
  eager training uses `.backward()`; compiled fullgraph first-order gradients
  are supported. They do not have the native two-kernel forward contract.
- Biases, second derivatives, hybrid export and dynamic=True are not qualified.

`autotune_()` installs a schedule only for the example shape/device/dtype.
`configure_shape_()` permits explicit per-shape numerical policy, including a
BF16 fallback. `policy_for()` returns the effective `(config, precision)`.
The bounded 128-entry map never retains input tensors and is not weight state.
Reinstall policies on another device and recompile/recapture after mutation.

See [the MLP manual](mxfp8_mlp.md) for constructor and persistence examples.

## Autotuning changes

The numerical policy and native SiLU arithmetic are fixed per context. Hybrid
search excludes inactive native GLU/recompute axes. Both projections retain
independent backward spaces, with additional compound proposals:

- `dw_split_count`: shape-relative FP32 workspace splitting and tree reduction;
- `dx_materialized_rows`: row-major quantize-once dX with configurable logical
  transpose tile rows, independent of the dW backend.

No winner is installed as a universal default. Context revision 32 separates
these studies from earlier search identities. Config IDs ignore provably
inactive fields (for example materialized quantizer settings while using a
fused backward leg); full configurations remain serialized. This avoids
spending trials timing identical executed schedules under different IDs.
Saved tuning results include precision, shape, phase, activation, device,
revisioned configuration and trial-store identity.

The 48-trial revision-30 decoder-shape study reduced native candidate timing
from approximately 180 to 91.4 µs; hybrid timing improved only slightly, to
77.0 µs. Revision 31 additionally preserves a fused dW backend when proposing
shape-relative splits. Both spaces remain available rather than forcing all
split reductions through materialized quantization.
Revision 32 also excludes partial output tiles from the experimental
cooperative forward following the failed ragged trial described below.

## Ragged performance diagnosis

RTX 5070 Ti, Torch 2.13.0+cu132. The public hybrid API was compiled fullgraph
with aggressive benchmark options. Numbers below are CUDA-graph replay
forward plus fully recomputed backward, excluding compilation and warmup.

| M,K,H,O | Activation | Fused backward | Materialized dX, fused dW |
|---|---|---:|---:|
| 129,128,256,128 | SiLU | 173.41 µs | 98.46 µs |
| 129,128,256,128 | GELU | 173.54 µs | 98.98 µs |
| 257,256,512,256 | SiLU | 297.72 µs | 144.29 µs |
| 257,256,512,256 | GELU | 298.73 µs | 145.66 µs |

Adding a two-way FP32 workspace split to the remaining fused dW reduced these
four candidate times further to **73.96, 74.45, 120.96 and 121.76 µs**,
respectively, with unchanged reported numerical errors. These additional
results are in `autotune_reports/mlp/api_ragged_split/`.

The first SiLU case was also measured in independent initial workers:
173.18 → 98.68 µs. Numerical results were unchanged within each A/B pair;
the largest relative-L2 discrepancy from the independent quantized reference
across the four confirmation contexts was 0.00356%, below the unchanged 0.3%
hybrid implementation gate. No non-input forward storage was retained.

The baseline Nsight Systems capture attributes 152.2 µs of 171.2 µs summed
kernel time to four fused backward GEMMs. This is primarily device work, not
Python dispatch. dX does not reduce over ragged M, so it can use existing
aligned-reduction materialized quantization without padding a BF16 transpose.
dW still reduces over M and retains its fused tail-capable path.

These are not BF16 wins: the previous 129-token BF16 result was approximately
16.4 µs. Improving masked transposed quantization for dW is still useful work;
it is not implemented by this schedule change. No DRAM/tensor saturation claim
is inferred from the Systems trace.

A focused Nsight Compute capture of the remaining dW kernel found only four
CTAs (128 threads each) on 70 SMs, 125 registers/thread, no local-load/store
spill sectors, and 1.86% device-average tensor-pipe activity (32.49% maximum
over SMs). This supports testing more independent split work, rather than
assuming a bandwidth or register-spill bottleneck. The profiler's 44.0 µs
duration is diagnostic and is not substituted for ordinary replay timing.
The raw capture is `api_ragged_dw_counters.ncu-rep`.

The same backward schedules passed native MXFP8 numerical checks on both
shapes/activations, with total times of 128–129 µs for the first shape and
231–233 µs for the second. Native forward/recompute still costs more than
hybrid on these small examples. Separately, the public hybrid API reproduced
the prior long-shape result at 415.64 µs for `8192,256,768,512`.

An additional cooperative quantize-once forward experiment hung on
`129,128,256,128`. Its isolated worker was terminated and the GPU returned to
idle. Partial output tiles are now rejected in the public runtime, kernel
constructor and tuner before cooperative launch. The exact internal cause
has not been localized; this is a conservative qualification restriction,
not a synchronization fix. Ordinary on-chip/native and hybrid forward paths
do not depend on that candidate. Its negative log is retained.

Artifacts are under `autotune_reports/mlp/api_ragged_confirmation/`, with the
baseline trace in `api_ragged_baseline.nsys-rep`. The profiling environment
audit is `api_nsight_environment.json` in the same report root.

## Short matched decoder training

The driver is `benchmarks/train_mlp_precision.py`: two decoder layers, width
256, hidden width 512, eight 128-token sequences, 300 updates (307,200 tokens).
Attention, QK normalization, embeddings, corpus windows, BF16 AdamW state,
initialization and depth-scaled residual weights are matched across policies.
Only the MLP recipe changes. FP32 loss/norm reductions remain unchanged.

Initial seed-123 study, untuned constructor schedules:

| Precision | Initial held-out loss | Final held-out loss | Median step |
|---|---:|---:|---:|
| BF16 | 5.54347 | 1.60853 | 0.826 ms |
| MXFP8 | 5.54552 | 1.61187 | 1.273 ms |
| BF16-gate hybrid | 5.54493 | 1.61185 | 1.306 ms |

All steps had finite loss and gradient norms. This short trajectory does not
establish convergence parity or a hybrid quality advantage. Timings include
compiled forward/backward and fused AdamW, but no whole-step graph replay;
they must not be compared directly with graph-replayed MLP-only timings.
Peak allocated memory was approximately 98.4/100.7/103.4 MB respectively.
These observations are not an allocator-leak test or a training speedup claim.

After the bounded schedule search, seeds 123/124/125 produced mean final
held-out losses of **1.61729 BF16, 1.61859 MXFP8, and 1.61958 hybrid**. Paired
loss differences versus BF16 ranged from +0.00018 to +0.00323 for MXFP8 and
+0.00055 to +0.00326 for hybrid. Three short runs are not enough to establish
statistical superiority; the hybrid's better local operator error did not
translate into a consistent loss advantage. Corpus-window hashes match across
all nine runs. These results are in `api_training_tuned_v2/`.

Whole-step CUDA graph replay was then tested separately for seed 123, including
AdamW and static input buffers updated with new corpus batches. Capture's
weight/state updates are reset in place before measured training:

| Precision | Graph median step | Final held-out loss |
|---|---:|---:|
| BF16 | 0.406 ms | 1.60853 |
| MXFP8 | 0.456 ms | 1.61308 |
| BF16-gate hybrid | 0.479 ms | 1.61185 |

Replay removes much of the host/launch overhead, but this tiny decoder still
favors BF16 by about 12% over MXFP8 and 18% over hybrid in runtime. Early
loss/gradient-norm records match the uncaptured runs; later trajectories are
not bitwise identical. Do not claim deterministic training parity from this
study. Graph peak allocated memory was approximately 156/161/169 MB; retained
graph workspaces differ from the uncaptured allocation footprint. Raw records
are in `api_training_graph_v1/`.

## Framework qualification

The combined 21-test suite passed on both local Torch 2.12.1+cu132 and
2.13.0+cu132 stacks. It covers multi-shape SiLU/GELU gradients, eager backward,
compiled fullgraph execution, inference and explicit BF16 shape fallback,
parameter-only gradients, registered down-linear opcheck, bounded shape maps,
conversion/configuration contracts, and documentation. A repeated 30-step
hybrid check showed no live-allocation growth beyond the 4 KiB test allowance
after warmup. This is not an all-shapes or extreme-value release matrix.
Torch 2.12 aggressive epilogue fusion has a separately documented numerical
caveat in [the prior compiler comparison](mxfp8_mlp_quality_runtime.md); these
API tests use default compile options, not that aggressive recipe.
An additional eager/compiled smoke test confirmed that the known unsafe
cooperative ragged shape raises before launch and CUDA remains healthy.

The driver writes flushed per-step JSONL and a final summary. For example:

```bash
python -m benchmarks.train_mlp_precision --precision bf16_gate \
  --steps 300 --output-dir autotune_reports/mlp/my_hybrid_training
python -m rtx.autotune.mlp --precision bf16_gate --shape 1024,256,512,256 \
  --trials 48 --seconds 120 --output-dir autotune_results/my_mlp \
  --result-json autotune_reports/mlp/my_hybrid_schedule.json
```

Pass `--mlp-config` to the training driver to test an explicitly selected
schedule. The driver refuses mismatched shape/activation/precision metadata.
Use `--cuda-graph` for the separate whole-step replay experiment. Model-level
graph capture remains caller-owned; the MLP does not silently capture an
optimizer or impose static input storage on an application.
