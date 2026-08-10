# Changelog

All notable public API and dataset-schema changes are recorded here. This
project follows semantic versioning while it is in active alpha development.

## 0.12.0

- Add independent, composable autotuning families for NVFP4 dynamic-X/AOT-W
  and fully prequantized inference, including state-aware traffic/resource
  features, calibrated harnesses, promotion, runtime winner lookup, and a
  balanced portable campaign manifest.
- Support native 64-row NVFP4 GEMM tiles, K=64, and predicated ragged fused
  shapes. Generalize shared resource models to use the true four-bit operand
  width and 16-value NVFP4 scale vectors.
- Add selectable exact or power-of-two tensorwise scaling while retaining
  delayed power-of-two scaling as the training default. Validate delayed
  convergence, one-generation recovery, zero/tiny inputs, and current-scale
  TorchAO packing equivalence.
- Harden delayed scale state with non-aliasing double buffers, stream-aware
  rebootstrap, train/load-state resets, and a direct Inductor lowering for the
  stateful forward launch. Resolve the same installed MXFP8 backward winner
  for both public linear frontends and verify bit-identical gradients.
- Add paired eager/compiled end-to-end training and controlled convergence
  tools. Make their timing safe for the backward runner's private CUDA
  streams and include explicit gradient checks and optional profiler output.

## 0.11.0

- Implement native SM120/SM121 NVFP4 forward execution for
  `rtx.NVFP4Linear`: fused BF16 quantization and E2M1 block-scaled MMA for
  training, standalone TorchAO-compatible packing, dynamic-X/prequantized-W
  inference, and fully prequantized inference. Training continues to use the
  MXFP8 backward kernels.
- Make delayed power-of-two tensorwise scaling the module training default.
  Each fused CTA consumes the previous generation's tiny per-CTA amax state,
  prepares scales cooperatively, and emits the next state in the same kernel;
  retain current/JIT scaling as an explicit policy and bootstrap or rebootstrap
  safely on the first call and after shape changes.
- Generalize the fused forward and prequantized GEMM machinery across MXFP8 and
  NVFP4 physical operand widths, scale-vector sizes, MMA atoms, and output
  scaling while preserving the established MXFP8 launcher contracts and
  metadata-only backward transpose layouts.
- Add a fully composable `nvfp4_fused_fwd` autotuning family, delayed-telemetry
  traffic/resource features, promotion and runtime-winner support, a balanced
  16-context campaign, and a paired release benchmark that requires at least
  1.5x the tuned MXFP8 training-forward speed on its default shapes.

## 0.10.0

- Add fused dynamic quantization/MMA MXFP8 backward by compiling the
  forward-class CuTe kernel over row-major or metadata-only logical-transpose
  tensor layouts. dX and dW expose the complete dynamic forward schedule
  independently without materializing E4M3 operands or E8M0 scales in global
  memory.
- Add executable split-K dW with FP32 partial workspaces and serial, tree, or
  persistent-tree FP32 epilogues, an FP32 atomic alternative, and dual-stream
  dX/dW execution.
- Stage logical-transpose TMA operands in CuTe MN-major shared-memory layouts,
  add a native `ldmatrix.m8n8.x4.trans.b16` 128-bit SMEM-to-register path with
  its measured lane/register-to-K mapping, and emit K-major MXFP8 MMA tiles.
  Validate all backward families on SM120 through the production matrix.
- Keep config interning opaque to Dynamo so eager and
  `torch.compile(fullgraph=True)` share the same registered forward/backward
  kernels.
- Preserve the decomposed quantize-plus-GEMM implementation as an explicitly
  selectable family and measured untuned runtime seed; fused families remain
  fully autotunable while cross-CTA quantization reuse is developed. Advance
  the backward kernel revision.
- Use the persistent dual-operand quantizer for each decomposed backward GEMM
  by default after paired races showed gains across representative shapes;
  retain independent quantizer launches as a separate search family.
- Implement launch-level interleaved decomposed backward execution and remove
  unimplemented graph, cluster-reduction, and legacy persistent-GEMM choices
  from the active backward search space instead of spending trials on rejects.
- Add compound 128x256/256x128 TMA reuse-tile basins that lower MXFP8 staging
  and consumer registers atomically, allowing the backward tuner to discover
  true intra-CTA operand reuse without crossing an illegal intermediate.
- Add a persistent four-operand backward quantizer that emits the dX and dW
  E4M3/E8M0 operand pairs in one launch despite their different reduction
  lengths. Make it the runtime seed together with concurrent dX/dW GEMMs after
  paired races improved representative SM120 shapes by roughly 4--14%; retain
  the former per-matmul dual quantizers as an independent search schedule.
- Encode SM120 SFA/SFB atom-layout, complete-quantizer-warpgroup, and wide-CTA
  register constraints before compilation, and benchmark wide-CTA reuse with
  paired confidence intervals and a dedicated register/stage sweep.
- Add executable CTA-cluster operand reuse for fused forward/backward. The
  cluster owner quantizes once and publishes packed native E4M3/E8M0 tiles to
  peer-local SMEM with DSMEM stores; keep it searchable after representative
  5070 Ti measurements showed the synchronization cost loses at M512.
- Extend clustered operand reuse to the oriented cp.async/ldmatrix transport:
  peer CTAs elide the shared BF16 load and quantization while the owner
  publishes only the current native E4M3/E8M0 K slice. Keep the family as an
  explicit autotuning coordinate rather than a default after 5070 Ti races
  showed cluster synchronization losing at both M512 and M8192.
- Specialize dX-only and dW-only autograd requests so each compiles and caches
  only its selected backward matmul and quantizer instead of constructing the
  full two-gradient runner. Cover eager and fullgraph partial-gradient paths in
  the production matrix.
- Add `rtx-autotune verify-winners` for independently confirming and racing the
  latest finalists left by a wall-time-limited campaign. Make verification
  journals idempotently resumable, deduplicate legacy confirmations before
  racing, and let post-hoc summaries supersede shallower anytime winners during
  runtime-cache promotion.
- Add a frozen revision-19 backward calibration manifest. A ten-minute 5070 Ti
  wave produced 264 valid observations across twelve hot/rotating contexts and
  confirmed shape-specific improvements of roughly 4--8% without changing the
  portable default.
- Add 8/16/32-bit quantized GMEM store schedules and a CuTe `CopyG2SOp`
  logical-transpose transport family. Promote the 128-row, four-value,
  32-bit-store register schedule for backward after paired races improved
  M512/M1536/M8192 and wide-N cases by about 6.2/10.6/1.5/2.3 percent while
  remaining neutral at M128. Keep asynchronous transport independently
  autotunable: it beats the old seed in one representative case but loses to
  the matched register schedule.
- Generalize logical-transpose staging to 32/64/128-K CTA tiles. The 128-K
  family retains four E8M0 codes per row in SMEM and emits them with one
  aligned 32-bit store directly into the tensor-core-native `mma128` layout.
  Keep all tile/transport/store combinations searchable; at M512 the 32-K
  promoted schedule remains faster than the wider alternatives.
- Add persistent multi-output scheduling to the materialized MXFP8 GEMM used
  by decomposed backward and both packed inference states. Expose one to eight
  output tiles per CTA with raster, same-A, same-B, and serpentine locality;
  reconstruct exact TMA pipeline phases across unrolled output work and cover
  partial final CTAs without duplicate stores. Wire the family into all
  relevant autotuners and portable cost-model features. Keep one tile per CTA
  as the default: 5070 Ti measurements were 13.83 us versus roughly 25/49 us
  for two/four tiles at a 48-tile shape, while the best 768-tile persistent
  cases remained statistically level to slightly slower than the baseline.
- Add FP32 workspace and FP32 atomic split-K directly to the prequantized GEMM
  so the quantize-once backward seed can parallelize long dW reductions without
  duplicating operand quantization. Compose both paths with quad quantization,
  dual-stream execution, and the existing serial/tree/persistent-tree reducer;
  move dW reduction near the front of autotuning and model its actual split
  grid, launch count, and FP32 workspace. On a 5070 Ti at M=8192,N=K=512,
  four-way workspace/atomic split-K won paired races by about 28.9%; excessive
  splits lost, and every tested N=K=1536 case also lost because its unsplit dW
  grid already filled the GPU. Keep full reduction as the portable default and
  let the shape/device-aware tuner select this conditional family.

## 0.9.0

- Add dispersion-gated adaptive timing: stable screens stop after 3 samples,
  confirmations after 5, and paired races after 7; noisy measurements retain
  the full 5/11/11 budgets and every stopping decision is recorded.
- Add `rtx-autotune audit`, crash-tail/interior-corruption and identity checks,
  and `rtx-autotune install-winners` for atomic promotion of verified,
  device/shape/layout-specific runtime winners.
- Add disjoint `evaluate-pretrained` evaluation, ZIP-native dataset ingestion,
  exact-shape prospective summaries, compile-waste metrics, probabilities of
  beating random, and bootstrap confidence intervals.
- Reserve CuTe's measured 1 KiB GEMM wrapper overhead in static SMEM legality,
  preventing the observed 102,400-byte launch from reaching compilation on a
  101,376-byte RTX limit.
- Add a production matrix covering eager and fullgraph training/inference,
  both packed inference states, M=8192 dW accuracy, multiple streams, variable
  shapes, and bounded runner-cache eviction.

## 0.8.1

- Make random search retry progressively larger candidate pools before
  declaring a finite conditional space exhausted.
- Add coordinate-local search to the learned bandit portfolio and guarantee
  configurable bootstrap pulls for local and learned arms before UCB takes
  over.
- Report prospective coverage bounds, category/regime breakdowns, and matched
  treatment deltas with deterministic bootstrap confidence intervals.

## 0.8.0

- Bound shape/stream-specific MXFP8 forward and backward runner caches with
  configurable LRUs, cache statistics, and a synchronized public
  `rtx.clear_runtime_caches()` release boundary.
- Add a durable local ask/tell deployment which restores observations, issues
  leaseable work, records every response through the ordinary tuning store,
  and resumes against total trial and wall-time budgets.
- Separate architecture, physical device, compiler, environment, calibration,
  and kernel-source identities in new machine snapshots while preserving the
  existing bundle-path `machine_id`.
- Load dataset-promoted fused-forward, prequant-forward, and backward winners
  through the runtime frontends, including opaque first-execution resolution
  under `torch.compile`.
- Reject GEMM candidates against the actual CuTe launch SMEM footprint and add
  an opt-in, append-only exact deterministic-failure ledger isolated by
  prospective treatment and replicate.
- Expand CPU CI to Python 3.11 through 3.13 and the complete test suite, add
  package metadata validation, and provide a manually triggered self-hosted
  SM120 GPU workflow.

## 0.7.0

- Add a backend-neutral conditional search-space schema with dependent
  parameters, normalization, named constraints, mutation, and sampling.
- Add portable staged kernel tasks spanning static, compile, correctness,
  benchmark, and application fidelities, including typed failure provenance.
- Add serializable, resumable asynchronous ask/tell sessions, worker leases,
  out-of-order completion, partial evaluation, and explicit promotion.
- Wrap existing kernel adapters as portable tasks without changing RTX kernel
  integrations, and move trial outcomes out of the legacy MXFP8 module.
- Add per-context residual journals with crash-tail recovery and experimental
  treatment/replicate isolation.
- Add prefix-balanced task rotation, explicit random/random-local/hybrid
  portfolios, 5070 3-hour/6-hour launch profiles, and fixed-budget prospective
  optimizer summaries.
- Contain sticky CUDA device faults at worker-process scope, preserving the
  responsible observation and automatically resuming with the remaining
  campaign deadline instead of logging poisoned-context fallout.
- Stabilize supervised-run compiler identity and libNVVM discovery by restoring
  the preferred CUDA 13.2 toolkit path when a remote shell omits `CUDA_HOME`.

## 0.6.0

- Add CPU-only cross-device pretraining with revision-scoped latency, ranking,
  feasibility, conditional-effect, leave-one-device-out, and exact-SKU
  context-held-out artifacts with content integrity checks.
- Allow pretrained proposal priors in GPU campaigns while retaining local
  adaptation, bandit exploration, validation, and device-specific winners.
- Narrow the supported hardware surface to SM120/SM121 RTX Blackwell; remove
  planned SM110/Jetson Thor profiling and support claims.
- Add a resumable, nonstationary contextual bandit over random, learned-global,
  and model-local search strategies with cost-aware rewards and capped
  cross-context priors.
- Add coverage-first contextual bandit allocation across dataset contexts with
  milestone fairness, durable decision journals, and CSV/Parquet export.
- Add CLI controls for composing both allocation levels and a dedicated
  cross-device hierarchical-bandit manifest.
- Preserve the v2 manifest and sequential/breadth-first behavior as compatible
  control policies.
- Separate reusable bandit policy math from execution and dataset campaign
  lifecycle, and document repository, manifest, and benchmark boundaries.

## 0.5.1

- Prefer official PyTorch CUDA 13.2 wheels in `requirements.txt` and add a
  CUDA 13.0 fallback requirements file.
- Require TorchAO 0.18.0 or newer and install CUTLASS DSL with its `cu13`
  runtime extra.
- Make PyArrow a core dependency and add Apache TVM FFI 0.1.13.post2 and
  Einops to both distribution metadata and requirements.

## 0.5.0

- Replace the project-local packed operand dataclasses with TorchAO's canonical
  `MXTensor` and `NVFP4Tensor` tensor subclasses.
- Accept TorchAO-produced row-major and blocked-scale MXFP8 operands directly,
  including zero-copy views into the CuTe kernel-native scale shapes.
- Preserve the `rtx.MXFP8Tensor` spelling as an alias and additionally export
  TorchAO's canonical `rtx.MXTensor` name.
- Keep packed module state dictionaries as raw qdata/scales and versioned RTX
  layout metadata, independent of tensor-subclass serialization internals.
- Remove the old project-local dataclass constructor signatures; callers which
  construct operands manually should use TorchAO factories and attributes.

## 0.4.0

- Add distinct AOT-weight and fully-packed MXFP8 tuning families.
- Remove inactive operand-quantizer coordinates from both inference spaces.
- Add calibrated hot/rotating inference harnesses which exclude AOT packing
  from authoritative timing.
- Persist promoted winners in a device, shape, state, regime, and physical
  layout-specific runtime cache.
- Add a bounded cross-device inference pilot manifest.

## 0.3.0

- Rename the distribution and public package identity to `rtx`.
- Add the public `rtx.MXFP8Linear` and `rtx.NVFP4Linear` frontends.
- Add versioned `MXFP8Tensor` and `NVFP4Tensor` packed-operand contracts.
- Support dynamic BF16/BF16, dynamic-X/prequantized-weight, and fully
  prequantized MXFP8 forward states.
- Keep NVFP4 execution explicitly unavailable until its RTX kernel exists,
  while registering its forward, fake, context, and MXFP8-backward boundary.
- Make frontend and autotuner imports architecture neutral; SM120 CuTe modules
  are selected lazily at kernel compilation.
- Preserve v1/v2 autotuning family names and resume identities.
