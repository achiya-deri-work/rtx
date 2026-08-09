# Changelog

All notable public API and dataset-schema changes are recorded here. This
project follows semantic versioning while it is in active alpha development.

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
- Add 8/16/32-bit quantized GMEM store schedules and a CuTe `CopyG2SOp`
  logical-transpose transport family. Promote the 128-row, four-value,
  32-bit-store register schedule for backward after paired races improved
  M512/M1536/M8192 and wide-N cases by about 6.2/10.6/1.5/2.3 percent while
  remaining neutral at M128. Keep asynchronous transport independently
  autotunable: it beats the old seed in one representative case but loses to
  the matched register schedule.

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
