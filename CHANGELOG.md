# Changelog

All notable public API and dataset-schema changes are recorded here. This
project follows semantic versioning while it is in active alpha development.

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
