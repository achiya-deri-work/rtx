# Blackwell prior v1 — research only

`blackwell_prior_v1.json.gz` is a JSON-only frozen prior trained from the three
completed combined campaigns (5070 Ti, laptop 5070, 5090). It contains schedules
and small models, **not** application weights or raw kernel tensors.

- Schema/feature revision: 1/1.
- Training dataset digest: `fb31cd809ef5789016fca395d138c36d46d0b2f749ea78afa2ad6dd521508f9a`.
- 62,446 raw observations; 1,090 explicitly unstable successful timings excluded.
- 24 semantic/cache-regime bundles: 24 latency rankers, 24 preparation-cost
  proxies, and four feasibility models. Other bundles have insufficient negative
  evidence for a feasibility model; they do not invent failure probabilities.
- Repeated successful context/config observations are median-aggregated.
- Latency targets are context-centered; measured compiler resource counters and
  provenance are not model inputs. Adapter-derived hardware geometry is retained.
- Numerical-policy, family, cache-regime and kernel-revision mismatches cannot
  select a bundle. Unknown contracts fall back to caller ordering.
- Learned scores never hard-reject a configuration. Existing adapter legality
  checks and numerical gates remain authoritative.
- Feasibility scores are uncalibrated soft hints. Preparation cost includes
  setup/cache/lease effects and must not be described as pure compilation time.

The artifact explicitly has `deployment_enabled=false`. Load it with
`FrozenPrior.load(path, research=True)` only for research. It does not install
runtime winners or change `rtx.MXFP8Linear`/`rtx.NVFP4Linear` defaults.

The default hybrid cycles through two retrieval, four ranked, and two diversity
slots. Slot counts and bounded feasibility/preparation penalties are configurable
through `FrozenPrior.order` and the prospective plan's `selector` object.

Device-held-out replay is documented in
[the prior study](../docs/portable_prior_study.md). It is development evidence,
not certification: observed pools are incomplete/adaptive, compiler caches were
shared, and policies were selected after inspecting these data. The next study
uses frozen weights with new observations and shape candidates.

Rebuild from raw datasets with:

```bash
python -m rtx.autotune.prior <5070-ti-dir> <laptop.zip> <5090.zip> \
  --output autotune_priors/blackwell_prior_v1.json.gz
```

Do not rebuild or edit an artifact used by an active study. Plans pin its file
SHA-256 and reject changes. Preserve each version for reproducibility.
