# MXFP8 MoE: useful eight-trial first-hit tuning

The measured output-store fixes are now ordinary autotuner proposals, not
frozen winners. On two fresh 512-wide held-outs, the final eight-trial search
matched the best sixteen-trial shared MLP result. Expert regret was 0% and
0.86%. These are finite-budget comparisons on one RTX 5090, not a guarantee
of the global optimum or cross-device qualification.

## Implementation

- `rtx/autotune/output_transport.py` generates coupled BF16 output-store,
  tile and buffering proposals from existing template axes. It tries inherited
  geometry first, then legal full-output divisors. The 64-row, single-buffer,
  consumer-scale quarantine remains enforced. Quantizers and register budgets
  are inherited. No logical shape is silently padded.
- Routed experts compose eligible down, gate-dX and down-dX stores using
  **physical expert capacity**, not total token count. Forward on-chip aliases
  are excluded. A legal native128 initial configuration can supply a store
  anchor even when indexed/fused backward preparation is incompatible.
- The shared MLP compares forward-only stores, all eligible row stores, and
  all stores crossed with a second existing dW partition plan. Only partition
  fields/reduction kind transfer from that donor: quantizers, GEMM geometry,
  registers and non-dW legs remain the anchor's. All accumulation policies
  remain FP32; partition changes may change rounding order within the existing
  numerical gates. FP32 dW workspace stores are never converted to BF16 TMA.
- For ordinary budgets of at least eight trials, the opening contains the
  incumbent, up to six distinct complete structural proposals, and one
  scheduler-selected trial. The existing shared UCB/local/random machinery
  continues afterward. Direct controls, explicit reservation settings and
  compatible research-prior openings remain supported.
- Progressive 8-to-16 continuation preserves the prefix without duplicate
  evaluations. Legacy refinement does not invent a strict checkpoint.
  MLP tuning/winners use revision **46**, structural seeds **14**, prepared
  opening **3**. The expert portfolio is revision **20**, with the shared
  output-transport helper included in its source fingerprint.

No new kernel, eager hot-path operation, private CUDA graph, allocation API or
user-managed stream was introduced. The mechanism is the existing TMA store
implementation identified in the [measured bottleneck study](mxfp8_moe_measured_bottlenecks_20261119.md).

## Why eight-versus-sixteen was not enough

An initial four-workload pilot had at most 1.55% expert and 0.51% shared-MLP
regret against sixteen trials. However, both budgets missed the same crossed
configuration: TMA stores with the other existing dW plan.

A separate seeded dense fixture measured 264.9 versus 222.1 microseconds for
that combination, passing the independent quantized reference. The final
opening therefore compares this missing cell directly. On the original public
module fixture, actual eight-trial shared-MLP replay improved from about
268.7 to **225.7 microseconds**. Whole-layer replay measured 3.542 versus
**3.494 ms**; this smaller whole-layer delta must not be represented as the
standalone MLP percentage. These are separate-process measurements, not a
new paired whole-layer causal estimate.

## Final frozen-source qualification

Torch 2.13.0+cu132, RTX 5090/170 SMs, GPU0 only. GPU1 and its unrelated workload
were untouched. Source digest:
`92b27da32d1f05696b09e4ecdfe8cdd8511fdfeffaa63cf2ab2a07a5aae2bb82`.

The final two held-outs were selected after freezing the strengthened policy;
their timings were not used to construct it. Both use D=512, E=64, K=8,
expert H=128 and shared H=384. Their expert capacity is 2240. The development
point uses D=384, E=32, K=4, expert H=128, shared H=256 and capacity 1600.

| Workload / independent sequence lengths | Expert best 8 / 16 (ms) | Shared best 8 / 16 (ms) | Maximum component regret |
|---|---:|---:|---:|
| Development: 5000 + 7480 | 0.6028 / 0.6028 | 0.2215 / 0.2215 | 0% |
| Fresh aligned: 7136 + 10528 | 2.0518 / 2.0518 | 0.3932 / 0.3932 | 0% |
| Fresh ragged: 7000 + 10568 | 2.0501 / 2.0327 | 0.5381 / 0.5381 | 0.86% |

These are journal-prefix best eligible CUDA-graph measurements, not seed-pool
membership. Every family ran eight actual trials and then eight additional
trials, with 16 unique configuration IDs. The six searches produced **96
successful search observations**, with no compile/runtime/correctness failures.
Eight-trial search calls took 20.9–24.3 seconds per component; the configured
admission cap was 180 seconds. Reference setup, compilation outside search,
and finalist qualification are additional first-call wall time. A shorter
user time cap can admit fewer trials.

Standalone best-component times above must not be summed. Independently
measured full-module replay after eight trials was:

| Workload | MXFP8 fwd+bwd | Checkpointed BF16 fwd+bwd | BF16 / MXFP8 |
|---|---:|---:|---:|
| Development | 3.494 ms | 3.751 ms | 1.074x |
| Fresh aligned | 6.415 ms | 7.647 ms | 1.192x |
| Fresh ragged | 6.476 ms | 7.497 ms | 1.158x |

Both use the same router and actual parameter/input backward, user-side
`torch.compile(fullgraph=True, dynamic=False)` and ordinary outer CUDA-graph
replay. Inductor max-autotune was **off** in this bounded qualification;
optimizer work is excluded. Two rotating phase visits each use seven timing
samples. This does not establish a 1.2x win over aggressively autotuned BF16
on all shapes. In particular, the ragged shared MLP remains slower than its
standalone BF16 comparator; its direct-store fallback is numerically correct,
not a claim that the remaining performance problem is solved.

## Numerical, runtime and orchestration checks

- Every trial retains the existing independent reference and replay gates.
  Selected expert kernels additionally match the independent FP64 quantized
  oracle on output, input gradient, both weight gradients and route gradient.
  Maximum selected-oracle NRMS over final cases is **0.000115**. This measures
  implementation agreement with MXFP8 semantics, not error versus BF16.
- Whole compiled replay overwrites fresh gradients and checks all leaves for
  finiteness. Five warmed allocation samples are identical in every final
  result: live growth is zero.
- Focused CPU tests cover compound legality, ragged fallback, native128,
  inactive on-chip aliases, both winning and losing TMA outcomes, actual
  eight-trial coverage, cold on-chip shared selection, forward-only selection,
  donor-policy preservation, explicit reservations and exact progressive
  prefixes. The persisted final CPU run passed **81 tests**, with one
  CUDA-dependent test skipped in that sandboxed run.
- The dedicated public GPU test passed separately: fullgraph first-hit
  selection, warm cache reuse and progressive refinement, two packed shapes,
  empty/ragged sequence boundaries and noncontiguous upstream gradients.
  It completed in 93.5 seconds. No timing job remains running on GPU0.

## Reproduction and artifacts

Use the ordinary public `AutotunePolicy` controls; there is no new tuning API.
For this qualification, from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_COMPILE_THREADS=4 \
python -m benchmarks.qualify_moe_store_prior \
  --output autotune_reports/my_eight_trial_check \
  --cases two_pool new_aligned new_ragged --budgets 8 16

python -m benchmarks.analyze_moe_store_prior \
  autotune_reports/my_eight_trial_check \
  --output autotune_reports/my_eight_trial_check/summary.json
```

Raw residuals, sidecar configurations, numerical/memory reports, logs,
summaries and source archives are retained under
`autotune_reports/moe_eight_trial_prior_20261119/` on the existing Storage4
artifact symlink. The four-workload pilot is kept separately from final r46
evidence. The optional `--opening-policy without_transport` driver mode
disables only compound-store proposals for future ablations; it is not a
claim that this turn reproduced the entire old autotuner.
