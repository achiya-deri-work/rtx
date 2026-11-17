# BalancedRouter versus compiled Torch fixed64

Same-device comparison on an AC-powered RTX 5070 Laptop, Torch 2.13.0+cu132,
CuTe 4.7.0. These are **router-only** times: no expert FFNs, optimizer or
communication. The GPU was idle before the study. No 5090 timings are mixed in.

## Results

The confirmation uses four alternating A/B rounds, 31 samples per window and
2,000 ms requested calibrated warmup. Entries are medians of the four window
medians. Compilation, tuning, correctness checks and capture are excluded.

| Tokens | Phase | Torch fixed64 | Tuned CuTe | Observed speedup | Strict stability/promotion gate |
|---|---|---:|---:|---:|---|
| 1024 | Forward, no-grad | 38.175 ms | 0.885 ms | 43.12× | Inconclusive |
| 1024 | Forward + backward | 39.481 ms | 1.104 ms | 35.76× | Inconclusive |
| 4096 | Forward, no-grad | 43.372 ms | 2.397 ms | 18.09× | Inconclusive |
| 4096 | Forward + backward | 46.436 ms | 3.438 ms | 13.51× | Passed |

All cases use D=768, E=128, K=8 and router dimension 64. Both implementations
use `torch.compile(fullgraph=True, dynamic=False)` with Inductor's internal
cudagraphs disabled, followed by explicit whole-call CUDA graph replay. This
does not penalize Torch with Python per-round launch overhead during timing.
Forward means no-grad inference; backward differentiates a shared weighted
route-weight loss with respect to the input and every router parameter.

**Interpret the unqualified ratios approximately.** Several laptop timing
windows failed the unchanged 3% split-drift / 10% sample-range gates; no retries
or threshold relaxation converted these into qualified wins. The large
directional difference is nevertheless present throughout the raw samples:
`min(all Torch samples) / max(all CuTe samples)` is respectively 36.75×,
31.78×, 15.60× and 13.05× in table order. Those are empirical extrema, not
confidence intervals or cross-device guarantees.

The first run separately qualified the 4096-token training comparison at
44.205 → 3.257 ms, 13.57×. Its other phases had unstable windows. That run had
only a train-context winner, so no-grad forward used the portable cache-miss
seed. The confirmation fixes this benchmark coverage issue by independently
tuning forward and training; it does not modify production kernel math.

## Matching and correctness

The baseline is BiMoE's unchanged standalone `SinkhornRouter`, loaded directly
from its portable handoff, with `scheduler_backend="dual"`,
`repair_execution="fixed"`, Sinkhorn5, QB5 and repair64. Its source SHA256 is
`f4872cd1f3df5b320bedb526e666415c8b59130560b080d84195ac43d35fada7`.
The CUDA reference predicate executes all 64 repair rounds. CuTe uses the same
64-round ceiling but stops the device-side conditional loop when balanced.

Both models receive identical BF16 input and parameter values, with FP32
routing math and identical scalar FP32 parameters. Seed is 1234; projection
weights are initialized with standard deviation `D**-0.5`, matching the older
BiMoE performance study's nonuniform-affinity recipe. TF32 is disabled for
both. The shared balanced tuner has at most 16 unique trials per context and
a soft 30-second allowance. Confirmation resumes the original training
history and adds separate inference contexts; no winner is manually chosen.

Before timing, both shapes pass exact expert-major token-ID and inverse-index
equality. Route weights agree within `rtol=5e-5, atol=5e-6`; the largest
observed relative L2 error is 1.23e-7. Every input/parameter gradient is finite
and passes `rtol=0.015, atol=0.001`; the largest observed gradient relative L2
error is 8.57e-5. This is synthetic tensor/gradient qualification, not training
convergence evidence or proof that every input repairs in a few rounds.

## Reproduction and evidence

```bash
PYTHONPATH=. python benchmarks/compare_balanced_router_torch.py \
  --baseline-source /path/to/BiMoE/routing-research-port-20261111/bidirectional_mixture_of_experts/router.py \
  --samples 31 --warmup-ms 2000 \
  --output autotune_reports/balanced_router_vs_torch/results.json
```

Raw per-sample timings, eligibility decisions, configs, errors and source
hashes are in the existing storage4-backed folder:
`autotune_reports/balanced_router_cute_20261115/5070/`.
Files are `torch-fixed64-comparison.{json,log}` (initial),
`torch-fixed64-confirmation.{json,log}`, `fixed64-environment.json`, the exact
baseline and driver Python sources, and `fixed64-tuning/` residuals/winners.
No hardware counters were needed for this latency comparison; no bandwidth,
occupancy or whole-MoE performance claim is made.
