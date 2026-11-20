# Iteration 3: packed-router opening qualification

The packed tuner moves a legal staged-sharded router already in the opening
to trial 2 when multiple pools are nonempty and the longest scheduled pool
does not fit either persistent path. The initial plan still runs first; the
same control, persistent and local proposals remain available. Packs whose
pools all fit persistent or contain only one nonempty pool, initial plans
already covering this family, and explicit client priors or reservations
retain their original admission order. No hardware/shape winner table,
affinity gate, routing math or hot-path boundary read was added.

[The packed tuner](../rtx/autotune/varlen_routing.py#L223) moves an existing
legal structural coverage proposal, not merely a seed in the pool. The trial
and soft deadline accounting are unchanged. Packed winner/residual identities
include opening revision 1, and the progressive checkpoint identifies the
actual ordered prefix. Kernel semantic revision 7 is unchanged so portable
prior compatibility is not invalidated by a search-order change.

## Matched replay method

- HOME GPU0, RTX 5090/170 SM, Torch 2.13.0+cu132. Every visit used a fresh
  supervised process holding the shared GPU0 lock. Baseline and candidate
  runtime snapshots differ only in `rtx/autotune/varlen_routing.py`:
  `ebec8ac46b0ad3b10902194ef56ce698988b2d5e19e3a1b01dd06ce766fec58b` →
  `b00e6737a3cba9603723bc3d4a185d7f9fd78598a765f7f2e7fc41a278506971`.
- Freeze expert, shared and affinity configurations. The original fixture
  reuses its previously qualified row-broadcast affinity; new fixtures use
  scalar affinity. Inputs, parameters, upstreams, hard IDs and inverse indices
  are checked across arms. Each arm starts with four fresh progressive mixed
  trials; a second process continues that exact journal to eight cumulative
  trials. Both paths use fullgraph/static Inductor max-autotune and warmed
  outer CUDA graph replay. Medians pool two rounds of seven ordinary CUDA-event
  batch samples; compilation and tuning are excluded.
- Checkpointed BF16 is the *same selected router* with BF16 FFNs, not the
  separate compacted BF16 reference. Faster routing benefits both paths and
  does not establish a native FFN improvement. Results are fixed-buffer,
  synthetic graph replay, not training convergence or general latency bounds.
- The E24 balanced holdout inherits its single-pool expert plan, whose TMA
  store requires full M tiles. The initial visit stopped on a host-side
  legality check at logical C=736. The timed visits in **both** arms round
  expert storage to C=768 (`capacity_multiple=128`); no precision, router
  capacity or FFN algorithm changed. The failed setup log is retained.

## Results

CUDA-event forward plus backward latency in ms. An arrow compares old and
new opening orders at the same cumulative trial count.

| Pack | Native, 4 trials | Router, 4 trials | Matched BF16, 4 trials | Native, 8 trials |
|---|---:|---:|---:|---:|
| Original uneven, T8192/E32/K4: 127+513+2001+767+4095+689 | 2.089 → 1.176 | 1.437 → .635 | 2.042 → 1.224 | 1.177 → 1.177 |
| Held-out skew, T8192/E32/K4: 33+95+257+769+1703+5335 | 2.557 → 1.431 | 1.809 → .832 | 2.471 → 1.478 | 1.431 → 1.431 |
| Held-out balanced, T2048/E24/K6: 8×256 | .658 → .658 | .379 → .379 | .665 → .664 | .658 → .658 |

On both skewed packs the same staged warp-row, 32-column-shard,
native-warp QB plan moved from admission 5 to 2. Radix cache capacity is 16
items/thread on the original and 32 on the held-out skew; cap selection is
derived from the longest scheduled pool. At four admissions the change
reduced whole native time by roughly 44% on both packs. The old opening
found the same plan at eight admissions; this is a first-hit coverage gain.
The balanced profile already reached the staged-sharded plan at trial 4
(cache 4). Moving it to trial 2 did not move steady-state latency there.

All 12 timed visits passed. The six fresh arms each admitted eight unique
trials; all 48 passed the existing router gates, including the independent
segmented FP64 route-weight and score-adjoint checks on caller and synthetic
scores. Both arms preserved their four-record prefixes at cumulative eight.
Hard IDs/inverse indices matched exactly; all public outputs and gradients
were finite. The independent quantized FP64 expert oracle (output, dX, both
weight gradients, route-weight adjoint) had worst NRMS **.01350%** and worst
absolute error .000977. The full output/gradient comparison to the old
four-trial result had worst NRMS **.16716%** and worst absolute error .001832;
NRMS divides the error norm by reference norm with floor `1e-12`. MXFP8 replay
had zero measured absolute error in outputs and gradients; checkpointed BF16
replay worst NRMS was .08723%. Live allocation grew by zero bytes over five warmed
iterations per visit. These checks do not validate router-gradient training
behavior against BF16 or imply a long-run allocator guarantee.

## Reproduction

[Runner](../benchmarks/qualify_varlen_router_opening.py) and
[focused tests](../tests/test_varlen_router_opening.py). CPU: 23 relevant
tests passed (91 deselected), including a simulated two-admission deadline,
2→4→8→16 exact progressive order versus one-shot 16, explicit prior and
reservation, rejection and deduplication, and short/single-pool controls.

```sh
env CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q \
  tests/test_varlen_router_opening.py tests/test_routing_varlen.py \
  tests/test_routing_persistent_pool.py -k 'opening or cold_hybrid or packed_search_times'

.venv/bin/python -m benchmarks.qualify_varlen_router_opening \
  --baseline /tmp/rtx-moe-iteration3-20261119/baseline \
  --candidate /tmp/rtx-moe-iter3-router-20261119/candidate \
  --evidence autotune_reports/balanced_moe_challenge_20261119 \
  --output /tmp/rtx-moe-iter3-router-20261119/qualification-repro
```

The skewed results and the failed initial balanced setup are under
`/tmp/rtx-moe-iter3-router-20261119/qualification/`; the final balanced visits
are under `qualification-balanced/` in the same artifact root. These contain
per-profile summaries, exact commands, journals, graph validation and
baseline operand/gradient references. The frozen baseline source is also
archived with the [45-context study](balanced_moe_challenge_20261119.md).
