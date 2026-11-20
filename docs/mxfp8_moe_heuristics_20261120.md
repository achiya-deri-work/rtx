# Two-HOME-GPU MoE heuristic and data-movement study

## Contract and status

The user authorized both HOME RTX 5090s for deeper automatic search and
controlled hand-designed schedule panels. The 31 large-shape contexts completed on GPU 0
(`09db6472-cc08-c4cb-90db-560de232e5aa`) and GPU 1
(`5d9b8137-a267-5c4d-6270-ccda65eedbb0`), each with its own exclusive lock.
The tiny control was repeated on separately versioned fixed source,
as explained below. Follow-up prospective and equal-token controls are tracked
separately from the completed main matrix.
These are two boards of the **same SKU**, not cross-device-family evidence.
No remote machine, unrelated project, driver or clock/power setting is changed.

Frozen production source:
`1fe29c84cb28bf0f5e654097424d1b6eb030e07aa93126d2b0645e9880d09ab2`.
Torch 2.13.0+cu132, CuTe 4.7.0, 170 SMs per GPU. Sources and the read-only
environment preflight are at `/tmp/rtx-moe-heuristics-20261120/`.

The [study driver](../benchmarks/study_moe_heuristics.py) covers 24 discovery
and eight held-out contexts. Every context requests **32 expert, 32 shared
MLP and 32 router trials**; affinity selection is a separate small comparison.
Actual distinct/eligible counts come from residual journals, not requested
caps. Setup/search/compilation are excluded from steady-state timings.
Each completed case retains its selected configs, numerical checks, ordinary
three-visit/seven-sample timing, all-leaf graph replay and flat live-memory
checks. Per-context results are atomic; residual JSONL and progress are fsynced.

Discovery covers D384/512, K4/6/8, E16/32/64/128, packed versus true flat,
B8/16/32 and primarily L1024. Held-outs add E24/48/56/96, new lengths and
hidden widths, uneven independent sequence pools and a small forward control.
All hidden widths are 64-aligned and active hidden `S+K*H` stays inside the
hard ±7.5% interval around `4D`. Shared remains one ordinary dense SwiGLU.

The initial two pilots are adopted only after exact source, workload,
initialization mode, measurement settings and completed-result checks. One
legacy-initialized pilot records `stable_experts=false`; its own MX/BF16
comparison is matched, but it is not presented as an initialization-matched
cross-shape A/B against the independently initialized contexts.

## What “routing movement” actually means

There are three different costs; they must not be conflated:

1. **Router math:** projection/affinities, saved-state Sinkhorn, nondifferentiable
   QB/repair, and differentiable selected mixture weights. This works on score
   matrices and routing metadata, not copies of `[T,D]` token vectors.
2. **Metadata:** IDs, inverse maps, stable grouping, padding/counts and compact
   slot maps. These are real small-array writes/scans, not merely Python views.
3. **Expert dispatch/preparation/combine:** packed operand gathering,
   independently oriented quantization, and reading/reducing expert outputs.
   These move full feature vectors and are part of expert execution, not the
   standalone router timing.

The selected initial pilots use `input_preparation='indexed_transpose'` and
`dy_preparation='fused'`. They do **not** make a full expert-slot BF16 copy of
X or a separate weighted BF16 dY buffer:

- X is quantized once per token. A gather kernel physically writes packed FP8
  payloads and E8M0 scales for expert consumption; K-way dispatch is not a
  zero-traffic logical view.
- Backward stores slot→token indices and reads original BF16 X through them
  to quantize the required reduction orientation. Logical transposed loading
  avoids a standalone BF16 transpose, but producing independently scaled FP8
  operands is still real work and traffic.
- Weighted dY gathering and row/transposed quantization are fused. Both
  quantized orientations still exist, because their scale groups differ.
- Forward expert outputs and per-expert dX contributions are materialized,
  then combined/reduced through inverse metadata with FP32 accumulation.
  Compact forward may use smaller physical rows; backward preserves original
  leading-32 scale membership rather than silently compacting its K groups.
- Shared MLP sees all tokens directly; sequence packing requires no shared
  token permutation. Parameter/shape views are logical aliases, not `cat`
  or transpose copies hidden in a custom-op launcher.

The older BF16-gather and materialized-dY families remain search choices;
their movement must be counted when they win. Absence of `cudaMemcpy` calls
is **not** evidence of zero movement: most movement here is done by kernels.

There are also explicit score/metadata copies in the router itself.
[`PackedHardPlan.launch`](../rtx/_routing_varlen.py) copies probabilities,
scheduled boundaries and sequence/local maps into fixed-address conditional
graph state and copies its status back; the flat
[`HardRoutingPlan.launch_into`](../rtx/_routing_hard.py) copies scores in and
assignments out. Packed transport materializes padded scores and saves FP32
normalization history. These are not token-feature permutations, but must not
be described as zero-cost views. For scale, unpadded Sinkhorn history alone is
`2 * iterations * T * E * 4` bytes (80 MiB for T16384/E128/five iterations),
before scheduled-row padding and other arrays. That is a storage calculation,
not measured transfer traffic or an attribution of latency.

## Initial current-source measurements

These completed 32-trial pilots are provisional examples, not the final
distribution or evidence that a heuristic transfers:

| Packed B16×1024; D/E/K/H/S | Whole MX F+B | Compact checkpoint BF16 | Standalone router F+B |
|---|---:|---:|---:|
| 384/128/6/192/320 | 3.709 ms | 3.691 ms | 1.571 ms |
| 512/64/8/192/512 | 3.900 ms | 4.250 ms | 1.004 ms |

Standalone phases have different compilation/autograd boundaries and overlap
in scope. Their times cannot be summed or treated as exact in-layer shares.
Both whole precisions use the same selected router. No comparison to the
previous run is called a causal speedup if router or initialization changed.

### Completed 31-context matrix

| Layout | Contexts | Median MX / compact-checkpoint BF16 speedup | Range |
|---|---:|---:|---:|
| Independent packed pools | 24 | 1.059× | 0.929–1.167× |
| One global flat pool | 7 | 1.054× | 0.991–1.125× |

Speedup is BF16 time divided by MX time. **This is not a universal 1.2× win.**
All 31 completed contexts passed their independent quantized FP64 expert
oracles, optimizer-visible gradient checks, compiled graph replay and zero
steady live-allocation growth. Compiled checkpointed compact BF16 is the
primary comparison; saved-intermediate BF16 is exported separately.

The independent-pool router F+B diagnostic was 20–45% of whole MX F+B time;
the flat diagnostic was 41–63%. These are ratios of separately compiled
scopes, **not an additive in-layer attribution**.

### Detailed routing cost

Three-visit/seven-sample ordinary graph timings, B16×1024, in milliseconds:

| D/E/K/H/S | Whole MX F+B | Router F+B | Sinkhorn F | Sinkhorn B | QB/repair F | Metadata F+B |
|---|---:|---:|---:|---:|---:|---:|
| 384/16/4/320/256 | 2.158 | 0.644 | 0.047 | 0.033 | 0.445 | 0.014 |
| 384/128/4/320/256 | 3.263 | 1.480 | 0.104 | 0.108 | 1.084 | 0.043 |
| 384/128/6/192/320 | 3.715 | 1.577 | 0.112 | 0.109 | 1.143 | 0.064 |

The hard-assignment scope includes selection/repair kernels and conditional
graph execution; do not call all of it GEMM time or memcpy bandwidth time.
Router source-labelled host traces found five explicit D2D copies, about
5.4 µs total for the E16 anchor and 7.7 µs for E128/K6. Those traces include
host gaps and instrumentation, so these copy durations are diagnostic, not
an exact percentage of ordinary graph latency. A separate node-level outer
graph trace is retained to distinguish graph execution from host overhead.

That node-level **outer-graph** trace is now complete for E128/K4: 221 kernel
activities, 23 D2D copies moving 12.29 MB in 16.8 µs total. The hard plan ended
after five repair rounds with zero load error. Eleven top-k kernels consumed
~514 µs of diagnostic kernel time, with ~102 µs of sorting and ~137 µs of
quota kernels. This identifies selection/repair as a much larger target than
removing the explicit copies. Node tracing perturbs execution, so these are
not uninstrumented per-operation latency guarantees or total HBM-byte counts.

On the E128/K4 expert-only F+B trace, the ~1.504 ms sum of kernel durations
was approximately 34.4% GEMM, 31.5% fused recompute/activation/route-gradient,
20.5% quantization/fused operand preparation, 9.4% dispatch/combine,
2.7% metadata/reduction and 1.4% framework copy. These categories are a
diagnostic kernel classification, not a roofline or independent whole-layer
speedup prediction. In particular, fused recomputation also moves data.

Selected NCU kernels show real traffic, not aliases:

| Operation | NCU duration | DRAM read/write | L2 bytes | Active warps | Local load/store sectors |
|---|---:|---:|---:|---:|---:|
| Compact FP8 gather | 16.8 µs | 2.06 / 19.03 MB | 63.33 MB | 79.6% | 0 / 0 |
| Compact combine | 52.5 µs | 7.67 / 24.35 MB | 89.20 MB | 92.6% | 0 / 0 |
| Backward packed input gather | 24.6 µs | 7.55 / 24.37 MB | 59.60 MB | 77.3% | 0 / 0 |
| Fused dY gather/quantization | 57.1 µs | 25.49 / 54.86 MB | 189.20 MB | 76.1% | 0 / 0 |
| Indexed BF16→FP8 transpose quantization | 37.3 µs | 13.21 / 28.87 MB | 104.63 MB | 85.3% | 0 / 0 |

NCU uses replay with unchanged caches/clocks. These selected-kernel counters
are not total model traffic and the ratio of DRAM to L2 bytes is not an L2 hit
rate. Ordinary graph timing remains the latency evidence.

## From data to initial heuristics

The [analysis](../benchmarks/analyze_moe_heuristics.py) reuses the existing
`rtx.autotune` observation loader, timing eligibility and portable features.
It exports actual 8/16/32 prefix regret, failure/preparation cost, strict
one-serialized-field matched effects, and interpretable tree/ensemble models.
Cross-validation groups by semantic component geometry across boards and
replicates, not hardware-specific context IDs; random row splits are forbidden.
Held-out geometry overlaps are labelled explicitly when different MoE models
share the same shared-MLP or router component shape.

Ranking tests protect the incumbent, count failed/unstable proposals against
the budget and compare seven model choices plus the incumbent with the actual
eight-trial prefix and random selection. Logged-pool ranking is **not** a
prospective first-hit search or proof of global optimality. Models/rules remain
research-only until a fresh held-out search and matched whole-layer check pass.

One source-proved search cleanup is implemented independently of empirical
ranking: `gather_mode=flat/warp` is canonicalized for prepared FP8 input
gathering, where both values emit identical raw forward/backward launch plans.
The actual BF16-gather and on-chip forward paths retain the axis. This removes
duplicate candidate programs without changing any kernel or quantization math;
CPU tests compare the raw planner, not two calls through normalization.
The main study used the immutable pre-cleanup source above, so its budget
evidence is not silently mixed with a changed search policy. GPU 0's fresh
prior comparison uses alias-cleaned runtime `d67ea25337ba993de20276ca99d2640d33ec9142f6712a8b3e875f088a8fe9ec`;
GPU 1's recovery comparisons use metadata-fixed runtime
`b72064211b28ebcd80949250687b1cfe3f6d7a3575220009ab3455028c6c79b4`.
Source identity is recorded per comparison rather than pooling these as one
unchanged-runtime dataset.

### Evidence for initial search priorities

The completed prefixes strongly distinguish components. Across all 31 large
contexts, actual eight-trial expert regret against the best observed 32-trial
result had median 0.17%, worst 3.74%; packed-router regret
had median 0.21%, worst 1.50%; shared-MLP regret had median 0.17%, worst 0.76%.
The seven flat routers were very different: median 76.5%, worst 81.6%.
These are logged-prefix regrets, not a guarantee about every future shape or
an exhaustive optimum. The final export includes all 31 large contexts.

Strict one-field matched comparisons provide useful **soft priorities**:

- Packed `native_warp` QB was about 16% faster than `native_column` at the
  median across 24 compared contexts, and about 20% faster than `native`.
- Cached `expert_keys` columns generally beat `row_major` (median latter /
  former about 1.073), but column cache capacity remains length-dependent.
- Fused dY preparation beat materialization in all 26 compared contexts
  (median materialized / fused about 1.048). Keep materialized as an alternative;
  this is not a proof that it is universally dominated.
- Movement threads 128 generally beat 64 by a small amount; 256/512 were
  near ties. Do not spend a tiny effort budget sweeping every width first.
- Proved inactive FP8 `gather_mode` aliases should not consume separate slots.

Three 96-trial controlled panels jointly varied existing geometry, transport,
pipeline depth and direct/hybrid stores. The best expert-only improvements
over their own initial candidate were just 0.52%, 0.56% and 0.73%. A measured
128×128 gate-dX tile won two many-expert anchors. Deeper buffering was not
consistently better, and cooperative alternatives in the small-E anchor were
slower. Some transports are statically illegal for particular retained fused
families; rejected templates are not counted as measured performance.
These coupled panels do not establish a one-variable causal TMA/stage rule.

One panel winner was also checked in the full E128/K4 MoE: four alternating
visits measured 3.2865 ms baseline versus 3.2722 ms candidate, just **1.0044×**.
Changed-upstream, every-parameter, graph replay and constant live-allocation
checks passed. The gain is smaller than the between-visit range (~0.5–0.6%);
it is not a basis for a universal default or a headline speedup.

An additional same-GPU equal-token A/B retained D384/E128/K6/H192/S320,
32,768 tokens and exact input/upstream/parameter hashes, with fresh 32-trial
searches in each arm:

| Independent partitions | Expert F+B | Router F+B | Whole MX F+B | Compact BF16 F+B |
|---|---:|---:|---:|---:|
| 32×1024 | 3.604 ms | 2.740 ms | 7.086 ms | 6.803 ms |
| 16×2048 | 3.271 ms | 2.921 ms | 6.870 ms | 6.966 ms |

At L1024 each expert has 48 scheduled rows per sequence, padded to 64 for
backward's 32-value scale groups; at L2048 it has 96, already aligned.
Expert time drops ~9.3% but whole time only ~3.0%. Longer-pool hard routing
offsets part of the gain. This motivates sequence/group-utilization features,
not silently coalescing independent samples. Their routing/quantization
partitions are intentionally different numerical workloads.

The practical allocation is therefore: preserve broad legal templates, remove
true aliases, cover routing basins early, and spend additional trials where
measured uncertainty/headroom justifies them. Do not replace this with a
per-SKU winner table or a hard ban on unexplored schedules.

A discovery-only portable research prior (2,352 observations, five component
families) was frozen **before** held-out results were used. Grouped tree and
ensemble comparisons and fresh eight-actual-trial control/prior searches are
separate artifacts. `deployment_enabled=false` remains explicit: logged-pool
recall is not enough to install a production default. A forward-geometry
preservation probe did not pass its strict full-router FP64 scalar-gradient
gate even for its newly constructed control; no candidate was promoted from
that experiment and its failed logs are retained.

### Fresh eight-trial qualification

Five complete control/prior pairs are now finished. Each arm used exactly
eight expert, eight router and eight shared-MLP trials, the same input,
upstream and BF16 parameters, and the same frozen discovery-only prior.
The fifth case is an additional unseen flat transfer probe, not a training
example. All arms passed their own numerical/replay/live-memory gates.

| Held-out geometry | Layout | Control whole F+B | Prior whole F+B | Control / prior | Hard IDs identical |
|---|---|---:|---:|---:|---|
| T18432/D384/E56/K4/H320/S256 | flat | 5.053 ms | 3.806 ms | 1.328× | no |
| T12288/D384/E24/K4/H320/S256 | packed | 1.761 ms | 1.742 ms | 1.011× | yes |
| T9728/D512/E48/K4/H448/S384 | packed | 2.388 ms | 2.396 ms | 0.997× | yes |
| T16384/D512/E48/K6/H256/S512 | packed | 3.367 ms | 3.383 ms | 0.995× | yes |
| T9728/D512/E48/K4/H448/S384 | flat | 3.570 ms | 2.777 ms | 1.285× | yes |

For the two flat cases, router F+B fell from 3.329→2.141 ms and
1.895→1.106 ms respectively (see exact exported values). The E56 hard-route
change is **not** presented as an identical-routing optimization. The E48
result preserves actual IDs/inverse on the tested input, but does not prove
universal hard-policy invariance on changed inputs. The existing strict
changed-feed/FP64 full-router promotion gate is not replaced by this study.

The evidence supports targeting the poor flat search opening. It does not
support replacing the strong packed opening with an ML model: packed changes
were only about −1.1% to +0.5% in time. Between-process arms are not a matched
interleaved timing experiment; sub-percent changes should not be overread.
The artifact therefore remains **opt-in research**, not an automatic new
production default. It ranks candidates through the existing composable
autotuner, and never bypasses legality, numerical validation, exploration or
the user's trial cap.

For an explicit research evaluation:

```python
from rtx.autotune import AutotunePolicy
from rtx.moe import BalancedMoE

policy = AutotunePolicy.preset(
    "balanced", max_trials=8,
    prior_artifact="autotune_reports/mxfp8_moe_heuristics_20261120/discovery-prior-v1.json.gz",
    allow_research_prior=True,
)
model = BalancedMoE(
    384, 32, 320, top_k=4, shared_hidden_features=256,
    device="cuda", autotune="balanced", tuning_policy=policy,
)
# Use the normal user-owned torch.compile wrapper; no per-call tiny graph.
```

## Correctness issue found during collection

The optional Torch packed metadata path used the conservative workspace row
bound as if every row were a balanced real scheduled assignment. For the
256-token, B2/E32/K4 control the workspace has 264 rows, while the actual
scheduled total is 256. Tail assignments corrupted ranks and could scatter
outside the output allocation. This was independently reproduced on CPU
using the untouched frozen implementation, avoiding another poisoned GPU.

The fix sorts inactive rows last, uses actual scheduled counts for rank
offsets, and adds zero rather than overwriting a real slot for inactive rows.
All operations remain compiler-visible Torch metadata expressions; no eager
token permutation or new custom-op allocation was introduced. This path is
an explicit alternative, not the selected CuTe metadata of the 31 large cases.
The cold evaluator also rejects malformed output IDs before indexing its
independent oracle. Source-derived winner identities invalidate old caches.

Validation: 63 focused CPU/fake/API tests passed, 14 CPU/CUDA metadata cases
passed, and the exact formerly faulting geometry passed an additional
compiled forward/gradient/replay test. A fresh 32-trial tiny study completed
on separately frozen recovery source. Its original failure record is retained,
and the fixed result is **not silently pooled into the old-source dataset**.

Controlled schedule panels use existing adapter axes: tall/thin and wide
tiles, cooperative versus TMA loads, 1/2/3/4 buffers and coupled direct/hybrid
output stores. They preserve scale rounding, quantization groups and dtype
contracts. There are no manually imposed register caps or measured-SKU lookup
tables. A faster component candidate still needs whole-layer qualification.

The [matched cost profiler](../benchmarks/profile_moe_routing_cost.py) separates
ordinary phase timings from source-labelled traces and saves actual expert
operands for counter-safe profiling without executing the router under NCU.
Logical buffer/arena sizes are explicitly distinguished from measured DRAM/L2
traffic. Profiler faults stop further work on the affected GPU; they cannot
silently turn into valid data.

## Reproduction

From the frozen source, in two isolated terminals, use `CUDA_VISIBLE_DEVICES=0`
or `1`, the corresponding per-GPU lock, and:

```bash
python -m benchmarks.study_moe_heuristics run \
  --source "$PWD" --output /tmp/moe-heuristics-gpu0 --shards 2 --shard 0
# Other GPU: distinct output directory and --shard 1.

python -m benchmarks.analyze_moe_heuristics \
  /tmp/moe-heuristics-gpu0 /tmp/moe-heuristics-gpu1 \
  --output /tmp/moe-heuristic-analysis
```

Current logs: `/tmp/rtx-moe-heuristics-20261120/study-gpu{0,1}.log`.
Source/winner/fixture identities, rejected arms and raw reports are retained.
This document is updated as qualification completes; an unmeasured heuristic
is not a shipping default.

Durable bundle (Storage4 via the repository symlink):
`autotune_reports/mxfp8_moe_heuristics_20261120/`.
The main matrix is `qualification-summary/whole_layer.csv` / `.parquet`;
`qualification-summary/prospective.json` records all five actual-budget pairs,
including hard-ID equality and exact source/prior identities. Grouped offline
model comparisons are in `analysis-final/`, and routing attribution is in
`recovery-routing-gpu1/attribution-nodes.json`. Frozen runtime snapshots,
adopted pilot journals, the original tiny fault and fixed-source recovery are
preserved. Adopted-pilot loading verifies hashes and supports relocating the
whole bundle without editing the original provenance documents.
