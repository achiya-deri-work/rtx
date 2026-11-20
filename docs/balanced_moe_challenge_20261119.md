# Balanced MXFP8 MoE: strengths and weaknesses across 45 contexts

## Conclusion

The challenge is not one uniformly slow GEMM. There are four distinct issues:
small per-expert batches make preparation expensive; narrow projections retain
substantial per-value overhead; routing selection can miss a much faster plan;
and packed-capacity padding can inflate both runtime and apparent speedups.

**There are real compute-heavy wins, but no universal 1.2x result.** The broad
HOME study reaches 1.24x checkpointed BF16 on its wide/long interaction case.
Several small-input designs lose substantially. A 1.26x packed result against
the original BF16 reference disappears against a stronger compacted reference.
These are measured results, not a prediction of global-optimum schedules.

## Scope and method

- **38 HOME contexts + 7 WORK supplements**, both RTX 5090/170 SM hosts;
  Torch 2.13.0+cu132, CUDA 13.2, CuTe 4.7.0, driver 595.99.02. HOME GPU1 was
  untouched. WORK used one persistent multiplexed SSH transport, closed after
  retrieval. No services, containers, drivers, clocks or power settings changed.
- Dynamic BF16 inputs/expert parameters/outputs; MXFP8 expert/shared GEMMs,
  FP32 activation/products/reductions and sensitive router arithmetic. One
  shared SwiGLU. Fixed K and independently balanced routing per packed sequence.
- Fresh bounded selection: up to **8 expert + 8 shared + 4 router trials**,
  90-second soft search admission per component. Actual counts are exported;
  one HOME router search admitted only two trials after an expensive compile.
  Expert and shared searches completed eight in every primary context.
- Both precision paths use fullgraph/static compilation with aggressive
  dynamic-weight Inductor settings, no freezing, and warmed outer CUDA graphs.
  Ordinary CUDA-event batches: two rounds, seven samples, at least 5 ms per
  batch. Compilation, tuning and warmup are excluded from reported runtime.
- Main reference: identical native router with **reentrant-checkpointed BF16
  FFNs**. Saved-intermediate BF16 is recorded separately, not the target.
  Forward-only columns measure the training forward and use the saved BF16
  reference; they are not a pure-inference or matched-checkpoint-forward claim.
- No optimizer, communication, full model, convergence, or rotating-cache test.
  These are warm same-buffer measurements. Source was frozen across collection:
  `ebec8ac46b0ad3b10902194ef56ce698988b2d5e19e3a1b01dd06ce766fec58b`.

Coverage spans D=128–1536, expert H=64–512, E=16–128, K=2/3/4/6/8,
shared H=128–512, 128–65536 tokens, and 1/2/6/8/32 independent sequences.
Feature widths are multiples of 32, including D576/H160/shared H320; sequence
lengths include arbitrary tails. This is a controlled sparse matrix with
interaction cases, **not** every Cartesian combination or every legal shape.
Models use synthetic inputs and fresh initialization, not trained-router
entropy extremes. Low timing variance on a fixed replay does not establish
latency variance across changing training batches.

All speedup columns below are **BF16 time / MXFP8 time**; above 1 favors MXFP8.
Do not pool raw HOME and WORK latencies into one machine's scaling curve.

## Where it works, and where it does not

Representative HOME results; the full 45-row export is linked below.

| T / D / H / E / K / shared H | MX F+B | Whole speedup | Routed-only speedup |
|---|---:|---:|---:|
| 128 / 384 / 128 / 32 / 4 / 256 | .475 ms | .720x | .577x |
| 512 / 384 / 128 / 32 / 4 / 256 | .515 ms | .876x | .614x |
| 2048 / 384 / 128 / 32 / 4 / 256 | .565 ms | .916x | .833x |
| 8192 / 384 / 128 / 32 / 4 / 256 | 1.187 ms | 1.018x | 1.042x |
| 32768 / 384 / 128 / 32 / 4 / 256 | 4.799 ms | 1.049x | 1.286x |
| 65536 / 384 / 128 / 32 / 4 / 256 | 9.250 ms | 1.070x | 1.402x |
| 32768 / 128 / 128 / 32 / 4 / 256 | 3.749 ms | 1.002x | 1.060x |
| 8192 / 768 / 256 / 48 / 6 / 512 | 2.463 ms | 1.144x | 1.217x |
| 8192 / 1536 / 128 / 32 / 4 / 256 | 2.422 ms | 1.079x | 1.190x |
| 32768 / 768 / 512 / 32 / 4 / 512 | 10.083 ms | **1.239x** | **1.445x** |
| 16140 / 576 / 160 / 24 / 3 / 320, two ragged pools | 2.585 ms | .906x | .877x |

The shared MLP matters independently. At T32768/D128 it achieves only .605x
BF16, erasing much of the routed gain. At T32768/D768/H512 it reaches 1.188x.
A fast routed branch does not establish a fast complete MoE.

WORK confirms that wider experts do not rescue a small per-expert batch:

| WORK design D/H/E/K/shared H | Tokens | MX F+B | Whole / routed speedup |
|---|---:|---:|---:|
| 768/256/48/6/512 | 512 | .820 ms | .723x / .570x |
| 512/128/64/8/384 | 512 | .621 ms | .827x / .589x |
| 384/128/24/6/256 | 512 | .463 ms | .912x / .774x |
| 768/256/48/6/512 | 2048 | 1.154 ms | .905x / .779x |
| 512/128/64/8/384 | 2048 | .820 ms | .939x / .842x |
| 384/128/24/6/256 | 2048 | .582 ms | .939x / .919x |
| 384/128/32/4/256, independent anchor repeat | 8192 | 1.170 ms | 1.025x / 1.047x |

## The useful axes are not just total tokens

### Rows per expert, C approximately T*K/E

At fixed T8192/D384/H128/K4/shared H256:

| Experts | Real rows/expert | Whole speedup | Routed speedup |
|---:|---:|---:|---:|
| 16 | 2048 | 1.039x | 1.097x |
| 32 | 1024 | 1.018x | 1.042x |
| 64 | 512 | .980x | .906x |
| 128 | 256 | .945x | .793x |

Increasing E with K fixed does not increase useful routed GEMM FLOPs, but does
increase router work and dynamic weight/gradient volume. Smaller C also changes
GEMM reduction efficiency. This is strong evidence for a per-expert-batch
feature, not a universal crossover threshold.

### Projection geometry and granularity

At T8192/E32/K4/H128, whole speedup moves from .938x at D128 to .989x at
D256, 1.018x at D384, 1.074x at D1024 and 1.079x at D1536.

Hidden width is not monotonic: H64/H128/H192/H256/H512 routed speedups are
.919x/1.042x/.967x/1.128x/1.233x. The selected down macro-tile model rounds
H64 to K128 and H192 to K256. This supports investigating alignment and
pipeline selection; it is not an instruction-count or saturation measurement.

The equal-work control changes E32/K4/H128 to E64/K8/H64: active K*H and total
expert parameter count stay constant, as do mean rows/expert and D. Whole MX
time nevertheless rises **1.187 -> 1.514 ms**; BF16 also slows. More experts
and routes increase routing/movement work even when nominal FLOPs are unchanged.

The asymptotic bookkeeping explains why tokens cannot amortize everything:
useful recomputed routed GEMM work scales roughly as `24*T*K*D*H`, whereas
preparation includes `T*D`, `T*K*(D+H)` and `E*D*H` terms. More tokens reduce
the relative weight-preparation tax; activation preparation and movement also
grow with tokens. D, H, K, C, geometry and cache state remain relevant.

## Packing: scheduling, selection and bounds are separate effects

Same T8192/D384/H128/E32/K4/shared H256, with identical input/weights/upstream:

| Independent lengths | MX F+B | Matched BF16 F+B | Static slot amplification | Expert backward typed arena model |
|---|---:|---:|---:|---:|
| 8192 | 1.187 ms | 1.208 ms | 1.000x | 116.7 MB |
| 2 x 4096 | 1.080 ms | 1.098 ms | 1.031x | 119.9 MB |
| 8 x 1024 | 1.113 ms | 1.185 ms | 1.219x | 139.3 MB |
| 32 x 256 | 1.296 ms | 1.627 ms | 1.969x | 217.1 MB |
| 127+513+2001+767+4095+689 | 2.094 ms | 2.049 ms | 1.156x | 132.9 MB |

Arena figures are shape-planner scratch models, not total VRAM or measured
traffic. Static amplification is not proof that every padded tile executed;
active-prefix execution can trim work without shrinking allocation shapes.

**Uneven is not intrinsically twice as slow.** With FFN policies frozen,
raising its router search from four to eight cumulative trials changes router
F+B from **1.442 -> .635 ms** and whole MX from **2.094 -> 1.178 ms**.
Input, parameter, upstream and hard-assignment hashes match. This is an
opening/selection weakness. Tiny-128 and small-512 router controls give much
smaller whole changes: .475 -> .465 ms and .515 -> .477 ms respectively.

### The stronger BF16 control changes the headline

The original reference intentionally inherits the native router's padded expert
capacity. It is a matched-schedule reference, not the best possible BF16 MoE.
An independent BF16 control compacts valid expert rows to a shape-only safe
logical bound, rounds capacity to 128, and includes its additional cumsum/
scatter work and checkpoint recomputation in timing. No production code changed.

Four rotating visits, seven samples per cell, paired separately on each host:

| 32-sequence control | HOME | WORK |
|---|---:|---:|
| Current selected MXFP8 | 1.297 ms | 1.343 ms |
| Original checkpointed BF16 | 1.628 ms | 1.667 ms |
| Compacted checkpointed BF16 | **1.192 ms** | **1.237 ms** |
| MXFP8 with existing active-prefix template | 1.245 ms | not measured |

BF16 capacity changes **2016 -> 1152**, with a safe logical bound of 1052.
Every output/gradient passes parity; worst BF16-relative NRMS is .0337%.
Native active-prefix output/all gradients are bitwise identical to current
native. Its structural compatibility move disables fragment GLU and persistent
dW: those implementations currently require static capacity. Thus this is a
complete legal template comparison, not an isolated Boolean or global optimum.

The apparent 1.25x speedup becomes approximately **.92x** against compacted
BF16. Active prefix recovers about 4% native time, but remains about .96x of
compacted BF16 on HOME. Padding/preparation and template composability need work;
the original packed headline must not be advertised as a general 1.25x win.

Changing sequence-local leading-axis scale groups would be a numerical-policy
change, not free compaction. This study preserves the existing MXFP8 recipe.

## Where the time goes, and a profiler trap

Complete isolated-expert traces show these diagnostic shares:

| Work group | T8192/D384/H128 | T32768/D768/H512 | T65536/D384/H128 |
|---|---:|---:|---:|
| Pure quantization/preparation | 20.8% | 12.4% | 22.1% |
| Gather/scatter | 14.6% | 9.1% | 18.6% |
| Fused recompute/activation/quantization | 14.9% | 18.2% | 13.5% |
| Other GEMMs | 32.7% | 39.7% | 29.2% |

The fused category includes necessary GEMM math: it must not be called pure
quantization overhead. Standalone phase times are not additive whole-layer
attribution, and percentages do not establish tensor-core saturation.

Torch's whole-graph trace omits some conditional repair-body activity. Nsight
Systems node tracing resolves it:

| Input, same model/configuration | Torch-visible kernels | Systems kernels | Repair rounds | Kernel+copy interval coverage |
|---|---:|---:|---:|---:|
| Normal | 163 | 202 | 3 | 95.4% |
| Sparse outliers | 163 | 177 | 2 | 95.9% |

Both Systems captures contain **one graph-launch API and zero per-kernel launch
APIs**. A synchronization closes the profiler boundary; it is not a sync inside
each launcher. The apparent 100–200 us holes in the Torch trace are not evidence
of eager Python gaps. Input-dependent repair work matters. No new Compute
counters were collected: DRAM/L2 hit rates, spills and tensor-pipe saturation
are not established by this study. Raw Systems reports/SQLite/CSVs are retained.

## Accuracy and bounded-search confidence

- All **45 contexts** pass the independent quantized FP64 expert reference,
  output/all-leaf finiteness, captured fresh-gradient replay and flat live
  allocation checks. Worst expert-oracle NRMS: **.0274%**. This is not a full
  FP64 router/model or a convergence qualification.
- Versus matched BF16, output NRMS is **6.32–6.88%**, input-gradient NRMS
  **6.44–6.59%**; worst nonscalar parameter-gradient NRMS is **8.13%** on
  correlated input. Normal, RMS-matched bounded, sparse 16x-outlier and highly
  correlated inputs plus an independent seed are included.
- **Scalar router gradients need attention.** At T65536, `router.score_bias`
  differs by **89.1%**: absolute error .01253 against reference magnitude
  .01407. All values are finite. Cancellation/small aggregate signal may amplify
  ordinary expert quantization noise, but that mechanism and training impact
  are not proved here. Do not hide this behind the output error or call it an
  implementation-oracle failure. A real-loss, multi-seed training check is needed.
- Actual progressive 8-to-16 FFN controls on tiny-128, thin-long, experts-128 and
  packed-32 preserve operands/routes and do not materially improve deployed
  policies. Each journal contains **16 cumulative observations**; the continuation
  evaluates eight new ones. This is not an exhaustive-family optimum.
- Five **candidate** row-broadcast affinity plans fail the current elementwise
  parity gate and are discarded; the selected fallback paths pass. Reported
  differences involve a handful of BF16-sized steps (.000244–.000488 absolute).
  This does not establish their cause or justify relaxing validation. Compare
  against an independent higher-precision affinity oracle before deciding
  whether this is a kernel error or an overly strict reduction-order gate.
- All 38 primary timing contexts satisfy the descriptive MAD/round-range gate.
  Of these, 11 exceed 1.05x, 20 fall in [.95,1.05], seven are below .95x, and
  two exceed 1.2x against the original reference. These deliberately sampled
  counts are not deployment-weighted performance; one of the two strong wins
  is padding-sensitive, as the stronger BF16 control demonstrates.
- Warm process peaks span .073–3.343 GB; five repeated live-allocation samples
  are flat per context. Peaks include other warmed reference state and are not
  a per-precision memory-saving comparison. Tooling tests: **18 passed, one skip**.

## What to prioritize next

1. **Router first-hit coverage.** Protect inexpensive, diverse row/column/sharded
   recipes for skewed packed inputs, rather than spending the prefix on nearby
   slow plans. Revalidate on new profiles; the measured 44% whole-time reduction
   is selection headroom, not a reason to hard-code this shape's winner.
2. **Packed preparation and composability.** Keep the compacted BF16 control.
   Make trimming coexist safely with current forward/dW wins and measure which
   padded preparation passes remain. Do not silently change scale grouping.
3. **Small-C/narrow structural paths, including the shared MLP.** Fewer tensor
   passes/launches and better weight reuse are more promising than merely doubling
   the same search prefix. Still test alternative families before declaring a
   hardware limit; no registers or device-specific winner tables were hand-tuned.
4. **Router-gradient quality under a real loss.** Repeat the scalar-gradient
   finding across seeds and training before claiming numerical readiness.
   Audit the five rejected affinity candidates independently as well; never
   promote them simply because the discrepancies are sparse.

## Artifacts and reproduction

Durable bundle: `autotune_reports/balanced_moe_challenge_20261119/` (Storage4).
It contains individual atomic results/residual journals, source snapshots,
exact commands, HOME/WORK separation, checksummed raw traces and both versions
of the compaction control. No production kernel was changed by this study.

- [All 45 contexts, CSV](../autotune_reports/balanced_moe_challenge_20261119/analysis/all_contexts.csv)
- [All 45 contexts, Parquet](../autotune_reports/balanced_moe_challenge_20261119/analysis/all_contexts.parquet)
- [Full analysis and controls](../autotune_reports/balanced_moe_challenge_20261119/analysis/summary.json)

```bash
flock /tmp/rtx-moe-study-gpu0.lock env CUDA_VISIBLE_DEVICES=0 \
  .venv/bin/python -m benchmarks.profile_moe_challenge run \
  --output autotune_reports/my_moe_challenge --max-seconds 7200

# Same command resumes only a matching frozen source/settings contract.
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m benchmarks.profile_moe_challenge \
  summarize --output autotune_reports/my_moe_challenge
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m benchmarks.report_moe_challenge \
  autotune_reports/my_moe_challenge
```

The reusable launcher covers the 38-context primary matrix; the exact WORK
supplement and focused-control launchers are in the evidence bundle.
