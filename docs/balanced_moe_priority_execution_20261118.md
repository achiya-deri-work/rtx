# Executing the independent MoE review

## Outcome

The first-hit opening now reaches useful geometry and fused routed-dY before
generic coordinate neighbors. Prepared shared MLPs compare both-orientation
and row-only preparation fusion early. Redundant CPU pool generation is
removed for explicitly opted-in, fully seeded pools.

On HOME GPU0, fused-dY improved matched complete-layer forward/backward by
**2.7% large / 2.2% mixed packed** over the previous eight-trial winner.
Shared preparation fusion passed numerics but did not improve these cases;
the tuner retains the faster separate path. **Neither case beats checkpointed
BF16.** This work does not claim 1.2× or a universal winning configuration.

No quantization/activation/reduction policy, register budget or public API
default was changed. There is no new eager numerical work in custom ops or
user-managed workspace/stream requirement. GPU1 was untouched.

## Runtime changes

- `rtx/autotune/moe.py`: incumbent remains first; compact/indexed geometry
  combined with fused-dY is trial two, its materialized control is trial three.
  A final 15-second probe motivated this order over the initially tested
  geometry-then-fused ordering. Tiny-capacity
  short-dW coverage is preserved. Six protected proposals leave one UCB
  opportunity within an eight-trial prefix. Local geometry turns rotate
  down/gate-dX instead of repeatedly favoring down. Portfolio revision 19.
- `rtx/autotune/mlp.py`: for already prepared training MLPs, both-orientation
  exchange precedes row-only exchange when legal. Ragged rows retain legal
  row-only fusion; on-chip 4/8-way and existing long dual-axis openings remain.
  Compatible frozen rankers and explicit neighborhood requests are respected.
  Structural context revision 12 fences the new search prefix.
- `rtx/autotune/sampling.py`: explicitly opted-in adapters can skip broad/local
  generation when legal explicit seeds already fill the pool. The selected
  candidates and their order are unchanged. The shared MLP opts in through a
  context tag; legacy adapters keep their prior audit/checkpoint behavior.
- `rtx/_sampling.py`: exact categorical distance uses precomputed set
  signatures rather than Python field loops. Proposal generation, selection
  ordering and tie handling are unchanged. Randomized equivalence tests cover
  missing fields and flattened nested values.
- Search wrappers preserve the shared engine's distinction between legacy
  continuation and deterministic progressive checkpoints. They do not invent
  strict progressive checkpoints for legacy schedules.
- Expert winner evidence now records reference setup, context setup, search
  call and invocation-to-winner wall time separately. The soft admission
  limit still excludes context setup and cannot preempt an in-flight compile.

These are search/planning changes over existing structural kernel templates,
not manually chosen register limits or a hard-coded per-SKU dispatch table.

## Controlled whole-layer experiment

D=384, E=32, K=4, expert H=128, one shared H=256; N=29,504 flat and independent
packed lengths `[512,2000,768,4096]`. Baseline is the previous qualified
eight-trial winner. Expert candidate changes only routed-dY preparation.
Shared candidate uses the existing legal both-orientation path for aligned M,
row-only path for the ragged packed M. Same native router in every variant.

| Variant | Large flat | Mixed packed |
|---|---:|---:|
| Previous native winner | 4.7351 ms | 1.2383 ms |
| Fused routed-dY only | 4.6114 ms | 1.2112 ms |
| Shared preparation fusion only | 4.7591 ms | 1.2386 ms |
| Both changes | 4.6327 ms | 1.2117 ms |
| Checkpointed BF16 experts and shared | 4.1624 ms | 1.0521 ms |

All variants use `torch.compile(fullgraph=True,dynamic=False)` with
max-autotune/subprocess GEMM tuning and outer CUDA Graph replay. Compilation,
warmup, checks and capture are excluded. Each uses set-to-none backward,
every input/parameter gradient, four alternating visits and seven samples
per visit. BF16 checkpoints FFNs, not routing; it does not save the hidden
connection as a privileged comparison.

Large native variants matched outputs and every gradient bitwise. Every cell
passed two changed-upstream replay comparisons, exact hard routes, finiteness
and five identical live-allocation samples. BF16 is checked against its own
fresh execution, not required to match quantized native tensors. Cumulative
allocation totals with multiple models/graphs alive are not independent peak
memory measurements. No convergence or cross-precision quality claim follows.

The large combined interaction is about −2.7 µs: shared fusion is simply
unhelpful here, not hiding a large synergistic win. The measured fused-dY
speedups are 1.0268× / 1.0224×. Native remains about 10.8% / 15.1% slower than
checkpointed BF16. Do not sum standalone component speedups.

## Bounded first-hit evidence

The final measured 30-second mixed searches admitted five expert candidates
on each case. The large useful geometry completed at 23.32 s and fused-dY at
23.93 s. Both passed the existing independent quantized FP64 expert oracle.
These are **caps of eight**, not claims that all eight finished.

The shared pool originally spent 12.72 s constructing/ranking candidates
before its first attempt; the fully seeded fast path reached that attempt at
0.41 s. It measured all eight large-case candidates within 18.66 s, compared
with seven in the earlier 30-second run. The entire tuned/qualified shared
call reported 20.50 s. This is not a cold CUDA compiler-cache speedup claim.

Shared both-orientation and row-only proposals were measured immediately after
the incumbent. The final selected shared policy remained separate on both
cases. Additional local choices can still compete; fusion is not mandatory.
Mixed packing completed six shared trials before its soft deadline. Pairwise
deployment qualification is additional work and remains reported separately.

The **final revision-19** 15-second legacy-mix probe completed eight expert
trials in 14.09 s and eight shared trials in 10.78 s (separate per-component
allowances). The complete fused geometry/traffic challenger was trial two,
finished at 6.65 s and measured 1.5775 ms in the expert evaluator. Expert
context setup was 0.416 s; invocation-to-winner was 14.52 s. Compiler caches
were warm: these numbers must not be attributed solely to the ordering change
or advertised as an empty-cache compile guarantee. All-leaf, independent
quantized FP64, changed-gradient replay and live-memory checks passed.

`candidate_mix='mixed'` still intentionally builds a broader expert pool;
its planning took roughly ten seconds here. The legacy mix avoids that broad
construction. A 15-second admission allowance is not a universal 15-second
cold-first-input latency guarantee. Exact winner-cache hits should skip this
search; compatible progressive refinement resumes measured history.

For a smaller explicit effort while retaining automatic first-input selection:

```python
from rtx.autotune import AutotunePolicy

policy = AutotunePolicy.preset(
    'quick', search_schedule='progressive', candidate_mix='legacy')
# Pass autotune='balanced', tuning_policy=policy to BalancedMoE.
# 'balanced' is the selection mode here; policy selects the effort.
```

The module's existing default remains unchanged; this is an optional recipe,
not a silent removal of broad mixed search.

## Evidence and reproduction

All intermediate failures, residuals, timings, configs, CPU planning profile
and source archives are preserved under
`autotune_reports/moe_priority_execution_20261118/` on local Storage4.
The full factorial measurement runtime hash is
`c9e5b151575ffbb13aecfc5425bbcafc8aa55226b46afde27ad13b04ebcd9c98`.
Its snapshot is `measured-source.tar.gz`. Subsequent compatibility tightening
scopes the same skip-generation behavior to versioned MLP contexts and avoids
adding strict checkpoints to legacy searches, then moves the complete
challenger to trial two. It changes no measured kernels, native factorial
configurations or numerical behavior. `final-source.tar.gz` preserves the
final implementation qualified by the revision-19 probe and integration tests.
Its runtime hash is
`181556131ba42f0ce86ea4f68488b0ff6fde61fa9e794b09144eb3d546645610`.

Targeted GPU tests passed **9 routed-dY tests** (including real compiled
cold→hot, durable winner reload, noncontiguous gradients, ragged/empty packed
segments and cumulative refinement) and **4 dual-orientation SiLU/FP32 tests**
(exact packed data/scales and mutable graph replay). No comparator was relaxed.
The earlier intermittent baseline allocation issue is not proven fixed by
these warm passes; no failed cold evidence was erased.

The final focused CPU suite passed **49 tests** with one GPU-only test skipped
in that CPU invocation. It covers 8→16 progressive prefix equivalence,
nonduplicating legacy refinement, actual bounded shared/expert openings,
legality, numerical-policy preservation, exact diversity distances and
unchanged default pool audits. API reference and diff-whitespace checks pass.

`benchmarks/prepare_moe_priority_eval.py` derives ablations with the real
adapters using metadata only; it is not a numerical or performance oracle.
`benchmarks/benchmark_moe_composition.py` now supports independent expert and
shared candidates, changed-upstream replay and a checkpointed BF16 control:

```bash
CUDA_VISIBLE_DEVICES=0 python -m benchmarks.benchmark_moe_composition \
  --baseline-result /path/to/qualified/result.json \
  --candidate-config /path/to/expert.json \
  --candidate-shared-config /path/to/shared.json \
  --match-inputs --max-autotune --changed-dy --bf16-checkpoint \
  --visits 4 --samples 7 --output /path/to/fresh/result.json
```

Next, the shared MLP needs phase-level preparation/GEMM measurements before
new kernel work. The current fusion templates alone do not erase its tax.
For routed experts, movement and few-K GEMM ownership remain relevant.
No broad data collection or learned-prior deployment was started.
