# Cross-SKU autotuning study

This study analyzes the six-hour NVFP4 campaigns collected on the RTX 5070
Laptop, RTX 5070 Ti, and RTX 5090. Its purpose is to identify useful portable
search priors without mistaking an adaptively sampled autotuning dataset for a
controlled experiment.

## Reproducing the study

Train the repository's bagged gradient-boosted tree ensemble and evaluate its
heads on held-out contexts:

```bash
rtx-autotune pretrain \
  autotune_datasets/nvfp4_full_power_6h_v1_rtx5070ti \
  nvfp4_full_power_6h_v1_rtx5070_laptop.zip \
  nvfp4_full_power_6h_v1_rtx5090.zip \
  --output autotune_models/nvfp4_cross_sku_full_power_v1 \
  --seed 20260817 --estimators 40 --ensembles 4 --max-depth 4 \
  --min-leaf 5 --max-features 128 --min-rule-support 10 --max-rules 512
```

Then produce the statistical report:

```bash
rtx-autotune study-sku \
  autotune_datasets/nvfp4_full_power_6h_v1_rtx5070ti \
  nvfp4_full_power_6h_v1_rtx5070_laptop.zip \
  nvfp4_full_power_6h_v1_rtx5090.zip \
  --artifact autotune_models/nvfp4_cross_sku_full_power_v1 \
  --output autotune_reports/nvfp4_cross_sku_full_power_v1 \
  --minimum-contexts 5
```

The report contains 36,199 observations. It normalizes latency by the median
inside each exact device, kernel-family, shape, cache-regime, and replicate
context. It therefore measures schedule quality rather than confusing a fast
GPU or small matrix with a good configuration. Confidence intervals resample
whole contexts, not individual timings.

## Hardware balance

| SKU | SMs | Bus | Theoretical bandwidth | Measured DRAM | Measured quant bandwidth | L2 |
|---|---:|---:|---:|---:|---:|---:|
| RTX 5070 Laptop | 36 | 128 bit | 384 GB/s | 340 GB/s | 754 GB/s | 32 MiB |
| RTX 5070 Ti | 70 | 256 bit | 896 GB/s | 796 GB/s | 1,698 GB/s | 48 MiB |
| RTX 5090 | 170 | 512 bit | 1,792 GB/s | 1,530 GB/s | 2,852 GB/s | 96 MiB |

The 5090 has 2.43 times the 5070 Ti's SM count, but only 1.92 times its
measured DRAM bandwidth and 1.68 times its measured quant bandwidth. It is
therefore substantially more sensitive to scale traffic, weak data reuse, and
small grids per SM. The fixed-shape native-MXFP8 probe should not be read as a
hardware peak: it reports 204 TFLOP/s on the 5090 and 211 TFLOP/s on the 5070
Ti while BF16 scales in the expected direction. That probe is exposing kernel
or shape utilization, not the devices' peak ordering.

Three devices are not enough to infer a reliable continuous law from SM count,
bus width, or bandwidth. Hardware fields should remain explicit model features,
but the current evidence supports SKU classes and conditional interactions more
strongly than a global linear hardware formula.

## Can one cost model serve every SKU?

No—not yet. Every portable latency and ranking head failed the deployment
gate. At four proposals, their median regret was usually competitive with
matched random search, but their p90 regret was worse on at least one SKU. The
artifact consequently records `selected_cost_head: none` for delayed, dynamic,
and JIT-region NVFP4. This is a successful safety result: the gate prevents a
plausible average model from making first-hit autotuning less reliable.

Feasibility heads remain useful for delayed and JIT-region kernels, and all
three families produced supported conditional rules. Dynamic NVFP4 had no
failed candidates, so a feasibility classifier is neither identifiable nor
needed for that sample.

An exact-SKU latency head passed only for the laptop delayed family. We should
therefore keep exact-SKU winner caches, learn separate SKU residual heads, and
use the portable artifact only for legality, conditional rules, and proposal
ordering until a prospective study clears the tail-regret gate.

## Winner portability

Winner transfer is evaluated only where the source winner was also benchmarked
on the target. Coverage is consequently low and is reported with regret.

| Source to target | Coverage | Median regret | P90 regret | Within 2% |
|---|---:|---:|---:|---:|
| Laptop to 5070 Ti | 27.2% | 2.70% | 7.83% | 44.2% |
| Laptop to 5090 | 27.8% | 1.38% | 6.51% | 59.1% |
| 5070 Ti to Laptop | 7.0% | 9.33% | 18.42% | 9.1% |
| 5070 Ti to 5090 | 12.7% | 1.59% | 14.30% | 55.0% |
| 5090 to Laptop | 5.1% | 9.28% | 28.80% | 12.5% |
| 5090 to 5070 Ti | 9.5% | 0.90% | 5.85% | 80.0% |

The desktop SKUs are related enough for warm starts, although their tail risk
still forbids blindly copying winners. Desktop-to-laptop transfer is poor.
Device identity and hardware balance must be part of the runtime-cache key.

## Good and bad regions of the search space

The strongest consistent findings are:

- JIT-region X geometries from 2 through 8 rows form a broad performance
  plateau. Five rows is close to neutral across all devices and remains a good
  numerical/performance default. One row is about 29% slower in the marginal
  study; 16, 32, and 64 rows become increasingly poor. W regions from 2 through
  8 rows are likewise broadly flat, with four rows a robust center. W=1 is
  strongly poor. Geometry changes the numerical policy, so even a small timing
  difference is not enough to silently change it.
- Dynamic quantization with one persistent wave is about 18% slower after
  context normalization. Two waves is still mildly weak; three through eight
  form the useful region. Four quantizer warps is about 9% weak, while eight is
  the safe center and sixteen is worth retaining. A parent-linked rule finds
  the combination of four warps and one persistent wave roughly 37% slower.
- Delayed and dynamic GEMM favor two B `ldmatrix` matrices. One costs roughly
  5–7%, and four costs roughly 6–11% in these families. JIT-region differs:
  four has a modest favorable association, so this coordinate must remain
  family-specific.
- Delayed scale loads have a sharp center at vector width four. Width one is
  approximately twice as slow in the marginal study, and width eight is also
  poor. Scale vectorization and GEMM stage count are the delayed model's
  strongest interaction.
- Three GEMM stages is the robust center. One stage is strongly poor and two is
  weak for dynamic quantization. Deeper schedules must stay available because
  shared-memory pressure and CTA residency change with shape and SKU.
- A supported parent-linked dynamic rule finds that a 128x128x128 tile with a
  2x2 atom layout is about 4.1% faster when weight quantization spans more than
  three effective CTA waves. This is a useful conditional proposal prior, not
  a universal default.

The tree models reinforce these relationships. Delayed ranking is dominated by
stage count, scale vector width, work tiles per CTA, and shared-memory fraction.
Dynamic ranking is dominated by B `ldmatrix`, quant persistent waves, stages,
tile M, and effective CTA waves. JIT-region ranking is dominated by X-region
rows, warp ownership, quant grid size, shared-memory fraction, K:N aspect, and
the memory roofline. Important interactions include region geometry with
ownership and memory pressure rather than region geometry alone.

Marginal effects can be badly confounded when several coordinates only occur
together in an anchor implementation. For example, some warp-ownership and
direct-epilogue anchors look extremely poor as single-coordinate effects. They
must not become hard exclusions without a parent-linked comparison or a
randomized ablation holding the other coordinates fixed.

## SKU-sensitive regions

- A 32-byte A/B swizzle is nearly neutral on the laptop but clearly weak on the
  5070 Ti and 5090. Larger Blackwell grids prefer wider transaction/swizzle
  choices.
- JIT grid swizzle one is weak on the laptop but improves the 5070 Ti and 5090,
  consistent with their much larger CTA grids.
- A 32-bit JIT-region amax load is weak on the laptop and favorable on both
  desktops.
- The factorized regional epilogue is weak on the laptop, closer on the 5070
  Ti, and slightly favorable on the 5090. This is consistent with the 5090's
  greater compute-to-bandwidth imbalance: reducing scale traffic/reusing
  factors becomes more valuable.
- One delayed quant persistent wave is weak on the laptop and 5070 Ti but nearly
  neutral on the 5090. This should stay an SKU-conditioned coordinate rather
  than a portable exclusion.

These are hypotheses supported across contexts on three machines. They should
guide proposal order, then be validated prospectively; they are not hardware
laws.

## Failure structure

The final bundles contain 539 JIT-region correctness failures, 36 delayed
correctness failures, and no dynamic failures. The feasibility models are
driven primarily by M/N/K tail fractions and their interaction with scale
layout, transport, reduction, and region geometry. This supports tightening
static legality around ragged tails and scale indexing. It does not support
pruning a schedule merely because a particular SKU made it slow.

## Changes to the autotuning policy

The evidence supports the following conservative policy:

1. Use feasibility prediction and known tail/layout rules before compilation.
2. Keep separate exact-SKU winner caches and SKU residual cost heads.
3. Share conditional rules between SKUs, but use them to rank proposals rather
   than delete the rest of the space.
4. Seed dynamic searches around tile M=128, three stages, B `ldmatrix`=2, at
   least eight quantizer warps, and at least three persistent quant waves.
5. Seed delayed searches around scale-load width four, B `ldmatrix`=2, and
   three stages.
6. Seed JIT-region searches in X=2..8 and W=2..8, retaining 5x4 as the default.
   Search ownership, factor reuse, and grid swizzle with SKU-aware ordering.
7. Preserve 15–25% exploratory proposals outside learned priors so the tuner
   can detect implementation changes, compiler changes, and new-SKU behavior.
8. Never deploy a cost head unless held-out p90 regret beats matched random on
   every target class in its declared portability domain.

The next dataset should contain randomized, paired coordinate ablations within
the same context, plus more SKUs. That design will separate causal schedule
effects from correlations introduced by adaptive search and will let hardware
ratios become statistically meaningful instead of being inferred from only
three points.

## Follow-up paired and shape-held-out evidence

The paired study resolved 15,952 measured parent→child moves. Of these, 9,293
were coupled or composite changes and are deliberately excluded from claims
about one coordinate. The remaining exact transitions provide substantially
stronger proposal-order evidence than the marginal tables above. Examples seen
consistently across devices include severe regressions from narrowing quantizer
vector ownership from 8 or 16 values to 2, and large JIT-region regressions when
moving otherwise competitive X-region geometries to one row.

A symmetric parent-pair classifier reached approximately 0.75–0.91 held-out
pair AUC. That did not translate into safe search: every portable and exact-SKU
family failed the eight-trial regret gate with complete M/N/K groups held out
after practically tied measurements (within 0.2%) were removed from the labels.
The artifact writer therefore emitted no pairwise model. This demonstrates why
pair classification accuracy cannot substitute for fixed-budget search replay.

The failure study found a deterministic revision-8 JIT-region boundary: all
539 correctness failures occurred on the two deliberately ragged N/K shapes,
and no candidate on those shapes succeeded. The failures reproduced on all
three GPUs with near-zero cosine, making this an implementation contract bug
rather than ordinary NVFP4 numerical loss. Focused CUDA isolation showed that
N tails were correct and every failure had an odd physical count of 1x16 blocks.
That makes the packed row stride an odd multiple of eight bytes, violating the
16-byte alignment assumed by the SM120 native load path on following rows.
Native RTX storage therefore uses the smallest 32-value K boundary, the
quantizer zero-fills the extra block, a CUDA regression covers the boundary,
and affected NVFP4 runtime-winner revisions were advanced. Arbitrary logical
shapes remain supported; the worst additional packed storage versus a
standalone NVFP4 encoding is only eight bytes per row.

Raw timing prefixes also expose strong SKU differences. On fixed candidate
cohorts where every candidate received at least fifteen measurements (twenty
when that larger cohort retained at least five contexts), the first sample
count meeting ≤1% p90 median error, ≤2% p90 winner regret, and ≥95% winners
within 2% was:

| Family | 5070 Laptop | 5070 Ti | 5090 |
|---|---:|---:|---:|
| Delayed | 15 | 5 | 15 |
| Dynamic | 15 | 5 | 9 |
| JIT-region | 20 | 3 | 5 |

These values describe this campaign's fixed confirmation cohort, not universal
defaults. They justify SKU-aware measurement budgets and stronger final races
on the laptop. The 5070 Ti is markedly stable.

Finally, proposal time is material. Across JIT-region contexts, gradient-boosted
pool construction accumulated roughly 2,600 seconds on the laptop, 3,400 on
the 5070 Ti, and 4,000 on the 5090; model-local ranking accumulated another
1,700–2,200 seconds. Random and coordinate proposal time remained tiny. Bandit
reward already charges this cost to its arm, and balanced first-hit tuning now
uses explicit 512-candidate/250-ms global and 256-candidate local caps. Offline
campaigns retain the ability to enlarge these limits deliberately.

The statistical contracts behind these conclusions are specified in
[Autotuning evidence methodology](autotune_evidence_methodology.md).
