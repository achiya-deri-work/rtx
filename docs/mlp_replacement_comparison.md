# Replacement-level MLP comparison

`python -m rtx.autotune.replacement_benchmark` compares ordinary compiled BF16,
the existing recomputed BF16 backend, and selected-direct MXFP8. Every backend
runs in a separate subprocess with identical seeded BF16 inputs/weights and
upstream gradients. No models, production caches, or external projects are
modified. MXFP8 uses the supplied pinned config, never a hidden tuning run.

```bash
python -m rtx.autotune.replacement_benchmark \
  --shape 11712,256,768,256 --mix 1,1,2 \
  --config path/to/pinned-winner.json --output fresh-report-directory
```

For repeated fresh-process evidence, add `--rounds 6` (or a positive multiple
of six). Every block covers all backend-order permutations, balancing position
and pairwise precedence. Each worker's full report/log is retained under
`round-NNN/`. The aggregate includes all process medians, maximum observed warm
peak allocation, and same-round workload ratios versus ordinary BF16. Those
ratios are descriptive observations, not confidence intervals. Sources,
configuration and contracts must remain unchanged across rounds.

New reports hash every Python source under `rtx/`, including kernel and runtime
helpers, before and after each worker. A source change during measurement is
an error, as is differing source provenance across backends or rounds. This
does not archive source contents or pin external compiler binaries. Preserve
the corresponding checkout/environment when sharing reports. Earlier September
17 reports retain their original smaller frontend-only hash set; they are not
retroactively upgraded to full source provenance.

`--mode outer_capture` measures the complete training pair and complete workload
under caller-owned CUDA graph replay for **all three** backends. Compilation,
capture and warmup are outside timing. Both captured outputs and every returned
gradient are compared with direct execution at original and changed X/weights;
the original inputs are restored before measurement. Captured resource owners
remain alive through measurement. The training-pair graph is released before
capturing the workload graph so two graph pools do not inflate its memory peak.

Isolated inference/training-forward/backward diagnostics remain direct in this
mode; `contract.timing_modes` records this per phase. Do not add these direct
phase times to explain a replayed end-to-end measurement. Graph-owned storage is
included in the workload memory report; it is not ordinary eager residency.

To reuse the explicitly named candidate from a paired experiment, pass
`--config paired-report.json --config-field challenger`; the default field is
`config`, with raw configuration objects also accepted. Source hash and selected
field are recorded so a comparison cannot silently substitute the default.

`--shape` is M,K,H,O; H is post-gating width. Mix counts are training pairs,
grad-enabled forwards, and inference forwards. This version requires at least
one training pair. Each public forward still computes just one MLP invocation.

## Contracts

- Measurement revision 2 clears reference-only cuBLAS workspaces in each
  disposable worker after numerical checking, before target rewarm/capture.
  Each backend then reacquires its own required workspace. The cleanup and
  released bytes are recorded; it is never performed by application launchers.
  This prevents independent-reference allocations from inflating the CuTe
  workload's live-memory peak. Allocator reserve can still contain freed
  reference blocks and must not be treated as required deployment storage.
  Revision-1 absolute warm-memory results below are historical and must not
  be combined with revision-2 results for capacity decisions.
- Fullgraph/static compilation with `max_autotune=True` and automatic CUDA
  graphs disabled. These are explicit comparator settings, not a claim to
  reproduce every AFRL compiler option.
- BF16 and recomputed BF16 use the same FP32 SiLU/product followed by BF16 cast.
  Inputs, parameters, outputs, and accumulated leaf gradients are BF16.
- Recipe correctness is independently checked for output and all gradients.
  MXFP8 uses the explicit quantized reference. BF16 divergence is recorded
  separately; passing the recipe check does not imply equality to BF16.
- Numerical-gate revision 2 checks candidate **and reference** finiteness,
  complete output/gradient coverage, finite nonnegative metrics and consistency
  of normalized RMS. The ordinary recipe gate remains 0.025 normalized RMS;
  references with RMS at most 1e-12 instead require maximum absolute error at
  most 1e-7, matching preflight's near-zero rule. Nonfinite references cannot
  pass through a NaN comparison. The revision is part of the comparison
  contract, so mixed old/new gate reports cannot be aggregated as one study.
- Timings include inference forward, training forward, backward measured on a
  freshly prepared forward graph, complete training pair, and complete mix.
  Phase medians must not be added or subtracted to claim an end-to-end result.
- The same MXFP8 configuration is used for all phases. This does not simulate
  separately selected teacher configs, teacher weights, or nested checkpoint
  activation lifetimes.
- Reports include first-training-pair wall time and allocator peak, warm
  resident and peak allocation, incremental warm peak, reserved memory, and
  repeated live-allocation samples. Fresh workers avoid cross-backend scratch
  contamination but not disk compiler-cache reuse.
- Resident allocation is measured, **not attributed resident scratch**.
  External/private allocation breakdown and full-model capacity remain open.

## Recommendation safety

`rtx.autotune.replacement.recommend_replacement` requires matching contracts,
valid evidence, and ordinary BF16 as the baseline. By default it does not accept
a slower candidate just because it saves memory. `ReplacementPolicy` requires
explicit `prefer_memory=True` with a bounded `max_slowdown` to permit that trade.
It returns a recommendation only; accepting another numerical backend is still
an explicit application decision.

The recommendation carries a `comparisons` entry for every backend, including
workload milliseconds, time ratio to BF16, warm peak allocation, and signed
bytes saved, plus the explicit `policy`. These descriptive measurements remain
available when the result is `unqualified`; they do not authorize promotion.
Negative bytes saved means greater allocation than the baseline.

The comparison CLI exposes the same policy:

```bash
python -m rtx.autotune.replacement_benchmark \
  --shape 11712,256,768,256 --mix 1,1,2 --rounds 6 \
  --config pinned-winner.json --prefer-memory --max-slowdown 0.25 \
  --minimum-speedup 0.02 --output NEW_REPORT_DIRECTORY
```

Here `0.25` permits at most 25% longer runtime **only for a lower measured
peak**, while `0.02` requires a 2% time reduction for speed-based eligibility.
Memory preference ranks eligible choices by peak, then time. Without
`--prefer-memory`, nonzero `--max-slowdown` is rejected before workers launch.
Defaults remain speed-oriented with a 1% minimum time reduction. The selected
policy is recorded in `decision.policy` for both single and balanced rounds.
These switches cannot make an unqualified report promotable, change numerical
precision, or install a runtime winner. Application acceptance remains explicit.

The current benchmark marks its reports unqualified for promotion: even with
repeated order-balanced fresh-process measurements it has not qualified contention.
It therefore returns `selected=null`, even if one diagnostic time is lower.
Do not manually flip that flag to manufacture promotion evidence.

## September 16 diagnostic evidence

### September 17 updated schedule

The final tile refresh is
`autotune_reports/replacement_recompute128_balanced_20260917/`, using
`recompute_tiles_20260917/m128n64a4b2s2.json` (`challenger` field). It repeats
the same six balanced backend orders, seven samples, M11712/K256/H768/O256,
mix `1,1,2`, and outer capture. All 18 workers pass output/all-gradient recipe,
changed-data replay, and live-allocation stability checks; worst recipe NRMS
across the workers is 0.0000826.

| Backend | F+B ms | Complete mix ms | Maximum warm peak MiB |
| --- | ---: | ---: | ---: |
| Ordinary BF16 | 0.554 | 1.021 | 121.13 |
| Recomputed BF16 | 0.674 | 1.176 | 121.13 |
| MXFP8, 128×64 recompute | 0.872 | 1.402 | 282.16 |

The median same-round MXFP8/BF16 workload ratio is **1.374** (range
1.370–1.376). The newer tile improves speed but does not change this harness's
memory disadvantage. Policy and signed per-backend memory savings are retained
in the decision. Status remains `unqualified`, selected replacement is null,
and no model or runtime cache was modified. To reproduce, use the command
below with this newer configuration path and a new output directory.

#### Preceding down-projection-only schedule

`autotune_reports/replacement_prequant_down_balanced_20260917/` reruns six
backend-order permutations (18 fresh workers) at M11712/K256/H768/O256 and
mix `1,1,2`, with caller-owned graph replay. It uses the explicit challenger
from `mlp_prequant_down_20260917/train.json`: prequantized gate/recompute/down,
prefetched derivative loads, and four-way FP32 dW splits. Seven samples per
worker; defaults and installed winners are untouched.

| Backend | F+B ms | Complete mix ms | Maximum warm peak MiB |
| --- | ---: | ---: | ---: |
| Ordinary BF16 | 0.554 | 1.020 | 121.13 |
| Recomputed BF16 | 0.674 | 1.176 | 121.13 |
| Improved MXFP8 | 0.944 | 1.475 | 282.16 |

The paired MXFP8/BF16 workload ratio is 1.444 (range 1.441–1.448), down from
the earlier 2.394 graph comparison. It remains slower and uses more memory in
this isolated captured workload; neither BF16 parity nor a memory-saving
replacement is established. All 18 workers passed recipe checks, changed-data
replay checks, and stable live-allocation checks. The report remains
`unqualified`, with no selected replacement, because contention qualification
is unresolved. Full-model activation lifetimes differ from this harness.

Reproduction:

```bash
.venv/bin/python -m rtx.autotune.replacement_benchmark \
  --shape 11712,256,768,256 --mix 1,1,2 --samples 7 --rounds 6 \
  --mode outer_capture \
  --config autotune_reports/mlp_prequant_down_20260917/train.json \
  --config-field challenger --output NEW_REPORT_DIRECTORY
```

### Historical baseline

`autotune_reports/replacement_smoke_20260916` contains a small aligned three-way
smoke run. `autotune_reports/replacement_pinned_11712_20260916` uses AFRL's pinned
M11712/K256/H768/O256 training winner, recording its configuration and source
hash. Both used Torch 2.13/CUDA 13.2 on the 5070 Ti. NVML still reported a driver
library mismatch, so these are diagnostic results, not uncontended qualification.

For the tall shape with mix `1,1,2`:

| Backend | F+B ms | Complete mix ms | Warm peak MiB |
| --- | ---: | ---: | ---: |
| Ordinary BF16 | 0.603 | 1.062 | 168.44 |
| Recomputed BF16 | 0.704 | 1.168 | 169.28 |
| Pinned MXFP8 | 2.470 | 4.661 | 224.25 |

All results were finite; worst training recipe NRMS was 0.000115 for MXFP8 and
0.000061 for the BF16 paths. MXFP8's worst BF16-relative NRMS was about 0.0650.
Post-warmup live allocations were stable. The isolated peak does not predict
activation savings across a checkpointed multi-layer application.

This establishes a working comparison harness and reproduces the direction of
the reported regression. It is not a kernel speedup or completed release gate.

## September 17 order-balanced comparison

`autotune_reports/replacement_balanced_11712_20260917/` contains six rounds
(18 fresh workers) on the same tall shape and mix `1,1,2`. MXFP8 uses the
explicit `challenger` from
`prequant_gate_20260916/train_outer_capture_prequant_recompute.json`: separate
gate quantization plus prequantized recomputation, not the original pinned
winner. All six backend permutations were executed; each worker retained its
own numerical checks, phase samples, memory evidence and source/config hashes.

| Backend | Median F+B ms | Median complete mix ms | Maximum warm peak MiB |
| --- | ---: | ---: | ---: |
| Ordinary BF16 | 0.596 | 1.053 | 169.28 |
| Recomputed BF16 | 0.701 | 1.179 | 169.28 |
| Improved MXFP8 | 1.800 | 2.764 | 227.59 |

MXFP8's same-round complete-workload ratio to BF16 ranged from 2.595 to 2.640
(median 2.604). This is still a substantial replacement-level regression,
despite the gains against the older RTX schedule. These selected-direct
compiled timings must not be substituted for the earlier outer-graph replay
measurements. The isolated memory peak provides no case for a memory-saving
replacement here; multi-layer activation savings need a separate workload.

Recipe output/all-gradient normalized RMS remained below 0.000115; BF16-relative
normalized RMS remained about 0.065. Every worker passed finiteness and stable
post-warmup live-allocation checks. The NVML mismatch still prevents contention
qualification, so the aggregate correctly returns `selected=null`. No runtime
winner or application precision was changed.

## Equivalent caller-owned graph replay

`autotune_reports/replacement_graph_balanced_11712_20260917/` repeats the same
six-round experiment with `--mode outer_capture`. All three backends replay the
complete training pair and mixed workload, after successful original/changed
X-and-weight replay checks. Results across 18 fresh workers:

| Backend | Median graph F+B ms | Median graph mix ms | Maximum warm graph allocation MiB |
| --- | ---: | ---: | ---: |
| Ordinary BF16 | 0.554 | 1.019 | 121.13 |
| Recomputed BF16 | 0.674 | 1.176 | 121.13 |
| Improved MXFP8 | 1.465 | 2.439 | 255.16 |

The same-round MXFP8/BF16 mixed-workload ratio is 2.387–2.399 (median 2.394).
Removing host dispatch reduces the MXFP8 absolute latency but does not close
the replacement gap. This supports investigating device scheduling/conversion
and backward work next; it does not itself attribute stalls to a specific
instruction. Do not infer full-model memory savings from these isolated graphs.

All workers passed recipe checks (worst normalized RMS 0.000115), changed-input
graph equivalence and stable live-allocation checks. NVML remained unavailable,
so these repeated results are diagnostic, not uncontended release qualification.
