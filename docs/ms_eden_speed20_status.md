# NVFP4 backward speed target — current handoff

Historical October 27 snapshot. See [the consolidated research index](nvfp4_research.md)
for later integration and training evidence; job/status statements below are archival.

Updated 2026-10-27. **Target not achieved.** Stable kernel defaults unchanged;
no research winner promoted and no commit/push performed by this work.

## Target and evidence

Complete compiled MS-EDEN dX+dW, including preparation and fresh RNG, must reach
at least 1.20x matched stable MXFP8 backward on representative decoder shapes.
Keep matched NVFP4 forward, original BF16 sources and FP32 arithmetic, numerical,
graph and memory gates. A qualified schedule must then translate to training.

Latest qualified TMA-plus-dual-finalizer work on the5060 reaches **1.11122x**
on `4096x1536x512` and **1.18068x** on `4096x2048x512`.
The narrow `4096x512x512` remains a provisional regression with unstable timing.
These are complete backward comparisons, not model-training speedups. Detailed
per-run values and gates are below; no representative-suite1.20x claim follows.

The 17-trial joint producer-grid study and confirmations passed their numerical,
RNG, current-buffer replay and live-memory checks. Shared-dY input reuse, inline
scale finalization and bounded direct reciprocal sharing did not qualify for
promotion. More same-space tuning has not demonstrated the requested gain.

## Current dependency

Latest resume: connectivity and counter access are restored. Windows now reports
`RmProfilingAdminOnly=0x0`. The GPU was idle before capture. A clean single-kernel
Nsight Compute run succeeded and same-input replay remained byte-exact. The older
permission/connectivity failures below are historical, not current blockers.

For the retained `4096x1536` transpose preparation producer (128 threads,
256 blocks, VPL16, async stage1), basic counters show 96 registers/thread,
41.67% theoretical and 37.07% achieved occupancy. A fresh detailed capture has
zero local/shared spilling requests, 34.87% scheduler cycles with no eligible
warp, and per-active-warp stall percentages: MIO throttle 15.93%, fixed-latency
wait 15.13%, short scoreboard 10.41%, long scoreboard 6.27%, barrier 3.66%,
math-pipe throttle 1.66%. These are diagnostic ratios, not additive wall-time
shares. Traffic was 12.62 MB DRAM reads and 24.87 MB L2; reported zero DRAM writes
does not mean no output stores (writes may remain cached during capture).
Profiled duration ~95.8us is not a new ordinary-timing result or speedup claim.

Next: attribute MIO/short-scoreboard pressure to the producer's shared-memory
loads, shuffle/reduction and conversion instructions before changing its layout
or buffering. Compare affected variants with these counters, then confirm the
complete backward; register reduction alone already failed to establish a win.
Raw reports and replay checks are under `counters-restored/` in the local report
root. The intermediate `producer-detail` capture hit an existing JSON-file guard
after correctness checks; the separate `rtx-mseden-stalls-20261027.MvZTag` capture
completed cleanly and is the detailed-counter reference.

### Counter-driven XOR layout experiment

The retained transpose loader had 602,678 shared-load bank conflicts and 799,286
wavefronts for 196,608 load instructions. A research-only pair-preserving XOR
layout (`rtx/kernels/ms_eden_bank_layout.py`) reduced conflicts to 8,560 and
wavefronts to 205,168. MIO throttle fell from 15.93% to 10.99%, but registers rose
from 96 to 123 and achieved occupancy fell from 37.07% to 29.35%. Instruction
count increased from 9,057,660 to 9,169,256. Counter captures are isolated-kernel
diagnostics, not complete-backward speedup measurements.

Ten tests passed on the 5060: address bijection/pair alignment/bank ownership plus
three shapes x three buffer counts with exact packed/scales/partials comparisons
and changed-input graph replay. The complete `4096x1536x512` comparison passed
FP32-reference, fresh-RNG, current-buffer, and live-memory guards (zero growth).
All AB/BA timing windows were stationary:

| Variant | MXFP8 us | FP4 us | Paired MXFP8 / FP4 |
|---|---:|---:|---:|
| XOR layout | 266.412 | 271.609 | 0.982145 |
| Fresh unchanged FP4 control | 268.824 | 266.986 | 1.006190 |

These are separate sessions, not a direct paired XOR/control confidence interval.
No promotion. Next experiment: reduce address-generation live ranges to retain
conflict reduction without the register/occupancy regression. Artifacts are in
`bank-layout/`; isolated remote snapshot is
`/home/achiya/rtx-mseden-bank-layout-20261027.0aetcL`. The initial full comparator
attempt lacked a test helper; supplying the snapshot's existing helpers resolved
that harness issue. `xor-retry.json` and `control.json` are completed runs.

Follow-up address hoisting computes one base for even rows and one for odd rows;
the remaining accesses use constant offsets. The mapping and arithmetic are
unchanged. All 19 CPU/GPU tests passed. Isolated counters show 96 registers/thread
(down from 123), 37.27% achieved occupancy, 8,984 shared-load conflicts and 205,592
load wavefronts. Complete backward finished in isolated snapshot
`/home/achiya/rtx-mseden-bank-hoist-20261027.QgunZX`: MXFP8 267.610us versus
candidate 268.133us, paired ratio 0.998264 (within-session 95% interval
[0.996456,0.999989]). All AB/BA windows were stationary. Same-draw dX/dW were
bitwise equal for Gaussian/outlier/zero inputs, changed-input graph checks passed,
and live allocation growth was zero. This recovers the first XOR variant's
regression but does not establish a win over retained FP4 or reach the goal.
Artifacts are in `bank-hoist/`. No variant is promoted.

Compact row-loop follow-up: all 28 tests passed (including the prior variants).
Registers decreased to 80/thread, achieved occupancy rose to 42.06%, and
shared-load conflicts were 6,823. Instruction count increased to 9,459,058.
The first full-backward process in
`/home/achiya/rtx-mseden-bank-compact-20261027.EsQlUz` remained live for almost
five minutes with one busy CPU core and no finalized JSON. Its log stopped after
the existing autograd stream warning; that does not establish the cause.
Read-only GDB attachment was denied. Only that verified experiment (PID3675) was
explicitly terminated; no host security settings were changed.

A fresh diagnostic rerun completed normally before its 45-second stack-dump
timer fired: MXFP8 271.128us versus compact FP4 261.501us, paired ratio 1.034261
(within-session 95% interval [1.032033,1.036577]), all AB/BA windows stationary.
Same-draw dX/dW, current-buffer graph, fresh-RNG and live-memory gates passed
(zero live growth). This is a modest single-session win, not 1.20x qualification;
the first stall's cause remains unknown. `compact-diagnostic.json` is the valid
result. An additional decoder1536 confirmation reached 260.005us versus
270.176us MXFP8, ratio 1.038926 with all AB/BA windows stationary. The subsequent
decoder512 process stalled in warmup synchronization; repeated Python stack
dumps located it at `compare_ms_eden_pipeline.py`'s warmup device synchronize.
Only that experiment and its queue were terminated, so decoder2048 was not run.
The unchanged control then reproduced the same synchronization stall and was
explicitly terminated. This is not evidence of a compact-layout-specific fault.

An opt-in diagnostic `--warmup-sync-each` synchronizes between baseline and
candidate graphs during warmup only. The unchanged decoder512 control completed
under it, but one candidate BA window was unstable (ratio 0.952694); no qualified
performance claim follows. The compact case is being checked under the same
protocol in `/home/achiya/rtx-mseden-replay-diagnostic-20261027.BBCfim` completed:
MXFP8 101.178us versus compact FP4 107.276us, ratio 0.942708, but both candidate
order windows were unstable. It is not qualified performance evidence.
Timed windows remain unchanged, the diagnostic flag is recorded in results,
and default warmup behavior remains unchanged. This is not a proven runtime fix
or evidence that arbitrary interleaved graph replay is reliable. Fourteen
focused CPU comparator tests passed. No public dispatch/default changes were made.

Bounded 15-second replay-only diagnostics in fresh processes then passed:
stable/stable 58,847 pairs; stable/default-FP4 54,056 pairs;
stable/selected-FP4 59,873 pairs; selected mixed replay after the exactness
preflight 61,460 pairs. All used alternating backward graphs with one sync per
pair, not the sync-each workaround, and passed final independent FP32 gradient
and zero-live-growth gates. These limit attribution: neither mixed policies,
the selected schedules nor preflight order reproduced the stall in isolation.
The earlier synchronization hangs remain unresolved; do not claim a production
fix. The diagnostic script is `benchmarks/probe_ms_eden_graph_interleave.py`;
run it with an external timeout because Python deadlines cannot bound CUDA waits.
A bounded decoder2048 complete comparison finished with original warmup
semantics: MXFP8 418.258us, compact FP4 376.729us, ratio 1.109078, within-session
95% interval [1.106736,1.111402]. All AB/BA windows were stationary; exact-draw,
changed-input graph, fresh-RNG, numerical and zero-live-growth gates passed.
This does not beat the historical retained-policy ratio (~1.126) on that shape;
those are different sessions and do not establish a direct regression either.
The 1.20x target is still unmet and no research variant is promoted. All runs
from this diagnostic sequence have exited; no experiment remains active.

### Packed Rademacher sign-cache probe

`ms_eden_packed_signs.py` is research-only and explicitly requires finite BF16
inputs and signs in {-1,+1}. It stores 16 cached signs as bits and applies their
sign bits before the inherited FP32 Hadamard/quantization/correction. Stable
producer code and public dispatch are unchanged. Exhaustive finite-BF16 CPU
sign equivalence plus 12 GPU cases (two shapes, three buffering depths, rolled/
unrolled rows, changed-input/sign graph replay) passed. Two additional
register-ceiling exactness cases passed.

The anticipated resource win did not materialize: uncapped packed signs still
use 80 registers/thread, 42.00% achieved occupancy and zero local spill sectors;
executed instruction count increased from compact layout's 9,459,058 to
9,605,490. Requested PTXAS register ceilings of 72 and 64 also yielded 80-register
kernels with identical instruction counts and zero spills. Do not treat the
requested ceilings as achieved resources or their profiled durations as a win.
Installed compiler source exposes `min_blocks_per_mp` as an `nvvm.minctasm`
launch hint. Unlike the PTXAS ceiling, this affected the generated resources:
seven-block hint -> 72 registers, 47.05% achieved occupancy; eight-block hint ->
64 registers, 51.97% achieved occupancy. Both had zero local spill sectors and
passed exact-output/changed-buffer graph tests. Executed instruction counts rose
to 9,937,266 and 10,106,226 respectively. Complete-backward confirmations in
`/home/achiya/rtx-mseden-occupancy-20261027.pWn0LX` are terminal. Hint7 hit the
recurring warmup device-synchronization stall and was terminated by its explicit
120-second timeout (status124; no timing result). Hint8 passed the full gates:
MXFP8 277.780us versus candidate 267.827us, paired ratio 1.038296, within-session
95% interval [1.036258,1.040315], all AB/BA windows stationary and zero live growth.
This does not establish an improvement over the earlier compact-layout ~1.039x
result. No variant is promoted.
No full-backward speed claim or promotion follows from this sign-cache probe.
Snapshots: `/home/achiya/rtx-mseden-packed-signs-20261027.vBHoTg` and
`/home/achiya/rtx-mseden-sign-cap-20261027.uaDxIy`. All these runs have exited.

## Hardware and next gate

### Compact-layout buffering interaction

The research driver now exposes `compact2` and `compact3`: only eligible
transpose producers use two or three stages with the compact XOR layout.
Stable dispatch and numerical policy are unchanged. On the idle 5060,
`4096x1536x512`, complete compiled dX+dW with fresh RNG measured:

| Buffers | MXFP8 us | NVFP4 us | Paired baseline/candidate | Within-session 95% interval |
|---|---:|---:|---:|---|
| 2 | 269.023 | 262.780 | 1.025773 | [1.024163, 1.027353] |
| 3 | 264.697 | 266.777 | 0.992948 | [0.991411, 0.994637] |

Both 24-round comparisons passed all AB/BA stationarity gates, same-draw
Gaussian/outlier/zero dX/dW checks, changed-buffer graph replay, fresh RNG,
independent FP32-reference numerical gates and zero live allocation growth.
Neither reaches 1.20x or establishes an improvement over the earlier one-stage
~1.039x result (separate sessions, not a direct inter-variant comparison).
Three stages are slightly slower than their matched MXFP8 baseline. Deeper
buffering therefore does not close the gap on this shape; no promotion.

Artifacts: `compact-stages/` locally and isolated remote snapshot
`/home/achiya/rtx-mseden-compact-stages-20261027.H90HkI`. An initial invocation
from the wrong script location failed the source-hash lookup before GPU work;
the retained log documents it, and both proper module invocations completed.
CPU address-layout test passed; GPU producer-stage equivalence had already
passed for stages 1/2/3, and these runs add full-backward gates. All workers
from this comparison have exited.

The 5060 remains the only authorized GPU for experiments. Do not use the 5070,
local AFRL GPUs, or occupied work5090. The requested read-only 5090 access check
verified authenticated sudo/Docker listing; no containers were modified and
that verification did not authorize GPU work there.

### Compact occupancy without packed signs

Research-only `min_blocks_hint=7/8` now also applies to the compact XOR loader
with ordinary FP32 sign multiplication, isolating resource control from packed
sign decoding. Two GPU exact-producer tests passed (packed bytes, scales,
partials, changed-source/sign graph replay). CPU benchmark/tuner/sign regression
selection passed 30 tests with 16 GPU-only cases skipped; layout test passed.

On `4096x1536x512`, complete backward with hint7 measured MXFP8 262.659us,
NVFP4 254.139us, paired ratio 1.029442, within-session interval
[1.025668,1.032518], all AB/BA windows stationary. Hint8 measured 273.187us
versus 262.693us, ratio 1.040370, but candidate BA relative range 0.107867
exceeded the 0.1 gate: that timing is unqualified. Both runs passed numerical,
same-draw, changed-buffer, fresh-RNG and zero-live-growth checks. Neither
demonstrates the target or improvement over earlier compact results.

Hint8 single-producer counters confirm 64 registers/thread, 52.196% achieved
occupancy, zero local load/store sectors and 9,959,794 executed instructions.
This is fewer instructions than packed signs with the same hint (10,106,226),
but still more than unhinted compact (9,459,058); occupancy alone does not
translate into the needed end-to-end gain. No promotion or training run follows.
Artifacts: `compact-occupancy/`; remote snapshot
`/home/achiya/rtx-mseden-compact-occupancy-20261027.IxKlcr`. All workers exited.

Earlier counter-permission denials and SSH timeouts are resolved. Historical
logs remain in `joint-grids/`; current successful counter captures are described
above. No host profiling/security settings were changed by the agent.

### Fused Philox with compact preparation and paired streams

The full pipeline benchmark now accepts the existing opt-in `rng_mode=philox`
and records it in both schedules; original-BF16/direct-reader requirements and
fresh-RNG checks remain mandatory. Explicit RNG remains the default. The
fixed-uniform preparation-only validator rejects Philox, rather than claiming
to have exercised fused RNG. The whole-gradient exact probe supports explicit
uniforms or shared per-leg Philox seed tensors and changes those buffers during
graph replay. This changes research coverage, not runtime policy or defaults.

Independent GPU qualification passed 16 uniform/finalizer checks (both operand
streams, packet sizes 1/4 and row-major/native layouts) and eight same-draw
gradient cases across zero/tiny/normal/large magnitudes. The initial gradient
tests lacked a snapshot helper; after copying that independent-reference file,
all eight passed. CPU harness/topology/RNG tests passed 227 cases in 2.33s;
the additional fixed-uniform rejection test passed in a 15-case harness rerun.

Complete compact-loader, operand-paired `4096x1536x512` comparisons:

| Philox packet | MXFP8 us | NVFP4 us | Paired ratio | All order windows stationary |
|---|---:|---:|---:|---|
| 4 | 263.116 | 274.706 | 0.960102 | yes |
| 1 | 273.350 | 267.252 | 1.021127 | no |

Packet4 within-session 95% interval is [0.957404,0.962606]; it loses to matched
MXFP8. Packet1 timing is unqualified. Both passed same-draw seeded dX/dW,
changed-buffer, fresh-RNG, FP32 numerical and zero-live-growth gates. The seeded
stream is not Torch's explicit-uniform bitstream; independent tests establish
same-draw equivalence using its own CPU Philox oracle. No speed promotion.

Artifacts: `compact-philox/`; snapshot
`/home/achiya/rtx-mseden-compact-philox-20261027.9oof7D`. All workers exited.
This closes another previously untested composition, but does not meet 1.20x.

### GEMM schedules with the compact preparation path

The existing first-hit tuner/residual store now has a `compact_gemms` research
portfolio: all nine dX/dW M-atom pairs from {2,4,8}, plus stage3 on either or
both GEMMs. Preparation, scales, original-BF16 sources and RNG policy are fixed.
Five portfolio/real-tuner CPU checks passed; combined adapter regression passed
20 tests. The remote snapshot initially contained an old adapter lacking the
strict order-quality argument; updating that isolated adapter and starting
`study-v2` preserved the initial failed coordinator log.

All 12 full-backward trials completed on 5060, `4096x1536x512`, in 211 seconds.
Each passed same-draw Gaussian/outlier/zero checks, compiled current-buffer and
fresh-RNG guards, FP32-reference numerical gates and zero live growth. Eleven
passed every AB/BA stationarity gate; dW-only stage3 failed and was ineligible.
Screening used eight paired rounds and five seconds of warmup per candidate.

Stationary paired MXFP8/candidate ratios ranged from 1.008626 to 1.053315.
The best paired ratio was both GEMMs at stage3 with M-atoms8/8. The tuner,
which minimizes candidate milliseconds rather than ratios across separate
processes, selected M-atoms2/4 at stage2: 257.562us and paired ratio1.043048.
This is not a confirmed inter-variant ranking. The retained control scored
1.034386; no candidate approaches1.20x. No schedule is promoted and no training
run is justified by this screen. It rejects a large missed win within this
specific warp-geometry/buffering portfolio, not the entire GEMM search space.

Artifacts: `compact-gemms/study-v2/` contains residuals and per-worker source,
config, guard and timing records. Remote snapshot:
`/home/achiya/rtx-mseden-compact-gemms-20261027.fW3D3S`. All workers exited.

### Exact FP32 cross-lane butterfly FMA

Historical ordinary SASS uses a sign-select (`FSEL`) followed by `FADD` for
each cross-lane butterfly. Research `butterfly_fma=True` in the compact loader
replaces that pair with explicit `fma.rn.f32(sign,own,other)`, with sign exactly
+/-1. For finite FP32 operands this preserves the rounding operation; no
narrowing, approximate reciprocal or correction reassociation is introduced.
Register-local butterflies and all scale/correction code are unchanged.

Four GPU cases passed byte-exact packed/scales/partial checks and changed-buffer
graphs at 128x128 and4096x1536, stages1/2, including zero, outliers, exponents
[-100,100] and arbitrary non-unit FP32 signs. Full4096x1536x512 backward passed
same-draw, numerical, fresh-RNG/current-buffer and zero-live-growth gates:
MXFP8 270.629us, candidate263.443us, paired1.027971 with within-session interval
[1.026047,1.029878], all AB/BA windows stationary. No improvement over earlier
compact results is established; no promotion or training rerun.

Counters validate the intended mechanism: executed producer instructions fell
from9,459,058 to8,890,738 (~6.0%), registers remain80, achieved occupancy41.904%,
and local load/store sectors remain zero. Lower instruction count alone did
not deliver the end-to-end target. Artifacts: `butterfly-fma/`; isolated snapshot
`/home/achiya/rtx-mseden-butterfly-fma-20261027.nQDn45`. All workers exited.

### Compact critical-path capture and enqueue-order confirmation

`profile_backward_subparts` can now scope the compact research producer around
the exact selected schedule, with compiler combo settings and source hashes
recorded. Five existing profile-driver tests passed. Ten warmed complete
backward replays were captured with Nsight Systems, excluding compilation;
raw report/SQLite/guard JSON are in `critical-path/`. For replays2..10, medians
relative to the first graph GPU kernel were:

| Work | Start us | End us | Duration us |
|---|---:|---:|---:|
| dX W producer | 7.329 | 21.281 | 14.049 |
| dW dY producer | 15.425 | 82.469 | 67.140 |
| dW X producer | 49.987 | 91.878 | 41.922 |
| dX dY producer | 80.357 | 138.761 | 58.883 |
| dW GEMM | 140.424 | 179.531 | 39.010 |
| dX GEMM | 177.387 | 218.382 | 40.323 |

These are independently summarized profiled distributions, not additive wall
times or ordinary latency. Late dX-row admission motivated two fresh complete
backward enqueue-order tests with selected M-atoms2/4 and compact preparation:

| Preparation order | MXFP8 us | NVFP4 us | Paired ratio |
|---|---:|---:|---:|
| dx_a,dw_a,dx_b,dw_b | 266.105 | 269.866 | 0.986338 |
| dx_a,dx_b,dw_a,dw_b | 262.619 | 263.126 | 0.997614 |

Both24-round experiments passed all AB/BA stability, numerical, same-draw,
fresh-RNG/current-buffer and zero-live-growth gates. Neither improves on the
retained order. This argues against treating the observed late start as free
removable latency: advancing it can contend with other preparation. No default
changed. Research CLI rejects combining an order override with a conflicting
dependency-frontier experiment; two focused CPU guard/restoration tests passed.

Remote captures: `/home/achiya/rtx-mseden-critical-path-20261027.YCA0ka`.
Ordering trials: `/home/achiya/rtx-mseden-compact-order-20261027.cSKP3V`;
local artifacts `compact-order/`. All workers exited.

### VPL32 structural transpose producer

Research `ms_eden_group32.py` moves the fifth H128 butterfly bit stage from
cross-lane exchange into registers, preserving the seven-stage association.
Each 32-value lane fragment is split into two independent 16-value scale groups
using the retained FP32 quantization/correction implementation. No 32-value
scaling or transformed global-memory intermediate is introduced. A separate
pair-preserving XOR layout remains conflict-free for this lane ownership.

Five CPU/GPU tests passed: layout bijection/alignment/bank ownership and four
producer cases (128x128/4096x1536, stages1/2), byte-exact packed values, scales
and partial maxima including wide exponents and arbitrary FP32 signs, plus
changed-buffer graph replay. Complete4096x1536x512 backward passed same-draw,
independent numerical, fresh-RNG/current-buffer, zero-live-growth and all AB/BA
stationarity gates: MXFP8 266.489us, candidate269.991us, paired0.985511,
within-session95% interval[0.982822,0.988162]. No promotion.

Counters show the tradeoff: instructions8,660,338 versus compact VPL16's
9,459,058, but registers120 versus80 and achieved occupancy29.225% versus
~42%; zero local spill sectors. Wider work ownership reduces instructions but
increases live state. A lower-register sign-loading variant is the next bounded
test, not an assumption of a win. Artifacts: `group32/`; isolated snapshot
`/home/achiya/rtx-mseden-group32-20261027.rcuuH4`. All workers exited.

Two bounded Nsight Systems captures of unchanged decoder512 control completed
successfully, with node-level and lower-overhead graph-level tracing. Neither
reproduced the intermittent synchronization stall. Raw captures and completed
result JSONs are preserved locally in `replay-timeline-node/` and
`replay-timeline-light/`; profiled timings are not speedup evidence.

The node trace attributes 115,052 host-launched kernels to PyTorch's Int64 fill
kernel (61.578ms aggregate GPU duration), versus 115,057 graph launches. This
is consistent with graph RNG-offset updates, not evidence of eager quantization
escaping capture. Installed `CUDAGeneratorImpl.h` documents the replay offset
fill. CPU launch cost and GPU kernel duration must not be conflated; do not
remove fresh RNG to eliminate this work.

An expanded reliability probe alternated backward and total graphs for both
policies, completing 35,969 iterations over its 20-second replay window with
independent FP32 gradient checks and zero live allocation growth. The targeted
performance driver explicitly selects backward-only, so adding total replay
does not explain its historical stalls. No synchronization workaround or
production behavior was changed. The probe now exposes `--scopes` and checks
gradients for each requested scope. Occasional faulthandler samples during this
successful run are not themselves evidence of a hang.

Next: use these healthy traces to isolate an actually stalled run before
attributing the problem to kernel scheduling. A timing win must
survive representative-shape and replay reliability checks before promotion or
matched short model training. No experiment remains active from this sequence.

### VPL32 reloaded-sign result (retrieved and verified)

The completed `group32_reload` experiment does not meet the speed target.
At 4096x1536x512 on the 5060, complete compiled backward including preparation
and fresh RNG measured MXFP8 268.243us versus candidate 270.654us. The paired
ratio is 0.991929, within-session 95% interval [0.989072, 0.994768], across
24 rounds; both AB and BA orders pass all stationarity checks.

Reloading signs achieved its resource objective: 80 registers/thread versus
cached VPL32's 120, achieved occupancy 42.216% versus 29.225%, and zero local
load/store spill sectors. Instructions were 8,807,784. Recovering occupancy
did not translate into a complete-backward win, so register pressure alone
does not explain the remaining gap. These counters describe the isolated
producer; they are not an end-to-end utilization measurement.

Same-draw dX/dW outputs were bitwise equal on Gaussian, outlier and zero inputs;
changed-buffer graph replay passed. Independent FP32 relative RMS errors were
approximately 0.1372/0.1370 for candidate dX/dW, versus 0.0376/0.0377 for MXFP8,
within the existing separate algorithm gates. Live replay growth was zero.
The two focused reloaded-sign producer GPU tests also passed.

Raw report, timing JSON, and logs are preserved in `group32-reload/` under the
local report root. Remote snapshot:
`/home/achiya/rtx-mseden-group32-reload-20261027.7Qsxgt`.
Process inspection confirmed the profiling job has exited. No candidate was
promoted and stable defaults remain unchanged. Next investigations should
target preparation work/traffic on the complete path rather than assuming
another occupancy improvement alone will close the gap. The representative
shape 1.20x target and translation to matched training remain unmet.

### Why reloading did not recover VPL32 performance

Three fresh, bounded NCU captures of the validated 4096x1536 transpose producer
completed on the idle 5060. All outputs remained byte-exact against the retained
producer. Each capture used the same input seed, grid and stage count.

| Variant | DRAM read MB | L2 MB | MIO throttle % | Long scoreboard % |
|---|---:|---:|---:|---:|
| Compact VPL16, cached signs | 12.607 | 24.471 | 10.079 | 5.710 |
| Compact VPL16, reloaded signs | 12.606 | 25.226 | 17.982 | 6.213 |
| VPL32, reloaded signs | 12.616 | 32.357 | 16.842 | 8.438 |

These are per-active-warp diagnostic stall ratios, not additive wall-time shares.
Profiled durations were respectively 92.832/97.664/102.208us; they are not
ordinary backward timing or an independently confirmed speedup comparison.
Zero measured DRAM writes does not imply no output stores: writes can remain
cached within the capture window.

The important new evidence is that reducing live registers trades against more
cache traffic and instruction-pipeline pressure. VPL32 reload has about 32%
more L2 traffic than compact cached-sign VPL16, while useful DRAM input reads
are unchanged. The source contains scalar sign loads and two separate 16-value
quantization/store fragments per VPL32 lane. Attribution of the extra sectors
between sign loads and packed/scales stores is not yet measured; do not call
it a bandwidth limit or assume all extra traffic is sign loading.

Next bounded experiment: collect operation-specific global load/store sectors
for these same captures, then optimize the responsible ownership/access pattern.
Do not promote VPL32 or spend another broad geometry sweep on it based solely
on its lower instruction count. Raw `traffic-*.ncu-rep`, replay manifests and
logs are in `group32-reload/`. All three profiler processes have exited.

### VPL32 store-sector attribution and combined-store experiment

Operation-specific profiles localized the VPL32 traffic increase. Global-store
sectors were 246,784 for compact VPL16 (both sign modes), versus 492,544 for
VPL32 reload; L2 write sectors were 169,325/184,501/405,305 respectively.
Global-load sectors were 524,288/1,966,080/1,179,648. Thus VPL32's extra L2
traffic is primarily associated with stores, not simply more sign loads.

An opt-in `combined_stores` research option now quantizes the two 16-value
halves into local register fragments, then writes one 128-bit packed fragment
and one 32-bit pair of BF16 temporary scales per lane. All original FP32
Hadamard, normalization, correction and maximum arithmetic is retained.
The final outer-scale/finalization policy is unchanged. It does not introduce
a new global intermediate or change production dispatch.

Four GPU tests passed (128x128 and4096x1536, cached/reloaded signs), including
wide exponents, arbitrary signs and changed-buffer graph replay; two CPU
ownership/alignment checks passed. The initial compile correctly rejected the
old 4-byte fake packed-pointer alignment for 128-bit stores; the combined mode
now declares its actual 16-byte output allocation contract. Failed logs remain
preserved rather than replacing the failure with a success claim.

The intended mechanism worked: global-store sectors dropped from 492,544 to
148,480, L2 write sectors from405,305 to148,449. Registers stayed80/thread,
achieved occupancy41.749%, and local spill load/store sectors were zero.
However complete4096x1536x512 backward measured MXFP8 275.118us versus
candidate271.646us, paired1.009825 (95% within-session interval
[1.005733,1.013331],24rounds). All per-order stationarity, exact-gradient,
independent numerical, fresh-RNG/current-buffer and zero-live-growth gates
passed. This does not establish superiority over the prior FP4 winners or
reach1.20x. No promotion or matched training run is warranted yet.

New research modes are `group32_stores` and `group32_reload_stores`; profiler
flag is `--combined-stores` with `--variant group32`. Artifacts live in
`group32-stores/`, source snapshot
`/home/achiya/rtx-mseden-group32-stores-20261027.sApAQb`.
Attribution reports remain in `group32-reload/sectors-*.ncu-rep`.
All launched experiments exited. Further work must target remaining preparation
arithmetic or complete-path scheduling rather than claiming that fewer memory
sectors alone establishes an end-to-end win.

### Compact VPL16 x 64-bit store interaction

The dependency/bank-layout comparison driver previously forced 32-bit stores,
preventing experiments with the schedule's existing64-bit store option. It now
exposes `--grouped-store-bits {8,16,32,64}` (default32 for historical continuity).
The compact research compiler now declares packed output alignment from the
effective store width, matching the canonical compiler's allocation contract.
This is not a new production axis: canonical64-bit stores had earlier been
qualified, but their composition with compact XOR loading was missing.

Two exact GPU tests passed (128x128 and4096x1536; wide exponents, arbitrary
signs and changed-input replay), plus200 focused CPU tests with45 CUDA skips.
The two complete public backward runs both passed all24-round AB/BA stationarity,
same-draw dX/dW equivalence, numerical/fresh-RNG/current-buffer and zero-growth
checks:

| Shape | MXFP8 us | Compact64 FP4 us | Paired speedup | Within-session95% interval |
|---|---:|---:|---:|---|
| 4096x1536x512 | 266.016 | 257.226 | 1.036050 | [1.033519,1.038511] |
| 4096x2048x512 | 403.029 | 355.927 | 1.130892 | [1.128586,1.133318] |

The larger-shape ratio is encouraging but below1.20. These are matched
MXFP8/candidate comparisons, not a direct paired proof that64-bit stores beat
compact32. Stable defaults remain unchanged; no new model training is justified
by this evidence yet. Artifacts: `compact64/`; isolated remote snapshot
`/home/achiya/rtx-mseden-compact64-20261027.uphiJ1`. Both workers exited.

### Compact64 four-producer concurrency retuning

Added the `compact64_grids` research space to the existing complete-backward
autotuner. It composes compact XOR loading and64-bit stores with the same
pairwise-balanced17-candidate producer-grid portfolio (64/128/256/512 blocks
for each of dX-A/dX-B/dW-A/dW-B). Six focused CPU tests passed, including
assertions that both serialized schedules retain64-bit stores. No production
dispatch or automatic winner installation was added.

The bounded5060 search on4096x1536x512 completed in447s:15 candidates finished
all correctness/replay/RNG/memory and separate AB/BA timing checks; two hit the
90-second isolated-worker deadline and remain runtime failures, not measured
slow configurations. They were not retried or silently omitted from residuals.

The absolute-time selection used grids[512,256,128,64],256.855us and paired
MXFP8/candidate1.038313. Control[256,256,256,256] measured257.665us,1.033145.
The largest paired ratio was1.050432 at[128,512,256,64], but its absolute time
was266.004us. Cross-session clocks therefore prevent treating the ranking
difference as a confirmed inter-candidate benefit. Other stationary ratios
ranged down to0.955556. No candidate approached1.20, so no promotion or model
training followed. The sub-percent absolute-time selection is not a confirmed
improvement over control.

All per-trial manifests, logs, result JSONs, failures and tuning residuals are
preserved in `compact64-grids/study/`; remote snapshot
`/home/achiya/rtx-mseden-compact64-grids-20261027.eCcGT8`. Coordinator exited0.
This closes the missing compact64/grid interaction but provides no basis for
another same-space sweep: the next experiment needs a different mechanism.

### Compact64 instruction attribution: staging is the next structural target

A bounded NCU InstructionStats/SourceCounters capture completed with byte-exact
producer replay. It records9,446,770 executed warp instructions, including
1,474,560 FFMA,1,449,984 FADD,614,400 FSEL,594,944 SHFL and503,808 FMUL.
No local/shared spilling requests were reported. The profiler's generic FMA
speedup estimate is not an end-to-end prediction and does not authorize
reassociating the FP32 correction math.

The largest individual not-issued sampling location (80 samples) is a USHF
immediately after `DEPBAR.LE SB0,0`; all80 are long-scoreboard samples. This
is evidence of waiting for outstanding async copies, not expensive integer
shifts. Aggregate sampled attribution also includes FFMA106, LDS56 and SHFL46;
these finite samples are diagnostic and cannot be converted into exact runtime
shares.

The reported98,304 excessive shared wavefronts are concentrated in LDGSTS
copy instructions (leading sites5,120 each), not the already-improved LDS
consumer bank layout. Of90,112 theoretical excessive global sectors,24,576
are assigned to the BF16 temporary-scale store; cached sign loads account for
many other sites (4,096 each). Theoretical counters must not be substituted for
measured bank-conflict counters or external DRAM traffic.

Next structural candidate: a TMA BF16 transpose-staging producer using existing
project TMA patterns, with a legal physical SMEM layout and preserved FP32
Hadamard/quantization/correction. This may replace many narrow cp.async
instructions and their staging dependencies, but is unimplemented/unproven;
prior deeper cp.async buffering did not yield a win. Exact output, graph and
full-path tests remain mandatory before any promotion. Raw source/SASS-capable
capture: `compact64/compact64-instructions.ncu-rep`; profiler exited successfully.

### TMA BF16 staging prototype: correct, consumer layout remains costly

Implemented opt-in `ms_eden_tma_preparation.py`, using existing CuTe TMA
construction patterns with a legal64-byte-swizzled SMEM layout. It preserves
original BF16 inputs, retained FP32 H128/EDEN arithmetic, per-warp partial
ownership and all output formats. Each buffering slot owns an independent
generation-counted transaction barrier; a CTA rendezvous protects slot reuse.
Research comparison modes `tma1/tma2/tma3` expose buffering; no stable dispatch
is changed.

The initial GPU test hung and was explicitly terminated. Compile-only PTX
inspection showed `elect.sync ..., -1` inside a thread==0 region. The fix
enters TMA copy lowering from the entire first warp and separately elects
one lane to arrive/expect transaction bytes. The subsequent six GPU tests
passed (128x128/4096x1536, stages1/2/3), with exact packed/scales/partials,
wide exponents, arbitrary FP32 signs and changed-buffer graph replay. Seven
CPU contract/tuner tests passed; six GPU cases were skipped locally. The old
failure log is retained, not treated as a successful test.

Complete4096x1536x512 backward,64-bit packed stores,24 paired rounds:

| TMA buffers | MXFP8 us | FP4 us | Paired ratio | Within-session95% interval |
|---|---:|---:|---:|---|
| 1 | 268.597 | 256.397 | 1.047342 | [1.043539,1.051103] |
| 2 | 268.751 | 258.589 | 1.039638 | [1.036290,1.042291] |
| 3 | 268.939 | 279.601 | 0.959203 | [0.955655,0.962246] |

All cases passed separate AB/BA stationarity, same-draw complete gradients,
independent FP32 error gates, fresh RNG/current-buffer guards and zero-live-growth
checks. These results do not establish a meaningful win over compact cp.async
staging, and three stages regress versus MXFP8. No promotion or training claim.

One-stage counters:66 registers/thread,46.591% achieved occupancy,9,571,240
executed instructions, zero local spill sectors, but1,383,528 shared-load bank
conflicts. This contrasts with compact XOR's roughly7,000 conflicts. Thus TMA
has removed narrow-copy orchestration while retaining a consumer ownership
poorly matched to the hardware-supported SMEM swizzle. A next prototype must
change the consumer access/ownership (for example, evaluate matrix-load-based
ownership while preserving butterfly association), not merely deepen buffering.
That next design is unimplemented and must be independently qualified.

Evidence: `tma/`; remote snapshot
`/home/achiya/rtx-mseden-tma-20261027.RoZS5x`. All benchmark and counter workers
exited. Current TMA support is research-only; alternate constructor swizzles
are not yet GPU-qualified and must not be presented as supported production modes.

### TMA consumer XOR permutation

Added opt-in `consumer_order='xor4'`: lane-dependent shared read order followed
by an exact two-level register permutation restoring the original16 values
before sign multiplication and the retained butterfly/quantizer. Partial
ownership is unchanged. CPU permutation inversion and four GPU cases passed
(two shapes, stages1/2; wide exponents/arbitrary signs/changed-buffer replay).
The focused CPU suite passed8 tests with10 CUDA cases skipped locally.

Measured shared-load conflicts fell1,383,528 ->598,017, registers66 ->72,
occupancy46.591% ->47.004%, instructions9,571,240 ->9,877,164; no local spills.
The two-buffer complete4096x1536x512 run under the original warmup protocol
passed all gates: MXFP8269.510us, candidate254.478us, paired1.057159,
24-round within-session95% interval[1.053280,1.060607]; all AB/BA windows
stationary and replay live growth zero. This still misses1.20.

One-buffer full comparison and its unchanged-settings retry both timed out
at post-replay warmup synchronization, with GPU utilization reported0%.
Both were confirmed terminal before subsequent work. A labeled diagnostic
with `--warmup-sync-each` completed all replay/numerical checks, but its
1.069406 timing ratio fails several per-order stationarity checks and is NOT
qualified performance evidence. Do not hide either timeout or promote this
variant based on the diagnostic. The two-buffer run did not need that option.

Artifacts (including failed logs): `tma-xor4/`; isolated snapshot
`/home/achiya/rtx-mseden-tma-xor4-20261027.3qAu4z`. All processes exited.
The comparator currently reuses the same autograd leaf objects between A/B
graph bundles and emits AccumulateGrad stream-mismatch warnings. Independent
leaf identities sharing the same storage are a concrete next harness diagnosis,
not yet an established cause or implemented workaround. Numerical policies,
timing gates and production defaults remain unchanged.

### Independent A/B autograd leaves: warning removed, bounded reliability restored

The comparator now detaches alias leaves when sharing A/B inputs. This creates
new autograd identities without copying data or changing pointers, strides,
contents, shared version tracking, or timed GPU work. Metadata records
`independent_autograd_leaves_shared_storage_v1`; source hashes distinguish it
from historical results. CPU tests assert independent gradient accumulation,
identical storage and reflected in-place updates.54 focused comparator/tuner
tests passed.

Four fresh-process full comparisons completed with the original unsynchronized
warmup protocol, all without the previously observed AccumulateGrad stream
warning or warmup stall. The formerly twice-stalling one-buffer XOR4 case
completed twice. This supports shared autograd identity as a contributing
factor; it is not proof that every historical CUDA stall has the same cause.
No per-graph synchronization workaround was made the default.

TMA XOR4, one buffer,64-bit stores, complete compiled backward:

| Shape/run | MXFP8 us | FP4 us | Paired ratio | All AB/BA stationary |
|---|---:|---:|---:|---|
| 4096x1536x512 first | 265.749 | 249.406 | 1.066296 | yes |
| 4096x1536x512 confirmation | 272.262 | 254.609 | 1.070662 | yes |
| 4096x2048x512 | 402.608 | 347.644 | 1.155483 | yes |
| 4096x512x512 | 99.858 | 104.079 | 0.954196 | no: candidate BA |

All passed same-draw, independent numerical, fresh-RNG/current-buffer and zero
live-growth guards. The narrow-shape timing is a provisional regression, not a
qualified precise ratio; do not exclude it from the report or claim success
across decoder projections. The two confirmed middle-shape intervals are
[1.063459,1.069409] and[1.068062,1.073354]; larger-shape interval
[1.151694,1.158741]. These are within-session paired intervals, not cross-run
comparisons against earlier FP4 variants.

The1.20x target remains unmet. Stable defaults unchanged; no model training
claim. Evidence:`independent-leaves/`; remote snapshot
`/home/achiya/rtx-mseden-independent-leaves-20261027.akCYY6`. All workers exited.

### XOR5 shape transfer retrieved and audited

The pending 5060 workers finished; no worker was restarted. Retrieved results
from `/home/achiya/rtx-mseden-tma-xor5-20261027.ikAoGd` into `tma-xor5/`.
One-buffer TMA XOR5,64-bit stores, complete compiled backward including RNG:

| Shape | MXFP8 us | FP4 us | Paired ratio | Timing qualification |
|---|---:|---:|---:|---|
| 4096x2048x512 |400.674|348.596|1.148543|all AB/BA windows stationary|
| 4096x512x512 |101.327|104.733|0.967020|baseline AB nonstationary|

The larger shape's24-round within-session95% interval is[1.146001,1.150931].
The narrow result is a provisional regression, not qualified precise timing.
Both passed the recorded same-draw Gaussian/outlier/zero checks, independent
gradient gates, current-buffer replay, fresh RNG and zero replay live growth;
these fields were explicitly audited after retrieval. No numerical gates changed.
Neither establishes the1.20x target or a model-training improvement.

Do not restart the packed-FP32 butterfly experiment: the execution journal's
"Paired FP32 Hadamard arithmetic" section already records exact48-case
qualification, scalar SASS lowering on SM120, registers96->244 and a complete
backward regression to0.77411x. PTX acceptance is not hardware throughput proof.
Further work must target another measured complete-path bottleneck rather than
repeating that rejected implementation. Stable defaults remain unchanged.

### Three-bit row permutation rejected before GPU execution

A proposed XOR7 permutation of row bits0/1/2 does not reduce XOR5's two-way
bank conflicts. The CPU byte-address model disproved the predicted one-way
accesses. Current columns `warp + 4*subrow` fix bank bit0 to `warp//2` for
every source row under SW64: only16 banks are reachable by32 lanes. The
additional row-bit permutation still gives two-way conflicts and adds register
permutation work. Removed the proposed kernel/dispatch variant before GPU
execution; retained tests for three-bit permutation inversion, its actual
two-way result, and exhaustive reachable-bank parity over all128 rows and8
subrows. Focused CPU suite:6 passed,14 GPU tests skipped. No GPU performance
or correctness claim for the discarded variant.

Next layout improvements must change column ownership or vector load width;
another row-only XOR search cannot break this floor. This is a layout-model
bound, not a claim that the complete backward is limited by bank conflicts.

### Column-remapped TMA: conflicts removed, target still missed

Implemented research-only `consumer_order='columns'` and
`tma_columns_1`/`tma_columns_2` experiment choices. Columns map to
`warp%2 + 2*(subrow%4) + 8*(warp//2) + 16*(subrow//4)`; three row-XOR bits
spread accesses across32 banks. Register permutations restore the original
values before unchanged FP32 arithmetic. Output packed bytes/scales remain
identical. Warp partial maxima change ownership, but the exact per-CTA maximum
and final global maximum remain identical. The dedicated producer test compares
this explicit auxiliary contract, not a relaxed numeric tolerance; old producer
tests still require per-warp byte equality.

Validation: CPU8 passed/53 CUDA skips across bank/TMA files; four5060 GPU cases
passed (small/4096x1536, stages1/2, wide exponents/arbitrary signs, changed-buffer
graphs). Complete same-draw Gaussian/outlier/zero gradients, independent FP32
gates, current-buffer/fresh-RNG guards and zero live-memory growth all passed.

Complete4096x1536x512 backward,24 paired rounds:

| Variant | MXFP8 us | FP4 us | Paired ratio | Timing gate |
|---|---:|---:|---:|---|
| One buffer first |271.400|254.909|1.062038|candidate AB unstable|
| One buffer confirmation |268.357|251.651|1.064728|all stationary|
| Two buffers |269.912|256.310|1.054656|all stationary|

Qualified within-session95% intervals:[1.061290,1.067833] and
[1.052723,1.056560]. No1.20x result, no new model-training claim, no default
promotion. Different historical runs are not a direct paired columns-vs-XOR5
comparison.

After timings finished, bounded Nsight Compute captured the one-buffer producer:
shared-load bank conflicts4,625 (prior XOR5 capture204,978),72registers,
47.193% occupancy,10,103,724 executed instructions (prior9,875,924),
148,480 global-store sectors and zero local load/store sectors. Counters confirm
the intended conflict reduction but not a complete-path improvement. Extra
index/permutation instructions are a plausible offset; this is not proof that
all remaining time belongs to arithmetic. Further work should target measured
whole-path costs, not continue reducing this now-small conflict counter.

Artifacts:`tma-columns/`; remote isolated snapshot
`/home/achiya/rtx-mseden-tma-columns-20261027.ACDVC8`. All timing/profiling workers
exited; no occupied5090 or other laptop was used.

### Refreshed exact-schedule timelines and TMA enqueue-order interaction

The profiling driver now accepts a completed comparison as its schedule
manifest, preserving actual candidate overrides (rowVPL16/store64) and pipeline.
Three CPU manifest tests passed. Research TMA choices and TMA source hashes are
recorded. Captured ten warmed node-level backward replays for stable MXFP8 and
TMA XOR5 in separate processes, including fresh caller RNG and scoped combo
fusion. Raw JSON/log/nsys-rep/SQLite artifacts:`tma-timelines/`. Node identity
multisets matched across all ten replays.

Diagnostic spans: MXFP8 three nodes236.718us; FP4 eighteen nodes215.989us.
These profiler spans are NOT ordinary latency acceptance or additive phase
timings. FP4 row-dY producer starts75.07us/ends133.62us; its maxima reduction
ends138.44us; its scale finalizer starts156.11us/ends173.52us. dW GEMM runs
134.09-173.90us, while dX GEMM runs173.68-215.99us. FP4 GEMMs take about40-42us
versus63-72us MXFP8. Preparation/admission and finalizer overlap remain the
gap, not evidence that FP4 GEMM arithmetic is intrinsically slower.

Two targeted TMA XOR5 row-dY-first enqueue orders were then measured with the
same24-round complete comparison and full guards:

| Order | MXFP8 us | FP4 us | Paired ratio |
|---|---:|---:|---:|
| dx_a,dx_b,dw_a,dw_b |273.152|266.251|1.024464|
| dx_a,dw_a,dx_b,dw_b |267.190|266.279|1.003007|

All order-specific timing windows stationary; exact-draw gradients, independent
numeric gates, matched forward, fresh RNG, current buffers and zero live growth
passed. Neither reaches the target or supports replacing retained enqueue order.
Artifacts:`tma-orders/`; both workers exited. Separate-process medians do not
establish a direct order/control confidence interval.

Next specific hypothesis: dW GEMM competes with late dX scale finalization.
Test a finish-admission dependency without serializing independent producers;
do not repeat the unsuccessful row-dY-first enqueue orders or assume profiling
overlap alone proves interference. Stable policies remain unchanged.

### Finish-admission dependency rejected on the TMA path

Added research-only `--finish-admission independent|dx_a|dx_ready`. All four
preparations retain their original concurrent streams/order. The latter two
insert a dW-stream wait after its preparation, before either dX GEMM launch:
on the caller's activation-preparation frontier, or after its join with dX's
weight-preparation stream. Existing lifetime/exception joins remain intact.
No stable runtime launcher changed. CPU topology/lock/failure tests cover all24
enqueue permutations and all three admission choices:581 passed in1.81s.

Fresh5060 TMA XOR5 one-buffer/store64 complete4096x1536x512 comparisons:

| Admission | MXFP8 us | FP4 us | Paired ratio |
|---|---:|---:|---:|
| independent control |269.937|250.149|1.077758|
| after dX activation preparation |264.149|259.928|1.016675|
| after both dX operands ready |274.615|268.202|1.023334|

All24-round AB/BA windows stationary. Same-draw Gaussian/outlier/zero gradients,
independent numerical gates, matched forward, fresh RNG, current-buffer replay
and zero live-memory growth passed. Within-session95% intervals respectively
[1.075345,1.080268], [1.013548,1.020225], [1.021689,1.025054]. These separate
MXFP8 comparisons reject promotion of the admission variants; they are not a
direct cross-process FP4/control confidence interval. Delaying GEMM admission
did not close the preparation gap. Preserve independent admission.

Artifacts:`finish-admission/`; isolated remote
`/home/achiya/rtx-mseden-finish-admission-20261027.9TASeq`. All workers exited.
Next work must reduce preparation/finalizer work or resource demand rather
than merely shift the demonstrated critical path with another wait. Target and
model-training qualification remain unmet; stable defaults unchanged.

### TMA and finalizer composition: qualified1.111x middle-shape result

Research driver now composes retained/paired conversion/dual-dX/dual-both
finalizers with TMA preparation. Incompatible launch-topology combinations and
implicit RNG are rejected; finalizer/launcher hashes recorded. No stable kernel
or policy changes. Eighteen5060 rounding-boundary/independent-format/exact-byte/
changed-buffer/memory tests passed in12.09s. Four focused CLI conflict tests
passed; the prior581 topology tests also passed after driver changes.

Complete4096x1536x512, TMA XOR5 onebuffer/store64,24 paired rounds:

| Finalizer | MXFP8 us | FP4 us | Paired ratio | Timing gate |
|---|---:|---:|---:|---|
| Paired conversion |268.072|259.095|1.035002|all stationary|
| Dual dX first |267.621|247.914|1.077416|candidate BA unstable|
| Dual both first |300.327|274.037|1.095506|all order windows unstable|
| Dual both confirmation |266.038|239.495|1.111219|all stationary|
| Dual dX confirmation |269.259|246.099|1.094068|all stationary|

Confirmed within-session95% intervals:dual both[1.108657,1.113822]; dual dX
[1.091598,1.096308]. Every run passed same-draw gradients, independent numerical
gates, matched forward, fresh RNG/current buffers and zero live-memory growth.
Keep failed timing runs visible; their point estimates are not qualification.
The confirmed both-gradient result is below1.20 and is not a direct paired
comparison with the separate-process retained-finalizer run.

Artifacts:`tma-finalizers/`; isolated snapshot
`/home/achiya/rtx-mseden-tma-finalizers-20261027.Lueign`. Cross-shape transfer
queue launched for decoder2048 then decoder512 (`transfer-*.json/log`), external
120s timeout per process, exec session31991. Verify the process/results before
relaunching; launch alone is not completion. No model training yet.

### Dual-finalizer transfer and targeted geometry tuning

The transfer queue completed and both results were retrieved/audited:

| Shape | MXFP8 us | FP4 us | Paired ratio | Timing gate |
|---|---:|---:|---:|---|
|4096x2048x512|400.139|338.440|1.180680|all stationary|
|4096x512x512|101.482|102.993|0.986657|baseline BA unstable|

Both pass full same-draw, independent numerical, matched-forward, RNG,
current-buffer and zero-live-growth gates. Larger-shape within-session95%
interval[1.178106,1.182631], still below1.20. Narrow shape is a provisional
regression, not qualified precise timing. No new training claim.

Extended the existing autotune-backed joint geometry driver with `tma_finalizers`:
retained TMA XOR5 and dual-both finalization,20 candidates comprising control,
16 strength-two balanced four-operand finalizer grids at26/52/104/208 CTAs,
and64/256/512-thread challengers. Per-leg fused operands retain equal thread
counts. Producer/GEMM/numerical settings unchanged. Tests assert only finalizer
geometry changes, valid topology and real residual-store integration:8 passed.

Launched isolated bounded study on5060:
`/home/achiya/rtx-mseden-tma-finalizer-grids-20261027.0RlN5L`, `study.log`,
`study/residuals/`,20trial/900second budget,90second worker timeout. Exec
session21139; parentPID37540 and first candidatePID37578 verified live.
Do not infer completion or restart from this note; poll current process/results.
Worker screening uses8 AB/BA rounds/5s warmup, full unchanged correctness gates
and mandatory per-order timing quality. Winners require independent24-round
confirmation; automatic runtime promotion remains disabled.

### Geometry study complete; matched training translation queued

The20-candidate study finished in357.74s:20 correctness-valid,18 stationary
timing-eligible,2 explicitly excluded for timing. All raw workers, manifests,
residuals and selection were retrieved under `tma-finalizer-grids/`.
Best screening absolute latency used finalizer grids dx104/26,dw104/104;
best paired ratio used64-thread finalizers. Independently confirmed both with
24 rounds and15s warmup; all AB/BA windows and full runtime/numerical gates pass:

| Finalist | MXFP8 us | FP4 us | Paired ratio |
|---|---:|---:|---:|
| Grids104/26/104/104 |268.592|243.091|1.105989|
|64-thread finalizers |270.841|244.836|1.105700|

Intervals[1.104034,1.107950] and[1.102018,1.109164]. No clear improvement over
the earlier qualified1.111219 default-grid fused result; no promotion. The
study does not justify claiming that more finalizer-grid search closes1.20.

Added opt-in training `--ms-eden-research-variant tma_dual`, requiring a complete
explicit schedule map, operand-paired launch, staged maxima and explicit RNG.
Contexts restore research patches on exit/failure. Checkpoint identity includes
variant and relevant kernel/launcher source hashes. Default training behavior
and identities remain unchanged.39 focused CPU tests passed; the strengthened
11-case schedule suite passed afterward. No GPU training claim from those tests.

Launched remaining decoder1024 complete qualification followed by two matched
seed941512-step21M transformer runs (NVFP4 forward/MXFP8 backward versus full
MS-EDEN), same initialization/data/optimizer/FP32-sensitive recipe and compiled
graphs/combo fusion. Explicit training schedules preserve per-shape choices,
row16 and64-bit stores; no selected geometry promoted. This is a short
2.1M-byte-token translation check per policy, not long convergence evidence.

Remote `/home/achiya/rtx-mseden-training-tma-20261027.cKeqAt`, `training/`,
exec session56692. Exact launch script and schedule map are local under
`training-tma-dual/`. Preflight120s and per-training1200s external deadlines.
Verify process and results before restarting; at launch neither training nor
its performance/numerical outcome is established. Other machines untouched.

### Two-seed matched21M training translation completed

Remaining decoder1024 projection passed full numerical/RNG/replay/memory gates
and all timing windows: MXFP8188.158us, FP4163.598us, paired1.149061,
within-session95% interval[1.145017,1.152867]. This completes the four model
projection smoke matrix, not a1.20x qualification (narrow timing still limited).

Two independent512-step runs per policy completed on5060,2,097,152 byte tokens
each, same21,121,024-parameter decoder and matched forward. Seed941 ran baseline
first; seed942 reversed order. Analyzer verified matching initialization/data/
optimizer/precision/configuration provenance and NVFP4 forward projections.
All final parameters finite, no interruption/error. Compilation/evaluation/
checkpointing excluded from synchronized training-block wall, reported separately.

| Seed | MXFP8-bwd train s | FP4-bwd train s | Total train ratio | Median block throughput ratio | MXFP8-bwd final val | FP4-bwd final val |
|---|---:|---:|---:|---:|---:|---:|
|941|10.3117|9.6986|1.06322|1.03032|0.991012|0.996070|
|942|9.8361|9.5079|1.03452|1.03521|0.944406|0.980010|

Model speed gain is modest, not20%; seed941's total ratio is influenced by its
slower first baseline block. Matched-step final degradation is+0.005058 and
+0.035604 respectively. Seed942's gap already grows to+0.028277 at step256.
Do not summarize this as identical convergence. Historical pre-TMA seed942
also had a larger gap (0.975585 versus0.949978); this is context, not a controlled
proof that the new implementation cannot affect trajectories. Same-draw kernel
qualification remains the direct arithmetic evidence.

End-to-end wall including cold compilation/validation/checkpoints was45.116/
43.918s baseline/candidate for941 and25.189/33.021s for942. Do not present the
training-only gain as a cold-start speedup. Raw JSONL/logs, schedules, launch
script and observed-checkpoint equal-work/equal-wall analyses for each seed
are retained in `training-tma-dual/`. No interpolation or long-convergence claim.

All training workers exited. The1.20x complete-backward objective remains unmet;
no stable default or research winner promoted. This result supports a small
bounded model-level benefit and explicitly exposes its numerical tradeoff.

### Exact probability arithmetic interaction and producer-grid search

Tested existing `probability_math=power2` with TMA XOR5/dual-both finalization.
No new approximate arithmetic: independent positive-E4M3 format/tie, fallback,
divide-versus-power2 bitwise and graph tests passed12/12 on5060. Initial run
without the explicit CUDA opt-in was CPU-only5pass/7skip; separate CUDA-enabled
qualification completed and is the GPU evidence. Four CLI tests passed locally.

Complete24-round comparisons, all timing/numerical/RNG/current-buffer/memory
gates passed:

| Shape | MXFP8 us | FP4 us | Paired ratio |
|---|---:|---:|---:|
|4096x1536x512|270.142|243.380|1.111556|
|4096x2048x512|401.612|339.280|1.183705|

Intervals[1.109559,1.113514] and[1.182388,1.185098]. No clear gain over the
retained arithmetic; no default change or training rerun justified. Evidence:
`tma-probability/`, remote`/home/achiya/rtx-mseden-tma-probability-20261027.hh6O9A`.
Both workers finished and artifacts retrieved.

Added `tma_producer_grids` to the existing autotune-backed geometry driver:
control plus16 strength-two balanced allocations of64/128/256/512 CTAs across
the four producers, with retained divide arithmetic, TMA XOR5, dual-both
finalization and64-bit stores. This specifically tests the new implementation
interaction rather than reusing the old compact-producer grid conclusion.
Nine CPU portfolio/store-integration tests passed.

Launched on5060 in`/home/achiya/rtx-mseden-tma-producer-grids-20261027.mEj0GA`:
`study.log`, `study/residuals/`,17trials/900seconds,90s worker timeout. Exec
session is recorded in the conversation tool response; verify process/result
state before restarting. Same8-round screening and mandatory24-round winner
confirmation contract as the finalizer geometry study. Stable defaults unchanged.

### Producer-grid retrieval and independent confirmation

Read-only remote inspection confirmed the producer-grid parent PID43899 has
exited and the study reached its final selection after302.12seconds. Its selected
candidate is`0fdd403a195bdc88422c9323-9e2c4695`, with screened complete backward
243.8915us. Producer blocks are dX A/B256/128 and dW A/B512/64; this is not a
qualified improvement over the retained composition.

Artifact retrieval is in progress to`tma-producer-grids/` (exec97786). A fresh
24-round confirmation was launched on the idle5060 in the same remote snapshot,
using`benchmarks.experiment_ms_eden_bank_layout`, frontier`tma_xor5_1`,
`--scale-finalizer dual_both --grouped-store-bits 64 --shape decoder1536`,
and the selected trial's`manifest.json`. Output`confirm-producer-winner.json`,
log`confirm-producer-winner.log`; exec95342. Inspect these existing handles and
remote state before any restart. No defaults or winners were promoted.

### Producer-grid confirmation result and next controlled interaction

All17 producer-grid residuals have status`ok`. Independent24-round confirmation
of the screening winner measured MXFP8 273.469us versus MS-EDEN248.943us,
paired ratio1.097809 (95% round-cluster CI1.092699–1.103244). Both orders and
both implementations were stationary. Gaussian/outlier/zero same-draw dX/dW
checks were bitwise, changed-buffer and fresh-RNG guards passed, and live
allocation growth was zero. This does not improve the retained1.111219 result;
no producer-grid promotion. Confirmation JSON/raw/log are local under
`tma-producer-grids/confirm-producer-winner.*`.

Added research-only`tma_gemms` to the existing joint geometry tuner: use the
same bounded two-leg warp-layout/deeper-buffering portfolio as`compact_gemms`,
but with TMA XOR5, dual-both finalization and64-bit stores. This tests scheduling
interaction with the new preparation implementation rather than assuming the
old preparation's GEMM result transfers. Ten focused CPU tests passed. Numerical
policy, full-path timing gates, stable defaults and promotion rules are unchanged.
Fresh remote snapshot:`/home/achiya/rtx-mseden-tma-gemms-20261027.bjwDPV`.
Launched `tma_gemms` on decoder1536,12trials/900seconds,120s worker timeout,
with the original retained selection from`rtx-mseden-joint-grids-20261027.I9FptI`
(not the rejected producer-grid selection). Exec95678,`study.log`,
`study/residuals/`. Prior producer-grid retrieval exec97786 completed successfully.
The first launcher exited with a snapshot packaging error: missing`tests` caused
`ModuleNotFoundError` for`tests.cache_memory_checks` before benchmark capture.
Copied the unchanged validation helpers from the prior snapshot and relaunched
into fresh`study-v2/` and`study-v2.log`, preserving the failed attempt. This was
not a kernel or numerical failure; no validation was bypassed.

### TMA GEMM interaction screen completed

The corrected `tma_gemms` study completed all12 trials in215.50seconds. All
reported correctness-valid outcomes;11 passed every per-order stationarity
gate. The dX-only stage3 candidate was timing-ineligible. The strongest screen
was both GEMMs at stage3 with M-atoms8/8:240.660us,paired ratio1.131250, versus
control246.762us/1.112459. These are separate-process screening results, not a
confirmed causal ranking. Winner `b63343adc4a4fde801b85d4f-d3fd41b4` is undergoing
independent24-round confirmation in the same snapshot:
`confirm-gemm-winner.json` / `.log`, exec11128. Full study retrieved locally as
`tma-gemms/study-v2/` and `tma-gemms-study-v2.tar.gz`.

A proposed tile-shape portfolio was rejected locally: the native mma128 scale
layout requires fixed128x128x128 tiles, and several other candidates exceeded
CTA-thread/SMEM limits. No legality rule was relaxed. Instead, prepared
`tma_loops`:28 one-leg/joint choices covering K unroll2/3/4/6/8/12, MMA preload,
and A/B ldmatrix x2. All28 pass the existing static legality adapter for this
shape;12 focused CPU tests pass. The isolated snapshot is
`/home/achiya/rtx-mseden-tma-loops-20261027.IvfvWp`; no GPU study launched there
yet, to avoid overlapping the confirmation. Stable defaults remain unchanged.

First GEMM confirmation: MXFP8 270.705us, candidate238.825us, paired1.132055
(CI1.126803–1.137008), all same-draw Gaussian/outlier/zero checks bitwise. However,
candidate AB timings failed stationarity, so this is not a qualified win.
Preserved locally as`tma-gemms/confirm-gemm-winner.*`. Launched one fresh-process
repeat into`confirm-gemm-winner-2.*`; no competing GPU study and no promotion.

Second confirmation also failed stationarity: recorded medians are
MXFP8 299.868us and MS-EDEN269.434us, paired1.113118. All four per-order timing
series failed, with4.36–5.32% split drift. Same-draw numerical checks remained
bitwise. Both attempts are preserved and neither qualifies the three-stage
schedule. No further repeat is justified before understanding timing conditions.
Read-only post-run SMI showed43C,0%utilization and no compute processes; this
idle snapshot cannot attribute in-run drift. Windows PowerLineStatus reports
`Online` (AC connected). No device settings were changed.

Launched the28-trial`tma_loops` study from the original retained stage2 selection,
not the unqualified stage3 winner: snapshot`rtx-mseden-tma-loops-20261027.IvfvWp`,
`study.log`,900s total/120s worker timeout,exec36121. A short read-only telemetry
capture is recorded separately as`diagnostic-telemetry.csv`; it is diagnostic,
not a replacement for the existing AB/BA stationarity gates.

### Research-only inline maxima inside paired finalization

Implemented `compile_dual_finalizer(..., inline_max=True)` behind an explicit
research CLI flag `--dual-inline-max` (requires a dual finalizer). It eliminates
the separate two-CTA maximum-reduction launch for each fused backward leg by
repeating the exact warp reduction inside scale finalization. Expected tradeoff:
two fewer complete-backward launches versus more partial-array reads and warp
reductions. No claim of lower total traffic or measured speedup yet.

Each warp reads the original producer partials, preserves FP32 max and scaling,
and uses the unchanged stochastic-rounding routine and uniforms. CTA0/thread0
writes the auxiliary maximum; no CTA reads that output, so no inter-CTA waiting
or grid synchronization is introduced. All original outputs and argument ABI
remain intact; staged reduction is still the default.

Focused CPU suites passed597 tests,16 GPU cases skipped. The GPU matrix now
compares both staged and inline variants, including unequal/ragged partial
counts, row-major/native scale layout, both probability modes, zero maxima,
changing source/uniforms under graph replay and zero live-allocation growth.
Isolated snapshot:`/home/achiya/rtx-mseden-dual-inline-20261027.3Dx1MG`.
Queued GPU qualification behind the verified live loop-sweep coordinator
PID50295. Local reproduction script:`run-dual-inline.sh` in the report root;
remote`run.sh`, exec29440. It waits for that handle, refuses to proceed if another
joint-geometry worker remains, runs17 finalizer tests with the GPU opt-in, then
only on success runs the24-round decoder1536 full-backward comparison. Outputs
are`qualification.log` and`inline-max-decoder1536.{json,raw.json,log}`. Shell
noclobber and the comparator's existing output guard preserve earlier results.

### Loop-scheduling results and inline-finalizer GPU qualification

All28 `tma_loops` trials completed;24 passed all per-order timing-stability gates.
The largest stationary paired ratio was1.116602 (dX K-unroll3, dW unchanged),
245.018us. Unroll2 on dX reached1.114179 and preload on dX1.113785. These are
screening observations only, not evidence of a new qualified winner or1.20x.
Complete artifacts are local under`tma-loops/study/` with residuals and raw
samples; archive`tma-loops-study.tar.gz` also contains diagnostic telemetry.
No loop/ldmatrix schedule is promoted from this screen.

After the sweep exited normally, the queued inline-finalizer GPU suite passed
**17 tests in6.79seconds** on5060. Both implementations preserve exact maximum,
FP8 scale bytes and FP32 outer factors across updated-input graph replay; the
live-allocation check passed. The gated full-backward comparison then started
under exec29440. No timing result is claimed until its JSON is inspected.

Inline-finalizer comparison finished and was retrieved: MXFP8 265.431us versus
candidate253.151us,paired1.048320 (CI1.047019–1.049671), with all four AB/BA
series stationary. This is worse than the retained staged-finalizer composition,
despite eliminating two launch nodes. Repeated reduction traffic/work is a
plausible explanation, not independently counter-attributed here. Reject as a
default optimization. Artifacts:`dual-inline/`, `dual-inline-results.tar.gz`.
All sweep/qualification/comparison jobs described above are terminal. Goal
remains unmet; stable defaults unchanged.

### CTA-shared inline maximum follow-up

Added research `inline_scope='cta'` alongside the rejected per-warp inline
reduction. All threads partition the partial array, each warp writes one FP32
maximum to a tiny shared array, then one CTA barrier makes those maxima visible
to every thread. This removes repeated partial-array loads across warps but adds
shared-memory work/barrier cost. All CTAs remain independent; auxiliary maximum
and outer-factor outputs retain single-writer ownership. Staged reduction stays
the default, and invalid scope/flag combinations fail before compilation.

CPU checks:585passed,24assigned-GPU cases skipped. On the idle5060, started
the25-case exactness/replay/memory matrix followed, only if it passes, by a
24-round full decoder1536 comparison with `--dual-inline-max
--dual-inline-scope cta`. Snapshot:
`/home/achiya/rtx-mseden-dual-cta-20261027.aN0NbO`; exec72155,
`qualification.log`, `cta-max-decoder1536.{json,raw.json,log}`.
No performance claim or promotion before inspecting those outputs.

CTA-sharing qualification passed25 GPU tests in12.31s. Complete decoder1536
comparison passed all timing-stability gates: MXFP8 270.534us, candidate244.082us,
paired1.109630 (CI1.105686–1.113522). It recovers the per-warp-inline regression
but does not establish an improvement over retained staged finalization.
All results retrieved under`dual-cta/` and`dual-cta-results.tar.gz`.
Started sequential full comparisons on decoder512 and decoder2048 in the same
snapshot, outputs`cta-max-decoder512.*` and`cta-max-decoder2048.*`. This tests
shape-dependent launch-cost benefit/regression; no promotion or1.20x claim.

Both transfers finished with all per-order timing gates passing:

| Shape | MXFP8 us | CTA-inline MS-EDEN us | Paired ratio |
|---|---:|---:|---:|
|4096x512x512|102.764|103.100|0.995828|
|4096x2048x512|404.361|335.436|1.202744|

The wider point estimate exceeds1.20, but its round/order bootstrap lower bounds
do not; this is borderline evidence, not a qualified target result. The narrow
shape remains at parity/slight regression. All transfer artifacts are local in
`dual-cta/` and`dual-cta-transfer-results.tar.gz`.

Added`tma_cta_finalizers` to the real residual-writing research tuner, reusing
the20-candidate finalizer block/thread portfolio with the exact CTA-inline path.
Thirteen focused CPU tests passed. Launched decoder1536 (the still-deficient
shape),20trials/900s/120s worker timeout, in
`/home/achiya/rtx-mseden-dual-cta-grids-20261027.j1dS3M`, `study.log` and`study/`.
This tests whether fewer CTAs avoid redundant per-CTA maximum scans. No defaults
changed and independent winner confirmation remains mandatory.

### Full-K H16 transform feasibility, not yet integrated

The prior BF16 H8 tensor-core path zero-pads half of K16. Added an independent
integer basis/installed-CuTe-layout proof for two full-K H16 MMA instructions
over two independent H128 blocks, followed by three FP32 butterfly stages.
All256 basis vectors recover exactly two separate H128 transforms with bijective
input/output ownership. This proves indexing/algebra, not floating-point or speed.

Implemented research-only`ms_eden_h16_hadamard.py`: original BF16 inputs, exact
unit signs, FP32 accumulation and finishing stages, odd-block masking, guarded
scalar fallback for unsafe input range/arbitrary signs. No quantizer/statistics,
RNG, backward dispatch or default uses it. Association differs from scalar FWHT;
GPU qualification uses the pre-existing pure-transform FP64 error thresholds,
not weakened complete-gradient gates. Added19 GPU cases for three shapes,
Gaussian/outlier/cancellation/zero inputs, fallback, current-input graph replay
and live memory. CPU mapping/configuration checks passed2 tests; GPU cases not
yet executed. Snapshot:`/home/achiya/rtx-mseden-h16-20261027.hG6XxU`.
The current CTA-finalizer grid search remains the only GPU job.

That grid search completed20 trials,19 stationary. Its original control was
best (242.720us,paired1.102455); grid/thread tuning found no better candidate.
All residuals/raw results are local under`dual-cta-grids/study/`; no promotion.

H16 pure-transform GPU qualification then passed20 tests in3.39s. No complete
MS-EDEN integration is qualified. Extended the existing diagnostic paired
transform benchmark with`--mma-kind bf16_h16`, pair-level fallback routing,
explicit source hashes and actual launch geometry. Eight CPU benchmark/API
tests passed. Launched a4096x1536 comparison against the original scalar
transform,24paired rounds,15s warmup,eight calls per captured graph, same128
threads/128-block caps, in the H16 snapshot:`transform-4096x1536.{json,log}`.
This includes transform output stores but excludes quantization/correction,
RNG and GEMM; it must not be presented as a backward speedup.

First H16 diagnostic completed with stationary timings: scalar109.244us,
H16 111.543us,paired0.978814 (CI0.978073–0.979579). Initial FP64-reference NRMS
was3.09385e-8 for scalar and3.09669e-8 for H16; changed-input/sign and memory
checks passed with zero live growth. Thus this implementation does not yet
justify integration. Source inspection identifies scalar scattered source/output
accesses as a plausible offset to fewer butterfly instructions; explicit paired
BF16 loads and FP32 stores should be tested before claiming arithmetic is the
limitation. No counter attribution or integrated backward claim follows.
Artifacts:`transform-4096x1536.{json,log}`,`h16-qualification.log`. All studies
and qualification/diagnostic jobs in this section have exited normally.

### H16 paired FP32 stores

Added opt-in`H16Config(output_store_bits=64)` and diagnostic CLI
`--h16-store-bits 64`;32-bit stores remain default. Only fast-path output
movement changes: adjacent FP32 values use explicit64-bit copies, with an8-byte
output-alignment contract. Arithmetic, guards and fallback remain unchanged.
CPU benchmark/config tests:8passed. On5060,23GPU tests passed in8.75s, including
bitwise scalar-store versus paired-store equality on odd tails and4096x1536,
changed inputs/signs, graph replay and zero live growth.

Paired transform diagnostic: scalar109.076us versus H16-store64 108.165us,
ratio1.008387 (CI1.007888–1.008949),stationary and zero live growth. This removes
the isolated-store regression but is only a0.8% transform advantage, not a
backward result. Artifacts:`h16-store/`, `h16-store-results.tar.gz`; remote
`/home/achiya/rtx-mseden-h16-store-20261027.2wUK9k`. All jobs completed normally.

The standalone transform writes a full FP32 intermediate, unlike fused
preparation. The next discriminating experiment is a row-preparation prototype
using H16's native register ownership, FP32 correction and direct packed output,
without any full transformed GMEM intermediate. It must pass independent
decoded-operand/scale gates before complete-backward measurement; layout and
correction-reduction ownership are unresolved implementation work, not a win.

### Fused H16 row preparation implemented and qualified

Added`ms_eden_h16_preparation.py`, preserving H16-native register ownership
through per16-value max, FP32 normalization/correction and direct packed FP4
stores. Temporary scales remain BF16 and producer maxima FP32. There is no
transformed FP32 GMEM tensor. Unsafe ranges/non-unit signs call the original
scalar `prepare_tile`; transposed production paths are not modified.

On5060,22 GPU/config cases passed in9.06s over3x128,256x384,4096x1536 and seven
distributions (Gaussian,outlier,cancellation,zero,arbitrary signs,tiny,huge).
Gates retain candidate decoded NRMS <= scalar+1e-4, candidate/scalar decoded
NRMS <=0.005, independent dense FP64 H128, exact packed/scales fallback,
partial-max consistency, current-input/sign graph replay and zero live growth.
No complete-gradient or speed qualification is inferred from these tests.

Added scoped research driver`experiment_ms_eden_h16_preparation.py`: selects
only eligible original-BF16 row operands, retains TMA XOR5 transposed producers,
records actual dispatch/source hashes and restores compiler patch on exit.
Two CPU API/dispatch tests passed. Started24-round complete decoder1536 backward
with dual staged finalizers, original retained schedules and all existing
fresh-RNG/numerical/graph/memory gates in
`/home/achiya/rtx-mseden-h16-prep-20261027.v58zfT`: exec63878,
`full-decoder1536.{json,raw.json,log}`,`qualification.log`.
Stable defaults remain unchanged; no end-to-end benefit claimed yet.

First full-path attempt stopped before timing with`KeyError: transpose` because
the existing bank-layout research wrapper did not preserve the compiler's
signature. Added`functools.wraps` to that wrapper and a real nested H16/TMA
dispatch/restoration test. Four focused CPU tests passed. Relaunched into fresh
`full-decoder1536-v2.{json,raw.json,log}`; failed first-attempt log is preserved.
No numerical gate was bypassed and the kernel source was unchanged by this fix.

The corrected integration reached mixed-graph warmup but stalled at
`torch.cuda.synchronize` (`compare_ms_eden_pipeline.py:237`). After verifying
PID60835 was our CPU-busy/GPU-idle trial, it was terminated; ptrace stack attach
had been denied without attaching. A diagnostic reproduction with periodic
Python stacks (PID61259) identified the synchronization location, but completed
normally before a guarded termination attempt (which refused the vanished
target). Its JSON exists and session25536 exited0. Logs/source artifacts remain
intact. The earlier long normal run's warmup behavior remains unresolved
normal-warmup graph behavior, not an achieved speed result. The next diagnostic
uses the existing`--warmup-sync-each` option to isolate replay interleaving;
its warmup contract must remain explicit and cannot stand in for normal-mode
qualification. No production/default changes.

Retrieved the normally completed diagnostic result: all per-order series were
stationary, MXFP8 272.508us versus H16-row MS-EDEN254.874us,paired1.068312.
Dispatch records confirm45 eligible H16-row builds and135 retained/transposed
builds; this is not an accidentally unchanged producer. Because Python stack
dumps were enabled, treat timing as diagnostic only. It does not suggest an
improvement over the retained1.111x composition. Launched a fresh uninstrumented
normal-warmup repeat into`full-decoder1536-v3.*`; no warmup-sync override was
used. The earlier proposed synchronize-each diagnostic was not necessary once
the instrumented reproduction completed.

The fresh normal run completed without instrumentation or warmup changes:
MXFP8 272.685us, H16-row candidate254.878us,paired1.069507
(CI1.067761–1.071290),all AB/BA stability gates passed. Thus the fused native
H16 ownership is currently slower than the retained preparation composition.
No promotion or training run is justified. Full artifacts, including failed
attempts, are local in`h16-prep/` and`h16-prep-results.tar.gz`.

Source-level next hypothesis: H16-native quantization distributes one16-value
scale group across four lanes, adding correction shuffles and repeated divide
instructions versus retained VPL16. A four-block batch can exchange register
bit1 with lane bit0 and register bit3 with lane bit1, then rename register
bits2/3, giving each lane one contiguous16-value scale group. Its logical tile
is`base+lane%4` and local row coordinate`lane//4`; no final lane permutation is
needed because VPL16 quantization has no subgroup shuffles. Added a CPU proof
of the complete512-value bijection (four CPU tests now pass). This is an
unimplemented kernel-layout experiment, not measured bottleneck attribution.

### H16 grouped preparation qualification (2026-10-27)

Implemented the four-block register regrouping as an opt-in research compiler
variant (`grouped_output=True`), retaining FP32 transform/statistics and the
existing 16-value correction and 64-bit packed stores. The research comparison
driver exposes `--h16-grouped-output`; stable runtime defaults are unchanged.

The initial GPU matrix found a declared pointer-alignment mismatch (32 bits
versus the 64-bit packed store). Corrected the grouped output pointer contract
to 8-byte alignment, matching the allocated output buffers. The second run
passed all 47 tests on the RTX 5060 Laptop: both H16 layouts, three shapes,
seven distributions, independent FP64 decode checks, fallback cases, changed
input/sign graph replay, and live-allocation checks. Logs are preserved in
`/home/achiya/rtx-mseden-h16-prep-20261027.v58zfT/qualification-grouped-v{1,2}.log`.
Complete compiled backward on 4096x1536x512 measured MXFP8 270.444 us versus
grouped H16 NVFP4 245.123 us: paired speedup 1.101638x, round-bootstrap 95%
interval [1.100260, 1.103019]. All aggregate and per-order stationarity gates
passed across 24 rounds; replay live allocation growth was zero. Candidate
dX/dW NRMS was 0.137005/0.137176 against the FP32 reference, within unchanged
experimental gates. This improves the earlier native-layout H16 result
(1.069507x) but does not beat the retained best 1.111219x comparison or reach
the 1.20x goal. Historical cross-session comparisons are not paired evidence.
No promotion or new training run is justified. Local result:
`autotune_reports/ms_eden_speed20_20261026_5060/h16-prep/full-decoder1536-grouped-v1.json`.

### Isolated row-producer diagnostic, in progress

Added `benchmarks/compare_ms_eden_h16_preparation.py`: same inputs, independent
FP64 decode gates, three implementations, randomized 24-round graph timings,
and live-allocation checks. This is not complete-backward qualification.
Initial completed sizes on the 5060 measured retained/native-H16/grouped-H16
medians of 22.480/25.932/20.644 us at 4096x512 and
58.602/66.332/53.350 us at 4096x1536. All three decoded NRMS values agreed at
each size, and live growth was zero. Timing outliers remain visible in raw
samples, so these are diagnostic medians rather than qualified speed claims.
The third size remained CPU-active at 100%, GPU-idle, without another result
for almost four minutes. Deliberately terminated only verified PID62637 with
SIGTERM; session18307 completed with status143. The third size is inconclusive,
not a numerical failure or completed timing result. Remote
`row-producers-v1.log` in the H16 snapshot. Partial log copied locally as
`h16-prep/row-producers-v1-progress.log`. The isolated gain versus the absent
full-path gain motivates investigating concurrent resource contention or
producer geometry; it does not establish either as the cause.

Added `h16_grouped_grids` to the existing joint-geometry tuner: grouped H16
eligible rows, retained TMA XOR5 transpose, dual finalizers, and the same
17-candidate four-grid balanced portfolio (64/128/256/512 blocks per operand).
Full-backward numerical checks and AB/BA timing-quality rejection remain
unchanged. Nineteen CPU tests passed (42 GPU tests skipped locally).
Launched `grouped-grid-study-v1` in the same isolated remote snapshot with
17 trials, 900-second budget, and 120-second per-worker timeout. Session29617.
No winner is promoted automatically; a selected result needs fresh 24-round
confirmation and representative-shape/training qualification.

The search completed normally in 311.8 seconds: 17 numerically valid trials,
12 timing-eligible, five rejected for timing instability. Selected producer
block counts were dX A/B=256/64 and dW A/B=256/256. Its eight-round screen was
249.690 us versus matched MXFP8 275.211 us (paired1.101895x).
Independent 24-round confirmation completed normally: MXFP8 268.664 us,
candidate245.515 us, paired1.095651x, CI[1.092435,1.098412]; all four per-order
stationarity flags passed and live allocation growth was zero. The grid search
did not close the gap and does not justify promotion or training. Artifacts
are local in `h16-prep/grouped-grid-study-v1-results.tar.gz` and
`h16-prep/grouped-grid-confirm-v1.json`. Both remote processes are terminal.
Added a numerically validated single-producer capture option to the isolated
driver for subsequent register/spill attribution; no counters captured yet.

### H16 register/spill attribution

Subsequent NCU2026.1.1 captures completed successfully on the idle 5060, with
independent decode checks before capturing one4096x1536 producer. Retained
versus grouped H16: registers/thread48/64, local-load sectors0/98304,
local-store sectors0/8192, executed warp instructions8,803,760/10,055,100,
active-warps percentage58.44/50.25, and L2 traffic17.590/17.434 MB. Profiler
durations98.496/96.192 us are diagnostic, not ordinary runtime measurements.
The local traffic does not imply an increase in DRAM traffic or prove the
cause of the full-pipeline regression.

Exported SASS localizes this traffic: two `STL` instructions store packed BF16
Hadamard coefficients during setup (after POPC/LEA/PRMT coefficient formation),
and one `LDL.64` reloads those two registers on each fast-path work iteration.
There are no other LDL/STL instructions in this export. Thus recomputing these
small coefficients near MMA use is a concrete next experiment to trade cheap
integer operations for persistent register lifetime/local loads. It must retain
the guarded fallback, pass numerics, and prove a complete-backward gain.
Raw reports, CSVs, numerical manifests and SASS are preserved locally in
`h16-prep/h16-resources-v1.tar.gz`; both captures are terminal.

### Coefficient recomputation result

Added opt-in `recompute_coefficients=True` for grouped H16 and matching research
CLI flags. Stable defaults remain cached. An opaque per-iteration lane operand
allows coefficient reconstruction without persistent coefficient registers.
Initial branch-variable typing and direct `%laneid` inline-PTX compilation
failures are preserved; using a lane input operand passes all68 GPU/CPU tests
in the then-current matrix. One additional CPU dispatch case subsequently
passes locally (6 passed,63 skipped). FP32 numerics and fallback gates unchanged.

NCU confirms local-load/store sectors both fall to zero (from98304/8192),
registers64->62, instructions10,055,100->10,342,844. This verifies the intended
resource transformation. Independent complete-backward24-round timing does
not show a useful win: MXFP8 272.623 us versus candidate247.887 us, paired
1.101420x, CI[1.099624,1.103284], all four AB/BAstationarity checks pass and zero
live-allocation growth. This is essentially the prior grouped H16 result,
still below the retained best and1.20 target. Removing these cached local
accesses alone did not resolve the full-path gap; no promotion/training run.
All artifacts are local in `h16-prep/recompute-evidence-v1.tar.gz` and
`h16-prep/recompute-backward-v1.json`; remote tests/profiler/timing are terminal.

### Retained dual-finalizer timeline and persistence interaction

Extended the diagnostic driver with scoped `--scale-finalizer dual_both`,
requiring an exact schedule manifest and recording source hashes. Seven CPU
profile/manifest tests passed. Captured the retained TMA XOR5 plus dual-finalizer
configuration from `tma-finalizers/confirm-dual_both.json`, not an H16 variant.
Ten warmed public replays, fresh caller RNG, same original BF16 inputs and
compiler combo fusion. Raw artifacts are in `h16-prep/retained-dual-timeline-v1.tar.gz`.

Median node start/end relative to each capture iteration (us): row dY
74.84/133.78; dX maximum reduction148.60/153.83; dX finalizer154.02/169.16;
dX GEMM169.33/209.90. dW finalizer115.63/129.46 and GEMM129.62/168.50 overlap
the late dX preparation/finalization. Fourteen kernels per iteration including
two caller seed-fill kernels; twelve stable graph-node identities plus those
two fills were verified across ten iterations. These separately summarized
profiler medians are nonadditive and not ordinary speed qualification.

Next bounded hypothesis: limit GEMM grid waves to improve finalizer coexistence,
instead of inserting more cross-stream waits. Added a nine-candidate
`tma_persistence` portfolio: independently0/1/2 persistent waves for dX/dW,
everything else retained. CPU tests verify only those two fields change.
Launched `persistence-study-v1` in the H16 remote snapshot, nine trials,
600-second budget,120-second per-worker timeout; exec session60709. Selection
is diagnostic until independent24-round confirmation. Stable defaults unchanged.

Persistence search completed normally in162.85 seconds: nine numerically valid
trials,eight timing-eligible. Selected dX persistent_waves=2,dW=0; eight-round
screen245.590us and paired1.114854x. Fresh24-round confirmation: MXFP8
272.906us,candidate244.789us,paired1.116215x,CI[1.113535,1.118847],all four
order-stationarity checks passed,zero live allocation growth. This is a small
cross-session ratio change versus historical1.111219x,not a direct paired
improvement claim and not target completion. Local artifacts:
`h16-prep/persistence-study-v1.tar.gz` and `h16-prep/persistence-confirm-v1.json`.
Both study/confirmation workers exited normally. Launched sequential
decoder2048/decoder1024/decoder512 transfer checks in the same remote snapshot
(`persistence-transfer-*-v1.json/log`); verify their terminal state before reuse.

All three transfers completed normally and all per-order stationarity flags
passed. Decoder2048: MX400.762us/FP4338.849us,paired1.181570x. Decoder1024:
184.856/161.114us,paired1.148104x. Decoder512:105.152/106.984us,
paired0.971295x,CI[0.961833,0.980277],a measured regression versus MXFP8.
Persistence does not achieve the representative-suite target and should not
be selected universally. JSONs are local in `h16-prep/persistence-transfer-*-v1.json`.

Implemented opt-in paired E4M3 conversion inside the dual finalizer
(`pair_values=True`,researchCLI `--dual-pair-values`), reusing the prior paired
rounding implementation without changing division, uniform indexing, native
layout, or FP32 outer scaling. Odd scale-array tails are masked.49 tests passed
on5060: scalar/paired, row-major/native, divide/power2probability, unequal
partial counts, staged/warp-inline/CTA-inline maxima, changed inputs/uniforms,
zero inputs,graph replay and live memory.21 focused localCPU tests passed before
GPU qualification. Launched a24-round full-backward comparison on decoder1536
with retained producer/GEMM schedule (`dual-pairs-backward-v1.json/log`,session63757).
No default
or numerical gate changes.

Paired-dual comparison completed normally: MX269.763us,FP4262.772us,
paired1.025750x,CI[1.023615,1.027760],all order-stationarity flags true and zero
live growth. This is substantially worse than retained dual finalization;
pairing conversions alone is rejected for promotion. Local artifacts:
`h16-prep/dual-pairs-backward-v1.json` and `qualification-dual-pairs-v1.log`.
All transfer/qualification/timing workers from this step are terminal.

### Paired-finalizer geometry and exact bracket proof

Added `tma_paired_finalizers` to the bounded joint tuner, composing paired
conversion with the existing20-candidate finalizer-grid/thread portfolio.
17 CPU tuner tests passed. Launched `paired-finalizer-grid-study-v1` in the same
remote snapshot,20trials/900seconds/120seconds per worker,exec session63230.
This tests whether launch geometry recovers the paired-conversion regression;
no automatic promotion or gate changes.

Separately, a CPU-only arithmetic test demonstrates exact positive-scale
E4M3 bracketing from FP32 bits: normal low code is `(bits>>20)-960`, subnormal
low code is `trunc(value*512)`, with exact decoded endpoints and high=low for
representable values. The independent reference uses actual Torch FP8
round-to-nearest conversion plus neighbor codes. Tested all127 finite positive
levels, all adjacent midpoints and immediate FP32 neighbors, and100000 seeded
log-distributed samples clamped to[0,448]; codes,endpoints,and rounding
probabilities match exactly. `tests/test_ms_eden_scale_bracketing.py` passed.
This is not yet a GPU implementation or evidence of faster execution; it
motivates replacing repeated scale conversions without changing SR uniforms
or FP32 division/probability in a future opt-in experiment.

Paired-finalizer geometry study completed normally in355.4seconds. Best eligible
screen261.601us uses finalizer blocks dX208/52,dW104/52; this did not recover
the retained finalizer's performance. Full study preserved locally as
`h16-prep/paired-finalizer-grid-study-v1.tar.gz`. No promotion.

Implemented `bit_brackets=True` as a separate opt-in dual-finalizer experiment
(`--dual-bit-brackets`,incompatible with paired conversion). It retains the
existing FP32 normalized division and rounding-probability implementation,
uniform indexing,physical layout and outer scale. Negative-sign values,
including signed zero, use the original path.74 GPU tests passed in19.68s,
including exact FP8 level/midpoint/adjacent-FP32 boundary comparisons at uniforms
0,0.5,1-2^-24,plus all existing row/native,odd-tail,partial-count,zero,changed-input,
graph and live-memory checks.22 focused CPU tests passed separately.
Launched full24-round decoder1536 comparison as `bit-brackets-backward-v1.json/log`
in the same isolated5060 snapshot. No numerical gates or stable defaults changed.

Exact-bit comparison completed normally: MX268.389us,FP4242.549us,
paired1.106352x,CI[1.103823,1.108773]. All four order-stationarity checks passed,
live growth zero. It avoids the paired-conversion regression but does not
establish an improvement over retained dual finalization or reach1.20x.
Local JSON and qualification log are in `h16-prep/bit-brackets-backward-v1.json`
and `qualification-bit-brackets-v1.log`. All jobs from this step are terminal.

### Shared-dY disjoint warp roles

Reviewed the earlier shared-input loss: vectorized row loads still left scalar
column loads and128 registers/thread with both direction states live. Added
opt-in `WarpRoleSharedDY` (`compile_shared_dy(...,warp_roles=True)`) assigning
half the warps to each orientation. One32KiB tile is loaded once as before;
each warp caches one sign vector and computes only its assigned direction.
Partial arrays retain the existing size; unused-role warp maxima are explicitly
zeroed. Whole-CTA barriers remain outside role divergence. No math or defaults
change. Research wrapper exposes `--warp-roles`.

48 GPU cases passed in32.11s on5060: two shapes,128/256threads,VPL8/16,
scalar/vector row loads,two swizzles,and both old/new ownership. Each case
checks Gaussian,outlier,tiny,zero data,distinct changed signs,input replay,
byte-exact packed data/corrected scales and exact global maxima against two
independent original producers,plus zero live growth. Log is local in
`h16-prep/qualification-shared-warp-roles-v1.log`. This establishes correctness,
not lower register count or better speed. Generated resources and timing must
be measured before composing into the current TMA/dual-finalizer full path.
GPU qualification process is terminal; no other job was launched this step.

### Shared warp-role timing and resources

Added a bounded isolated comparison/capture driver with exact large-shape
output/maxima checks. At4096x1536,128blocks,VPL16,128-bit row loads,swizzle0,
24 randomized graph timing samples gave128threads: shared102.356us versus
split-role104.676us;256threads: shared93.816us versus split-role95.014us.
These diagnostic isolated medians do not justify full-backward integration.

NCU256-thread counters show why the proposed register benefit did not appear:
registers/thread only128->124,active warps29.74%->29.54%,zero local traffic for
both,instructions17,432,699->17,254,523. Profiled times172.480/172.512us are
not ordinary latency. Thus caching one sign vector in source did not materially
lower the compiled resource envelope. Both ownership implementations remain
research-only. Raw timing,counter reports,manifests,logs are preserved locally
as `h16-prep/shared-roles-evidence-v1.tar.gz`; all jobs are terminal.

## Artifacts

- Full record: [execution journal](ms_eden_speed_target_20261026.md).
- Local reports: `autotune_reports/ms_eden_speed20_20261026_5060/`.
- Earlier study and confirmations: `joint-grids/study-v3/`, `joint-grids/confirmations/`.
- Current experiments: `bank-compact/`, `replay-diagnostic/`, `packed-signs/`, `sign-cap/`, `occupancy/`.
- Historical denial: `joint-grids/counter-recheck.log`.
- Remote study: `/home/achiya/rtx-mseden-joint-grids-20261027.I9FptI`.
- Remote fresh denial: `/home/achiya/rtx-mseden-counter-recheck-20261027.vzEVsR`.

The research scorer now honors per-order timing instability; the joint study
requires explicit AB/BA quality evidence under a new measurement identity.
26 focused CPU adapter/tuner tests passed. Historical artifacts remain intact.
