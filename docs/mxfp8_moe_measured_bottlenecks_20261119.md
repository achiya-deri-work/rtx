# Why the current thin MXFP8 MoE trails BF16

## Conclusion

Thin shapes make preparation harder to amortize, but **the dominant measured
gap is not simply that BF16 also fails to use tensor cores**. The selected
MXFP8 direct epilogue generates four times the logical output's store-sector
traffic. Existing TMA epilogues remove that problem. Bounded tuning missed
these configurations, including a coupled geometry/buffering/store move.

On the complete large MoE, diagnostic store configurations remove **75.4% of
the BF16 latency gap**, without changing quantization or numerical policies.
This is a measured configuration A/B, not a production implementation change
or a newly installed runtime winner.

## Scope and measurement

HOME RTX 5090 GPU0, SM120, Torch 2.13.0/cu132, CuTe 4.7.0. GPU1 was untouched.
Large model fixture: N=29,504, D=384, E=32, K=4, expert H=128 and shared H=256.
Both FFNs are recomputed in native; BF16 independently checkpoints the FFNs,
not routing. Max-autotune/fullgraph compilation, all parameter/input gradients,
set-to-none backward and outer CUDA Graph replay are matched.

Ordinary timings exclude compilation, warmup and profiling. The final full
comparison uses four alternating visits and seven samples per visit. Pure
GEMM timing captures 32 launches per graph to amortize host replay overhead.
Six legal geometry/pipeline templates were measured per isolated geometry;
additional epilogue panels explicitly compare the existing store families.
Profiler durations are diagnostic, not substituted for ordinary latency.

## 1. The BF16 kernels are using tensor cores effectively

These counters profile the actual expert branch with the real hard routing
frozen beforehand. No router executes inside the injected process.

| Actual expert kernel | Ordinary trace | Tensor-pipe active, elapsed | DRAM throughput |
|---|---:|---:|---:|
| BF16 down | 78.6 µs | 60.8% | 1,461 GB/s |
| MXFP8 down, direct | 213.9 µs | 5.4% | 530 GB/s |
| BF16 gate-dX | 109.5 µs | 92.0% | 1,459 GB/s |
| MXFP8 gate-dX, direct | 239.2 µs | 9.7% | 517 GB/s |

Measured streaming-add bandwidth was 1,533 GB/s. Thus BF16 down is a mixed
memory/compute case, not a kernel with negligible tensor work. These ratios
are hardware activity metrics, not percent of theoretical useful FLOP peak.

The down kernels execute **2,850,816 BF16 versus 1,425,408 MXFP8 tensor
instructions**. MXFP8 really halves that instruction count, but loses elapsed
time elsewhere. Its grid has 2,784 CTAs, so lack of a device-wide work wave is
not the cause. No local-memory spills were observed. MXFP8 down has *higher*
active-warp occupancy than BF16 (33.5% versus 16.2%), yet far lower tensor
activity; occupancy alone does not explain or fix the deficit.

## 2. Direct output stores are a large, demonstrated cause

Each down output contains 91,226,112 logical BF16 bytes. BF16 issues 2,850,816
32-byte global-store sectors, exactly that volume. Native direct output
issues **11,403,264 sectors**, or 364,904,448 sector bytes: **4× amplification**.
This describes sector requests/coalescing, not proof of four separate logical
writes or four times the physical DRAM traffic.

Changing only down's existing epilogue at the same 128×128, one-stage geometry:

| Same packed GEMM | Ordinary time |
|---|---:|
| MXFP8 direct output | 213.4 µs |
| MXFP8 staged/TMA output | 70.8 µs |
| Matched functional BF16 | 82.5 µs |

TMA counters show zero LSU global-store sectors (the output goes through TMA),
L2 traffic reduced from 467 MB to 188 MB, and tensor activity increasing from
5.4% to 24.2%, despite occupancy falling. Counter replay changes cache state;
its duration is not the ordinary 70.8 µs figure.

The same store control reduced gate-dX from about 237.1 to 95.6 µs in the
isolated matched problem. For shared down, changing to a **legal 64-row,
two-stage TMA** configuration reduced about 58.7 to 22.3 µs; functional BF16
took 30.5 µs.

## 3. There is a real search barrier, not a missing hardware primitive

For shared down, M=29,504 is divisible by 64 but not 128. The current backend
rejects TMA output for a partial 128-row tile. A 64-row single-buffer consumer-
scale configuration is correctly quarantined after prior numerical failures;
that gate was not bypassed.

Measured legal configurations:

- 128-row, one-stage, direct: 58.7 µs.
- 64-row, two-stage, direct: 61.4 µs, slightly worse.
- 64-row, two-stage, TMA: about 22.3 µs, much better.

A one-coordinate greedy walk can reject the intermediate geometry and never
reach the winning store path. Geometry, safe buffering and epilogue need
coupled proposals. The aligned expert down already permits a same-geometry
TMA move, but that move was absent from its useful bounded opening.

## 4. Preparation remains a second, independent cost

The actual shared FFN took 0.591 ms native versus 0.373 ms checkpointed BF16.
Its trace contains approximately **142 µs of standalone quantization** across
11 kernels, plus fused preparation/activation inside other kernels. Those
142 µs are not an exact counterfactual saving: fusion changes scheduling and
resource use, and every required orientation still needs valid independent
scales.

The isolated expert and shared steps have more than 99.7% first-to-last kernel
coverage under graph replay. Their deficits are not mainly Python/eager launch
gaps. Native experts have 18 launches versus BF16's 23; fewer launches alone
are plainly insufficient.

Holding dimensions fixed independently also confirms that reduction length
matters more than adding output columns: at direct output, increasing K from
128 to 512 barely increases native time (213→248 µs), while BF16 rises
82→234 µs. The native path was dominated by non-MMA costs. Increasing N alone
does not remove the poor-store behavior. Strong functional BF16 still beat the
original direct-output templates in all six tested geometries; width alone
does not excuse the implementation deficit.

## 5. Full-layer causal check

Only existing configurations were composed: expert down and gate-dX TMA
epilogues, and the measured safe shared-forward-down TMA configuration.

| Complete forward/backward | Median |
|---|---:|
| Current native schedule | 4.5980 ms |
| Expert store changes only | 4.2923 ms |
| Shared forward-down store change only | 4.5642 ms |
| Combined diagnostic configuration | 4.2601 ms |
| Matched checkpointed BF16 | 4.1496 ms |

This is a **7.93% speedup**, removing 75.4% of the original gap. Native remains
2.66% slower here. Native outputs and every compared gradient matched bitwise;
all variants passed two changed-upstream replay checks, exact routing and five
identical live-allocation samples. None of these configurations was installed
as a runtime default or cache winner.

## Implications

First fix search coverage of output transport and coupled safe geometry/store
recipes. Then quantify and reduce the remaining shared preparation traffic.
Tiny pools still have genuine launch/padding/amortization limits; these results
do not promise every small shape will beat BF16. Broader shape/device and cold-
cache qualification is required before changing production selection.

## Evidence and caveats

Raw timings, configs, exact fixture, traces, NCU reports and source are under
`autotune_reports/moe_measured_causes_20261119/` on local Storage4. Primary files:
`whole/attribution.json`, `micro-final/results.json`, `counter-summary-final.json`,
`epilogue-expert_thin/result.json`, `shared-safe64/result.json`,
`gate-dx-stores/result.json`, and `whole-store-factorial.json`.

The initial microbenchmark's compiled BF16 `out=` path emitted a separate
116.7 µs copy after its GEMM, despite identical final strides. That is **not**
the production comparison. `micro/` and `micro-v2/` ratios are superseded by
the functional-BF16, graph-amortized `micro-final/` results. Noncontiguous
`out=` was rejected by Dynamo, not replaced with eager execution.

One NCU-injected conditional-routing attempt hit a device assertion/Xid43 and
was terminated by the fault supervisor. It produced no admitted counter data.
Ordinary qualification had passed before it, a fresh CUDA context passed
afterward, and subsequent profiling used fixed routes only. The failed attempt
is retained and marked ineligible. No driver repair/reset or GPU1 operation
was performed.

One initial BF16 counter index selected gate-dW, not gate-dX; its small output
volume exposed that attribution error. It is retained but excluded from dX
comparisons; `fixed-bf16-gate_dx-correct` is the verified final dX capture.
