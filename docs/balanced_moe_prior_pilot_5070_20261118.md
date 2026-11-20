# BalancedMoE RTX 5070 first-hit pilot

Two fresh contexts use the same frozen runtime and study drivers as the HOME
pilot: runtime SHA256
`bf5b7615dbdcbd083304446b36599e0604b346fb71f8c4e9051c1069da187d10`.
The laptop used its already archived source plus a reviewed nine-file
source-only overlay; no models, prior artifacts, or old tuning data were
transferred. The baseline archive and overlay SHA256 values are recorded
with the evidence.

The driver runs progressive mixed search with up to eight expert and eight
router trials per shape, fixed shared schedule, Inductor max-autotune, and
two warmed CUDA Graph replay rounds of seven samples. Each visit starts a
fresh process with a fresh study residual directory. No learned prior or
runtime rule was promoted. These are local first-hit results, not a
measurement of transferred prior benefit.

The data gates include independent FP64 quantized expert forward/backward
on the selected routes, real compiled all-leaf gradients, replay checks,
and no postwarm live-allocation growth. Route hashes are recorded per
context; different legal GPU-selected schedules are not assumed to yield
bitwise-identical results across devices.

## Results

Both contexts completed with eight eligible router and eight eligible expert
trials. The 36 saved observations also include four affinity candidates:
three eligible and one accuracy rejection. The medium row-broadcast
candidate failed its existing gate on 6 of 1,024,000 tensor entries
(max absolute error 2.44e-4); the qualified run used scalar affinity. Both
finished visits used scalar affinity and staged transport.

| Context | Worker wall | MXFP8 whole step | Checkpointed BF16 | MXFP8 routed experts | BF16 routed experts |
| --- | ---: | ---: | ---: | ---: | ---: |
| D384/E32/K4/H128, N128 | 123.8 s | 0.864 ms | 0.530 ms | 0.231 ms | 0.131 ms |
| D512/E64/K8/H128, N2000 | 119.8 s | 4.854 ms | 2.957 ms | 1.617 ms | 1.877 ms |

Timings are medians of the two event-round medians. The tiny native full-step
rounds were 0.932 and 0.797 ms, so its magnitude is descriptive under WSL.
The medium expert-only step was about 14% faster, but the complete layer
was slower; isolated times are not additive. The best observed tuner router
latencies were 0.338 and 0.674 ms, and expert latencies were 0.231 and
1.593 ms, respectively.

Both results had zero live-allocation growth, with worst FP64 oracle
relative L2 at most 9.02e-5 and worst replay relative L2 at most 0.001924.
The tiny input, upstream, router/expert parameter, and hard-route hashes
matched HOME; three shared parameter hashes did not. This supports a
same-contract comparison, not bitwise-matched full-layer device scaling.
The residual analyzer reports two raw device keys because its affinity
context lacks a bus width; both keys here are from the one physical 5070.

[Pilot manifest, command lines, raw residuals and traces, source overlay,
and CPU analysis](../autotune_reports/moe_next_priorities_20261118/prior_pilot_5070/).
The source overlay is applied to the existing
`routing_hybrid_5070/qualification-source.tar.gz` baseline archive
(SHA256 `dc474cbfee53a48e45e3cd0b5f7b885c172e6b6ee58f5868b30d0f0c832a2846`).
GPU utilization and memory returned to 0% and 0 MiB after both workers
exited; the laptop was released without changing any device settings.
