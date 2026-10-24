# Main-branch qualification

Updated 2026-10-25. This is a handoff for the documented support surface,
not a 1.0 release announcement or a guarantee of universal speedup.

## Completed evidence

| Requirement | Evidence and boundary |
|---|---|
| MXFP8/NVFP4 production contracts | 49-case production matrices pass on RTX 5090 and both available laptops (5060 and 5070), with coverage of both supported Torch stacks. See the detailed report for exact revision/device combinations. |
| GLU and compiled integration | Local full suites, laptop deployment tests, independent quantized FP32 references, eager/compiled execution, gradients, ragged shapes, and cache/lifetime checks. These are finite test matrices, not all possible shapes. |
| Modest-budget first use | Both laptops completed three-shape 8/16/32-trial MLP studies and fresh-process reuse checks. The 5070's paired deployment improvements were 2.095x and 1.148x over RTX incumbents on two shapes; the small shape retained its incumbent. These are not BF16 comparisons. |
| Matched training | Eight policies, 256 compiled steps each, on 5060 and 5090 with matched Torch 2.12.1, seed, pinned Ettin/WikiText assets and BF16 optimizer states. All recorded losses/states were finite. This is pretrained fine-tuning, not from-scratch or long-run convergence certification. |
| Package/install | Clean dependency environments and installed-wheel GPU checks on both Torch stacks; rebuilt `e7bac04` wheel/sdist pass validation and source inspection. Later changes are tests/documentation. |
| Publication | Main and GLU integration are published. Promotion revision 3 fixes unstable observations re-entering finalist selection; focused CPU and cross-process CUDA checks pass on desktops and laptops. |

NVFP4 is not numerically interchangeable with BF16: Ettin retains a material
NVFP4 quality gap. MXFP8 is much closer on the measured workloads. Keep the
policy-specific quality results when choosing precision; do not infer equal
quality merely because both satisfy their prespecified gates.

## Approved noisy-timing behavior

When timing evidence is unstable, production may use a correctness-checked
initial schedule **for the current process**, with a warning. It never installs
that unmeasured schedule as a durable winner. MXFP8 current-input validation
and NVFP4's standard harness validation have different scopes; neither promises
optimal performance. Explicit offline tuning still fails without eligible
timings. The strict singleton clock test is opt-in through
`pytest -m timing_environment`; correctness/fallback/cache tests remain mandatory.

## Focused integration follow-up

A 5060 full-suite test observed a 22,016-byte live-allocation increase.
It did not reproduce in 120 same-adapter evaluations across three environments,
the traced 30-test deployment file plus four extra repetitions, or the untraced
30-test deployment file. Those checks retained the original memory assertions.
No root cause or kernel leak fix is claimed.

The uninstrumented integration run at `9e0fa6f` reproduced the same increase
in `test_repeated_forward_mix[selected_direct]`: three samples of 67,394,560
bytes followed by 67,416,576 bytes. The original plain-training case passed.
The run also exposed a stale interrupted-recovery assertion demanding two
finalist rounds when only one timing-eligible candidate remains. The broad
run was stopped after preserving failure events; it is not a passing suite.
Focused diagnosis localized the transient allocations to gradient-sized blocks.
In a 160-evaluation mixed-workload probe, an increase returned to the exact
baseline during idle cleanup, without another model call or cache eviction.
This is evidence of delayed cleanup, not persistent growth; it does not identify
the precise internal owner or establish a kernel leak fix.

The deployment test now observes bounded idle cleanup (one-second polling
deadline), records raw samples and preserves its exact allocation bound.
Persistent growth still fails. Six CPU protocol cases cover these guarantees.
The corrected singleton-finalist assertion and deployment checks pass together
on the 5060: **32 tests and four subtests**. Selected-direct deployment checks
also pass on both 5090 Torch stacks (seven each); the local interrupted-recovery
file passes two tests and four subtests. These focused results close the observed
failures; the stopped broad run is not retroactively reported as green.

## Evidence and release boundary

[Detailed qualification report](postmerge_qualification_20261024.md) records
revisions, commands, limitations and artifact names. Raw artifacts live under
`autotune_reports/postmerge_20261024/` and are intentionally not published as
package contents. [Release checklist](release_checklist.md) remains authoritative
for a future tag/package release. No 1.0 tag or package upload has been made.
