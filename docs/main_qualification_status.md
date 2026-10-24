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

## Remaining integration gate

A historical 5060 full-suite test observed a 22,016-byte live-allocation increase.
It has not reproduced in 120 same-adapter evaluations across three environments,
the traced 30-test deployment file plus four extra repetitions, or the untraced
30-test deployment file. Those checks retained the original memory assertions.
No root cause or kernel leak fix is claimed.

One final uninstrumented library-suite run is active on the 5060 at frozen
commit `9e0fa6f`, with the approved singleton timing exclusion. Its result is
required before closing this integration gate. No duplicate full suite is
running on the other devices.

## Evidence and release boundary

[Detailed qualification report](postmerge_qualification_20261024.md) records
revisions, commands, limitations and artifact names. Raw artifacts live under
`autotune_reports/postmerge_20261024/` and are intentionally not published as
package contents. [Release checklist](release_checklist.md) remains authoritative
for a future tag/package release. No 1.0 tag or package upload has been made.
