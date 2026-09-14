# First-hit autotuning acceptance audit

This audits the six numbered implementation/evaluation requirements against
the final local worktree and executed artifacts. No kernel or search source
was edited after the final consolidated regression began. Documentation-only
handoff updates followed it. No commit or push was authorized or performed.

## Requirements and evidence

| Requirement | Implementation and executed evidence | Disposition |
|---|---|---|
| 1. Deterministic progressive prefixes, protected local coverage, diverse retrieval, reserved exploration, resume equivalence | `rtx/autotune/effort.py`, strategies and runtime coordinate projection. `test_progressive_effort.py` compares disk-reopened 8/16/32/64 prefixes to uninterrupted 64, checks interrupted durable observations and abandoned attempts. Retrieval/opening and matched-pool tests verify deduplication, coverage and policy separation. Final CPU search batch: 38 tests/11 subtests passed. | Delivered. Equivalence means the same recorded evidence, not identical independent noisy timings. |
| 2. Paired finalist confirmation and durable safe promotion | Separate confirmation events and raw observations; promotion revision 2, balanced rounds, numerical rejection, explicit timing-stability rejection and incumbent retention. Final `test_paired_confirmation.py`, `test_winner_promotion.py`, cache contracts and CUDA refinement passed. Earlier unstable backward evidence remains quarantined. | Delivered. Unknown timing-quality metadata from external generic adapters is not a calibration guarantee. |
| 3. Effort-aware reuse/refinement with compatible evidence | Explicit `reuse`/`refine`, cumulative per-pass caps, revision/device/workload/measurement identities, persisted scheduler state. Fresh-process tests forbid retuning and require compiler-cache hits; host/graph modes share forward but not backward evidence. Concurrent same-context, interruption and eviction tests passed. | Delivered. In-flight compilation is not forcibly preempted by a soft deadline. |
| 4. Matched ordinary/portfolio/rules/ML evaluation | Frozen identical pools and protected prefixes, rankings fixed before timing, independent numerical admission. Historical forward/NVFP4 studies include positive and negative findings. Latest graph backward study: 512 search trials, 3,607 full-pool reference trials, 100 balanced final measurements; clean accounting and independent three-seed winner checks. | Delivered. No training admission of inspected evaluations; no claim of global optimum. |
| 5. Local effort-level and integration qualification | Final Torch 2.13 matrix: 97 tests/42 subtests, zero failures/errors/skips. Includes MXFP8/NVFP4 all retained scaling policies, dynamic/packed states, first backward, aligned/ragged/new shapes, full 8/16/32/64 long-reduction ladders, capture, concurrency, interruptions, delayed state, bounded caches, GLU and recomputed MLP integration. Focused final Torch 2.12.1: 19 tests, zero failures/skips, supplementing the earlier full matrix. | Delivered over the documented local support matrix. This is not exhaustive shape/SKU/software certification. |
| 6. Promote only with prospective quality, tail-risk and startup-cost support | Raw startup/search/compiler/capture/confirmation/model costs are separated. Latest two backward cases have exact selection ties between all budgets/methods and the best paired reference. Forward results are mixed; generalized learned-policy/tail superiority is not established. | Decision gate completed: do not promote an unsupported universal learned default. Keep priors and experimental controls opt-in. |

The sixth requirement is an evidence gate, not a requirement to make ML win.
No universal quality or tail-regret guarantee is part of the delivered API.
Lack of evidence for a universal learned default is recorded as a negative
promotion decision, not hidden by another unbounded research campaign.

## Final artifacts

- `/tmp/rtx_first_hit_full_torch213_v3.{log,xml}`: 97 tests and 42 subtests,
  511.52 seconds; XML contains 139 cases, zero failures/errors/skips.
- `/tmp/rtx_wrapup_torch212_gpu_v1.{log,xml}`: 19 tests, 71.27 seconds,
  zero failures/errors/skips.
- `/tmp/rtx_wrapup_torch212_cpu_v1.{log,xml}`: 23 tests/10 subtests;
  five deliberate GPU skips.
- `/tmp/rtx_wrapup_search_contracts_v1.{log,xml}`: 38 tests/11 subtests;
  one GPU skip and one research-environment skip. The latter was run and
  passed separately in `/tmp/rtx_wrapup_research_rankings_v1.log`.
- `autotune_reports/matched_backward_graph_20260914_v1/analysis.json`:
  exact selection ties, clean journal/budget audits, independent numerics,
  search and confirmation costs and source-artifact hashes.
- `autotune_reports/matched_backward_graph_20260914_v1/oracle_analysis.json`:
  complete reference screening and stable paired top-five comparisons.
- `autotune_reports/matched_nvfp4_jit_20260914_v1/`: earlier frozen forward
  comparison, full reference, independent winner checks and clean audit.
- `autotune_reports/matched_backward_20260914_v1/timing_quality_audit.json`:
  preserved quarantine of earlier unstable evidence, not relabeled data.

## Handoff

Use [first-hit autotuning](first_hit_autotuning.md) for the user-facing recipe,
budgets, cache behavior and graph boundaries. The
[qualification matrix](autotune_qualification_matrix.md) records the supported
tests, and the longer [goal audit](autotune_goal_audit.md) preserves the sequence
of intermediate findings and fixes. Their older pending-work notes are
historical; this acceptance audit records the final disposition of the six
numbered requirements. Further generalization research is optional future work,
not an implied learned-policy promotion or a claim that every shape is optimal.
