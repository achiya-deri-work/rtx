# Clean HOME GPU0 recollection

The completed qualified prefix is now finalized; see the
[gap and prior follow-up](balanced_moe_gap_and_prior_20261118.md) for the
40-visit audit, shared-inclusive dataset and bounded-search correction.

This replaces neither raw evidence nor the quarantine markers described in
the [GPU0 incident](balanced_moe_gpu0_incident_20261118.md). No old study,
prior, seed sidecar or runtime winner is used to initialize this campaign.

## Gates and scope

`benchmarks/collect_moe_clean.py` wraps the existing study, not a new tuner.
Each isolated visit uses its existing progressive/mixed shared-UCB search,
full max-autotune and subprocess GEMM tuning. After completion it requires:

- all-leaf finiteness and the five independent quantized FP64 expert checks;
- two fresh-gradient CUDA Graph replay checks for native/BF16 whole/expert paths;
- five identical synchronized live-allocation samples;
- clean device-fault logs, journal/identity integrity and no new GPU0 Xids.

Only then does it publish a new immutable CSV/Parquet export snapshot and
admit another GPU visit. Fatal CUDA faults terminate the owned process group;
failed post-visit gates quarantine the study. Unknown kernel-log access is
not treated as zero faults. HOME GPU1 and unrelated workloads are untouched.
The launcher must restrict visibility to GPU0; this HOME recipe checks PCI
`0000:01:00`, rather than treating faults on GPU1 as faults from the study.

There are 32 shape/packing contexts: four intended MoE architectures crossed
with tiny, medium, mixed, many-short, paired-total, empty/boundary and large
token profiles. Architecture coverage precedes deeper cumulative 8→16→32
router/expert budgets. Shared MLP gets eight trials instead of untuned
defaults. Its journal is separate in each budget visit; shared trials are
not falsely described as cumulative 8→16→32 search.

No optimizer/convergence test, speed target, learned-prior deployment or
exhaustive search is implied. Exported curves use best *observed* eligible
candidates; interrupted or missing budgets remain missing. The time bound
controls admissions and each visit has a hard timeout.

## Pilot attempts

`autotune_reports/moe_clean_recollection_20261118_home0_v1` is preserved as a
failed pilot. Its guard stopped before expansion on a host-side TypeError in
shared-MLP mixed-pool generation, not a CUDA fault. A normalized compound
backward schedule serialized `quant_b=None` (inherit A); the updater treated
it as a dictionary. The corrected updater handles this as a reset to updated
A, while omitted/empty B updates preserve an independent B policy. A focused
configuration regression covers both meanings.

The corrected immutable snapshot is in
`autotune_reports/moe_clean_recollection_20261118_home0_v2/source`, with a
fresh study, runtime cache and compiler cache. Pilot and subsequent collection
share exactly this source and manifest. Source changes require a new bundle.

The v2 GPU pilot completed all gates with eight router, eight expert, eight
shared-MLP and two affinity observations: **26 raw observations in four tuning
contexts**. The common audit reports zero errors, warnings and malformed
tails; the five live-allocation samples are each 53,516,288 bytes, and worst
expert-oracle normalized error is 1.20e-7. No new GPU0 Xids were present.
These are validation results, not a new speedup/convergence claim.

The first postprocessing attempt incorrectly rejected `journalctl`'s exit 1
with the exact `-- No entries --` response. The reader now distinguishes this
from permission/unknown failures. The completed GPU result was independently
re-audited without rerunning it; its initial negative host-reader marker is
preserved at `checks/visit-001/superseded-reader-marker.json`. Only that exact
non-CUDA condition was superseded. Genuine CUDA-fault quarantines were not
changed. The scientific runtime and benchmark drivers stayed immutable;
the corrected operational controller is preserved separately as `controller.py`.

After that successful pilot, a **90-minute**, nohup/setsid/disowned continuation
was launched on HOME GPU0. It reused the qualified first point and started
the second architecture. The manifest contains 32 contexts and cumulative
8/16/32 budgets; the time limit does not promise completion of every point.

## Commands and artifacts

Run from an immutable checkout/snapshot, with its path on `PYTHONPATH`:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/source python \
  /path/to/source/benchmarks/collect_moe_clean.py \
  --source /path/to/source --output /path/to/new-bundle \
  --max-visits 1 --max-seconds 600 --shared-trials 8

# Only after the first visit qualifies; resumes the same journals/manifest:
CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/path/to/source python \
  /path/to/source/benchmarks/collect_moe_clean.py \
  --source /path/to/source --output /path/to/new-bundle \
  --max-visits 96 --max-seconds 5400 --shared-trials 8
```

`latest.json` is the last qualified checkpoint, not a claim that all work has
finished. `checks/visit-NNN/` contains numerical/memory, log-health and common
journal audits. `exports/visit-NNN/` contains corresponding CSV, Zstd Parquet
and summary snapshots. `study/` contains independently durable results and
append-only residuals. Pilot/launch logs and source archive are retained.
The older quarantined studies remain unchanged and excluded from ingestion.
The MoE-specific export has 18 pilot rows (expert/router/affinity); the eight
shared-MLP rows remain in their original journals for the common loader and
future whole-library modeling. Do not confuse the narrower CSV scope with
missing raw observations. For this particular v2 bundle, use its archived
`controller.py` with `PYTHONPATH` pointing at `source/` rather than the original
pre-correction controller embedded in the source archive. `launch.json`
records its hash and exact command/environment.
