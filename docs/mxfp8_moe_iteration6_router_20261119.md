# Iteration 6B: flat Sinkhorn forward geometry

Changing only forward Sinkhorn geometry from **128/1/4 to 256/1/1**
cut the matched E32 standalone router F+B from **2.733 to 1.751 ms**
on HOME RTX 5090 rank 0. This is router time shared with BF16, not
an MXFP8-only improvement. The same geometry failed the release gate on
the held-out E56 fixture: it changed the hard route on mutable X. **Do not
promote the new opening.** The production flat tuner and its existing
tests were restored byte-for-byte to the frozen starting point.
The previously selected backward **256/1/1**
is fixed in every E32 arm. No kernel, solver, affinity, metadata, quota,
or packed-routing algorithm changed.

## Matched forward intervention

T16384/D384/E32/K4/H320/S192 uses one global pool, the `combined` cell
from iteration 5 `flat-bc.json` (4.111 ms whole native MoE at the prior
measurement), and its exact X, dY, parameters, hard IDs, and inverse.
The frozen runtime was
`759c030b5867e029ebdd40def2bac8d8ebee82154075d06a2a79c31b324a79b9`.
Torch 2.13.0+cu132, 170 SMs, row-broadcast affinity, Torch QB, CuTe
stable metadata, solver budgets 5/5/64. Full-router measurements use
user-side fullgraph/static compilation with max-autotune, not the
tuner's detached-score objective.

| Forward threads / row groups / column groups | One real column pass (µs) | Complete Sinkhorn forward with history (ms) | Router F+B (ms) |
|---|---:|---:|---:|
| 128/1/4, improved baseline | 247.735 | 1.277232 | 2.733116 |
| 128/1/1 | 77.808 | .426436 | 1.874640 |
| 256/1/1 | 45.078 | .299377 | 1.751144 |

Each phase used four alternating-order visits × seven ordinary CUDA-event
graph batches calibrated to at least 10 ms. All 36 visits met the existing
stationarity gate. The best candidate's full-router savings were
975.7–982.7 µs on every paired visit. The column measurement starts at the
actual final row shift, computed by the preceding four iteration pairs and
fifth row pass. Its history slice matched the complete execution bitwise.
These scopes overlap; do not add their gains.

The separate matched whole-layer integration on this same cell measured
**4.112 → 3.121 ms** native MoE F+B (24.1% less),
with the archived input/gradient/parameter and hard-route identities and
changed-dY checks. The four paired savings were 987.5–991.5 µs. With the identical improved router, compact checkpoint
BF16 was **3.344 ms** and saved BF16 was **3.003 ms**. Benchmark-only
capture-GC isolation was enabled for that integration factorial; it is not
a runtime policy. See `/tmp/rtx-moe-iteration6-20261119/integration/flat-forward.json`.

The forward column does three scans (max, sum, history) and a group of four
columns/CTA at 128 threads schedules 512 rows per lane, versus 128 rows per
lane at 128/1/1 and 64 at 256/1/1. There are 8 versus 32 column CTAs; these
are source-level launch/work counts, not achieved occupancy or traffic. The
same-thread comparison isolates column grouping; the wider comparison also
changes the row block size. The complete-forward result establishes that
this did not offset the column win. No new profiler-counter claim is made.

For caller and seeded synthetic scores, independent FP64 gates remained
transport `(rtol=5e-5, atol=5e-6)`, score adjoint `(5e-4, 8e-6)`, and
selected-softmax weights `(5e-5, 5e-7)`. Largest E32 absolute errors across
all arms were `1.31e-6`, `8.86e-7`, and `2.45e-7`, respectively. Each
arm scheduled the hard assignment from **its own actual probabilities**.
Assignment, expert IDs and inverse hashes matched the control on both
calibration fixtures; compiled public-router IDs and inverse also matched
on replay with original input, changed input/upstream, and restoration.
The matched weights may differ in FP32 rounding. The independent complete
router oracle used FP64 affinity, transport and softmax on the actual
nondifferentiable hard slots. It checked dX, the projection and expert-token
gradients (worst NRMS .00178), and both scalar gradients with the existing
`2e-4/1e-5` scalar gates, including a nearly constant changed upstream.
All values were finite. Live allocation was exactly flat across six
synchronized post-warmup iterations in each arm. Process peaks include
earlier timing graphs retained for the paired visits and are not
independent per-arm footprint comparisons.

### E128 is not a matched win

On the archived D384/E128/K4/T16384 flat shape with backward 256/1/2 fixed,
the fresh isolated Inductor cache did not reproduce the incumbent's archived
hard hashes. Its trace (`e128-forward.log`) is a reconstruction negative,
not a result against either challenger. One fresh-process retry using the
original shared Inductor cache reconstructed the exact incumbent. The two
forward alternatives passed the FP64 checks but changed the compiled
public-router IDs/inverse on both original and changed X. They also changed
actual hard assignments on the evaluator's synthetic fixture. Reject both
for matched comparison; the raw column/forward/router times in
`e128-forward.json` are diagnostic even though stationary. No E128 winner
sidecar or whole-layer claim was produced.

## Search policy

The discarded flat tuner proposal (in `opening-src/rtx/autotune/routing.py`)
preserved the existing first four admissions, including both hard alternatives
and the same-hard backward control. It replaced the second static structural coverage slot
with a same-plan forward challenge derived from the improved backward arm;
the displaced plan remains available through ordinary structural search.
The one-column plan is considered only when the current one-row-group,
four-column-group forward has at least 256 scheduled rows per lane and
one CTA per expert fits in one SM wave. The widest legal block leaving at
least 48 rows per lane is tried, not imposed as the runtime winner. This
preserves the local and exploration slots, explicit priors/predictors and
user reservations. The counterfactual check for changed forward geometry
holds the actual hard policy fixed while separately comparing the two
transports. It runs at qualification, outside the timed and runtime paths;
the existing common-oracle-probability hard-backend check stays in place.

The proposal changed the flat winner source identity; the packed source identity stayed
`ff84b82703fc63e11e3cd2c1baad3e9a68a2d7cb2d342bd133c6e25788580f68`
in both snapshots. The config token, operator revision and prior schema are
unchanged. Targeted proposal CPU tests: 37 passed, 3 CUDA cases skipped. They cover
real distinct prefix admissions, both SM and scan-length sides, unchanged
legacy/forward/short/opening controls, explicit prior and reservation
order, deadline and timing rejection, and a small regression proving
common oracle probabilities miss a hard ordering change.
After restoring the existing tuner and tests, the same focused suite
passed 32 tests with three CUDA cases skipped.

### Eight genuine held-out admissions

T14336/D384/E56/K4/H320/S256, B8×1792 diagnostic partition with one global
pool and active width `4×320+256=4D`. Two fresh processes continued the
same journal from four to eight unique attempts, preserving the four raw
prefix lines byte-for-byte. Invocation including setup took 14.12 s and
14.92 s respectively. Trials 1–7 were correct and stationary; the eighth
was rejected without timing when actual probabilities changed a hard
assignment. There was no timing fallback.

| Trial | Config ID | Admission | Detached objective (ms) |
|---:|---|---|---:|
| 1 | `044b0167967033012957d9d0` | Cold native-column QB | 3.642 |
| 2 | `4823230bc5eece9bba6c9ac9` | Torch QB control | 3.085 |
| 3 | `6af25cebf5e33033508ef410` | Same hard plan, backward 256/1/1 | 2.573 |
| 4 | `374c70dbc161be07d91c744a` | Retained native QB | 3.616 |
| 5 | `4e9ab12146f8d17f2981c8cf` | Same backward/hard plan, forward 256/1/1 | **1.731** |
| 6 | `296d903bb09a814b03851555` | Retained structural | 3.595 |
| 7 | `88358ea605b22a42da5644e3` | Coordinate-local | 2.294 |
| 8 | `8a2245402c9ef2ab8eeba08e` | Random exploration | Hard mismatch |

The tuner objective covers two detached-score fixtures; it is not an
absolute compiled-router or whole-layer timing. Winner #5 recorded exact
actual-transport hard outputs for both fixtures. B's opening-only source
snapshot runtime digest is
`01414bd0f2a258ffefc6bfb2a26d293afd6efe77d0f43f022e926787bd6fb2ce`.

**Replay rejection:** In a fresh runtime process, the disk winner matched
trial #5 and successfully bound to the fullgraph public router. At fixed
addresses, the original/restored hard IDs and inverse matched the old
forward plan; the changed input's IDs and inverse did not. Independent
FP64 transport/weights and every router gradient passed for each arm on
its own schedule, and graph outputs matched fresh execution. This is a
cross-geometry hard mismatch, not a stale graph. The 2.709 → 1.871 ms
standalone router observation on this shape therefore remains diagnostic.
The first cache audit recorded one late +57,344 B live-allocation step;
a single fresh repeat logged eight prewarm and six postwarm samples flat
at 461,196,288 B for the selected arm. Preserve both logs; neither memory
behavior nor small FP64 transport error overrides the hard-route gate.

## Artifacts

Raw output, exact config and immutable snapshots are under
`/tmp/rtx-moe-iteration6-20261119/router/`. `e32-forward.json` contains
values, all timing samples, validation errors, hashes and memory samples;
`e32-candidate.router.json` can be passed to the whole-layer paired
composition runner. Keep the selected router identical for both the compact
checkpoint and saved BF16 comparisons.

From `e32-src`, replay the measured component matrix under the shared lock:

```bash
out=/tmp/rtx-moe-iteration6-20261119/router
fixture=/tmp/rtx-moe-iteration6-20261119/fixtures/flat384.json
router_python=/home/achiya/vs_code_main/rtx/.venv/bin/python
CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR="$out/inductor" \
  flock /tmp/rtx-moe-backward-20261119-gpu0.lock timeout 540s \
  "$router_python" -m benchmarks.qualify_flat_router_forward \
  --fixture "$fixture" --forward 128,1,1 --forward 256,1,1 \
  --output "$out/e32-forward-repro.json"
```

From `opening-src`, run both caps with one fresh journal:

```bash
cold=/tmp/rtx-moe-iteration5-20261119/router/cold-initial.router.json
for budget in 4 8; do
  CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR="$out/inductor" \
    flock /tmp/rtx-moe-backward-20261119-gpu0.lock timeout 540s \
    "$router_python" -m benchmarks.qualify_flat_router_forward \
    --fixture "$fixture" --heldout --tune-trials "$budget" \
    --tuning-dir "$out/heldout-tuning-repro" --tune-initial "$cold" \
    --output "$out/heldout-${budget}-repro.json"
done
```

From `heldout-audit-src`, verify the persisted eight-trial winner in a new
process with the exact same cache, including mutated input/upstream:

```bash
CUDA_VISIBLE_DEVICES=0 TORCHINDUCTOR_CACHE_DIR="$out/inductor" \
  flock /tmp/rtx-moe-backward-20261119-gpu0.lock timeout 540s \
  "$router_python" -m benchmarks.qualify_flat_router_forward \
  --fixture "$fixture" --heldout --tuning-dir "$out/heldout-tuning" \
  --tune-initial "$cold" --candidate-router "$out/heldout-candidate.router.json" \
  --output "$out/heldout-replay-audit-repro.json"
```

Inspect `validation.selected.eligible_for_matched_timing` as well as the
process exit status: the runner deliberately stores diagnostic timings for
hard-mismatched arms instead of treating them as matched comparisons.
