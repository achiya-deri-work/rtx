# Recomputed MLP autotuning: coverage before exploitation

The tuner minimizes whole-component forward or forward-plus-recomputed-backward
CUDA-graph replay time. It does not select precision on the user's behalf.
Both `mxfp8` and `bf16_gate` are compiled with `fullgraph=True, dynamic=False`
before capture. Compiler defaults are used consistently for both policies;
this is not an aggressive-Inductor-options comparison.

## Why the search changed

Earlier 48-trial studies exposed a large space but left all four backward
GEMMs decomposed, 128-by-128, and unsplit. Declaring a coordinate does not
demonstrate that it was explored. Two important barriers were quad-quantization
requirements preventing a switch to fused backward, and native scale layouts
preventing a switch to smaller tiles.

The tuner now starts with complete backend transition schedules, including
per-matmul quantization where fused backward requires it, row-major 64-row dX,
and split-2 FP32 dW. It then interleaves legal coordinate probes across
components and independently across dX/dW. The remainder goes to the existing
bandit-driven random, coordinate and learned search portfolio.

Coverage trials use the same correctness gates, confirmation policy,
append-only observations and budget as refinement. Resume skips already
observed identities. Interrupted candidate attempts retain the existing
orchestrator recovery behavior.

## Scope

Search includes the existing gate, down, recompute and four backward-leg
schedules: backend, geometry, staging, quantization, scale transport, register
limits, ldmatrix, swizzles, epilogues, persistence, split reductions and stream
scheduling where implemented. Materialized quantizers now also expose launch,
register-limit and scale-store choices. Layout transitions update quantizers
and consumers together.

Some options intentionally are not search choices:

- Precision, activation and the configured SiLU arithmetic policy are fixed.
- Packed-X/dynamic-W on-chip down currently implements scalar transport only.
- Cooperative GLU with partial output tiles remains rejected because its
  synchronization is not qualified.
- Inactive branches are excluded from candidate identity. Direct GLU stores
  do not acquire new identities merely by changing TMA store-buffer counts.

This is **not exhaustive Cartesian or pairwise coverage**, nor proof that every
declared option is implemented efficiently. Legality can depend on several
simultaneous changes. The audit explicitly records rejected and unreachable
single-coordinate transitions so they can be investigated rather than hidden.

## Running and auditing

```bash
python -m rtx.autotune.mlp \
  --shape 1024,256,512,256 --precision mxfp8 --phase train \
  --trials 256 --seconds 900 --coverage-fraction 0.75 \
  --warmup 3 --samples 7 --calls-per-sample 3 \
  --confirmation-repeats 2 --output-dir autotune_results/mlp
```

Use `--precision bf16_gate` for the explicit hybrid policy. Increase trial/time
budgets for deeper searches. A 32- or 48-trial run cannot cover hundreds of axes.
`--coverage-fraction 1` spends the trial budget on coverage before refinement;
it still does not exhaust the full configuration space. Planning and adapter
initialization occur before the search timer; the coverage plan records its
own planning time. An individual compile/candidate can exceed the search
deadline; use an external process watchdog for experimental campaigns.

Each context directory contains:

- `coverage_plan.json`: full finite candidate plan, selected prefix, declared
  axes and each seed/axis legality, duplicate or update-error diagnosis.
- `coverage_summary.json`: observed status for planned candidates, explicitly
  marking unmeasured ones, plus values of active fields in successful trials.
  Written on normal completion or a catchable failure, not guaranteed after
  a process kill; durable observations remain the recovery source.
- Existing residual observations and events: numerical errors, repeated
  timings, compilation costs, failures and configuration identities.

Correctness uses an independent quantized reference and seeded nonconstant
upstream gradients. Numerical thresholds are unchanged. Timing observations
from the old eager-native benchmark contract are invalidated by context
revision 33 and new compiler/gradient-probe tags.

## Local validation, RTX 5070 Ti

The repaired 48-trial native study at `1024,256,512,256` passed all 48 numerical
checks. It exercised both backward backends for all four legs, 64-row dX and
split-2 dW, unlike the older search. Initial replay latency was 179.49 us and
the best candidate observation was 71.84 us. This is a within-study result,
not a fresh-process speedup claim against BF16 or a global optimum.

Artifacts: `autotune_reports/mlp/coverage_v4_native/` and its adjacent result
JSON. Coverage reports are intentionally not shipped as runtime winners.

The corresponding `bf16_gate` study also passed all 48 candidates, with
86.46 us initially and a 73.30 us winning observation. Maximum relative-L2
error against the independent quantized reference across candidates and
outputs/gradients was 0.0149% native and 0.0100% hybrid. These are implementation
parity errors, **not** errors against an unquantized BF16 model.

Fresh-process confirmations used 15 timing samples with five calls per replay,
plus the existing two repeat confirmations: 73.34 us native and 75.12 us hybrid.
The stronger measurements are the appropriate handoff numbers; the lower
search observations should not be presented as guaranteed steady-state times.
Validation also passed 35 CPU orchestration/documentation tests and 13 CUDA
frontend/compiled-autograd tests on Torch 2.13.0+cu132.
