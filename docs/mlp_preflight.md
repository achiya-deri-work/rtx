# Disposable MLP first-hit preflight (experimental)

## Public candidate bundles

Use `rtx.MLPCandidate` to transport an exact-shape proposal without extracting
private frontend fields. Import reports produced by
`tools/qualify_mlp_deployment.py` in selected-direct mode:

```python
import shlex
import subprocess
import rtx

candidate = rtx.MLPCandidate.from_qualification(
    "autotune_reports/afrl_actual_shapes_20260917/combined_m11712_paired.json"
)
candidate.save("candidate-m11712.json")  # new file only
candidate = rtx.MLPCandidate.load("candidate-m11712.json")
request = candidate.prepare("autotune_reports/prepare-m11712-new")
print(shlex.join(request["command"]))
# Run before allocating the application, when the GPU is available:
subprocess.run(request["command"], check=True)
inspection = candidate.inspect("autotune_reports/prepare-m11712-new")
print(inspection["status"], inspection["gaps"])
```

The report path above is a local research artifact, not a shipped default.
For an explicitly chosen configuration, construct directly:

```python
candidate = rtx.MLPCandidate(
    (128, 256, 768, 256), config, phase="forward", activation="gelu"
)
```

Shape order is **M,K,H,O**, with H the post-gating width. The current bundle
contract is contiguous 2D, BF16 public operands, MXFP8 precision, and compiled
selected-direct execution. Training and no-grad inference are separate
contexts. Other precision/mode contracts are rejected rather than inferred.
The paired-report importer knows that tool's SiLU/MXFP8 contract; it is not a
generic reader for arbitrary benchmark formats.

`prepare()` is CPU-only and never launches, installs a winner or changes model
parameters. It writes `candidate.json`, `spec.json` and `request.json` to a new
directory. The returned argv launches cold tuning, strict fresh-process reuse
and separate independent-reference checks. Reports live in `qualification/`;
the default cache is `cache/` inside that request directory. An explicitly
supplied `cache_dir` is used instead and can be populated by the launched
qualification. Eight trials are the default; pass a public `AutotunePolicy`
as `policy=` to control effort. `reserve_mib=` and `execution_mix=` forward
the explicit capacity/objective contracts. Preparation does not allocate the
synthetic reserve; its workers do.

Bundles always remain **proposals**, even when their research report says
`promoted`. Provenance includes the source manifest and original artifact hash.
`source_matches_evidence` describes compatibility **at preparation time**;
changed sources require requalification. Preflight independently pins its
actual launch sources. Successful synthetic preflight does not establish a
BF16 replacement win, arbitrary-shape coverage or full-application capacity.
No neighboring-shape transfer or model installation is performed. A resolved
multi-module coverage/installation plan remains separate work.

`candidate.inspect(request_directory)` is read-only and CPU-only. It requires
all three worker reports and checks candidate/spec identity, exact shape,
phase, activation, mode, environment agreement, source consistency, strict
fresh-process reuse and requested synthetic occupancy. It re-evaluates raw
numerical and capacity/live-memory gates rather than trusting a pass flag.
Missing, inconsistent or failing evidence raises; nothing is installed.

The returned `preflight_passed` status describes the recorded run.
`current_sources_match` and `requires_requalification` distinguish that
historical pass from the current checkout; `launch_sources_match_preparation`
discloses source changes between export and execution. The report includes
capacity peaks, numerical metrics and explicit gaps. It does **not** initialize
CUDA to verify the current GPU, inspect current durable-cache contents,
identify every module's selected winner, or approve deployment. Archive files
are local evidence, not cryptographically authenticated attestations. Do not
interpret this single-context inspection as a freeze/installation plan.

New worker reports also contain `selected_plan`: the complete configuration
read from the actual forward's host plan token and its canonical SHA-256.
Preflight rejects selection changes within a worker or between cold, strict
reuse and numerical workers. Inspection returns this plan and
`matches_initial_serialization`. A false value can mean resolved shape-dependent
defaults, not necessarily a different tuning choice; compare the explicit
configurations. This observes execution without searching the cache or adding
a production launcher hook. Older reports without selection evidence return
`selected_plan=None` and an explicit gap; they cannot prove that every worker
executed the same configuration. An incomplete or inconsistent mixture of new
selection records fails inspection.

After successful qualification on the deployment checkout/environment, check
the current device's durable winner explicitly:

```python
coverage = candidate.check_cache("autotune_reports/prepare-m11712-new", device="cuda:0")
if coverage["status"] != "covered":
    raise RuntimeError(coverage)  # reprepare; do not start a live search here
print(coverage["selected_config"], coverage["winner_path"])
```

Unlike `inspect`, this may initialize CUDA to fingerprint the requested GPU.
It uses strict-cache runtime eligibility without tuning, compilation, kernel
launches, cache writes or initial-config fallback. It requires recorded
selection evidence, matching current sources, and the qualified Torch/CUDA/GPU
environment. `winner_changed` means an eligible entry no longer matches the
executed qualification plan; `missing_or_ineligible` covers absent, malformed
or incompatible entries. Both require new preparation/qualification, not
silent adoption. `covered` is a point-in-time exact-context observation; the
cache can subsequently change, so this is not a locked freeze, an application
approval or a multi-module coverage guarantee. Use the consuming module's
`require_cached_()` before compilation to prohibit live tuning on a miss.

For an existing automatically configured `MXFP8MLP`, validate a declared set of
requests and switch it to strict cached execution in one startup operation:

```python
report = rtx.require_qualified_mlp_cache_(layer, [training_request_directory,
                                               inference_request_directory])
print(report["contexts"])
compiled = torch.compile(layer, fullgraph=True, dynamic=False,
                         options={"triton.cudagraphs": False})
```

Supply only contexts actually prepared; use one request for training-only
deployment. Every request must match the module's exact constructor config,
tuning policy, cache path, activation and execution mixes (apart from the
cache-only mode/miss handling). Feature dimensions, BF16 parameter shapes and
CUDA device are checked. Use the same explicit `AutotunePolicy` and initial
configuration when preparing requests and constructing the consuming module.
The module must already be on its qualified device, with no manual shape
overrides. No tensor-sized example allocation is required by this operation.

All requests are checked before changing the module. Duplicate contexts or
any mismatch/gap fail without changing its policy. Successful calls preserve
Parameter identities/values, configuration, precision and optimizer ownership;
only live-tuning/fallback permission changes through `require_cached_()`.
The detached result lists coverage and remaining requirements. Run at startup
without concurrent execution, before caller compilation/capture; recompile
existing callers after mutation. This does not convert BF16 modules, approve
an application replacement, lock a cache, or qualify undeclared contexts.
An undeclared context with a preexisting eligible winner can still run; an
actual strict miss raises instead of tuning. A fully pinned deployment plan
with enforced context whitelisting remains separate work.

The existing module-based spec export below remains available when preparing
the application's own configured initial policy.

## Module-based preparation and qualification

Preflight tunes in a worker process, exits, and runs a second fresh worker with
retuning forbidden and an eligible disk winner required. The parent does not
initialize CUDA. Process exit releases candidate scratch/private allocator pools;
durable winner files remain available to the eventual training process.

The parent pins a SHA-256 manifest of RTX Python sources before launching its
workers. Each worker records `source_hashes_before` and `source_hashes_after`;
the parent requires both to match its manifest and the current source tree.
Missing or changed identity fails qualification, even when memory gates pass.
Do not edit RTX during a preflight. Failed reports and durable winners are
preserved, but a passing cold worker with mismatched sources does not launch
reuse. This is local source provenance, not a dependency lock or verification
that a later application still uses the same environment. Archive reports and
the tested source together; requalify after relevant changes. Older reports
without these fields remain historical evidence, not passes of this new gate.

Export the exact deferred specification from the intended `MXFP8MLP` (before
moving the model to CUDA), using an explicit shared `autotune_cache_dir`:

```python
# layer is your configured rtx.MXFP8MLP with autotune="balanced".
layer.export_tuning_spec("mlp-spec.json")
```

```bash
python -m rtx.autotune.mlp_preflight \
  --shape 11712,256,768,256 --spec mlp-spec.json \
  --output autotune_reports/mlp-preflight-new
```

The output directory must be new. `cold/` and `reuse/` contain worker logs,
reports, and allocator snapshots before the first forward and backward.
`cold` means a fresh process, not necessarily empty disk winner/compiler caches.
The shared cache path, initial configuration, tuning policy, shape, and execution
mix must match the consuming application. This command checks compiled
selected-direct execution with contiguous 2D inputs. Default `--phase train`
checks output and all three gradients; `--phase forward` checks no-grad inference.
Use `--activation silu|gelu|relu` to match the consuming module (default SiLU).
Phase and activation are explicit CLI contracts, not inferred from the tuning
specification. Run separate preflights for training and inference. Private
replay, arbitrary leading dimensions and noncontiguous layouts are not covered.
Changing those contracts can legitimately require different qualification.

Add `--check-numerics` to run a third, strict-cache worker after both capacity
workers pass. It compares compiled selected-direct output (and dX, dGateUp,
dDown for training) against the independent FP32 MXFP8-rounding recipe at
seeds 932 and 933. It cannot tune or fall back on a cache miss. The parent
recomputes acceptance from recorded per-output metrics rather than trusting
a pass flag. This measures agreement with the **quantized recipe**, not BF16
training convergence. The existing tuner-level normalized RMS gate is 0.025;
references with RMS at most 1e-12 instead require maximum absolute error at
most 1e-7. Finiteness of both candidate and reference is mandatory.

`numerics/report.json` records maximum absolute error, RMS error, reference RMS,
normalized RMS, source identity and environment. This worker holds no synthetic
reserve, and oracle allocations never enter the cold/reuse capacity reports.
It may require more memory than deployment because the reference materializes
intermediates and currently computes all reference gradients even for inference.
Failure is retained and fails the command. These two fixed seeds do not replace
the broader ragged/extreme-value, changing-weight or concurrency test matrices.

```bash
python -m rtx.autotune.mlp_preflight \
  --shape 128,128,128,128 --phase forward --activation gelu \
  --spec mlp-spec.json --output NEW_INFERENCE_REPORT_DIRECTORY
```

`export_tuning_spec()` returns a detached JSON-compatible dictionary and can
optionally write a new file. It rejects missing explicit cache directories,
disabled automatic tuning and manual shape overrides rather than exporting a
misleading partial configuration. It does not initialize CUDA or verify coverage.
This is configuration serialization, not `torch.export` of a model.

After preparing and verifying the desired contexts, disable live tuning and
initial-config fallback before compiling the application:

```python
layer.require_cached_()
compiled = torch.compile(layer, fullgraph=True, dynamic=False,
                         options={"triton.cudagraphs": False})
```

Parameters, initial search identity, objective and winner files are preserved.
A missing compatible winner raises with the shape, phase, activation, device
and execution-mode contract; it never starts a search. Recompile/recapture any
already compiled callers after this mutation. This method **does not prove
complete coverage or qualify winners**. Training and inference or private
replay can require separate winners. Ordinary `autotune="cache"` retains its
legacy initial-config fallback unless `require_cached_()` is explicitly called.

For an isolated capacity study with a fresh eight-trial cache and an explicitly
named candidate configuration:

```bash
python tools/run_mlp_capacity_study.py \
  --config autotune_reports/mlp_dw_splits_20260917_v2/split4.json \
  --field challenger --shape 11712,256,768,256 \
  --reserve-mib 12288 --trials 8 --output new-capacity-study
```

This creates a new spec/cache under the output directory rather than touching
an application's cache. It preserves the exact command and runs both cold and
reuse workers. The example candidate is a local research artifact, not a
configuration guaranteed to exist in an installed package.

Optional `--reserve-mib N` retains N MiB of synthetic CUDA storage in each
worker while it performs first-hit tuning and four training pairs or four
no-grad forwards, according to `--phase`. This models
occupied capacity, not full-model allocation ordering/fragmentation or activation
lifetimes. Reports give the reserve, first-pair peak, synchronized live storage,
runtime-cache counters and specification hash. The check requires finite output
and every gradient; it is not an independent numerical-reference test.
The command also enforces a live-memory gate: after one cold pair, the next
three synchronized live-allocation samples must be identical. The report
records `live_memory_gate` with the metric, spread, tolerance, and result.
Unstable residency exits nonzero and preserves the samples; it is not silently
reported as a successful capacity check. This short fixed-workload gate does
not establish long-training stability or diagnose the reason for growth.
Reports also distinguish peak **allocated** from peak **reserved** storage and
record allocator retry/OOM counter deltas. A successful run can include allocator
retries; do not describe that as ample headroom or an OOM-free allocation path.
`capacity_gate` separately reports completion and OOM-free completion. Recovered
finalist OOMs fail that gate even if the internal worker exits successfully with
a usable winner. The public parent command recomputes capacity/live-memory gates
from the raw report and exits nonzero on missing, inconsistent or failed gates;
it does not launch the reuse worker after a failed cold gate. Reports, snapshots
and any durable winner remain intact for diagnosis. Allocation retries without
an OOM are reported but do not fail the capacity gate. This deployment gate is
not an independent numerical-reference or full-application qualification.

The September 17 evaluator-stream revision reuses one stream across serial
candidates (with fresh candidate leaves). On the local 5070 Ti, fresh eight-trial
studies with 12 GiB synthetic occupancy completed with zero OOMs/retries on both
Torch versions. Peak reserved storage was 12.922 GiB (2.12.1) and 12.979 GiB
(2.13), versus 15.043/15.047 GiB before this change. Peak allocated storage was
12.520 GiB. Fresh durable reuse remained 12.169 GiB allocated/12.211 GiB reserved.
All warm live-memory samples were stable. Reports are under
`autotune_reports/evaluator_stream_capacity_torch{213,2121}_20260917/`.
The evaluator context tag changes to keep old measurement evidence separate;
existing durable files are not deleted. This is not an application capacity claim.

## OOM evidence

The disposable worker enables bounded allocator history and attaches a
process-lifetime PyTorch OOM observer. When reached, it records allocator data,
cache ownership counters and a snapshot **before Python stack unwinding**.
Do not install this hook casually in a long-lived application: the underlying
private PyTorch API has no removal interface. Snapshot pickles can include
source stack paths; treat them as diagnostic artifacts, not public datasets.

On the September 16 local driver mismatch, PyTorch's error-formatting path
asserts in `nvmlInit_v2_` before reaching the observer. The impossible-allocation
test therefore reports an explicit expected failure, not a successful pre-OOM
qualification. Pre-phase snapshots still work; they are not snapshots at the
exact failing allocation. No inference about AFRL fragmentation or leaked
graphs follows from this failure. Driver repair requires separate coordination.

That driver limitation is historical. After the user's September 17 reboot,
both supported Torch stacks passed the deliberate oversized-allocation test:
the observer captured 1 MiB of still-live allocation from the throwing frame
before unwinding. Evidence is in
`autotune_reports/post_reboot_oom_torch{213,2121}_20260917/` and
`post_reboot_preflight_torch{213,2121}_20260917.xml` (10 passing tests each).
The copied event JSON retains its original temporary snapshot path; the
preserved `oom-0000.pickle` is beside that JSON. This validates the observer,
not an explanation of the original AFRL OOM.

This tool does not launch or modify AFRL, automatically alter precision, promote
BF16 replacement decisions, or prove full-application capacity. It supplies an
isolated first-hit workflow and reproducible evidence for that next qualification.

## September 17 measured capacity envelope

The study was repeated with the faster prequantized-down/prefetched-recompute
schedule as the initial candidate. Artifacts:
`autotune_reports/prequant_down_capacity_torch{213,2121}_20260917/`.
Both versions completed eight valid trials and four finite training pairs;
the selected disk winner retained `down_backend="prequant"`. A second fresh
worker required that winner and forbade retuning. Every worker's four live
allocation samples were identical. Cold/reuse peaks matched the earlier
envelope below: cold 12.620 GiB allocated, 15.047/15.043 GiB reserved (2.13/2.12.1),
with two/one allocator retries; reuse 12.169 GiB allocated and 12.211 GiB
reserved without retries. No terminal allocator OOM was recorded. Matching
search peaks do not mean the two configurations have identical scratch needs:
the peak includes the other evaluated candidates and finalist qualification.
This is synthetic capacity evidence, not full-application OOM resolution.

Reproduce this updated study with:

```bash
python tools/run_mlp_capacity_study.py \
  --config autotune_reports/mlp_prequant_down_20260917/train.json \
  --field challenger --shape 11712,256,768,256 \
  --reserve-mib 12288 --trials 8 --output NEW_CAPACITY_DIRECTORY
```

### Earlier split-dW schedule

On the local 5070 Ti (15.47 GiB device memory), the M11712/K256/H768/O256
eight-trial search completed while retaining a **12 GiB synthetic reservation**.
It used the numerically verified four-way FP32 dW split candidate as its initial
configuration, six paired finalist rounds, compiled selected-direct training,
and four finite-output/all-gradient training pairs after selection. Both versions
recorded eight valid search candidates. A second fresh worker required the disk
winner and forbade retuning.

| Torch | Worker | Peak allocated GiB | Peak reserved GiB | Allocation retries | Terminal allocator OOMs |
| --- | --- | ---: | ---: | ---: | ---: |
| 2.13 | Cold | 12.620 | 15.047 | 2 | 0 |
| 2.13 | Reuse | 12.169 | 12.211 | 0 | 0 |
| 2.12.1 | Cold | 12.620 | 15.043 | 1 | 0 |
| 2.12.1 | Reuse | 12.169 | 12.211 | 0 | 0 |

Live allocation was stable over the four post-selection observations in each
worker. The reservation is included in every number. This is a successful,
pressure-tested envelope, **not** a claim that cold search is comfortably small:
its allocator reserve approaches the device limit and requires retries.

Artifacts, including exact specs, fresh caches, all observations, snapshots and
worker logs:

- `autotune_reports/mlp_capacity_12g_torch213_20260917_v2/`
- `autotune_reports/mlp_capacity_12g_torch2121_20260917/`

The earlier non-v2 Torch 2.13 run also passed, but did not yet record peak reserve
or retry counters. Synthetic occupancy does not reproduce a full model's
fragmentation, interleaved lifetimes, larger search budgets, or concurrent
workloads. The pre-reboot broken NVML terminal-OOM path is historical; see the
post-reboot observer qualification above. Neither test diagnoses the original
AFRL OOM.
