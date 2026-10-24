# Post-merge qualification

Work in progress; this is not a release announcement. The GLU integration was
published at `24d13e6`. Follow-up campaign/setup fixes are on `main`.

## Matched longer training trajectories

Two RTX 5090 workers completed all eight policies for 256 compiled steps each,
using 1,024 WikiText training paragraphs and 128 validation paragraphs at
sequence length 256. Within a worker, checkpoint, masks, data order, optimizer
and seed are identical across policies. Parameters, gradients and AdamW moments
are BF16; cross entropy is FP32. All checked gradients and optimizer states
remained finite. Validation uses copies to preserve delayed training state.

| Policy | Torch 2.13, seed 4810: initial → final NLL | Torch 2.12.1, seed 5810: initial → final NLL |
|---|---:|---:|
| BF16 | 2.0828 → 2.0489 | 2.1350 → 2.0954 |
| MXFP8 dynamic | 2.0972 → 2.0582 | 2.1427 → 2.1142 |
| NVFP4 JIT-region | 2.3803 → 2.3573 | 2.4392 → 2.4124 |
| NVFP4 delayed | 2.3878 → 2.3573 | 2.4568 → 2.4114 |
| NVFP4 current | 2.3876 → 2.3536 | 2.4568 → 2.4240 |
| NVFP4 block | 2.3757 → 2.3723 | 2.4322 → 2.4470 |
| NVFP4 block Four Over Six | 2.3476 → 2.3312 | 2.4246 → 2.4115 |
| MXFP8 recomputed MLP | 2.0976 → 2.0628 | 2.1541 → 2.1160 |

These are two matched within-run comparisons, not an experiment isolating
Torch-version effects: seed and version differ together. The checkpoint is
already pretrained and this is a short fixed-mask fine-tuning trajectory, not
training from scratch or long-run convergence certification. Block-only worsens
on one validation set. The substantial NVFP4/BF16 gap remains. Per-step timings
include compilation, synchronous gradient/state checks and logging; they must
not be reported as production throughput.

Reproduce using `benchmarks/run_postmerge_validation.sh`; raw JSONL, checkpoint
revision and source identities are retained in the output directory.

## First-hit refinement finding

The small-shape 8/16/32-trial sequence and fresh-process reuse passed both
stacks. The larger `M,K,H,O=2048,768,1536,768` case exposed a runtime bug at
8 → 16 trials: injecting the previously deployed winner into the opening
portfolio changed the progressive checkpoint key. The correction preserves
the opening portfolio and uses the winner only as the deployment comparison
baseline; its measured observation remains in history. Checkpoint compatibility
validation is not weakened. Resuming the actual failed journals at 16 and 32
trials now passes on both Torch stacks, with exactly 16/32 observations and one
compiled graph. A fresh-process 32-trial reuse adds no observations and invokes
no tuner. The targeted MLP first-hit/graph/gradient and progressive-search
regression suites pass 15 tests on each supported Torch stack. The ragged
`511,384,576,384` budget progression and fresh-process reuse also pass both
stacks. Laptop qualification remains in progress.

## Linear first-hit budget sweep

All 54 compiled inference probes passed: three families (MXFP8 materialized,
NVFP4 block, NVFP4 JIT-region), three M,N,K shapes (`256,768,768`,
`2048,768,1536`, `511,576,384`), three progressive budgets (8/16/32), and
both supported Torch stacks on separate RTX 5090s. Each probe checks finite
outputs against a BF16-cast FP32 reference, trial accounting, warm reuse and
CUDA-graph capture/replay. Error gates remain NRMS <0.07 for MXFP8 and <0.20
for NVFP4. This is inference coverage, not backward qualification.

Median first-call times across the nine family/shape cells were 3.98/4.69/6.65
seconds for Torch 2.13 and 4.35/4.75/6.91 seconds for Torch 2.12.1 at
8/16/32 trials respectively. The later budgets resume previous observations;
their times are additional refinement costs, not independent cold-start totals.
Compiler disk caches were not cleared, and process startup is excluded.

Most replay timings were approximately flat with additional effort. One
Torch 2.13 ragged MXFP8 cell changed from 10.24 to 8.18 microseconds (about
1.25x), while most cells changed by less than 2%. These are unpaired
cross-process measurements, not proof of monotonic speedup or global optimality.
The two stacks sometimes selected different schedules with materially different
latencies; that does not isolate a Torch-version effect. No research winner
was installed into the user's default runtime cache.

## Cross-device and installation status

Isolated source snapshots on the RTX 5060 and 5070 laptops are running release
qualification. Initial attempts had missing test dependencies and an import-path
setup error; those are failed attempts, not kernel passes. The corrected runner
explicitly imports its own checkout, and Pillow is now a declared test dependency.
Both initial Ettin model/short-training subprocesses passed, but their parent
qualification runs did not. The 5060's subsequent partial suite was deliberately
stopped to include the runtime refinement fix. The 5070 WSL instance restarted
externally during its partial suite; the cause is not established. Those partial
logs remain intact. Full suites restarted at `7ec987e` with four compiler workers
by default (caller-overridable), avoiding excessive parallel compiler memory use.
Fresh final local suites on both Torch stacks are also running against this code.
The restarted laptop suites have reported early failures; complete traces are
pending suite completion. They are not qualified, and release remains blocked
until those failures are understood and resolved.

The built wheel imports MXFP8Linear, NVFP4Linear and MXFP8MLP outside the source
tree and passes runtime version validation. That check reused the qualified
environment's dependency directory; it is not a fresh network dependency solve.
Both dependency requirement files are additionally being installed into genuinely
fresh virtual environments. The Torch 2.13 fresh install has passed `pip check`,
runtime version validation and public frontend imports outside the checkout;
its CLI help and all 58 public exports also load. Torch 2.12.1 installation is
still in progress. The `c728aae` wheel/sdist pass build and twine checks;
archive inspection found no ignored source artifacts (136 wheel / 622 sdist
entries). No release tag or package upload has been made.

Subsequent qualification runs record failures immediately in
`test-events.jsonl`, including collection failures and tracebacks. The optional
pytest plugin refuses existing output files and fsyncs failure/final records.
Its passing/failing-probe and overwrite-protection tests pass on both stacks.
This does not change any numerical or runtime gate.
