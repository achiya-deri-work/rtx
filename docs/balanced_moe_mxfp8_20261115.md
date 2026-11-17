# Initial BalancedMoE MXFP8 qualification — work RTX 5090

Date: 2026-11-15. Scope: MXFP8 only. **Functional experimental implementation,
not a performance release.** No NVFP4 experts were implemented or benchmarked.

This is the historical initial bring-up. The subsequent
[speed study](balanced_moe_speedup_20261115.md) supersedes its performance
assessment and records the prepared/TMA-output improvements and remaining gaps.

## What now exists

- `rtx.BalancedMoE`: fixed-K router, contiguous VirtualLayer expert banks and
  one ordinary shared SwiGLU in the initial designs.
- Uniform expert-batched GEMMs, without per-expert Python launch loops or
  physical BF16 transposes. Grouped TMA descriptors include an expert mode so
  zero-filled logical tails do not read another expert's data.
- Native gather, FP32 weighted combine, weighted upstream-gradient/route-weight
  adjoint and FP32 input-gradient reduction.
- MXFP8 hidden storage, FP32 gate activation/multiply, BF16 external weights,
  activations and parameter gradients. Fully recomputed expert backward; saved
  differentiable Sinkhorn state is not recomputed.
- A real H=128 on-chip hidden family. Gate/up operand allocations are reused
  for the MXFP8 hidden handoff and down GEMM. Down output tiles can run serially
  or with producer/consumer ownership and 1/2/3/4 weight buffers. Expert-slot
  output staging and token combine remain separate.
- Whole-routed-expert UCB tuning through `rtx.autotune`, independent candidate
  numerical gates, residual journals, source/device-safe persistent winners
  and compile-compatible first-hit selection.

See [the API guide](balanced_moe.md) for constructor controls and limitations.

## Environment and reproducibility

Authorized isolated source/venv:
`work_5090:/home/achyad/rtx-balanced-moe-20261115.awFVLD`.

RTX 5090, SM120, 170 SMs, 32 GiB; driver 595.99.02; Python 3.13.17;
Torch 2.13.0+cu132; TorchAO 0.18.0+cu132; CuTe DSL 4.7.0.
Resident detector services were left running and untouched. They held about
12.3 GiB at setup and were observed idle; this is not a dedicated bare-GPU
qualification or a promise about future service traffic.

Raw evidence is under
`autotune_reports/balanced_moe_mxfp8_20261115/work_5090/`, backed by Storage4.
It contains numerical JSON, append-only timing/trial JSONL, profiler traces,
test logs, environment audit and a source archive. Profiler traces are
diagnostic; headline timings use CUDA events over warmed graph replay, with
alternating implementation order and three measurement rounds. Compilation,
tuning, capture, input creation and warmup are excluded.

Measured numerical/core/tuner source digest:
`f5bfd0044591cc5dcfbe746d2d1dd55df92667d55af8f574498a38ea52e695b3`.
Final host-side expert-grid/weight-index validation adds digest
`20fcb62452739e65af68e9d76957f07ecb1ac6ab3fe16026063000671b815232`;
CUDA numerical kernels and schedules are unchanged by those bounds checks.
Each benchmark JSON separately records source and driver hashes. Earlier
exploratory results below have a different tuner revision; they must not be
silently treated as current cache winners.

## Correctness, compilation and ownership

The compact GPU matrix reported **29 passed, 2 skipped, 26 subtests passed**
in 79.38 seconds. The two skips are deliberately misaligned TMA physical row
strides; aligned TMA row/transpose/transpose-transpose cases ran. Existing
linear/GLU tests were included because grouped execution extends the shared
forward core. A separate three-case routed-gradient run then passed, adding
odd E=5, D=35, H=49, dummy padding and noncontiguous upstream gradients.
The final-source MoE-only rerun (including those additions and host bounds)
reported **29 passed, 2 skipped** in 40.42 seconds. Final CPU/documentation
checks also passed, including source-derived API and local-link checks.

Covered contracts:

- Independent quantized GEMM and explicit operational STE oracles, every
  differentiable input/weight/route coefficient, logical leading-axis reduction
  per expert and dummy-slot masking based on inverse membership.
- H=128 on-chip serial and all four buffered schedules, partial capacity C=65,
  buffer wraparound and changing inputs, compared against staged MXFP8.
- Real `torch.library.opcheck`, fake/autograd contracts, fullgraph compilation,
  first-hit tuning during compiled execution, backward after selection-cache
  clearing and compatible disk-winner reuse.
- Complete module backward and two AdamW updates, finite gradients for every
  parameter, separate 2D optimizer parameters and valid contiguous packed banks.
- Caller-owned CUDA graph replay after changing X and weights, plus eight
  synchronized post-warmup eager expert iterations without growing live
  allocated workspace (64 KiB drift gate). Reserved allocator memory is not
  used as a leak criterion.
- CPU reference/fullgraph and shared-engine search tests, including resumable
  trial budgets and canonicalization of inactive coordinates.

No whole-model learning/convergence conclusion follows from two optimizer
updates. Higher-order derivatives, distributed MoE and automatic whole-expert
Inductor graph replay are not qualified.

### Numerical boundaries

Candidate gates are output NRMS <=0.003 and gradient NRMS <=0.006 against an
independent Torch/FP64 quantized operational oracle, with an absolute-error
floor of 2e-7 for near-zero references. These are implementation-equivalence
checks, not errors against unquantized BF16.

For the current D384/E24/K6/H128, N512 winner, operational NRMS was:

| Output | NRMS |
|---|---:|
| Routed output | 7.54e-8 |
| dX | 9.30e-6 |
| Gate/up dW | 7.81e-6 |
| Down dW | 6.14e-6 |
| Route-weight adjoint | 1.25e-7 |

The earlier four-context study's errors versus compiled BF16 were approximately
6.4–6.6% NRMS for outputs/adjoints. That includes compounded MXFP8 quantization
and different intermediate rounding boundaries. It is not a convergence claim
or evidence that the kernel's implementation error is 6.5%.

## Measured performance — do not promote as a speed win

Expert attribution freezes identical learned-style routes and compares native
routed experts with a compiled **batched BF16** reference, not Python loops.
Router/shared work is excluded. Native backward recomputes intermediates;
ordinary BF16 autograd saves them, so this is not a matched-memory checkpoint
comparison.

Exploratory 24-trial / 30-second-soft-budget searches, prior tuner digest
`48b28c77e3f1c72bd72e3d29c3a4129369bf2ed91424cf594aa3c1fbcf3eb197`:

| N / D / E / K / H | BF16 forward | MXFP8 forward | BF16 fwd+bwd | MXFP8 fwd+bwd |
|---|---:|---:|---:|---:|
| 512 / 384 / 24 / 6 / 128 | 20.48 us | 86.77 us | 59.38 us | 378.86 us |
| 2000 / 384 / 24 / 6 / 128 | 49.15 us | 167.94 us | 165.63 us | 829.62 us |
| 512 / 768 / 48 / 6 / 256 | 48.83 us | 211.34 us | 179.58 us | 1027.16 us |
| 2000 / 768 / 48 / 6 / 256 | 177.41 us | 506.06 us | 586.31 us | 2637.27 us |

All four selected staged execution, although on-chip H=128 candidates passed
the numerical gate and beat the initial scalar seed in earlier trials.
Untuned numerical/timing collection also covered all 12 initial fixture
workloads; that does not mean all 12 received extensive tuning.

The **current-source** 16-trial full-layer study at N512/D384/E24/K6/H128:

| Scope | BF16 forward | MXFP8 forward | BF16 fwd+bwd | MXFP8 fwd+bwd |
|---|---:|---:|---:|---:|
| Routed experts only | 20.48 us | 95.93 us | 59.39 us | 407.30 us |
| Complete layer | 433.99 us | 587.36 us | 560.06 us | 1042.13 us |

Complete-layer variants use the same CuTe router, the same weights, and one
shared SwiGLU; both routed and shared FFN precision changes. Router/shared
autotuning was off for this diagnostic. Forward is grad-enabled. Complete
MXFP8 is about **1.86x slower** in this measured training-step comparison.
These bounded searches are not an exhaustive optimum, and the current 16-trial
winner is not asserted better than the earlier 24-trial winner.

### What the investigation found

1. Initially wide-load vector seeds were rejected because logical-transpose
   quantization requires staged ldmatrix x4, unswizzled BF16 staging and paired
   stores. The seed builder now constructs that complete family, including
   TMA/scalar tail alternatives where cp.async full-tile constraints fail.
2. Extra explicit seeds could remain in pool metadata without becoming search
   proposals beyond the two-seed opening. MoE now preserves them in the shared
   UCB structural arm and round-robins implementation families before depth
   variants. Quick cold budgets retain the staged family as designed.
3. Impossible dynamic role-register totals are rejected before compilation.
   Scalar/cp.async execution no longer constructs unused TMA descriptors that
   impose unrelated transpose/alignment constraints.
4. Warmed traces attribute most expert time to grouped projections and
   recomputation/adjoint GEMMs, not gather/combine. No Nsight tools/counters were
   installed in this isolated environment. There is **no measured claim** of
   bandwidth saturation, tensor-core saturation, occupancy or spill behavior.

## Next implementation priorities

The core is correct enough to continue measured optimization, not to stop at
frontend consolidation. Highest-value follow-ups:

1. Quantize operands once per appropriate expert/orientation, reuse them across
   output tiles/projections, and compare preparation versus packed GEMM cost.
   Current fused GEMMs still duplicate quantization across CTA output tiles.
2. Fuse recompute/derivative preparation and independent dX/dW operand emission,
   borrowing the stable MLP dual-orientation recipe without mixing experts.
3. Broaden true fine-grained ownership and expert-group/weight-reuse families;
   H=256 currently has staged execution only. Confirm spills/traffic before
   claiming the on-chip family is physically GMEM-free beyond its explicit
   hidden store contract.
4. Add split-FP32 dW reduction for long capacities and qualify larger token
   dimensions, instead of extrapolating the current small-C studies.
5. Integrate automatic CUDA-only launch replay around host selection tokens,
   then qualify complete-layer timing with router/shared/expert tuning together.

Do not change precision or routing policy to manufacture a speedup. Keep the
BF16 reference, independent oracle, raw negative results and staged fallback.

## Reproduction

From the isolated environment or a supported RTX checkout:

```bash
PYTHONPATH=. python -m pytest -q tests/test_mxfp8_moe.py
PYTHONPATH=. python benchmarks/benchmark_balanced_moe.py \
  --case-ids d384_e24_k6_h128 --tokens 512 \
  --effort balanced --trials 16 --rounds 3 --samples 9 \
  --whole-layer --profile --output autotune_results/moe/complete.json
```

The exact original commands, runtime audit, phase journals and configuration
objects are retained with the artifacts. No unrelated project, service,
Docker container or driver setting was changed.
