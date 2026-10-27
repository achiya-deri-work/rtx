# Conditional backward estimator and matched training, 2026-10-25

This is current-source, bounded research evidence on RTX 5060 Laptop, Torch
2.13.0+cu132/CUDA 13.2. It is not a claim of unbiased gradients, production
qualification, or full transformer convergence. No public defaults changed.

Correction discovered in final review: the initial artifacts below requested
`forward_outer_mode='exact'`, but the regional quantizer consumed its dynamic
config's default `power2`. Thus the initial numerical/teacher/decoder artifacts
are **matched power2 JIT forward**, despite their original JSON exact labels.
The original artifacts are retained unchanged for provenance. The experimental
wrapper is now fixed to pin the resolved mode in the consumed dynamic config;
an explicit requested mode takes precedence, and absent requests retain an
explicit config's mode (including `exact_four_over_six`). Exact follow-up results
are distinguished below. Fixed original-source gradient-oracle arithmetic does
not depend on forward mode, so the estimator conclusions remain applicable to
the tested backward policies, without converting the old runs into exact-forward
execution evidence.

## Contract and coverage

Base commit: `7f73446c08fb7d5d337147a5378c0a3f004b77e0`, plus the dirty research
overlay fingerprinted in `autotune_reports/nvfp4_backward_20261025/` source
manifests. Remote isolated snapshot: `/tmp/rtx-estimator-20261025.Q4S6DH`.

The independent oracle is direct eager FP64 `dY @ W` and `dY.T @ X` using the
original BF16 operands, not decoded forward operands, production preparation,
or production GEMM. All four `(dX,dW)` policy combinations use bitwise-identical
NVFP4 JIT-region forward (actual power2 in the initial runs); every repetition checks
that identity and gradient finiteness. The MXFP8 baseline is explicitly pinned
to `DEFAULT_MXFP8_BWD_CONFIG`. FP4 samples resample shared Rademacher rotations
and independent operand scale uniforms. Inputs/upstream gradients stay fixed.

One operand seed (807), Gaussian/sparse/outliers, shapes `(M,N,K)`
`(128,128,128)`, `(128,2048,128)` (long dX reduction), and
`(2048,128,128)` (long dW reduction), 32 repetitions, four policy combinations,
both gradients: nine bounded cases. Selected small sparse case extends the same
randomization prefix to 128 repetitions. This is 1,664 paired backward calls,
not 1,664 independent test cases. Sparse operands have 10% retention; outliers
have a 1% chance of 16x amplification. Dimensions are aligned, not ragged.

## Numerical result

Percentages below use L2 error divided by the fixed FP64 reference L2 norm
(floor `1e-12`). FP4 per-draw RMS averages squared errors across repetitions;
the sample-mean error is not the error of the runtime single-draw policy.

| Distribution, reduction | Gradient | MXFP8 error | FP4 per-draw RMS | FP4 32-mean error | Mean standard error |
|---|---|---:|---:|---:|---:|
| Gaussian, 128 | dX | 3.819% | 13.762% | 2.412% | 2.433% |
| Gaussian, 128 | dW | 3.801% | 13.774% | 2.447% | 2.434% |
| Gaussian, 2048 | dX | 3.782% | 13.617% | 2.420% | 2.407% |
| Gaussian, 2048 | dW | 3.788% | 13.787% | 2.439% | 2.437% |
| Sparse, 128 | dX | 3.822% | 13.704% | 2.893% | 2.406% |
| Sparse, 128 | dW | 3.651% | 14.062% | 2.901% | 2.471% |
| Outliers, 128 | dX | 4.021% | 15.559% | 2.742% | 2.751% |
| Outliers, 128 | dW | 3.898% | 16.742% | 2.984% | 2.959% |

Sparse 128-repetition dX/dW mean errors are 1.962%/1.990%, while their estimated
mean standard errors fall to 1.207%/1.238%. Subtracting estimated Monte Carlo
variance from squared mean error yields residual magnitudes about
1.547%/1.558%, close to the 32-repetition diagnostics 1.607%/1.520%.
Long-reduction sparse cases also give about 1.55–1.60%. This is evidence of a
persistent conditional sparse error component, not evidence of unbiasedness.
The correction is a noisy squared-bias diagnostic, not a confidence interval;
negative values remain negative in JSON and must not be read as proof of zero
bias. Gaussian/outlier mean errors are much closer to their Monte Carlo noise.

Deterministic MXFP8 has zero repetition variance but nonzero quantization error.
Single-draw FP4 errors are materially larger; averaging 32/128 samples is a
diagnostic, not a cost-matched runtime win. Mixed policies retain deterministic
MXFP8 in their selected phase and FP4 stochastic behavior in the other phase.
They do not change forward or reuse quantized forward operands.

## Integration

The compiled two-layer teacher smoke (20 AdamW steps, width128) passed all four
policies with finite upstream and final-layer gradients and nonzero parameter
updates. Initial loss is identical, `3.144437e-4`; final losses span
`1.929e-5`–`1.946e-5`; heldout NRMS spans `0.4812`–`0.4833`. This is smoke,
not a policy ranking or transformer qualification.

The initial matched TinyStories decoder uses the existing two-layer width256, sequence128,
batch16 recipe, 100 optimizer steps, fixed corpus and initialization hashes,
BF16 parameters/AdamW states, four research policy modes, and 16 validation
batches. Initial full-logit hashes and first compiled loss/backward warmup times
are emitted separately. All four completed 100 steps, finite gradient-norm
checks, and finite final parameters. All four have identical initial weights,
corpus prefix, and full initial logits hashes (`8e6608ed...88b6582`).

| dX / dW policy, actual power2 forward | Final validation cross entropy | First compile/backward warmup |
|---|---:|---:|
| MXFP8 / MXFP8 | 1.990308 | 10.79 s |
| FP4 / FP4 | 1.990129 | 11.26 s |
| MXFP8 / FP4 | 1.982354 | 9.23 s |
| FP4 / MXFP8 | 1.971786 | 10.35 s |

This one-seed, 100-step difference is a shortlist signal only: no statistically
supported policy ranking, no long-horizon claim. Reported post-compile elapsed
times include validation/checks, vary with cache state and launch order, and are
not a throughput comparison. All raw JSONL/logs are in the artifact directory.

### Corrected exact-forward follow-up

The final wrapper builds/interns the resolved regional dynamic configuration in
a static helper that returns only a request-key string before the autograd
boundary. The existing compiler-visible regional forward and its registered
operator are unchanged; the experimental autograd Function continues to own
backward. Production APIs/defaults are unchanged. Four targeted CUDA regressions
(eager/compiled × implicit/explicit power2 config) verify bitwise equality with
an independently called public forward using an explicit exact dynamic config,
and verify the chosen input distinguishes exact from power2 output. Initial
development attempts failed compiled dataclass guards; their logs are retained,
and only the final passing version was used for the following results.

One fresh exact-forward small sparse32 diagnostic passed all four policy and
finite/forward-identity checks. Its backward mean/std/standard-error/gain metrics
are bitwise identical to the original fixed-source sparse32 experiment, as
expected when only forward scaling changes and the upstream gradient is fixed.

All four corrected exact-forward decoder runs completed 100 steps with finite
gradients/parameters. They use the same original initialization/corpus hashes as
the power2 runs, and have a new identical-within-policy initial full-logit hash
`ce808670...1abb1bcf`, distinct from the prior power2 hash.

| dX / dW policy, corrected exact forward | Final validation cross entropy | First compile/backward warmup |
|---|---:|---:|
| MXFP8 / MXFP8 | 1.980456 | 7.09 s |
| FP4 / FP4 | 2.002889 | 7.53 s |
| MXFP8 / FP4 | 1.977818 | 6.35 s |
| FP4 / MXFP8 | 1.969916 | 7.36 s |

These remain one-seed short trajectories, not a statistically supported ranking
or a claim that either mixed policy is production-ready. Exact artifacts have
the `exact_` prefix and a separate `exact-source-manifest.txt`.
The final current-runtime CUDA batch passed 14/14 (10 mixed policy/RNG/checkpoint/
missing-gradient contracts plus four exact-mode regressions) in 13.84 s. The
5060 was released with 0% utilization and 0 MiB allocated after this work.

## Reproduction and limits

```sh
python -m benchmarks.study_ms_eden_estimator --shape 128 128 128 --distribution sparse --samples 128 --output fresh.json
python -m benchmarks.train_ms_eden_probe --matched-mixed --steps 20 --width 128 --output fresh-teacher.json
python -m benchmarks.train_quartet2_decoder --mode research_mx_mx --steps 100 --tokens training_data/tinystories-train-00000.byte_tokens.npy --output fresh-decoder.jsonl
```

Other decoder modes: `research_fp4_fp4`, `research_mx_fp4`, `research_fp4_mx`
(names order dX,dW). Focused CPU estimator/wrapper/policy tests: 38 passed. The
estimator/training evidence alone does not qualify checkpoint replay or allocator
behavior; separate integration tests provide the bounded checkpoint/RNG coverage
described above. This report does not qualify ragged dimensions, other
devices/stacks, long training, performance, general allocator behavior, or public
API release.
