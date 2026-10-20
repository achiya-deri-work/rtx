# Phase-specific MLP deployment

Use `rtx.bind_qualified_mlp_plans_` to pin independently preflight-qualified
grad-enabled and no-grad plans without a private selection hook. This keeps the
selected-direct operator path and its forward-owned backward token. It does
not replace Parameters, change numerical policy, install disk winners, or run
an autotuner during application execution.

## Startup recipe

First create and execute separate `MLPCandidate.prepare(...)` requests for
`phase='train'` and `phase='forward'`; see [preflight](mlp_preflight.md).
Each must pass current-source inspection and cache coverage. Different initial
configs, objectives and request-local cache directories are allowed. Do this
before allocating the full model when capacity is tight.

```python
import json
import torch
import rtx

# Existing bias-free RTX module; do not reconstruct or replace its Parameters.
manifest = rtx.bind_qualified_mlp_plans_(image_mlp, [
    ("qualification/image-train", (32, 640, 256)),
    ("qualification/image-forward", (32, 640, 256)),
])
with open("image-binding.json", "x") as f:
    json.dump(manifest, f, indent=2)

compiled_model = torch.compile(model, fullgraph=True, dynamic=False,
                               options={"triton.cudagraphs": False})
```

Both requests in this example have candidate shape `(20480, 256, H, O)`.
The binding is exact-rank/shape, contiguous BF16, current CUDA device and
activation. `(20480,256)` is **not** interchangeable with `(32,640,256)` in
the binding even though the mathematical flattening agrees. Unknown shapes,
phases and devices fail rather than tuning or falling back. Grad phase follows
grad-enabled execution with at least one differentiable operand, not
`module.training`; checkpoint and teacher calls can therefore use different
plans. Changing gradient masks still requires application validation.

Binding checks eligible disk winners once, then pins copied config metadata in
the module. Subsequent execution performs no disk-winner lookup. Editing or
removing that cache cannot silently replace a running plan. No cache-hit rate
is meaningful for this path: the expected result is an exact in-memory binding
hit or an error. Manual shape overrides cannot bypass it. Binding is not
`state_dict` state: rerun startup checks after loading, device moves or source
and environment changes. Recompile/recapture after any binding change.

These requests qualify selected-direct, so automatic private replay is rejected;
do not enable `triton.cudagraphs` and silently treat it as the same plan. Caller
capture is a separate qualification task. The ordinary autotuned API is unchanged.

## Evidence and acceptance

The returned manifest records source fingerprints, request paths, inspected
cache contracts and pinned configs. It deliberately says
`application_qualified=False` and `actual_input_layout_qualified=False`:
candidate preflight currently uses 2D tensors. After binding, independently test
the actual 3D inputs, outputs, all gradients, caller compilation, checkpointing
and teacher mix. Record those results next to—not in place of—the manifest.

Keep four distinct evidence sections in the application report:

1. Quantized-recipe correctness and BF16 numerical difference (separate gates).
2. Standalone forward/backward and directly measured invocation-mix timings.
3. Full-application steady update timings, warmup excluded, preferably balanced
   repeated fresh-process comparisons.
4. Full-model peak/live memory, explicit cached workspace and allocation-lifetime
   evidence. Isolated peak savings are not full-model capacity savings.

Promotion requires a repeatable application benefit and acceptable application
peak memory and training quality. Passing preflight or beating an archived RTX
plan does not meet those requirements by itself.

## September 18 AFRL decision

The tokenizer proposal reduces the isolated old/new RTX mix from 52.966 to
17.442 ms and beats compiled BF16's mix by 7.4%. Full application time is
55.234 versus 55.380 s/update: no demonstrated benefit. Peak allocated memory
increases 307.69 MiB, including 80.07 MiB more retained explicit workspace.
No evictions occur. Keep the tokenizer BF16; do not expand the same-space
search budget or increase cache caps as a presumed fix. The remaining peak
delta needs matched allocation-lifetime evidence, not fragmentation speculation.

These are historical local application measurements; the raw feedback and
datasets are not distributed with the library. Reproduce the comparison with
your own application before accepting a replacement.

The retained `baseline/updates.jsonl` and `candidate/updates.jsonl` from that
study split the explicit-storage increase into **48.09375 MiB paired** and
**31.974609375 MiB partial** runner storage. These are final-update cache-byte
deltas, not attribution of allocations at the application peak. Source
inspection shows `operand_storage='serial_reuse'` shares operands inside a
paired backward runner; independently constructed partial dX/dW runners do
not share that arena. Cross-partial lifetime-safe reuse is therefore a concrete
structural investigation target, not a demonstrated saving or an argument to
alias live buffers blindly. Validate serial/overlapping streams and retained
graph ownership before attempting it.

## Binding validation (September 18)

On the RTX 5070 Ti, the following suite passes **34 tests on each** of Torch
2.13.0+cu132 and 2.12.1+cu132:

```bash
python -m pytest tests/test_mlp_bound_plans.py tests/test_mlp_candidate.py \
  tests/test_mlp_public_spec.py tests/test_deployment_qualification.py -q
```

The new integration test runs real disposable train/forward preflights, binds
distinct phase configs, compiles fullgraph/static with replay disabled and
checks `(2,64,128)` inputs at two seeds against an independent quantized recipe
(normalized RMS < 0.025 for output and all three gradients). It checks executed
plan tokens, Parameter identities, no runtime winner reads/tuning, unknown
shape rejection, noncontiguous rejection and incompatible replay rejection.
This is a small-shape API regression, not the full tokenizer or application
memory/performance qualification. Fifteen warnings per suite are existing JIT
deprecations and an existing candidate-test scalar-conversion warning.

Final artifacts: `autotune_reports/afrl_actual_shapes_20260917/bound_final213.xml`
and `bound_final2121.xml`. Earlier failed artifacts remain: the first used an
incorrect expected exception type for noncontiguous input; a subsequent run
correctly rejected a maintainer source edit during qualification. Final runs
used frozen implementation sources and passed all gates.
