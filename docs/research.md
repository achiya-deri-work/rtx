# Research and qualification

For using the library, start with the [manual](README.md). These pages are
evidence and experimental design, not a second API. Dated results apply to
their saved source/device/protocol; older pending-work statements are historical.

## Autotuning

- [Qualification matrix](autotune_qualification_matrix.md) and [engineering acceptance audit](first_hit_completion_audit.md).
- [Advanced effort details and chronology](autotune_effort.md).
- [Protected search and conditional rules](autotune_protected_prior_results.md).
- [Audited-data results](autotune_evidence_results.md) and [evidence pipeline](autotune_evidence_pipeline.md).
- [Cross-SKU study](autotune_cross_sku_study.md) and [methodology](autotune_evidence_methodology.md).
- [Model tournament](model_tournament.md), [predictor validation](portable_predictor_validation.md), [prospective study](predictor_prospective_study.md).
- [Coverage repair](autotune_coverage_repair.md), [mixed pools](mixed_pool_study.md), [shape-stratified study](shape_stratified_study.md).
- [Backward timing diagnosis](backward_timing_diagnosis.md), [development chronology](autotune_goal_audit.md), [older notebook](../AUTOTUNING.md).

Tested cache/effort infrastructure is distinct from a generalizing learned
prior. Conditional trends exist, but the statistical-to-sampling connection
and broad policy superiority are not established by the engineering audit.
Acceptance does not mean ML cannot help or every shape is optimized.

## GLU/MLP and numerics

- [GLU historical benchmarks](mxfp8_glu_history.md) and [MLP historical qualification](mxfp8_mlp_history.md).
- [First-hit ownership](mlp_first_hit_plan_contract.md) and [effort qualification](mlp_first_hit_effort_qualification.md).
- [Stress tests](mxfp8_mlp_numerics.md), [divergence](mxfp8_mlp_divergence.md), [TorchAO parity](mxfp8_mlp_recipe_parity.md).
- [Selective precision](mxfp8_mlp_selective_precision.md), [precision API evidence](mxfp8_mlp_precision_api.md), [quality/runtime](mxfp8_mlp_quality_runtime.md).
- [Multi-shape results](mxfp8_mlp_multishape.md), [optimization](mxfp8_mlp_optimization.md), [coverage](mxfp8_mlp_autotuning_coverage.md).
- [DINOv3 regression](dinov3_regression.md) and [mixed precision](dinov3_mixed_precision.md).

## Documentation ownership

Put workflows in the manual, shared constructor controls in `api.md`, and exact
signatures/defaults in generated `api_reference.md`. Keep dated timings and
failed experiments here instead of copying them into quickstarts.
Regenerate with `python tools/doc_reference.py --write`, then run
`python -m pytest tests/test_documentation.py`. Keep historical paths reachable;
deleting evidence is not consolidation.
