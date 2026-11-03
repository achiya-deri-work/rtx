#!/usr/bin/env bash
# Focused correctness smoke, not a benchmark or exhaustive research qualification.
# Caller selects an assigned, idle SM12x GPU with CUDA_VISIBLE_DEVICES.
set -euo pipefail
python_exe=${1:-python}
output_dir=${2:-autotune_reports/nvfp4_research_qualification}
if [[ -e "$output_dir" ]]; then
  echo "refusing to overwrite $output_dir" >&2
  exit 2
fi
"$python_exe" -c 'import torch; assert torch.cuda.is_available(), "CUDA required"; assert torch.cuda.get_device_capability()[0] == 12, "SM12x required"; print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())'
mkdir -p "$output_dir"
export RTX_RUN_PAIRED_CUDA=1
export RTX_RUN_DUAL_DY_PREPARATION_CUDA=1 RTX_RUN_QUAD_PREPARATION_CUDA=1
export RTX_RUN_INLINE_SCALES_CUDA=1 RTX_RUN_SHARED_DY_CUDA=1
export RTX_RUN_H8_PREPARATION_CUDA=1 RTX_RUN_H16_PREPARATION_CUDA=1
export RTX_RUN_H16_CUDA=1 RTX_RUN_TF32_H8_CUDA=1
export RTX_RUN_GROUPED_REFINE_CUDA=1 RTX_RUN_DIRECT_REFINE_CUDA=1
export RTX_RUN_PAIRED_FP32_CUDA=1
"$python_exe" -m pytest -xq \
  tests/test_ms_eden_paired_pipeline.py \
  tests/test_ms_eden_dual_dy_preparation.py \
  tests/test_ms_eden_quad_preparation.py \
  tests/test_ms_eden_cta_finalizer.py \
  tests/test_ms_eden_inline_scales.py \
  tests/test_situ_decoder.py \
  'tests/test_ms_eden_shared_dy.py::test_shared_dy_exact[False-128-1-16-128-shape1]' \
  'tests/test_ms_eden_shared_dy.py::test_shared_dy_exact[True-128-1-16-128-shape1]' \
  'tests/test_ms_eden_h8_preparation.py::test_fused_preparation_oracle_graph_and_memory[gaussian-False-direct-1]' \
  'tests/test_ms_eden_h8_preparation.py::test_fused_preparation_oracle_graph_and_memory[gaussian-True-coalesced_x8-2]' \
  'tests/test_ms_eden_h16_preparation.py::test_h16_fused_preparation_fp64_decode_graph_and_memory[False-False-gaussian-shape0]' \
  'tests/test_ms_eden_h16_preparation.py::test_h16_fused_preparation_fp64_decode_graph_and_memory[True-True-gaussian-shape1]' \
  'tests/test_ms_eden_h16_hadamard.py::test_h16_fp64_quality[gaussian-shape0]' \
  tests/test_ms_eden_h16_hadamard.py::test_h16_guarded_fallback_matches_original \
  tests/test_ms_eden_h16_hadamard.py::test_h16_graph_current_inputs_and_no_live_growth \
  'tests/test_ms_eden_tf32_hadamard.py::test_tf32_full_transform_fp64_quality_and_scalar_association_report[tf32-gaussian]' \
  'tests/test_ms_eden_tf32_hadamard.py::test_tf32_full_transform_fp64_quality_and_scalar_association_report[bf16_padded-gaussian]' \
  'tests/test_ms_eden_grouped_refine.py::test_exact_producer_and_changed_graph[fragment-gaussian-8-False]' \
  'tests/test_ms_eden_grouped_refine.py::test_exact_producer_and_changed_graph[streamed-gaussian-16-True]' \
  tests/test_ms_eden_direct_refine.py::test_direct_refine_division_probe \
  'tests/test_ms_eden_direct_refine.py::test_direct_refine_producer[gaussian-16-True]' \
  'tests/test_ms_eden_paired_fp32.py::test_exact_paired_statistics[both-gaussian-16-True]' \
  --junitxml="$output_dir/results.xml" 2>&1 | tee "$output_dir/pytest.log"
