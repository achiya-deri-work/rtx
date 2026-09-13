#!/usr/bin/env bash
# Replay the same frozen task plan on each SKU; no training datasets required.
set -euo pipefail
study_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$study_root"
study_output="${1:?Usage: bash run_mixed_pool_study.sh OUTPUT [HOURS] [FROZEN_PLAN]}"
study_hours="${2:-2}"
study_plan="${3:-autotune_datasets/mixed_pool_local_v1/plan.json}"
if [[ ! -f "$study_output/plan.json" ]]; then
  test -f "$study_plan"
  mkdir -p -- "$study_output"
  cp -- "$study_plan" "$study_output/plan.json"
fi
# Artifact relocation is hash-checked by the worker. Override with
# RTX_STUDY_ARTIFACT if the identical artifact is stored elsewhere.
exec bash run_predictor_study.sh "$study_output" "$study_hours" "${RTX_STUDY_DEVICE:-cuda:0}"
