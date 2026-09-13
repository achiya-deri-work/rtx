#!/usr/bin/env bash
# Replay an already frozen plan; no training or automatic winner promotion.
set -euo pipefail
study_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$study_root"
study_output="${1:?Usage: bash run_shape_stratified_study.sh OUTPUT [HOURS] [FROZEN_PLAN]}"
study_hours="${2:-4}"
study_plan="${3:-autotune_datasets/shape_stratified_local_v2/plan.json}"
exec bash run_mixed_pool_study.sh "$study_output" "$study_hours" "$study_plan"
