#!/usr/bin/env bash
# Fully frozen cross-SKU study. No training, package upgrades, or winner install.
set -euo pipefail
study_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$study_root"
study_sku="${1:?Usage: bash run_shape_overnight.sh 5090|laptop|5070ti [HOURS]}"
case "$study_sku" in
  5090) study_default_hours=5 ;;
  laptop|5070ti) study_default_hours=7 ;;
  *) echo 'Expected 5090, laptop, or 5070ti' >&2; exit 2 ;;
esac
study_hours="${2:-$study_default_hours}"
study_python="${RTX_STUDY_PYTHON:-$study_root/.venv/bin/python}"
"$study_python" -c 'import sys, torch, numpy, pyarrow; from rtx.runtime import validate_runtime_environment; validate_runtime_environment(); assert torch.cuda.is_available(), "CUDA unavailable"; name=torch.cuda.get_device_name(); print("GPU:", name); label=sys.argv[1]; normalized=name.lower().replace(" ", ""); valid=("5090" in normalized if label=="5090" else "5070ti" in normalized if label=="5070ti" else "5070" in normalized and "laptop" in normalized); assert valid, f"Requested {label}, found {name}"' "$study_sku"
study_output="${RTX_STUDY_OUTPUT:-autotune_datasets/shape_overnight_v1_${study_sku}_$(hostname)_$(date -u +%Y%m%dT%H%M%SZ)}"
export RTX_STUDY_PYTHON="$study_python"
exec bash run_shape_stratified_study.sh "$study_output" "$study_hours" \
  autotune_manifests/shape_stratified_overnight_v1/plan.json
