#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 {laptop-3h|ti-6h}" >&2
  exit 2
}

profile="${1:-}"
case "$profile" in
  laptop-3h)
    wall_time="3h"
    ;;
  ti-6h)
    wall_time="6h"
    ;;
  *)
    usage
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ -x ".venv/bin/python" ]]; then
  # Import the just-pulled checkout even if the environment was originally
  # installed non-editably.
  autotune_cmd=(".venv/bin/python" "-m" "rtx.autotune.dataset")
else
  autotune_cmd=("${RTX_AUTOTUNE_BIN:-rtx-autotune}")
fi

output_dir="${RTX_AUTOTUNE_OUTPUT_DIR:-autotune_datasets}"
calibration="${RTX_AUTOTUNE_CALIBRATION:-hardware_calibration_${profile}.json}"
log_dir="${RTX_AUTOTUNE_LOG_DIR:-autotune_logs}"
mkdir -p "$log_dir"

if [[ ! -f "$calibration" ]]; then
  "${autotune_cmd[@]}" calibrate \
    --device cuda:0 \
    --output "$calibration" \
    --samples 5 \
    --target-ms 30
fi

echo "profile=$profile wall_time=$wall_time output=$output_dir calibration=$calibration"
"${autotune_cmd[@]}" run \
  autotune_manifests/autotuner_prospective_5070_v1.json \
  --device cuda:0 \
  --output-dir "$output_dir" \
  --format none \
  --calibration "$calibration" \
  --wall-time "$wall_time" \
  --context-slice 45s \
  --trial-milestones 4,8,16,32,64 \
  --initial-promote 1 \
  --strategy-orchestration manifest \
  --context-orchestration breadth_first \
  2>&1 | tee -a "$log_dir/autotuner_${profile}.log"
