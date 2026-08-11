#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: $0 {laptop-3h|ti-6h}" >&2
  exit 2
}

profile="${1:-}"
case "$profile" in
  laptop-3h)
    wall_seconds=10800
    ;;
  ti-6h)
    wall_seconds=21600
    ;;
  *)
    usage
    ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

# Keep compiler identity and libNVVM discovery stable across interactive,
# remote, and supervised shells. CUDA 13.2 is the preferred project toolkit.
if [[ -z "${CUDA_HOME:-}" && -d "/usr/local/cuda-13.2" ]]; then
  export CUDA_HOME="/usr/local/cuda-13.2"
elif [[ -z "${CUDA_HOME:-}" && -d "/usr/local/cuda" ]]; then
  export CUDA_HOME="/usr/local/cuda"
fi
if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
  case ":$PATH:" in
    *":$CUDA_HOME/bin:"*) ;;
    *) export PATH="$CUDA_HOME/bin:$PATH" ;;
  esac
fi
export PYTHONUNBUFFERED=1

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

deadline_epoch=$(($(date +%s) + wall_seconds))
attempt=0
while true; do
  remaining_seconds=$((deadline_epoch - $(date +%s)))
  if ((remaining_seconds <= 0)); then
    echo "DEADLINE supervisor wall-time exhausted"
    break
  fi
  attempt=$((attempt + 1))
  echo "profile=$profile attempt=$attempt remaining=${remaining_seconds}s output=$output_dir calibration=$calibration"
  set +e
  "${autotune_cmd[@]}" run \
    autotune_manifests/autotuner_prospective_5070_v1.json \
    --device cuda:0 \
    --output-dir "$output_dir" \
    --format none \
    --calibration "$calibration" \
    --wall-time "${remaining_seconds}s" \
    --context-slice 45s \
    --stall-timeout "${RTX_AUTOTUNE_STALL_TIMEOUT:-180s}" \
    --trial-milestones 4,8,16,32,64 \
    --initial-promote 1 \
    --strategy-orchestration manifest \
    --context-orchestration breadth_first \
    --adopt-existing-context-identity-if-present \
    2>&1 | tee -a "$log_dir/autotuner_${profile}.log"
  status=${PIPESTATUS[0]}
  set -e
  if ((status == 0)); then
    break
  fi
  if ((status != 75)); then
    exit "$status"
  fi
  echo "RESTART CUDA fault or watchdog stall; residuals are durable"
done
