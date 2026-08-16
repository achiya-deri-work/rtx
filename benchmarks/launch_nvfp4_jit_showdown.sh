#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="autotune_manifests/nvfp4_jit_sibling_showdown_v1.json"
campaign="nvfp4_jit_sibling_showdown_v1"
duration="${1:-2h}"
node="${RTX_AUTOTUNE_NODE:-$(hostname -s)}"
output_root="${RTX_AUTOTUNE_OUTPUT_DIR:-autotune_datasets/${campaign}_${node}}"
report_root="${RTX_AUTOTUNE_REPORT_DIR:-autotune_reports/${campaign}_${node}}"
log_root="${RTX_AUTOTUNE_LOG_DIR:-autotune_logs}"
calibration="${RTX_AUTOTUNE_CALIBRATION:-hardware_calibration_${node}.json}"
log_file="$log_root/${campaign}_${node}.log"
pid_file="$log_root/${campaign}_${node}.pid"

if [[ "${2:-}" != "--worker" ]]; then
  cd "$repo_root"
  mkdir -p "$log_root" "$output_root" "$report_root"
  if [[ -f "$pid_file" ]]; then
    old_pid="$(<"$pid_file")"
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "already running PID=$old_pid log=$log_file" >&2
      exit 1
    fi
  fi
  nohup "$repo_root/benchmarks/launch_nvfp4_jit_showdown.sh" \
    "$duration" --worker >"$log_file" 2>&1 </dev/null &
  pid=$!
  echo "$pid" >"$pid_file"
  disown "$pid" 2>/dev/null || true
  echo "launched PID=$pid log=$log_file dataset=$output_root"
  exit 0
fi

cd "$repo_root"
export PYTHONUNBUFFERED=1
if [[ -z "${CUDA_HOME:-}" && -d /usr/local/cuda-13.2 ]]; then
  export CUDA_HOME=/usr/local/cuda-13.2
fi

echo "START campaign=$campaign duration=$duration commit=$(git rev-parse HEAD)"
.venv/bin/python -m rtx.autotune.dataset validate "$manifest"
if [[ ! -f "$calibration" ]]; then
  .venv/bin/python -m rtx.autotune.dataset calibrate \
    --device cuda:0 --output "$calibration" --samples 7 --target-ms 40
fi

.venv/bin/python -m rtx.autotune.dataset run "$manifest" \
  --device cuda:0 --output-dir "$output_root" --format none \
  --calibration "$calibration" --wall-time "$duration" \
  --context-slice 75s --trial-milestones 8,16,32,64,128,256,512 \
  --initial-promote 4 --strategy-orchestration bandit \
  --strategy-bandit-exploration 0.5 --context-orchestration breadth_first \
  --bandit-min-trials 32 --max-milestone-lead 1 \
  --reuse-deterministic-failures --stall-timeout 240s

.venv/bin/python -m rtx.autotune.dataset audit "$output_root" \
  --output "$report_root/audit.json" --allow-errors
.venv/bin/python -m rtx.autotune.dataset collect "$output_root" \
  --output "$report_root/dataset" --format both
.venv/bin/python -m rtx.autotune.dataset verify-winners "$manifest" \
  --device cuda:0 --output-dir "$output_root" --promote 8
echo "DONE output=$output_root reports=$report_root"
