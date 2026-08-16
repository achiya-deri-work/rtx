#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
manifest="autotune_manifests/blackwell_all_kernels_scaling_v2.json"
campaign="blackwell_all_kernels_scaling_v2"

duration_to_seconds() {
  local value="$1"
  case "$value" in
    *h) echo $(( ${value%h} * 3600 )) ;;
    *m) echo $(( ${value%m} * 60 )) ;;
    *s) echo $(( ${value%s} )) ;;
    *)  echo "$value" ;;
  esac
}

worker() {
  local duration="$1"
  local output_root="$2"
  local report_root="$3"
  local calibration="$4"
  local wall_seconds
  wall_seconds="$(duration_to_seconds "$duration")"
  local deadline_epoch=$(( $(date +%s) + wall_seconds ))
  local python_bin="$repo_root/.venv/bin/python"
  local attempt=0

  cd "$repo_root"
  export PYTHONUNBUFFERED=1
  if [[ -z "${CUDA_HOME:-}" && -d /usr/local/cuda-13.2 ]]; then
    export CUDA_HOME=/usr/local/cuda-13.2
  elif [[ -z "${CUDA_HOME:-}" && -d /usr/local/cuda ]]; then
    export CUDA_HOME=/usr/local/cuda
  fi
  if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
    export PATH="$CUDA_HOME/bin:$PATH"
  fi

  echo "START campaign=$campaign duration=$duration output=$output_root"
  echo "COMMIT $(git rev-parse HEAD)"
  "$python_bin" -c "import rtx; print(rtx.validate_runtime_environment())"
  "$python_bin" -m rtx.autotune.dataset validate "$manifest"

  if [[ ! -f "$calibration" ]]; then
    "$python_bin" -m rtx.autotune.dataset calibrate \
      --device cuda:0 \
      --output "$calibration" \
      --samples 5 \
      --target-ms 30
  fi

  while true; do
    local remaining=$(( deadline_epoch - $(date +%s) ))
    if (( remaining <= 0 )); then
      echo "DEADLINE campaign wall time exhausted"
      break
    fi
    attempt=$((attempt + 1))
    echo "ATTEMPT $attempt remaining=${remaining}s"
    set +e
    "$python_bin" -m rtx.autotune.dataset run "$manifest" \
      --device cuda:0 \
      --output-dir "$output_root" \
      --format none \
      --calibration "$calibration" \
      --wall-time "${remaining}s" \
      --context-slice "${RTX_AUTOTUNE_CONTEXT_SLICE:-60s}" \
      --trial-milestones "${RTX_AUTOTUNE_MILESTONES:-4,8,16,32,64,128}" \
      --initial-promote "${RTX_AUTOTUNE_INITIAL_PROMOTE:-2}" \
      --strategy-orchestration bandit \
      --strategy-bandit-exploration "${RTX_AUTOTUNE_STRATEGY_EXPLORATION:-0.4}" \
      --context-orchestration breadth_first \
      --max-milestone-lead 1 \
      --reuse-deterministic-failures \
      --adopt-existing-context-identity-if-present \
      --stall-timeout "${RTX_AUTOTUNE_STALL_TIMEOUT:-180s}"
    local status=$?
    set -e
    if (( status == 0 )); then
      break
    fi
    if (( status != 75 )); then
      echo "FAIL campaign exited with status=$status"
      return "$status"
    fi
    echo "RESTART watchdog or fatal CUDA context; residual journals are durable"
  done

  mkdir -p "$report_root"
  "$python_bin" -m rtx.autotune.dataset audit "$output_root" \
    --output "$report_root/audit.json" \
    --allow-errors
  "$python_bin" -m rtx.autotune.dataset collect "$output_root" \
    --output "$report_root/dataset" \
    --format both
  echo "DONE output=$output_root reports=$report_root"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  worker "$@"
  exit $?
fi

duration="${1:-4h}"
node="${RTX_AUTOTUNE_NODE:-$(hostname -s)}"
output_root="${RTX_AUTOTUNE_OUTPUT_DIR:-autotune_datasets/${campaign}_${node}}"
report_root="${RTX_AUTOTUNE_REPORT_DIR:-autotune_reports/${campaign}_${node}}"
log_root="${RTX_AUTOTUNE_LOG_DIR:-autotune_logs}"
calibration="${RTX_AUTOTUNE_CALIBRATION:-hardware_calibration_${node}.json}"
pid_file="$log_root/${campaign}_${node}.pid"
latest_log="$log_root/${campaign}_${node}.log"

cd "$repo_root"
git pull --ff-only
if [[ ! -x .venv/bin/python ]]; then
  echo "missing $repo_root/.venv; create the supported CUDA 13.2 environment first" >&2
  exit 1
fi
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -c "import torch; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability()[0] == 12"

mkdir -p "$log_root" "$output_root" "$report_root"
if [[ -f "$pid_file" ]]; then
  old_pid="$(<"$pid_file")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "campaign is already running as PID $old_pid; log=$latest_log" >&2
    exit 1
  fi
fi

nohup "$repo_root/benchmarks/launch_all_kernel_autotune.sh" \
  --worker "$duration" "$output_root" "$report_root" "$calibration" \
  >"$latest_log" 2>&1 </dev/null &
pid=$!
echo "$pid" >"$pid_file"
disown "$pid" 2>/dev/null || true

echo "launched PID=$pid"
echo "log=$latest_log"
echo "dataset=$output_root"
echo "reports=$report_root"
echo "follow: tail -f $latest_log"
