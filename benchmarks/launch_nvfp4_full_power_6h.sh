#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
campaign="nvfp4_full_power_6h_v1"

duration_to_seconds() {
  local value="$1"
  case "$value" in
    *h) echo $(( ${value%h} * 3600 )) ;;
    *m) echo $(( ${value%m} * 60 )) ;;
    *s) echo $(( ${value%s} )) ;;
    *) echo "$value" ;;
  esac
}

remaining_seconds() {
  local deadline="$1"
  local remaining=$(( deadline - $(date +%s) ))
  (( remaining > 0 )) && echo "$remaining" || echo 0
}

run_phase() {
  local phase="$1"
  local manifest="$2"
  local allocation_s="$3"
  local global_work_deadline="$4"
  local output_root="$5"
  local report_root="$6"
  local calibration="$7"
  local python_bin="$repo_root/.venv/bin/python"
  local phase_output="$output_root/$phase"
  local phase_report="$report_root/$phase"
  local phase_deadline=$(( $(date +%s) + allocation_s ))
  local attempt=0

  if (( phase_deadline > global_work_deadline )); then
    phase_deadline="$global_work_deadline"
  fi
  mkdir -p "$phase_output" "$phase_report"
  echo "PHASE_START name=$phase allocation=${allocation_s}s manifest=$manifest"

  while true; do
    local remaining
    remaining="$(remaining_seconds "$phase_deadline")"
    if (( remaining <= 0 )); then
      echo "PHASE_DEADLINE name=$phase"
      break
    fi
    attempt=$(( attempt + 1 ))
    echo "PHASE_ATTEMPT name=$phase attempt=$attempt remaining=${remaining}s"
    set +e
    "$python_bin" -m rtx.autotune.dataset run "$manifest" \
      --device cuda:0 \
      --output-dir "$phase_output" \
      --format none \
      --calibration "$calibration" \
      --wall-time "${remaining}s" \
      --context-slice "${RTX_AUTOTUNE_CONTEXT_SLICE:-75s}" \
      --trial-milestones "${RTX_AUTOTUNE_MILESTONES:-8,16,32,64,128,256,512,1024}" \
      --initial-promote "${RTX_AUTOTUNE_INITIAL_PROMOTE:-4}" \
      --strategy-orchestration bandit \
      --strategy-bandit-exploration "${RTX_AUTOTUNE_STRATEGY_EXPLORATION:-0.48}" \
      --context-orchestration breadth_first \
      --bandit-min-trials 24 \
      --max-milestone-lead 1 \
      --reuse-deterministic-failures \
      --adopt-existing-context-identity-if-present \
      --stall-timeout "${RTX_AUTOTUNE_STALL_TIMEOUT:-240s}"
    local status=$?
    set -e
    if (( status == 0 )); then
      echo "PHASE_COMPLETE name=$phase remaining_global=$(remaining_seconds "$global_work_deadline")s"
      break
    fi
    if (( status != 75 )); then
      echo "PHASE_FAIL name=$phase status=$status"
      return "$status"
    fi
    echo "PHASE_RESTART name=$phase reason=watchdog_or_cuda_context"
  done

  "$python_bin" -m rtx.autotune.dataset audit "$phase_output" \
    --output "$phase_report/audit.json" --allow-errors
  "$python_bin" -m rtx.autotune.dataset collect "$phase_output" \
    --output "$phase_report/dataset" --format both
  echo "PHASE_DONE name=$phase output=$phase_output report=$phase_report"
}

worker() {
  local duration="$1"
  local output_root="$2"
  local report_root="$3"
  local calibration="$4"
  local total_s
  total_s="$(duration_to_seconds "$duration")"
  local reserve_s=300
  if (( total_s < 1200 )); then
    reserve_s=60
  fi
  local started_epoch
  started_epoch="$(date +%s)"
  local global_deadline=$(( started_epoch + total_s ))
  local work_deadline=$(( global_deadline - reserve_s ))
  local release_s=$(( (total_s - reserve_s) * 25 / 100 ))
  local topology_s=$(( (total_s - reserve_s) * 35 / 100 ))
  local deep_s=$(( (total_s - reserve_s) * 30 / 100 ))
  local python_bin="$repo_root/.venv/bin/python"
  local overflow_round=0

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
  echo "CLOCK start=$started_epoch work_deadline=$work_deadline deadline=$global_deadline reserve=${reserve_s}s"
  "$python_bin" -c "import json, rtx, torch; print(json.dumps({'environment': rtx.validate_runtime_environment(), 'gpu': torch.cuda.get_device_name(0), 'capability': torch.cuda.get_device_capability(0)}, sort_keys=True))"

  local manifest
  for manifest in \
    autotune_manifests/nvfp4_revision8_release_v1.json \
    autotune_manifests/nvfp4_full_power_topology_v1.json \
    autotune_manifests/nvfp4_full_power_jit_deep_v1.json \
    autotune_manifests/nvfp4_full_power_overflow_v1.json; do
    "$python_bin" -m rtx.autotune.dataset validate "$manifest" >/dev/null
  done

  if [[ ! -f "$calibration" ]]; then
    "$python_bin" -m rtx.autotune.dataset calibrate \
      --device cuda:0 --output "$calibration" --samples 7 --target-ms 40
  fi

  run_phase \
    release autotune_manifests/nvfp4_revision8_release_v1.json \
    "$release_s" "$work_deadline" "$output_root" "$report_root" "$calibration"
  if (( $(remaining_seconds "$work_deadline") > 0 )); then
    run_phase \
      topology autotune_manifests/nvfp4_full_power_topology_v1.json \
      "$topology_s" "$work_deadline" "$output_root" "$report_root" "$calibration"
  fi
  if (( $(remaining_seconds "$work_deadline") > 0 )); then
    run_phase \
      jit_deep autotune_manifests/nvfp4_full_power_jit_deep_v1.json \
      "$deep_s" "$work_deadline" "$output_root" "$report_root" "$calibration"
  fi

  while (( $(remaining_seconds "$work_deadline") > 0 )); do
    overflow_round=$(( overflow_round + 1 ))
    local overflow_remaining
    overflow_remaining="$(remaining_seconds "$work_deadline")"
    run_phase \
      "overflow_${overflow_round}" \
      autotune_manifests/nvfp4_full_power_overflow_v1.json \
      "$overflow_remaining" "$work_deadline" "$output_root" "$report_root" "$calibration"
  done

  echo "CLOCK_WORK_DONE remaining_total=$(remaining_seconds "$global_deadline")s"
  "$python_bin" -m rtx.autotune.dataset audit "$output_root" \
    --output "$report_root/audit_all.json" --allow-errors || true
  echo "DONE output=$output_root reports=$report_root elapsed=$(( $(date +%s) - started_epoch ))s"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  worker "$@"
  exit $?
fi

duration="${1:-6h}"
node="${RTX_AUTOTUNE_NODE:-$(hostname -s)}"
output_root="${RTX_AUTOTUNE_OUTPUT_DIR:-autotune_datasets/${campaign}_${node}}"
report_root="${RTX_AUTOTUNE_REPORT_DIR:-autotune_reports/${campaign}_${node}}"
log_root="${RTX_AUTOTUNE_LOG_DIR:-autotune_logs}"
calibration="${RTX_AUTOTUNE_CALIBRATION:-hardware_calibration_${node}.json}"
pid_file="$log_root/${campaign}_${node}.pid"
log_file="$log_root/${campaign}_${node}.log"

cd "$repo_root"
git pull --ff-only
if [[ ! -x .venv/bin/python ]]; then
  echo "missing $repo_root/.venv; create the supported CUDA 13.2 environment first" >&2
  exit 1
fi
.venv/bin/python -m pip install --no-deps -e .
.venv/bin/python -c \
  "import torch; assert torch.cuda.is_available(); assert torch.cuda.get_device_capability()[0] == 12"

mkdir -p "$log_root" "$output_root" "$report_root"
if [[ -f "$pid_file" ]]; then
  old_pid="$(<"$pid_file")"
  if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
    echo "campaign already running PID=$old_pid log=$log_file" >&2
    exit 1
  fi
fi

nohup "$repo_root/benchmarks/launch_nvfp4_full_power_6h.sh" \
  --worker "$duration" "$output_root" "$report_root" "$calibration" \
  >"$log_file" 2>&1 </dev/null &
pid=$!
echo "$pid" >"$pid_file"
disown "$pid" 2>/dev/null || true

sleep 2
if ! kill -0 "$pid" 2>/dev/null; then
  echo "worker exited during launch; inspect $log_file" >&2
  tail -80 "$log_file" >&2
  exit 1
fi

echo "launched PID=$pid duration=$duration"
echo "log=$log_file"
echo "dataset=$output_root"
echo "reports=$report_root"
echo "follow: tail -f $log_file"
