#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

parse_duration_seconds() {
  local value="$1"
  case "$value" in
    *h) echo $(( ${value%h} * 3600 )) ;;
    *m) echo $(( ${value%m} * 60 )) ;;
    *s) echo $(( ${value%s} )) ;;
    *[!0-9]*) echo "duration must be an integer such as 4h, 240m, or 14400" >&2; return 2 ;;
    *) echo "$value" ;;
  esac
}

machine_tag() {
  local host gpu_uuid
  host="$(hostname -s 2>/dev/null || hostname)"
  gpu_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i 0 2>/dev/null | head -n 1 || true)"
  printf '%s_%s' "$host" "${gpu_uuid:-cuda0}" | tr -cs '[:alnum:]._-' '_'
}

if [[ "${1:-}" != "--worker" ]]; then
  duration="${1:-4h}"
  wall_seconds="$(parse_duration_seconds "$duration")"
  if (( wall_seconds < 1800 )); then
    echo "The release campaign requires at least 30 minutes." >&2
    exit 2
  fi

  git pull --ff-only

  log_dir="${RTX_AUTOTUNE_LOG_DIR:-autotune_logs}"
  mkdir -p "$log_dir"
  tag="$(machine_tag)"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log_path="$(realpath -m "$log_dir/release_5070_${tag}_${stamp}.log")"
  pid_path="$log_dir/release_5070_${tag}.pid"

  if [[ -f "$pid_path" ]]; then
    old_pid="$(sed -n '1p' "$pid_path")"
    if [[ "$old_pid" =~ ^[0-9]+$ ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "A release campaign is already running: pid=$old_pid"
      echo "Log: $(ls -1t "$log_dir"/release_5070_"$tag"_*.log 2>/dev/null | head -n 1)"
      exit 1
    fi
  fi

  unit_tag="$(printf '%s' "$tag" | tr -cs '[:alnum:]-' '-')"
  unit_name="rtx-autotune-release-${unit_tag}"
  if command -v systemd-run >/dev/null 2>&1 \
      && systemctl --user is-system-running >/dev/null 2>&1; then
    if systemctl --user is-active --quiet "$unit_name.service"; then
      worker_pid="$(systemctl --user show --property=MainPID --value "$unit_name.service")"
      echo "A release campaign is already running: pid=$worker_pid unit=$unit_name.service"
      exit 1
    fi
    systemd-run --user --unit="$unit_name" --collect \
      --working-directory="$repo_root" \
      --property="StandardInput=null" \
      --property="StandardOutput=append:$log_path" \
      --property="StandardError=append:$log_path" \
      "$repo_root/run_5070_4h.sh" --worker "$wall_seconds" "$tag"
    for _ in {1..20}; do
      worker_pid="$(systemctl --user show --property=MainPID --value "$unit_name.service" 2>/dev/null || true)"
      [[ "$worker_pid" =~ ^[1-9][0-9]*$ ]] && break
      sleep 0.1
    done
  else
    nohup setsid "$repo_root/run_5070_4h.sh" --worker "$wall_seconds" "$tag" \
      </dev/null >>"$log_path" 2>&1 &
    worker_pid=$!
    disown "$worker_pid" 2>/dev/null || true
  fi
  if [[ ! "$worker_pid" =~ ^[1-9][0-9]*$ ]]; then
    echo "Detached service started but no worker PID was reported." >&2
    exit 1
  fi
  echo "$worker_pid" >"$pid_path"
  echo "Started detached release campaign: pid=$worker_pid duration=$duration"
  echo "Log: $log_path"
  echo "Watch: tail -f $log_path"
  exit 0
fi

wall_seconds="${2:?missing worker wall time}"
tag="${3:?missing worker machine tag}"
log_dir="${RTX_AUTOTUNE_LOG_DIR:-autotune_logs}"
pid_path="$log_dir/release_5070_${tag}.pid"
trap 'rm -f "$pid_path"' EXIT

if [[ -z "${CUDA_HOME:-}" && -d /usr/local/cuda-13.2 ]]; then
  export CUDA_HOME=/usr/local/cuda-13.2
elif [[ -z "${CUDA_HOME:-}" && -d /usr/local/cuda ]]; then
  export CUDA_HOME=/usr/local/cuda
fi
if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
  case ":$PATH:" in
    *":$CUDA_HOME/bin:"*) ;;
    *) export PATH="$CUDA_HOME/bin:$PATH" ;;
  esac
fi
export PYTHONUNBUFFERED=1

if [[ -x .venv/bin/python ]]; then
  python_cmd=(.venv/bin/python)
  autotune_cmd=(.venv/bin/python -m rtx.autotune.dataset)
else
  python_cmd=(python)
  autotune_cmd=("${RTX_AUTOTUNE_BIN:-rtx-autotune}")
fi

manifest=autotune_manifests/blackwell_release_candidate_5070_v1.json
output_dir="${RTX_AUTOTUNE_OUTPUT_DIR:-autotune_datasets}"
campaign_root="$output_dir/blackwell_release_candidate_5070_v1"
report_dir="${RTX_AUTOTUNE_REPORT_DIR:-autotune_reports}"
mkdir -p "$output_dir" "$report_dir"

commit="$(git rev-parse --short=12 HEAD)"
calibration="hardware_calibration_release_5070_${tag}_${commit}.json"
deadline_epoch=$(( $(date +%s) + wall_seconds ))
postprocess_reserve=600

echo "START release candidate machine=$tag commit=$commit wall=${wall_seconds}s"
echo "CUDA_HOME=${CUDA_HOME:-unset}"
"${autotune_cmd[@]}" validate "$manifest"
"${autotune_cmd[@]}" probe --device cuda:0

echo "GATE full CPU/CUDA test suite"
"${python_cmd[@]}" -m unittest discover -s tests -v

echo "GATE unified production matrix (quick long-reduction shape)"
"${python_cmd[@]}" benchmarks/validate_production.py --quick \
  --output "$report_dir/production_matrix_${tag}_${commit}.json"

if [[ ! -f "$calibration" ]]; then
  echo "CALIBRATE $calibration"
  "${autotune_cmd[@]}" calibrate --device cuda:0 \
    --output "$calibration" --samples 5 --target-ms 30
fi

attempt=0
while true; do
  remaining=$(( deadline_epoch - $(date +%s) - postprocess_reserve ))
  if (( remaining <= 0 )); then
    echo "DEADLINE no tuning time remains after release gates"
    break
  fi
  attempt=$((attempt + 1))
  echo "TUNE attempt=$attempt remaining=${remaining}s residuals=$campaign_root"
  set +e
  "${autotune_cmd[@]}" run "$manifest" \
    --device cuda:0 \
    --output-dir "$output_dir" \
    --format none \
    --calibration "$calibration" \
    --wall-time "${remaining}s" \
    --context-slice 45s \
    --trial-milestones 4,8,16,32,64 \
    --initial-promote 1 \
    --strategy-orchestration bandit \
    --strategy-bandit-exploration 0.35 \
    --context-orchestration bandit \
    --bandit-min-trials 8 \
    --context-bandit-exploration 0.35 \
    --max-milestone-lead 1 \
    --adopt-existing-context-identity-if-present \
    --stall-timeout "${RTX_AUTOTUNE_STALL_TIMEOUT:-180s}" \
    --reuse-deterministic-failures
  status=$?
  set -e
  if (( status == 0 )); then
    break
  fi
  if (( status != 75 )); then
    echo "TUNE failed status=$status"
    exit "$status"
  fi
  echo "RESTART CUDA fault or watchdog stall; durable residuals retained"
done

remaining=$(( deadline_epoch - $(date +%s) ))
if (( remaining > 60 )); then
  echo "VERIFY current finalists remaining=${remaining}s"
  set +e
  timeout --signal=INT --kill-after=30s "${remaining}s" \
    "${autotune_cmd[@]}" verify-winners "$manifest" \
      --device cuda:0 --output-dir "$output_dir" \
      --calibration "$calibration" --promote 2
  verify_status=$?
  set -e
  if (( verify_status != 0 )); then
    echo "VERIFY incomplete status=$verify_status; saved residuals remain valid"
  fi
fi

echo "AUDIT $campaign_root"
"${autotune_cmd[@]}" audit "$campaign_root" \
  --output "$report_dir/release_candidate_${tag}_${commit}.audit.json"

echo "EXPORT CSV and Parquet"
"${autotune_cmd[@]}" collect "$campaign_root" \
  --output "$report_dir/release_candidate_${tag}_${commit}" --format both

echo "INSTALL verified device-local runtime winners"
"${autotune_cmd[@]}" install-winners "$campaign_root" --minimum-support 1

echo "COMPLETE machine=$tag campaign=$campaign_root"
