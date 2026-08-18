#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

duration_seconds() {
  case "$1" in
    *h) echo $(( ${1%h} * 3600 )) ;;
    *m) echo $(( ${1%m} * 60 )) ;;
    *s) echo $(( ${1%s} )) ;;
    *[!0-9]*) echo "Use an integer duration such as 6h, 360m, or 21600." >&2; return 2 ;;
    *) echo "$1" ;;
  esac
}

machine_tag() {
  local host uuid
  host="$(hostname -s 2>/dev/null || hostname)"
  uuid="$(nvidia-smi -i 0 --query-gpu=uuid --format=csv,noheader 2>/dev/null | head -n1 || true)"
  printf '%s_%s' "$host" "${uuid:-cuda0}" | tr -cs '[:alnum:]._-' '_'
}

if [[ "${1:-}" != "--worker" ]]; then
  duration="${1:-6h}"
  seconds="$(duration_seconds "$duration")"
  if (( seconds < 1800 )); then
    echo "The prospective study requires at least 30 minutes." >&2
    exit 2
  fi
  git pull --ff-only
  mkdir -p autotune_logs
  tag="$(machine_tag)"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log="$(realpath -m "autotune_logs/prospective_v2_${tag}_${stamp}.log")"
  pid_file="autotune_logs/prospective_v2_${tag}.pid"
  if [[ -f "$pid_file" ]]; then
    old_pid="$(sed -n '1p' "$pid_file")"
    if [[ "$old_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "Prospective v2 is already running: pid=$old_pid" >&2
      exit 1
    fi
  fi
  nohup setsid "$repo_root/run_blackwell_prospective_v2.sh" \
    --worker "$seconds" "$tag" </dev/null >>"$log" 2>&1 &
  worker_pid=$!
  disown "$worker_pid" 2>/dev/null || true
  echo "$worker_pid" >"$pid_file"
  echo "Started prospective v2: pid=$worker_pid duration=$duration"
  echo "Log: $log"
  echo "Watch: tail -f '$log'"
  exit 0
fi

wall_seconds="${2:?missing worker duration}"
tag="${3:?missing machine tag}"
pid_file="autotune_logs/prospective_v2_${tag}.pid"
trap 'rm -f "$pid_file"' EXIT

if [[ -z "${CUDA_HOME:-}" && -d /usr/local/cuda-13.2 ]]; then
  export CUDA_HOME=/usr/local/cuda-13.2
elif [[ -z "${CUDA_HOME:-}" && -d /usr/local/cuda ]]; then
  export CUDA_HOME=/usr/local/cuda
fi
if [[ -n "${CUDA_HOME:-}" && -x "$CUDA_HOME/bin/nvcc" ]]; then
  export PATH="$CUDA_HOME/bin:$PATH"
fi
export PYTHONUNBUFFERED=1

if [[ -x .venv/bin/python ]]; then
  autotune=(.venv/bin/python -m rtx.autotune.dataset)
else
  autotune=(rtx-autotune)
fi

manifest=autotune_manifests/blackwell_prospective_v2.json
output_root="${RTX_AUTOTUNE_OUTPUT_DIR:-autotune_datasets}"
campaign="$output_root/blackwell_prospective_v2"
report_root="${RTX_AUTOTUNE_REPORT_DIR:-autotune_reports}"
pairwise="$repo_root/rtx/autotune/artifacts/blackwell_diversity_atlas_v1_pairwise"
commit="$(git rev-parse --short=12 HEAD)"
calibration="hardware_calibration_prospective_v2_${tag}_${commit}.json"
deadline=$(( $(date +%s) + wall_seconds ))
reserve=600
mkdir -p "$output_root" "$report_root"

echo "START prospective_v2 machine=$tag commit=$commit wall=${wall_seconds}s"
"${autotune[@]}" validate "$manifest" >/dev/null
"${autotune[@]}" probe --device cuda:0
if [[ ! -f "$calibration" ]]; then
  "${autotune[@]}" calibrate --device cuda:0 \
    --output "$calibration" --samples 7 --target-ms 40
fi

run_args=(
  run "$manifest"
  --device cuda:0
  --output-dir "$output_root"
  --format none
  --calibration "$calibration"
  --pairwise-artifact "$pairwise"
  --context-slice 60s
  --trial-milestones 4,8,16,32,64
  --initial-promote 1
  --strategy-orchestration bandit
  --strategy-bandit-exploration 0.4
  --context-orchestration breadth_first
  --bandit-min-trials 8
  --max-milestone-lead 0
  --adopt-existing-context-identity-if-present
  --stall-timeout "${RTX_AUTOTUNE_STALL_TIMEOUT:-240s}"
  --reuse-deterministic-failures
)
if [[ -n "${RTX_AUTOTUNE_PRETRAINED_ARTIFACT:-}" ]]; then
  run_args+=(--pretrained-artifact "$RTX_AUTOTUNE_PRETRAINED_ARTIFACT")
fi

attempt=0
while true; do
  remaining=$(( deadline - $(date +%s) - reserve ))
  if (( remaining <= 0 )); then
    break
  fi
  attempt=$(( attempt + 1 ))
  echo "TUNE attempt=$attempt remaining=${remaining}s campaign=$campaign"
  set +e
  "${autotune[@]}" "${run_args[@]}" --wall-time "${remaining}s"
  status=$?
  set -e
  if (( status == 0 )); then
    break
  elif (( status != 75 )); then
    echo "TUNE failed status=$status" >&2
    exit "$status"
  fi
  echo "RESTART watchdog/device-context failure; residual journals are durable"
done

echo "AUDIT campaign=$campaign"
"${autotune[@]}" audit "$campaign" \
  --output "$report_root/prospective_v2_${tag}_${commit}.audit.json"
echo "EXPORT campaign=$campaign"
"${autotune[@]}" collect "$campaign" \
  --output "$report_root/prospective_v2_${tag}_${commit}" --format both
echo "SUMMARY campaign=$campaign"
"${autotune[@]}" summarize-tuners "$campaign" \
  --output "$report_root/prospective_v2_${tag}_${commit}_tuners" --format both
echo "DONE campaign=$campaign reports=$report_root"
