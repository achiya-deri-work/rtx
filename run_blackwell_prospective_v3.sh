#!/usr/bin/env bash
set -Eeuo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$repo_root"

duration_seconds() {
  case "$1" in
    *h) echo $(( ${1%h} * 3600 )) ;;
    *m) echo $(( ${1%m} * 60 )) ;;
    *s) echo $(( ${1%s} )) ;;
    *[!0-9]*) echo "Use an integer duration such as 4h, 240m, or 14400." >&2; return 2 ;;
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
  duration="${1:-4h}"
  seconds="$(duration_seconds "$duration")"
  if (( seconds < 1800 )); then
    echo "The prospective study requires at least 30 minutes." >&2
    exit 2
  fi
  git pull --ff-only
  mkdir -p autotune_logs
  tag="$(machine_tag)"
  stamp="$(date -u +%Y%m%dT%H%M%SZ)"
  log="$(realpath -m "autotune_logs/prospective_v3_${tag}_${stamp}.log")"
  pid_file="autotune_logs/prospective_v3_${tag}.pid"
  if [[ -f "$pid_file" ]]; then
    old_pid="$(sed -n '1p' "$pid_file")"
    if [[ "$old_pid" =~ ^[1-9][0-9]*$ ]] && kill -0 "$old_pid" 2>/dev/null; then
      echo "Prospective v3 is already running: pid=$old_pid" >&2
      exit 1
    fi
  fi
  nohup setsid --fork "$repo_root/run_blackwell_prospective_v3.sh" \
    --worker "$seconds" "$tag" </dev/null >>"$log" 2>&1
  worker_pid=""
  for _ in $(seq 1 50); do
    candidate="$(sed -n '1p' "$pid_file" 2>/dev/null || true)"
    if [[ "$candidate" =~ ^[1-9][0-9]*$ ]] && kill -0 "$candidate" 2>/dev/null; then
      worker_pid="$candidate"
      break
    fi
    sleep 0.1
  done
  if [[ -z "$worker_pid" ]]; then
    echo "Prospective v3 worker failed to publish a live PID; inspect $log" >&2
    exit 1
  fi
  echo "Started prospective v3: pid=$worker_pid duration=$duration"
  echo "Log: $log"
  echo "Watch: tail -f '$log'"
  exit 0
fi

wall_seconds="${2:?missing worker duration}"
tag="${3:?missing machine tag}"
pid_file="autotune_logs/prospective_v3_${tag}.pid"
echo "$$" >"$pid_file"
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
  python_bin=.venv/bin/python
  autotune=(.venv/bin/python -m rtx.autotune.dataset)
else
  python_bin=python3
  autotune=(rtx-autotune)
fi

manifest=autotune_manifests/blackwell_prospective_v3.json
output_root="${RTX_AUTOTUNE_OUTPUT_DIR:-autotune_datasets}"
campaign="$output_root/blackwell_prospective_v3"
report_root="${RTX_AUTOTUNE_REPORT_DIR:-autotune_reports}"
pairwise="$repo_root/rtx/autotune/artifacts/blackwell_diversity_atlas_v1_pairwise"
pretrained="${RTX_AUTOTUNE_PRETRAINED_ARTIFACT:-$repo_root/autotune_models/blackwell_current_portable_v3}"
archive="$repo_root/blackwell_current_portable_v3.zip"
if [[ ! -f "$pretrained/manifest.json" && "$pretrained" == "$repo_root/autotune_models/blackwell_current_portable_v3" && -f "$archive" ]]; then
  mkdir -p "$repo_root/autotune_models"
  unzip -q -o "$archive" -d "$repo_root"
fi
if [[ ! -f "$pretrained/manifest.json" ]]; then
  echo "Missing portable prior: $pretrained/manifest.json" >&2
  echo "Copy blackwell_current_portable_v3.zip to the repository root, or set RTX_AUTOTUNE_PRETRAINED_ARTIFACT." >&2
  exit 2
fi
artifact_id="$("${autotune[@]}" validate "$manifest" >/dev/null; "$python_bin" -c 'import json,sys; print(json.load(open(sys.argv[1]))["artifact_id"])' "$pretrained/manifest.json")"
if [[ "$artifact_id" != "3317a6fef969ec5129c56732" ]]; then
  echo "Unexpected pretrained artifact $artifact_id; expected 3317a6fef969ec5129c56732" >&2
  exit 2
fi

commit="$(git rev-parse --short=12 HEAD)"
calibration="hardware_calibration_prospective_v3_${tag}_${commit}.json"
deadline=$(( $(date +%s) + wall_seconds ))
reserve=900
mkdir -p "$output_root" "$report_root"

echo "START prospective_v3 machine=$tag commit=$commit artifact=$artifact_id wall=${wall_seconds}s"
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
  --pretrained-artifact "$pretrained"
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
  --active-stall-timeout "${RTX_AUTOTUNE_ACTIVE_STALL_TIMEOUT:-20m}"
  --reuse-deterministic-failures
)

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
  --output "$report_root/prospective_v3_${tag}_${commit}.audit.json"
echo "EXPORT campaign=$campaign"
"${autotune[@]}" collect "$campaign" \
  --output "$report_root/prospective_v3_${tag}_${commit}" --format both
echo "SUMMARY campaign=$campaign"
"${autotune[@]}" summarize-tuners "$campaign" \
  --output "$report_root/prospective_v3_${tag}_${commit}_tuners" --format both
echo "EVALUATE artifact=$artifact_id campaign=$campaign"
"${autotune[@]}" evaluate-pretrained "$pretrained" "$campaign" \
  --output "$report_root/prospective_v3_${tag}_${commit}_pretrained.json"
echo "DONE campaign=$campaign reports=$report_root"
