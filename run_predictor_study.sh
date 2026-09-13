#!/usr/bin/env bash
# Launch an already frozen plan; does not pull, train, or install winners.
set -euo pipefail
study_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$study_root"
study_output="${1:?Usage: bash run_predictor_study.sh OUTPUT_DIR [HOURS] [DEVICE]}"
study_hours="${2:-2}"
study_device="${3:-cuda:0}"
study_python="${RTX_STUDY_PYTHON:-$study_root/.venv/bin/python}"
test -f "$study_output/plan.json"
study_log="$study_output/launch-$(date -u +%Y%m%dT%H%M%SZ).log"
nohup bash -c '
  "$1" -u -m tools.predictor_study run --output "$2" --hours "$3" --device "$4"
  study_status=$?
  "$1" -u -m tools.analyze_predictor_study "$2"
  exit "$study_status"
' bash "$study_python" "$study_output" "$study_hours" "$study_device" \
  > "$study_log" 2>&1 < /dev/null &
study_pid=$!
disown "$study_pid"
printf 'PID=%s\nLOG=%s\n' "$study_pid" "$study_log"
