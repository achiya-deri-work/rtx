#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
hours="${1:-4}"
output="${2:-autotune_datasets/first_hit_anchors_$(hostname)}"
python_bin="${RTX_PYTHON:-$repo_root/.venv/bin/python}"
# Updating is explicit: never pull over somebody's running/dirty experiments.
if [[ "${RTX_GIT_PULL:-0}" == 1 ]]; then
    git pull --ff-only
fi
mkdir -p "$output"
log="$output/launch-$(date -u +%Y%m%dT%H%M%SZ).log"
nohup "$python_bin" -u -m rtx.autotune.first_hit_study run \
  autotune_manifests/blackwell_first_hit_anchors_v1.json \
  --hours "$hours" --device "${RTX_DEVICE:-cuda:0}" --output "$output" \
  > "$log" 2>&1 < /dev/null &
worker_pid=$!
disown "$worker_pid"
echo "PID=$worker_pid LOG=$log OUTPUT=$output"
echo "Follow: tail -f '$log' (per-context details are in worker-*.log)"
