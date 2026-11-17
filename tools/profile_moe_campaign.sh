#!/usr/bin/env bash
# Run only in an isolated RTX snapshot; no driver/power or other-project changes.
set -euo pipefail
output_dir=${1:?Output directory required}
shift
mkdir -p "$output_dir"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_CACHE_DIR="$PWD/.inductor"
ulimit -c 0
moe_python=${RTX_PROFILE_PYTHON:-.venv/bin/python}
moe_smi=$(command -v nvidia-smi || true)
if [[ -z "$moe_smi" && -x /usr/lib/wsl/lib/nvidia-smi ]]; then moe_smi=/usr/lib/wsl/lib/nvidia-smi; fi
if [[ -n "$moe_smi" ]]; then
  "$moe_smi" --query-gpu=timestamp,power.draw,clocks.sm,clocks.mem,temperature.gpu,utilization.gpu,memory.used \
    --format=csv,nounits --loop-ms=1000 > "$output_dir/telemetry.csv" &
  moe_telemetry_pid=$!
  trap 'kill "$moe_telemetry_pid" 2>/dev/null || true' EXIT
fi
"$moe_python" -m benchmarks.profile_moe --calibrate-memory --output "$output_dir/memory-bandwidth.json"
"$moe_python" -m benchmarks.run_moe_profile_study --output "$output_dir" "$@"
