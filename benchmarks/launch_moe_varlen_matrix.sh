#!/usr/bin/env bash
# Detach only this study. The matrix writes independent resumable contexts.
set -euo pipefail
repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
study_python="${RTX_PYTHON:-python}"
study_args=("$@")
study_output=""
for ((i=0; i<${#study_args[@]}; i++)); do
    if [[ "${study_args[i]}" == --output && $((i+1)) -lt ${#study_args[@]} ]]; then
        study_output="${study_args[i+1]}"
    fi
done
if [[ -z "$study_output" ]]; then
    echo 'Required: --seeds <config-sidecar-directory> --output <study-directory>' >&2
    exit 2
fi
mkdir -p -- "$study_output"
study_log="$study_output/launch-$(date -u +%Y%m%dT%H%M%SZ).log"
nohup env "PYTHONPATH=$repo_dir${PYTHONPATH:+:$PYTHONPATH}" \
    "$study_python" -u "$repo_dir/benchmarks/tune_moe_varlen_matrix.py" \
    --source "$repo_dir" "${study_args[@]}" >"$study_log" 2>&1 </dev/null &
study_pid=$!
disown "$study_pid"
printf 'PID=%s LOG=%s\n' "$study_pid" "$study_log"
