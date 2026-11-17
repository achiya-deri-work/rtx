#!/usr/bin/env bash
# Controlled structural A/Bs, one disposable GPU worker per context.
set -euo pipefail
output_dir=${1:?Output directory required}
mkdir -p "$output_dir"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_COMPILE_THREADS=2
export TORCHINDUCTOR_CACHE_DIR="$PWD/.inductor"
ulimit -c 0
moe_python=${RTX_PROFILE_PYTHON:-.venv/bin/python}
seeds=(--seed-studies seeds/prepared-final-current-source-wide.json seeds/prepared-final-tma-shapes.json)
for context in d768_e48_k6_h256:2000 d768_e48_k6_h256:8192 d384_e24_k6_h128:8192; do
  case_id=${context%:*};tokens=${context#*:};label="$case_id-n$tokens"
  echo "VARIANTS $label"
  "$moe_python" -m benchmarks.profile_moe --case-id "$case_id" --tokens "$tokens" \
    "${seeds[@]}" --variants all --rounds 2 --samples 9 --evict \
    --output "$output_dir/$label.json" > "$output_dir/$label.log" 2>&1
done
echo 'BOUNDED_TUNE d768_e48_k6_h256-n128'
"$moe_python" -m benchmarks.profile_moe --tokens 128 "${seeds[@]}" \
  --tune-trials 16 --rounds 3 --samples 9 --evict \
  --output "$output_dir/d768_e48_k6_h256-n128-tuned.json" \
  > "$output_dir/d768_e48_k6_h256-n128-tuned.log" 2>&1
echo "DONE followups=$output_dir"
