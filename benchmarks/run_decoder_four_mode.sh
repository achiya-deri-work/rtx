#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${RTX_PYTHON:-$repo_root/.venv/bin/python}"
output_dir="${RTX_DECODER_OUTPUT:-training_results/tinystories_four_mode_rc1}"

"$python_bin" benchmarks/train_decoder.py \
  --precision bf16 mxfp8 nvfp4_delayed nvfp4_block \
  --optimizer fp32_master \
  --batch-size 24 \
  --steps 37500 \
  --warmup-steps 375 \
  --log-interval 25 \
  --validation-interval 625 \
  --checkpoint-interval 625 \
  --output "$output_dir" \
  "$@"

"$python_bin" benchmarks/check_decoder_throughput.py \
  "$output_dir" \
  --report-only \
  >"$output_dir/summary.json"
cat "$output_dir/summary.json"
