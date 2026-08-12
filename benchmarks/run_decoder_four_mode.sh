#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

python_bin="${RTX_PYTHON:-$repo_root/.venv/bin/python}"
output_dir="${RTX_DECODER_OUTPUT:-training_results/tinystories_four_mode_rc1}"

"$python_bin" benchmarks/train_decoder.py \
  --precision bf16 mxfp8 nvfp4_delayed nvfp4_block \
  --optimizer fp32_master \
  --steps 300000 \
  --warmup-steps 3000 \
  --log-interval 100 \
  --validation-interval 5000 \
  --checkpoint-interval 5000 \
  --output "$output_dir" \
  "$@"

"$python_bin" benchmarks/check_decoder_throughput.py \
  "$output_dir" \
  --minimum-mxfp8 0 \
  --minimum-nvfp4-delayed 0 \
  --minimum-nvfp4-block 0 \
  >"$output_dir/summary.json"
cat "$output_dir/summary.json"
