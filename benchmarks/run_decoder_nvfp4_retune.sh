#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

autotune_bin="${RTX_AUTOTUNE_BIN:-$repo_root/.venv/bin/rtx-autotune}"
python_bin="${RTX_PYTHON:-$repo_root/.venv/bin/python}"
manifest="autotune_manifests/decoder_batch64_nvfp4_materialized_v2.json"
output_root="${RTX_NVFP4_RETUNE_OUTPUT:-autotune_datasets/decoder_batch64_nvfp4_retune_v2}"
budget_seconds="${RTX_NVFP4_RETUNE_SECONDS:-600}"
stall_seconds="${RTX_NVFP4_RETUNE_STALL_SECONDS:-120}"
deadline_epoch=$(( $(date +%s) + budget_seconds ))
attempt=0

while true; do
  remaining_seconds=$((deadline_epoch - $(date +%s)))
  if ((remaining_seconds <= 0)); then
    echo "DEADLINE NVFP4 retune supervisor wall-time exhausted"
    break
  fi
  attempt_stall_seconds="$stall_seconds"
  if ((attempt_stall_seconds > remaining_seconds)); then
    attempt_stall_seconds="$remaining_seconds"
  fi
  attempt=$((attempt + 1))
  echo \
    "CAMPAIGN NVFP4 native retune attempt=$attempt "\
    "remaining=${remaining_seconds}s stall=${attempt_stall_seconds}s"
  set +e
  "$autotune_bin" run "$manifest" \
    --device cuda:0 \
    --output-dir "$output_root" \
    --format both \
    --wall-time "${remaining_seconds}s" \
    --context-slice 60s \
    --trial-milestones 4,8,16,32 \
    --initial-promote 3 \
    --strategy-orchestration bandit \
    --context-orchestration breadth_first \
    --reuse-deterministic-failures \
    --stall-timeout "${attempt_stall_seconds}s"
  status=$?
  set -e
  if ((status == 0)); then
    break
  fi
  if ((status != 75)); then
    exit "$status"
  fi
  echo "RESTART compiler/CUDA worker stall; residuals are durable"
done

"$autotune_bin" verify-winners "$manifest" \
  --device cuda:0 \
  --output-dir "$output_root" \
  --promote 3
"$autotune_bin" audit "$output_root" \
  --output autotune_reports/decoder_batch64_nvfp4_retune_v2_audit.json
"$autotune_bin" install-winners "$output_root" --force

"$python_bin" benchmarks/benchmark_nvfp4_end_to_end.py \
  --compile \
  --nv-scaling block \
  --nv-backend auto \
  --mx-backend auto \
  --shape 32768,2304,768 \
  --shape 32768,768,768 \
  --shape 32768,3072,768 \
  --shape 32768,768,1536 \
  --warmup 3 \
  --rounds 9 \
  --calls 3 \
  --output autotune_reports/nvfp4_decoder_shapes_retuned_v2.json

"$python_bin" benchmarks/train_decoder.py \
  --precision bf16 mxfp8 nvfp4_block \
  --optimizer fp32_master \
  --batch-size 64 \
  --steps 12 \
  --warmup-steps 1 \
  --log-interval 1 \
  --validation-interval 12 \
  --validation-batches 1 \
  --checkpoint-interval 12 \
  --output training_results/post_nvfp4_retune_batch64 \
  --no-resume
"$python_bin" benchmarks/check_decoder_throughput.py \
  training_results/post_nvfp4_retune_batch64 \
  --tail 7 \
  --report-only \
  >autotune_reports/post_nvfp4_retune_batch64.json
