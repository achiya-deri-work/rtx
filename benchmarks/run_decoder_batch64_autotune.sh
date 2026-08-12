#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

autotune_bin="${RTX_AUTOTUNE_BIN:-$repo_root/.venv/bin/rtx-autotune}"
python_bin="${RTX_PYTHON:-$repo_root/.venv/bin/python}"
output_root="${RTX_BATCH64_TUNE_OUTPUT:-autotune_datasets/decoder_batch64_tuning_v1}"
pretrained="${RTX_BATCH64_PRETRAINED:-autotune_models/blackwell_release_5070ti_train_v1}"
mx_seconds="${RTX_BATCH64_MX_SECONDS:-720}"
nv_seconds="${RTX_BATCH64_NV_SECONDS:-720}"
stall_timeout="${RTX_BATCH64_STALL_TIMEOUT:-120s}"

run_campaign() {
  local manifest="$1"
  local budget_seconds="$2"
  local deadline_epoch=$(( $(date +%s) + budget_seconds ))
  local attempt=0
  local remaining_seconds
  local status

  # Exit 75 is a deliberate worker recycle after a CUDA fault or a compiler
  # stall. Residual journals are durable, so resume without extending the
  # campaign's original wall-time budget.
  while true; do
    remaining_seconds=$((deadline_epoch - $(date +%s)))
    if ((remaining_seconds <= 0)); then
      echo "DEADLINE $manifest supervisor wall-time exhausted"
      break
    fi
    attempt=$((attempt + 1))
    echo "CAMPAIGN $manifest attempt=$attempt remaining=${remaining_seconds}s"
    set +e
    "$autotune_bin" run "$manifest" \
      --device cuda:0 \
      --output-dir "$output_root" \
      --format both \
      --wall-time "${remaining_seconds}s" \
      --context-slice 75s \
      --trial-milestones 8,16,32,64 \
      --initial-promote 2 \
      --strategy-orchestration bandit \
      --context-orchestration breadth_first \
      --pretrained-artifact "$pretrained" \
      --reuse-deterministic-failures \
      --adopt-existing-context-identity-if-present \
      --stall-timeout "$stall_timeout"
    status=$?
    set -e
    if ((status == 0)); then
      break
    fi
    if ((status != 75)); then
      return "$status"
    fi
    echo "RESTART compiler/CUDA worker stall; residuals are durable"
  done

  "$autotune_bin" verify-winners "$manifest" \
    --device cuda:0 \
    --output-dir "$output_root" \
    --promote 2
}

run_campaign autotune_manifests/decoder_batch64_mxfp8_v1.json "$mx_seconds"
run_campaign autotune_manifests/decoder_batch64_nvfp4_v1.json "$nv_seconds"

"$autotune_bin" audit "$output_root" \
  --output autotune_reports/decoder_batch64_tuning_audit.json
"$autotune_bin" install-winners "$output_root" --force

for batch in 24 64; do
  "$python_bin" benchmarks/train_decoder.py \
    --precision bf16 mxfp8 nvfp4_delayed nvfp4_block \
    --optimizer fp32_master \
    --batch-size "$batch" \
    --steps 12 \
    --warmup-steps 1 \
    --log-interval 1 \
    --validation-interval 12 \
    --validation-batches 1 \
    --checkpoint-interval 12 \
    --output "training_results/post_tune_batch${batch}" \
    --no-resume
  "$python_bin" benchmarks/check_decoder_throughput.py \
    "training_results/post_tune_batch${batch}" \
    --tail 7 \
    --report-only \
    >"autotune_reports/post_tune_batch${batch}.json"
done
