#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

autotune_bin="${RTX_AUTOTUNE_BIN:-$repo_root/.venv/bin/rtx-autotune}"
python_bin="${RTX_PYTHON:-$repo_root/.venv/bin/python}"
output_root="${RTX_BATCH64_TUNE_OUTPUT:-autotune_datasets/decoder_batch64_tuning_v1}"
pretrained="${RTX_BATCH64_PRETRAINED:-autotune_models/blackwell_release_5070ti_train_v1}"

run_campaign() {
  local manifest="$1"
  "$autotune_bin" run "$manifest" \
    --device cuda:0 \
    --output-dir "$output_root" \
    --format both \
    --wall-time 12m \
    --context-slice 75s \
    --trial-milestones 8,16,32,64 \
    --initial-promote 2 \
    --strategy-orchestration bandit \
    --context-orchestration breadth_first \
    --pretrained-artifact "$pretrained" \
    --reuse-deterministic-failures \
    --stall-timeout 180s
  "$autotune_bin" verify-winners "$manifest" \
    --device cuda:0 \
    --output-dir "$output_root" \
    --promote 2
}

run_campaign autotune_manifests/decoder_batch64_mxfp8_v1.json
run_campaign autotune_manifests/decoder_batch64_nvfp4_v1.json

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
