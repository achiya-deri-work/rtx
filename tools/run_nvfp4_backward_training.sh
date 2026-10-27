#!/usr/bin/env bash
# Run only after tuning has exited: these timings must not overlap other GPU work.
set -euo pipefail
if (( $# != 4 )); then
  echo "usage: $0 PYTHON TOKENS SELECTED_DESIGN_DIR OUTPUT_DIR" >&2
  exit 2
fi
python_exe=$1
tokens=$2
designs=$3
output=$4
if [[ -e "$output" ]]; then echo "refusing to overwrite $output" >&2; exit 2; fi
mkdir -p "$output"
for shape in decoder1536 decoder512 decoder2048 decoder1024; do
  echo "CONFIRM training-design $shape"
  # One independent attempt: do not retry until a favorable timing is observed.
  # Any numerical, runtime or timing failure stops before model training.
  trial="$output/confirm-$shape"
  timeout --kill-after=15s 300s "$python_exe" -u -m benchmarks.tune_ms_eden_design \
    --selection "$designs/$shape.json" --output "$trial" \
    --shape "$shape" --confirm-only --rounds 24 --warmup-ms 15000 \
    --seconds 240 --trial-timeout 180 > "$trial.log" 2>&1
done
for seed in 941 942; do
  modes='mxfp8 nvfp4_mxfp8 nvfp4_full'
  if [[ "$seed" == 942 ]]; then modes='nvfp4_full nvfp4_mxfp8 mxfp8'; fi
  for mode in $modes; do
    extra=()
    if [[ "$mode" == nvfp4_full ]]; then extra=(--ms-eden-designs "$designs/designs.json"); fi
    echo "TRAIN seed=$seed mode=$mode $(date -u +%FT%TZ)"
    timeout --kill-after=30s 1200s "$python_exe" -u -m benchmarks.train_precision20m \
      --tokens "$tokens" --output "$output/$mode-seed$seed.jsonl" \
      --mode "$mode" --seed "$seed" --steps 1024 --warmup-steps 64 \
      --eval-every 256 --eval-batches 32 --log-interval 128 --checkpoint-interval 1024 \
      --cudagraphs --combo-kernels "${extra[@]}" > "$output/$mode-seed$seed.log" 2>&1
  done
  "$python_exe" -m benchmarks.analyze_precision20m \
    "$output/mxfp8-seed$seed.jsonl" "$output/nvfp4_mxfp8-seed$seed.jsonl" \
    "$output/nvfp4_full-seed$seed.jsonl" --output "$output/all-modes-seed$seed.json"
  "$python_exe" -m benchmarks.analyze_precision20m --baseline-mode nvfp4_mxfp8 \
    "$output/nvfp4_mxfp8-seed$seed.jsonl" "$output/nvfp4_full-seed$seed.jsonl" \
    --output "$output/matched-forward-seed$seed.json"
done
echo "FINISHED matched training $(date -u +%FT%TZ)"
