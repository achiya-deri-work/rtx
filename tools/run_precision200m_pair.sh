#!/usr/bin/env bash
# Run from an immutable RTX source snapshot. One policy per visible desktop GPU.
set -euo pipefail
if (( $# != 3 )); then
  echo "usage: $0 ABSOLUTE_PYTHON ABSOLUTE_TOKENS NEW_OUTPUT_DIRECTORY" >&2
  exit 2
fi
python_exe=$1
tokens=$2
output=$3
if [[ -e "$output" ]]; then echo "refusing to overwrite $output" >&2; exit 2; fi
mkdir -p "$output"
common=(--tokens "$tokens" --corpus-tokens 100000000 --layers 16 --width 1024
  --intermediate 2816 --heads 16 --batch 16 --sequence 512 --seed 941
  --learning-rate 0.0003 --cudagraphs --combo-kernels --cpu-threads 4
  --mlp-activation situ --situ-beta 4 --situ-linear-beta 25)
run_pair() {
  local phase=$1
  shift
  local pids=() failed=0
  local modes=(nvfp4_mxfp8 nvfp4_full)
  for gpu in 0 1; do
    local mode=${modes[$gpu]}
    local extra=()
    if [[ "$mode" == nvfp4_full ]]; then extra=(--ms-eden-backward-pipeline paired); fi
    echo "START $phase gpu=$gpu mode=$mode $(date -u +%FT%TZ)"
    CUDA_VISIBLE_DEVICES=$gpu "$python_exe" -u -m benchmarks.train_precision20m \
      "${common[@]}" --mode "$mode" --output "$output/$phase-$mode.jsonl" \
      "${extra[@]}" "$@" > "$output/$phase-$mode.log" 2>&1 &
    pids+=("$!")
  done
  printf '%s\n' "${pids[@]}" > "$output/$phase.pids"
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then failed=1; fi
  done
  if (( failed )); then echo "FAILED $phase; inspect logs; no subsequent phase launched"; return 1; fi
  "$python_exe" -m benchmarks.analyze_precision20m --baseline-mode nvfp4_mxfp8 \
    "$output/$phase-nvfp4_mxfp8.jsonl" "$output/$phase-nvfp4_full.jsonl" \
    --output "$output/$phase-comparison.json"
  echo "DONE $phase $(date -u +%FT%TZ)"
}
# Independent smoke initialization; no smoke optimizer state enters the study.
run_pair smoke --steps 8 --warmup-steps 2 --eval-every 8 --eval-batches 4 \
  --log-interval 4 --checkpoint-interval 8
run_pair train --steps 12288 --warmup-steps 512 --eval-every 1024 --eval-batches 32 \
  --log-interval 128 --checkpoint-interval 4096
echo "FINISHED matched ~206M decoder study $(date -u +%FT%TZ)"
