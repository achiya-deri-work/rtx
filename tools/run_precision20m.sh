#!/usr/bin/env bash
# Run one matched three-policy sequence; launch with nohup for SSH resilience.
set -euo pipefail
if (( $# < 4 )); then
  echo "usage: $0 PYTHON TOKENS OUTPUT_DIR 'MODE MODE MODE' [trainer arguments...]" >&2
  exit 2
fi
python_bin=$1
tokens=$2
output_dir=$3
mode_order=$4
shift 4
mkdir -p "$output_dir"
failed=0
for mode in $mode_order; do
  case "$mode" in
    mxfp8|nvfp4_mxfp8|nvfp4_full) ;;
    *) echo "unknown precision mode: $mode" >&2; exit 2 ;;
  esac
  if [[ -e "$output_dir/$mode.jsonl" || -e "$output_dir/$mode.log" ]]; then
    echo "refusing to overwrite $output_dir/$mode" >&2
    exit 2
  fi
  echo "START $(date -u +%FT%TZ) $mode"
  if "$python_bin" -u -m benchmarks.train_precision20m \
      --mode "$mode" --tokens "$tokens" --output "$output_dir/$mode.jsonl" \
      "$@" > "$output_dir/$mode.log" 2>&1; then
    echo "DONE $(date -u +%FT%TZ) $mode status=0"
  else
    status=$?
    echo "DONE $(date -u +%FT%TZ) $mode status=$status log=$output_dir/$mode.log"
    failed=1
  fi
done
exit "$failed"
