#!/usr/bin/env bash
# Fresh-process, reversed-order full-model confirmation; no profile injection.
set -euo pipefail
if (( $# < 3 || $# > 4 )); then
  echo "usage: $0 PYTHON TOKENS OUTPUT_DIR [grid|signs|philox|combined|pipeline]" >&2
  exit 2
fi
python_bin=$1
tokens=$2
output_dir=$3
experiment=${4:-grid}
case "$experiment" in
  grid|signs|philox|combined|pipeline) ;;
  *) echo "unknown experiment: $experiment" >&2; exit 2 ;;
esac
mkdir -p "$output_dir"
for name in baseline1 candidate1 hybrid1 hybrid2 candidate2 baseline2; do
  extra=()
  mode=nvfp4_full
  if [[ "$experiment" != grid ]]; then
    extra=(--ms-eden-producer-blocks 256)
  fi
  if [[ "$experiment" == pipeline ]]; then
    extra+=(--ms-eden-producer-signs register)
  fi
  case "$name" in
    candidate*)
      extra=(--ms-eden-producer-blocks 256)
      case "$experiment" in
        signs) extra+=(--ms-eden-producer-signs register) ;;
        philox) extra+=(--ms-eden-rng-mode philox) ;;
        combined) extra+=(--ms-eden-producer-signs register --ms-eden-rng-mode philox) ;;
        pipeline) extra+=(--ms-eden-producer-signs register --ms-eden-backward-pipeline paired) ;;
      esac ;;
    hybrid*) mode=nvfp4_mxfp8; extra=() ;;
  esac
  if [[ -e "$output_dir/$name.jsonl" || -e "$output_dir/$name.log" ]]; then
    echo "refusing to overwrite $output_dir/$name" >&2
    exit 2
  fi
  echo "START $(date -u +%FT%TZ) $name"
  "$python_bin" -u -m benchmarks.train_precision20m \
    --tokens "$tokens" --output "$output_dir/$name.jsonl" --mode "$mode" \
    --steps 512 --warmup-steps 32 --eval-every 128 --eval-batches 32 \
    --log-interval 128 --checkpoint-interval 512 --cudagraphs \
    "${extra[@]}" > "$output_dir/$name.log" 2>&1
  echo "DONE $(date -u +%FT%TZ) $name"
done
