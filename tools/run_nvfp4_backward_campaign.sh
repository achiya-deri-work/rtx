#!/usr/bin/env bash
# Device-local research search; no winner is promoted into runtime defaults.
set -euo pipefail
if (( $# != 3 )); then
  echo "usage: $0 PYTHON INITIAL_SELECTION OUTPUT_DIR" >&2
  exit 2
fi
python_exe=$1
selection=$2
output=$3
if [[ -e "$output" ]]; then
  echo "refusing to overwrite $output" >&2
  exit 2
fi
mkdir -p "$output"
failed=0
# Prioritize the actual decoder's four projections before scale extremes.
for shape in decoder1536 decoder512 decoder2048 decoder1024 tiny long wide; do
  echo "START search $shape $(date -u +%FT%TZ)"
  if timeout --kill-after=15s 2100s "$python_exe" -u -m benchmarks.tune_ms_eden_design \
      --selection "$selection" --output "$output/search-$shape" --shape "$shape" \
      --trials 64 --seconds 1800 --trial-timeout 180 --rounds 12 \
      --warmup-ms 10000 --allow-row-major > "$output/search-$shape.log" 2>&1; then
    echo "DONE search $shape"
  else
    echo "FAILED search $shape; evidence preserved"
    failed=1
    continue
  fi
  echo "START confirm $shape $(date -u +%FT%TZ)"
  if timeout --kill-after=15s 300s "$python_exe" -u -m benchmarks.tune_ms_eden_design \
      --selection "$output/search-$shape/selection.json" \
      --output "$output/confirm-$shape" --shape "$shape" --confirm-only \
      --rounds 24 --warmup-ms 15000 --seconds 240 --trial-timeout 180 \
      > "$output/confirm-$shape.log" 2>&1; then
    echo "DONE confirm $shape"
  else
    echo "FAILED confirm $shape; not a qualified winner"
    failed=1
  fi
done
echo "FINISHED search/confirmation campaign failures=$failed $(date -u +%FT%TZ)"
exit "$failed"
