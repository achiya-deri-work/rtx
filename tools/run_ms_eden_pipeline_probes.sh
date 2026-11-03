#!/usr/bin/env bash
set -euo pipefail
python_exe=${1:?Python executable}
output_dir=${2:?Fresh output directory}
mkdir -p "$output_dir"
for shape in tiny decoder2048 long; do
  echo "PAIRED $shape"
  "$python_exe" -m benchmarks.compare_ms_eden_pipeline --shape "$shape" \
    --candidate-pipeline paired --candidate-json '{}' --warmup-ms 3000 \
    --output "$output_dir/paired-$shape.json" > "$output_dir/paired-$shape.log" 2>&1
done
for shape in decoder2048 long; do
  for stages in 1 2 3; do
    echo "BUFFER $shape stages=$stages"
    "$python_exe" -m benchmarks.compare_ms_eden_pipeline --shape "$shape" \
      --candidate-json "{\"producer_async_copy\":true,\"producer_stages\":$stages}" \
      --warmup-ms 3000 --output "$output_dir/buffer$stages-$shape.json" \
      > "$output_dir/buffer$stages-$shape.log" 2>&1
  done
done
