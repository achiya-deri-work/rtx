#!/usr/bin/env bash
set -euo pipefail
python_exe=${1:?Python executable}
output_dir=${2:?Fresh output directory}
mkdir -p "$output_dir"
for shape in tiny decoder2048 long wide; do
  echo "PURE PAIRED $shape"
  "$python_exe" -m benchmarks.compare_ms_eden_pipeline --shape "$shape" \
    --candidate-pipeline paired --candidate-json '{}' --warmup-ms 3000 \
    --output "$output_dir/paired-inline-$shape.json" > "$output_dir/paired-inline-$shape.log" 2>&1
done
for mode in buffer2; do
  config='{"producer_async_copy":true,"producer_stages":2}'
  for shape in tiny long; do
    echo "PAIRED $mode $shape"
    "$python_exe" -m benchmarks.compare_ms_eden_pipeline --shape "$shape" \
      --candidate-pipeline paired --candidate-json "$config" --warmup-ms 3000 \
      --output "$output_dir/paired-$mode-$shape.json" > "$output_dir/paired-$mode-$shape.log" 2>&1
  done
done
echo "TUNER SMOKE"
"$python_exe" -m benchmarks.tune_ms_eden_pipeline --shape tiny \
  --output "$output_dir/tuner-smoke" --trials 2 --rounds 4 --pool-size 8 \
  > "$output_dir/tuner-smoke.log" 2>&1
echo "TIMELINE"
nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
  --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node \
  -o "$output_dir/paired-timeline" "$python_exe" -m benchmarks.profile_backward_subparts \
  --mode public --shape 4096,2048,512 --policy fp4_fp4 --backward-pipeline paired \
  --profile --output "$output_dir/paired-timeline.json" > "$output_dir/paired-timeline.log" 2>&1
nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv \
  "$output_dir/paired-timeline.nsys-rep" > "$output_dir/paired-timeline-stats.csv" 2>&1
