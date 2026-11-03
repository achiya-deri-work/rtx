#!/usr/bin/env bash
# Run after the phase suite, never concurrently on its GPU.
set -euo pipefail
python_exe=${1:?Python executable required}
output_dir=${2:?New output directory required}
mkdir -p "$output_dir"
for shape in 512,512,512 4096,2048,512 32768,512,512 4096,2048,2048; do
  for policy in stable_mxfp8 fp4_fp4; do
    label="${shape//,/_}-$policy"
    echo "PROFILE $label"
    nsys profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
      --capture-range=cudaProfilerApi --capture-range-end=stop \
      --cuda-graph-trace=node -o "$output_dir/$label" \
      "$python_exe" -m benchmarks.profile_backward_subparts --mode public \
      --shape "$shape" --policy "$policy" --profile \
      --output "$output_dir/$label.json" > "$output_dir/$label.log" 2>&1
    nsys stats --report cuda_gpu_kern_sum,cuda_api_sum --format csv \
      "$output_dir/$label.nsys-rep" > "$output_dir/$label-stats.csv" 2>&1
  done
done
mkdir -p "$output_dir/../paired_graph8"
for shape in 512,512,512 4096,2048,512; do
  label=${shape//,/_}
  echo "GRAPH8 $label"
  "$python_exe" -m benchmarks.profile_backward_subparts --mode paired --graph-batch 8 \
    --shape "$shape" --output "$output_dir/../paired_graph8/$label.json" \
    > "$output_dir/../paired_graph8/$label.log" 2>&1
done
mkdir -p "$output_dir/../paired"
for shape in 512,512,512 4096,512,512 4096,1536,512 4096,2048,512 4096,512,1024 32768,512,512 4096,2048,2048 8192,1536,1536; do
  label=${shape//,/_}
  echo "PAIRED $label"
  "$python_exe" -m benchmarks.profile_backward_subparts --mode paired \
    --shape "$shape" --output "$output_dir/../paired/$label.json" \
    > "$output_dir/../paired/$label.log" 2>&1
done
