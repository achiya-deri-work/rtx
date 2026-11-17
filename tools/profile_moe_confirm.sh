#!/usr/bin/env bash
# Warmed-counter confirmation and compiled-public/whole-layer timelines.
set -euo pipefail
output_dir=${1:?Output directory required}
mkdir -p "$output_dir"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/.inductor"
ulimit -c 0
moe_python=${RTX_PROFILE_PYTHON:-.venv/bin/python}
moe_module=${RTX_MOE_PROFILE_MODULE:-benchmarks.profile_moe}
nsys_exe=${RTX_NSYS:-/usr/local/bin/nsys}
ncu_exe=${RTX_NCU:-/opt/nvidia/nsight-compute/2026.1.1/ncu}
seeds=(--seed-studies seeds/prepared-final-current-source-wide.json seeds/prepared-final-tma-shapes.json)
if [[ ${2:-} == --minimal-only ]]; then
  for context in d384_e24_k6_h128:expert_step d384_e24_k6_h128:public_expert_step d768_e48_k6_h256:whole_step; do
    case_id=${context%:*};phase=${context#*:};label="$case_id-n2000-$phase"
    options=()
    if [[ "$phase" == whole_step ]]; then options+=(--whole-layer);fi
    "$nsys_exe" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
      --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node \
      -o "$output_dir/$label-timeline" "$moe_python" -m "$moe_module" \
      --case-id "$case_id" --tokens 2000 "${seeds[@]}" "${options[@]}" --skip-baseline \
      --capture timeline --minimal-ranges --capture-phase "$phase" \
      --output "$output_dir/$label-timeline.json" > "$output_dir/$label-timeline.log" 2>&1
    "$nsys_exe" stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum --format csv \
      "$output_dir/$label-timeline.nsys-rep" > "$output_dir/$label-timeline-stats.csv" 2>&1
  done
  echo "DONE minimal=$output_dir";exit
fi
for context in d384_e24_k6_h128:public_expert_step d768_e48_k6_h256:whole_step d768_e48_k6_h256:bf16_whole_step; do
  case_id=${context%:*};phase=${context#*:};label="$case_id-n2000-$phase"
  options=()
  if [[ "$phase" != public_expert_step ]]; then options+=(--whole-layer);fi
  echo "TIMELINE $label"
  "$nsys_exe" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
    --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node \
    -o "$output_dir/$label-timeline" "$moe_python" -m benchmarks.profile_moe \
    --case-id "$case_id" --tokens 2000 "${seeds[@]}" "${options[@]}" \
    --capture timeline --capture-phase "$phase" --output "$output_dir/$label-timeline.json" \
    > "$output_dir/$label-timeline.log" 2>&1
  "$nsys_exe" stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum --format csv \
    "$output_dir/$label-timeline.nsys-rep" > "$output_dir/$label-timeline-stats.csv" 2>&1
done
metrics=gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_bytes.sum,lts__t_sector_hit_rate.pct,launch__registers_per_thread,launch__shared_mem_per_block_allocated,sm__warps_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum
for phase in expert_forward expert_backward; do
  label="d768_e48_k6_h256-n2000-$phase-warmed"
  echo "COUNTERS $label"
  timeout --kill-after=15s 300s "$ncu_exe" --profile-from-start off --replay-mode application \
    --app-replay-match grid --clock-control none --cache-control none --metrics "$metrics" \
    --export "$output_dir/$label" "$moe_python" -m benchmarks.profile_moe --tokens 2000 \
    "${seeds[@]}" --fixture results/hardware-captures/d768_e48_k6_h256-n2000.pt \
    --skip-baseline --capture counters --capture-phase "$phase" \
    --output "$output_dir/$label.json" > "$output_dir/$label.log" 2>&1
  test -f "$output_dir/$label.ncu-rep"
  "$ncu_exe" --import "$output_dir/$label.ncu-rep" --csv --page raw > "$output_dir/$label.csv"
done
echo "DONE confirmation=$output_dir"
