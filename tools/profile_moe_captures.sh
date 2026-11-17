#!/usr/bin/env bash
# Sequential diagnostic captures, never concurrent with an ordinary timing job.
set -euo pipefail
output_dir=${1:?New capture directory required}
mkdir -p "$output_dir"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# No idle compile-worker child under Compute injection: it can deadlock the
# profiler's multi-process serialization while another context is replaying.
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHINDUCTOR_CACHE_DIR="$PWD/.inductor"
ulimit -c 0
moe_python=${RTX_PROFILE_PYTHON:-.venv/bin/python}
ncu_exe=${RTX_NCU:-/opt/nvidia/nsight-compute/2026.1.1/ncu}
nsys_exe=${RTX_NSYS:-/usr/local/bin/nsys}
seeds=(--seed-studies seeds/prepared-final-current-source-wide.json seeds/prepared-final-tma-shapes.json)
metrics=gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_bytes.sum,lts__t_sector_hit_rate.pct,launch__registers_per_thread,launch__shared_mem_per_block_allocated,launch__grid_size,launch__block_size,launch__waves_per_multiprocessor,sm__warps_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct,smsp__warp_issue_stalled_barrier_per_warp_active.pct
"$ncu_exe" --query-metrics > "$output_dir/available-metrics.txt"
# The actual returned CSV columns are authoritative; unavailable metrics must
# remain absent, not be interpreted as zero by analysis.
for context in d768_e48_k6_h256:2000 d384_e24_k6_h128:2000 d768_e48_k6_h256:8192; do
  case_id=${context%:*};tokens=${context#*:};label="$case_id-n$tokens"
  fixture="$output_dir/$label.pt"
  if [[ ! -f "$fixture" ]]; then
    "$moe_python" -m benchmarks.profile_moe --case-id "$case_id" --tokens "$tokens" \
      "${seeds[@]}" --skip-baseline --prepare-only --save-fixture "$fixture" \
      --output "$output_dir/$label-fixture.json" > "$output_dir/$label-fixture.log" 2>&1
  fi
  if [[ -x "$nsys_exe" && ! -f "$output_dir/$label-timeline.nsys-rep" ]]; then
    echo "TIMELINE $label"
    "$nsys_exe" profile --trace=cuda,nvtx --sample=none --cpuctxsw=none \
      --capture-range=cudaProfilerApi --capture-range-end=stop --cuda-graph-trace=node \
      -o "$output_dir/$label-timeline" "$moe_python" -m benchmarks.profile_moe \
      --case-id "$case_id" --tokens "$tokens" "${seeds[@]}" --fixture "$fixture" \
      --skip-baseline --capture timeline --capture-phase expert_step \
      --output "$output_dir/$label-timeline.json" > "$output_dir/$label-timeline.log" 2>&1
    "$nsys_exe" stats --report cuda_gpu_kern_sum,cuda_api_sum,nvtx_sum --format csv \
      "$output_dir/$label-timeline.nsys-rep" > "$output_dir/$label-timeline-stats.csv" 2>&1
  fi
  for phase in expert_forward expert_backward; do
    name="$label-$phase"
    if [[ ! -f "$output_dir/$name.ncu-rep" ]]; then
      echo "COUNTERS $name"
      timeout --kill-after=15s 300s "$ncu_exe" --profile-from-start off \
        --replay-mode application --app-replay-match grid --clock-control none --cache-control none \
        --metrics "$metrics" --export "$output_dir/$name" \
        "$moe_python" -m benchmarks.profile_moe --case-id "$case_id" --tokens "$tokens" \
        "${seeds[@]}" --fixture "$fixture" --skip-baseline --capture counters --capture-phase "$phase" \
        --output "$output_dir/$name.json" > "$output_dir/$name.log" 2>&1
      if [[ ! -f "$output_dir/$name.ncu-rep" ]]; then
        echo "No counter report; inspect $output_dir/$name.log" >&2;exit 1
      fi
    fi
    "$ncu_exe" --import "$output_dir/$name.ncu-rep" --csv --page raw > "$output_dir/$name.csv"
  done
done
echo "DONE captures=$output_dir"
