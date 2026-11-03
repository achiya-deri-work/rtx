#!/usr/bin/env bash
# Fixed-input diagnostic capture; never use profiler duration as a speed claim.
set -euo pipefail
python_exe=${1:?Python executable required}
output_dir=${2:?New output directory required}
ncu_exe=${3:-ncu}
mkdir -p "$output_dir"
metrics=gpu__time_duration.sum,dram__bytes_read.sum,dram__bytes_write.sum,lts__t_bytes.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active
for item in fp4:dx fp4:dw fp8:dx; do
  mode=${item%:*}
  phase=${item#*:}
  label="$mode-$phase"
  echo "COUNTERS $label"
  "$ncu_exe" --profile-from-start off --replay-mode kernel \
    --metrics "$metrics" --export "$output_dir/$label" \
    "$python_exe" -m benchmarks.profile_backward_subparts --mode "$mode" \
    --shape 4096,2048,512 --phase "$phase" --profile \
    --output "$output_dir/$label.json" > "$output_dir/$label.log" 2>&1
  "$ncu_exe" --import "$output_dir/$label.ncu-rep" --csv --page raw \
    > "$output_dir/$label.csv"
done
