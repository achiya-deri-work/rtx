#!/usr/bin/env bash
# Optional, bounded counter checks after ordinary timing; never run concurrently.
set -euo pipefail
moe_source=${1:?Isolated source root}
moe_python=${2:?Python executable}
moe_seeds=${3:?Completed sweep root}
moe_output=${4:?Output directory}
moe_ncu=${5:-/usr/local/cuda-13.2/bin/ncu}
mkdir -p "$moe_output"
export PYTHONPATH="$moe_source"
export TORCHINDUCTOR_COMPILE_THREADS=4
ulimit -c 0
for moe_context in tokens_128 tokens_7376; do
  moe_tokens=${moe_context#tokens_}
  moe_base="$moe_seeds/$moe_context/result"
  # Conditional router graphs do not run reliably under counter injection.
  # Prepare and preserve the real route in an uninjected process, then use
  # the existing expert-only profiler with exactly those operands/configs.
  "$moe_python" "$moe_source/benchmarks/benchmark_moe_varlen.py" \
    --source "$moe_source" --output "$moe_output/$moe_context-fixture.json" \
    --case-id d384_e32_k4_h128 --lengths "$moe_tokens" --max-autotune \
    --expert-config "$moe_base.expert.json" --router-config "$moe_base.router.json" \
    --shared-config "$moe_base.shared.json" --prepare-only \
    --save-expert-fixture "$moe_output/$moe_context.pt" \
    > "$moe_output/$moe_context-fixture.log" 2>&1
  if "$moe_ncu" --profile-from-start off --replay-mode kernel --graph-profiling node \
    --clock-control none --cache-control none \
    --metrics gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_bytes.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active \
    --kernel-name-base demangled -k 'regex:ExpertJointRecompute' --launch-count 1 \
    --force-overwrite -o "$moe_output/$moe_context" \
    "$moe_python" "$moe_source/benchmarks/profile_moe.py" \
    --output "$moe_output/$moe_context.json" --case-id d384_e32_k4_h128 --tokens "$moe_tokens" \
    --config "$moe_base.expert.json" --fixture "$moe_output/$moe_context.pt" \
    --skip-baseline --capture counters --capture-phase expert_step --minimal-ranges \
    > "$moe_output/$moe_context.log" 2>&1; then
    "$moe_ncu" --import "$moe_output/$moe_context.ncu-rep" --csv --page raw \
      > "$moe_output/$moe_context.csv" 2>&1
  else
    echo "COUNTER_FAILED $moe_context; inspect $moe_output/$moe_context.log"
  fi
done
# One capture per fresh process avoids conflating profiler state with execution.
for moe_phase in bf16_checkpoint_step expert_bf16_checkpoint_step; do
  moe_base="$moe_seeds/fanout_8/result"
  "$moe_python" "$moe_source/benchmarks/benchmark_moe_varlen.py" \
    --source "$moe_source" --output "$moe_output/$moe_phase.json" \
    --case-id d384_e32_k4_h128 --case-spec '{"top_k":8}' --lengths 2000 --expert-phases --max-autotune \
    --expert-config "$moe_base.expert.json" --router-config "$moe_base.router.json" \
    --shared-config "$moe_base.shared.json" --phases "$moe_phase" \
    --profile torch --profile-phase "$moe_phase" --profile-replay \
    > "$moe_output/$moe_phase.log" 2>&1
done
