#!/usr/bin/env bash
# Isolated fanout study plus durable, compressed HOME artifact collection.
# Run under the GPU lock and a detached/transient job manager, not SSH lifetime.
set -uo pipefail
moe_source=${1:?Immutable source snapshot}
moe_python=${2:?Python executable}
moe_output=${3:?Run directory}
moe_archive=${4:?Archive directory}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=4
ulimit -c 0
cd "$moe_source" || exit 1
printf 'START source=%s output=%s cuda=%s\n' "$moe_source" "$moe_output" "$CUDA_VISIBLE_DEVICES"
"$moe_python" -u -m benchmarks.profile_moe_active_budget run --study fanout \
  --source "$moe_source" --output "$moe_output/study" --max-seconds 8400
moe_status=$?
# Preserve partial evidence even when a numerical/device/timing gate stops it.
"$moe_python" -m benchmarks.profile_moe_active_budget summarize --output "$moe_output/study"
moe_export_status=$?
mkdir -p "$moe_archive" || exit 1
cp "$moe_output/source.tar.gz" "$moe_output/home-environment.json" "$moe_archive/" || exit 1
for moe_format in json csv parquet; do
  if [[ -f "$moe_output/study/summary.$moe_format" ]]; then
    cp "$moe_output/study/summary.$moe_format" "$moe_archive/" || exit 1
  fi
done
tar -czf "$moe_archive/results.tar.gz" -C "$moe_output" study || exit 1
printf 'COLLECTED output=%s archive=%s study_exit=%s export_exit=%s\n' \
  "$moe_output" "$moe_archive" "$moe_status" "$moe_export_status"
if (( moe_status != 0 )); then exit "$moe_status"; fi
exit "$moe_export_status"
