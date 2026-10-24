#!/usr/bin/env bash
# Usage: bash benchmarks/run_release_qualification.sh .venv/bin/python [output-dir]
# Uses the selected environment unchanged; does not install, pull, or publish.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
# Qualify this checkout even when the interpreter belongs to another environment.
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
# Bound independent compiler processes on laptop/WSL hosts; callers may override.
export TORCHINDUCTOR_COMPILE_THREADS="${TORCHINDUCTOR_COMPILE_THREADS:-4}"
runtime_python="${1:-.venv/bin/python}"
report_dir="${2:-autotune_reports/release_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$report_dir"
if [[ -e "$report_dir/status.txt" || -e "$report_dir/tests.log" ]]; then
    echo "Refusing to overwrite qualification evidence: $report_dir" >&2
    exit 2
fi
"$runtime_python" -c 'import rtx; print(rtx.validate_runtime_environment())' \
    | tee "$report_dir/runtime.log"
"$runtime_python" -c 'import json; from rtx.autotune.dataset import _source_snapshot; print(json.dumps(_source_snapshot(), indent=2))' \
    > "$report_dir/source.json"
git rev-parse HEAD > "$report_dir/commit.txt"
git diff --stat > "$report_dir/worktree.txt"
echo "START tests: $report_dir/tests.log"
test_status=0
timeout 3600s "$runtime_python" -m pytest -q --junitxml="$report_dir/tests.xml" \
    > "$report_dir/tests.log" 2>&1 || test_status=$?
# A separate process prevents a failed CUDA context contaminating this matrix.
echo "START production matrix (tests exit=$test_status): $report_dir/production.log"
matrix_status=0
timeout 600s "$runtime_python" benchmarks/validate_production.py \
    --output "$report_dir/production.json" \
    > "$report_dir/production.log" 2>&1 || matrix_status=$?
echo "START Ettin prospective inference: $report_dir/ettin.log"
model_status=0
timeout 900s "$runtime_python" -m benchmarks.ettin_regression \
    --compile --corpus wikitext --paragraph-offset 128 --paragraphs 128 \
    --max-length 511 --seeds 4801 --require-quality \
    --output "$report_dir/ettin.jsonl" > "$report_dir/ettin.log" 2>&1 || model_status=$?
echo "START Ettin short training: $report_dir/ettin_training.log"
training_status=0
timeout 900s "$runtime_python" -m benchmarks.train_ettin_qualification \
    --compile --steps 12 --output "$report_dir/ettin_training.jsonl" \
    > "$report_dir/ettin_training.log" 2>&1 || training_status=$?
printf 'tests_exit=%s\nmatrix_exit=%s\nmodel_exit=%s\ntraining_exit=%s\n' \
    "$test_status" "$matrix_status" "$model_status" "$training_status" \
    | tee "$report_dir/status.txt"
echo "DONE evidence=$report_dir"
[[ "$test_status" -eq 0 && "$matrix_status" -eq 0 && "$model_status" -eq 0 && "$training_status" -eq 0 ]]
