#!/usr/bin/env bash
# Usage: bash benchmarks/run_release_qualification.sh .venv/bin/python [output-dir]
# Uses the selected environment unchanged; does not install, pull, or publish.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
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
timeout 1800s "$runtime_python" -m unittest discover -s tests -v \
    > "$report_dir/tests.log" 2>&1 || test_status=$?
# A separate process prevents a failed CUDA context contaminating this matrix.
echo "START production matrix (tests exit=$test_status): $report_dir/production.log"
matrix_status=0
timeout 600s "$runtime_python" benchmarks/validate_production.py \
    --output "$report_dir/production.json" \
    > "$report_dir/production.log" 2>&1 || matrix_status=$?
printf 'tests_exit=%s\nmatrix_exit=%s\n' "$test_status" "$matrix_status" \
    | tee "$report_dir/status.txt"
echo "DONE evidence=$report_dir"
[[ "$test_status" -eq 0 && "$matrix_status" -eq 0 ]]
