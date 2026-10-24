#!/usr/bin/env bash
# One serial GPU worker. Launch independent workers only on idle devices.
# bash benchmarks/run_postmerge_validation.sh PYTHON OUTPUT SEED
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
runtime_python="${1:-.venv/bin/python}"
report_dir="${2:?provide a fresh output directory}"
study_seed="${3:-4810}"
mkdir "$report_dir"
git rev-parse HEAD > "$report_dir/commit.txt"
git diff > "$report_dir/worktree.patch"
"$runtime_python" -m benchmarks.train_ettin_qualification --compile \
    --steps 256 --length 256 --train-paragraphs 1024 \
    --validation-paragraphs 128 --validate-every 64 --seed "$study_seed" \
    --output "$report_dir/training.jsonl" > "$report_dir/training.log" 2>&1
for shape in 256,256,512,256 2048,768,1536,768 511,384,576,384; do
    for budget in 8 16 32; do
        "$runtime_python" -m tools.qualify_mlp_first_hit \
            --shape "$shape" --budget "$budget" --seconds 300 \
            --cache-root "$report_dir/cache-$shape" \
            --output "$report_dir/first-hit-$shape-$budget.json" \
            > "$report_dir/first-hit-$shape-$budget.log" 2>&1
    done
    "$runtime_python" -m tools.qualify_mlp_first_hit \
        --shape "$shape" --budget 32 --seconds 300 --reuse-only \
        --cache-root "$report_dir/cache-$shape" \
        --output "$report_dir/reuse-$shape.json" \
        > "$report_dir/reuse-$shape.log" 2>&1
done
echo "DONE $report_dir"
