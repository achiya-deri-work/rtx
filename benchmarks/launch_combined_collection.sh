#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
hours="${1:-6}"
output="${2:-autotune_datasets/combined_first_hit_v1_$(hostname -s)}"
python_bin="${RTX_PYTHON:-$repo_root/.venv/bin/python}"
"$python_bin" -c 'import torch; assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.get_device_capability()[0] == 12, "SM12x required"'
"$python_bin" -m rtx.autotune.dataset validate autotune_manifests/blackwell_diversity_atlas_v1.json
mkdir -p "$output"
log="$output/launch-$(date -u +%Y%m%dT%H%M%SZ).log"
nohup "$python_bin" -u -m rtx.autotune.combined_collection \
  --hours "$hours" --output "$output" --device "${RTX_DEVICE:-cuda:0}" \
  --slice-minutes "${RTX_COLLECTION_SLICE_MINUTES:-15}" \
  > "$log" 2>&1 < /dev/null &
worker_pid=$!
disown "$worker_pid"
echo "PID=$worker_pid LOG=$log OUTPUT=$output"
echo "Follow: tail -f '$log'"
