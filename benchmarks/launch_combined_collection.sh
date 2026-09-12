#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
hours="${1:-6}"
output="${2:-autotune_datasets/combined_first_hit_v1_$(hostname -s)}"
python_bin="${RTX_PYTHON:-$repo_root/.venv/bin/python}"
if [[ ! -x "$python_bin" && -z "${RTX_PYTHON:-}" ]]; then
  python3 -m venv "$repo_root/.venv"
fi
# Installation happens before the collection clock starts, in this venv only.
# Use RTX_INSTALL_DEPS=0 for an already provisioned supported environment.
if [[ "${RTX_INSTALL_DEPS:-1}" == 1 ]]; then
  "$python_bin" -m pip install -r "$repo_root/requirements.txt"
  "$python_bin" -m pip install --no-deps -e "$repo_root"
fi
"$python_bin" -c 'import cutlass.cute, torch, torchao, tvm_ffi, einops, pyarrow, rtx; print(rtx.validate_runtime_environment()); device = __import__("os").environ.get("RTX_DEVICE", "cuda:0"); assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.get_device_capability(device)[0] == 12, "SM12x required"; print("GPU:", torch.cuda.get_device_name(device))'
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
