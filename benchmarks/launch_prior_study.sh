#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
hours="${1:-4}"
output="${2:-autotune_datasets/prior_study_v1_$(hostname -s)_$(date -u +%Y%m%dT%H%M%SZ)}"
python_bin="${RTX_PYTHON:-$repo_root/.venv/bin/python}"
prior="${RTX_PRIOR:-$repo_root/autotune_priors/blackwell_prior_v1.json.gz}"
if [[ ! -x "$python_bin" && -z "${RTX_PYTHON:-}" ]]; then
  python3 -m venv "$repo_root/.venv"
fi
if [[ "${RTX_INSTALL_DEPS:-1}" == 1 ]]; then
  "$python_bin" -m pip install -r requirements.txt
  "$python_bin" -m pip install --no-deps -e .
fi
"$python_bin" -c 'import cutlass.cute, torch, torchao, tvm_ffi, pyarrow, einops, rtx, os; print(rtx.validate_runtime_environment()); assert torch.cuda.is_available(), "CUDA unavailable"; assert torch.cuda.get_device_capability(os.environ.get("RTX_DEVICE", "cuda:0"))[0] == 12, "SM12x required"'
mkdir -p "$output"
if [[ ! -f "$output/plan.json" ]]; then
  "$python_bin" -m rtx.autotune.prior_study plan --prior "$prior" --output "$output/plan.json"
fi
log="$output/launch-$(date -u +%Y%m%dT%H%M%SZ).log"
nohup setsid "$python_bin" -u -m rtx.autotune.prior_study run "$output/plan.json" \
  --hours "$hours" --output "$output/results" --device "${RTX_DEVICE:-cuda:0}" \
  > "$log" 2>&1 < /dev/null &
worker_pid=$!
disown "$worker_pid"
# Give nohup/setsid time to take effect before the invoking shell exits.
sleep 2
kill -0 "$worker_pid" || { tail -n 50 "$log"; exit 1; }
echo "PID=$worker_pid LOG=$log OUTPUT=$output"
echo "Follow: tail -f '$log'"
