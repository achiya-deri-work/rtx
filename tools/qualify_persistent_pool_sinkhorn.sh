#!/usr/bin/env bash
# Reproduce the RTX 5070 fixed-schedule confirmation in isolated snapshots.
# Usage: bash tools/qualify_persistent_pool_sinkhorn.sh RUN_ROOT PYTHON [same-source]
# RUN_ROOT contains baseline/, candidate/, seeds/tokens_128/result.{router,expert,shared}.json
# and evidence/{tiny,medium}/ref.pt from the original saved-reference passes.
# With same-source, only candidate/ is used and references are created here.
set -euo pipefail
route_run_root="$(realpath "$1")"
route_python="$2"
export TORCHINDUCTOR_CACHE_DIR="$route_run_root/inductor"
export OMP_NUM_THREADS=1
route_case='{"d_model":384,"num_experts":32,"top_k":4,"expert_hidden_dim":128,"shared_hidden_dim":256}'
route_wide='{"packed_sinkhorn_backend":"persistent","packed_persistent_threads":1024}'
run() {
    local kind="$1" label="$2" ref="$3" fields="$4" reference_mode="$5"
    shift 5
    local src="$route_run_root/$kind" seed="$route_run_root/seeds/tokens_128"
    local result="$route_run_root/evidence/confirmation/$label"
    mkdir -p "$result"
    local -a command=("$route_python" "$src/benchmarks/benchmark_moe_varlen.py"
        --source "$src" --output "$result/result.json" --case-id d384_e32_k4_h128 --case-spec "$route_case"
        --lengths "$@" --router-config "$seed/result.router.json" --expert-config "$seed/result.expert.json"
        --shared-config "$seed/result.shared.json" "$reference_mode" "$route_run_root/evidence/$ref/ref.pt"
        --router-overrides "$fields" --phases transport_forward transport_backward router_step mxfp8_step
        --rounds 2 --samples 7)
    printf '%q ' "${command[@]}" > "$result/command.sh"
    printf '\n' >> "$result/command.sh"
    (cd "$src"; PYTHONPATH="$src" "${command[@]}") > "$result/run.log" 2>&1
    sed -n '/^TIME /p' "$result/run.log"
}
if [[ "${3:-}" == same-source ]]; then
    run candidate tiny_a1 tiny '{}' --save-reference 128
    run candidate tiny_b1 tiny "$route_wide" --reference 128
    run candidate tiny_b2 tiny "$route_wide" --reference 128
    run candidate tiny_a2 tiny '{}' --reference 128
    run candidate medium_a medium '{}' --save-reference 255 0 129
    run candidate medium_b medium "$route_wide" --reference 255 0 129
else
    run candidate tiny_b1 tiny "$route_wide" --reference 128
    run baseline tiny_a1 tiny '{}' --reference 128
    run baseline tiny_a2 tiny '{}' --reference 128
    run candidate tiny_b2 tiny "$route_wide" --reference 128
    run candidate medium_b medium "$route_wide" --reference 255 0 129
fi
