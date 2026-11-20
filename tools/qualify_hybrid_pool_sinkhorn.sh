#!/usr/bin/env bash
# Usage: bash tools/qualify_hybrid_pool_sinkhorn.sh ISOLATED_SOURCE PYTHON
# The snapshot must contain seeds/result.{router,expert,shared}.json.
set -euo pipefail
route_root="$(realpath "$1")"
route_python="$2"
export OMP_NUM_THREADS=1 TORCHINDUCTOR_CACHE_DIR="$route_root/inductor"
case_spec='{"d_model":384,"num_experts":32,"top_k":4,"expert_hidden_dim":128,"shared_hidden_dim":256}'
wide='{"packed_sinkhorn_backend":"persistent","packed_persistent_threads":1024}'

run() {
    local label="$1" ref="$2" fields="$3" reference_mode="$4"
    shift 4
    local output="$route_root/evidence/visits/$label"
    mkdir -p "$output"
    local -a command=("$route_python" "$route_root/benchmarks/benchmark_moe_varlen.py"
        --source "$route_root" --output "$output/result.json" --case-id d384_e32_k4_h128 --case-spec "$case_spec"
        --lengths "$@" --router-config "$route_root/seeds/result.router.json"
        --expert-config "$route_root/seeds/result.expert.json" --shared-config "$route_root/seeds/result.shared.json"
        "$reference_mode" "$route_root/evidence/references/$ref.pt"
        --router-overrides "$fields" --phases transport_forward transport_backward router_step mxfp8_step
        --rounds 2 --samples 7)
    printf '%q ' "${command[@]}" > "$output/command.sh"
    printf '\n' >> "$output/command.sh"
    (cd "$route_root"; PYTHONPATH="$route_root" "${command[@]}") > "$output/run.log" 2>&1
    sed -n '/^TIME /p' "$output/run.log"
}

run short_a1 short '{}' --save-reference 128 128 128 128 128 128 128 128
run short_b1 short "$wide" --reference 128 128 128 128 128 128 128 128
run short_b2 short "$wide" --reference 128 128 128 128 128 128 128 128
run short_a2 short '{}' --reference 128 128 128 128 128 128 128 128
run mixed_a mixed '{}' --save-reference 768 128 119 9 0 0 0 0
run mixed_b mixed "$wide" --reference 768 128 119 9 0 0 0 0
run long_a long '{}' --save-reference 512 512 0 0 0 0 0 0
run long_b long "$wide" --reference 512 512 0 0 0 0 0 0
