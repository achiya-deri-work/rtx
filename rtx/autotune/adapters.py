"""Adapters connecting every current MXFP8 tuning family to the shared engine."""

from __future__ import annotations

from dataclasses import replace
from typing import Callable, Iterable, Mapping

from .core import DiscreteKernelAdapter, KernelContext
from .legacy import DeviceFingerprint
from .outcomes import TrialOutcome
from .hardware import (
    geometry_features,
    launch_resource_features,
    profile_value,
    traffic_features,
)
from ..kernels.mxfp8 import (
    DEFAULT_MXFP8_FWD_CONFIG,
    FWD_SEARCH_SPACE,
    MXFP8_FWD_KERNEL_REVISION,
    MXFP8FwdConfig,
    MXFP8Problem,
    SM120_FUSED_RUNTIME_SMEM_RESERVE_BYTES,
    SM120_GEMM_RUNTIME_SMEM_RESERVE_BYTES,
    fwd_config_from_dict,
    fwd_config_id,
    fwd_config_to_dict,
    normalize_fwd_config,
)


# The fused kernel carries its persistent output sequence in a constexpr loop.
# Empirical SM120 compiles remain bounded through 72 slots, while 192-slot
# candidates repeatedly exceed the worker watchdog in NVVM. Keep some margin
# for portable campaigns without hiding the useful 16--96-slot region.
MAX_FUSED_PERSISTENT_CONSTEXPR_WORK_TILES = 96


def _device_dict(
    device: DeviceFingerprint | Mapping[str, object] | None,
) -> Mapping[str, object]:
    if device is None:
        return {}
    return device.as_dict() if isinstance(device, DeviceFingerprint) else dict(device)


def _context(
    family: str,
    revision: int,
    problem: MXFP8Problem,
    device: DeviceFingerprint | Mapping[str, object] | None,
    regime: str,
    tags: Mapping[str, object] | None,
) -> KernelContext:
    return KernelContext(
        family=family,
        kernel_revision=revision,
        workload={"m": problem.m, "n": problem.n, "k": problem.k},
        device=_device_dict(device),
        regime=regime,
        tags={} if tags is None else dict(tags),
    )


def _sm120_scale_bytes(tile_m: int, tile_n: int, tile_k: int, stages: int) -> int:
    return stages * (
        ((tile_m + 127) // 128) * 128
        + ((tile_n + 127) // 128) * 128
    ) * (tile_k // 32)


def _fused_smem_bytes(config: MXFP8FwdConfig) -> int:
    operands = config.mxfp8_stages * (
        config.tile_m + config.tile_n
    ) * config.tile_k * config.native_operand_bits // 8
    scales = config.mxfp8_stages * (
        ((config.tile_m + 127) // 128) * 128
        + ((config.tile_n + 127) // 128) * 128
    ) * (config.tile_k // config.scale_vector_size)
    bf16 = 0
    if config.load_engine in ("cpasync", "tma"):
        bf16 = (
            config.bf16_stages
            * (config.tile_m + config.tile_n)
            * config.bf16_tile_k
            * 2
        )
    epilogue = (
        config.epilogue_stages * config.tile_m * config.tile_n * 2
        if config.epilogue != "direct"
        else 0
    )
    reserve = (
        SM120_FUSED_RUNTIME_SMEM_RESERVE_BYTES
        if config.load_engine != "scalar"
        else 0
    )
    delayed_scale = 32 if getattr(config, "collect_amax", False) else 0
    return operands + scales + bf16 + epilogue + delayed_scale + reserve


def _fused_grid_geometry(
    problem: MXFP8Problem,
    config: MXFP8FwdConfig,
    device: DeviceFingerprint | Mapping[str, object] | None,
) -> tuple[int, int, int]:
    profile = _device_dict(device)
    natural_ctas = (
        (problem.m + config.tile_m - 1) // config.tile_m
    ) * ((problem.n + config.tile_n - 1) // config.tile_n)
    grid_ctas = natural_ctas
    if config.persistent:
        sm_count = max(
            1, int(profile_value(profile, "multiprocessor_count", 1) or 1)
        )
        grid_ctas = min(natural_ctas, sm_count * config.persistent_waves)
        while grid_ctas > 1 and natural_ctas % grid_ctas:
            grid_ctas -= 1
    return natural_ctas, grid_ctas, natural_ctas // grid_ctas


def _gemm_smem_bytes(config: object) -> int:
    q_bytes = (
        config.stages
        * (config.tile_m + config.tile_n)
        * config.tile_k
        * config.native_operand_bits
        // 8
    )
    scale_bytes = config.stages * (
        ((config.tile_m + 127) // 128) * 128
        + ((config.tile_n + 127) // 128) * 128
    ) * (
        config.tile_k // config.scale_vector_size
    )
    out_bytes = (
        config.epilogue_stages * config.tile_m * config.tile_n * 2
        if config.epilogue == "tma"
        else 0
    )
    return q_bytes + scale_bytes + out_bytes


def _gemm_launch_smem_bytes(config: object) -> int:
    # CuTe's generated wrapper reserves an additional 1 KiB for pipeline
    # barriers/descriptors on the current SM120 GEMM. The raw operand estimate
    # alone allowed a 101,376-byte candidate whose actual launch requested
    # 102,400 bytes on a 101,376-byte device limit.
    return _gemm_smem_bytes(config) + SM120_GEMM_RUNTIME_SMEM_RESERVE_BYTES


def _gemm_smem_rejection(
    config: object,
    device: DeviceFingerprint | Mapping[str, object] | None,
) -> str | None:
    profile = _device_dict(device)
    smem_limit = int(
        profile_value(profile, "shared_memory_per_block_optin", 0)
        or profile_value(profile, "shared_memory_per_block", 0)
        or 0
    )
    required = _gemm_launch_smem_bytes(config)
    if smem_limit and required > smem_limit:
        return (
            f"GEMM launch requires {required} bytes of CTA SMEM including "
            f"runtime overhead, device limit is {smem_limit}"
        )
    return None


def _specialized_register_budget(
    *,
    consumer_warps: int,
    consumer_registers: int,
    producer_warps: int,
    producer_registers: int,
    quantizer_warps: int = 0,
    quantizer_registers: int = 0,
) -> int:
    # setmaxnreg values are limits rather than final compiler allocation.  The
    # feature is explicitly a budget; compiled metadata supersedes it.
    return 32 * (
        consumer_warps * consumer_registers
        + producer_warps * producer_registers
        + quantizer_warps * quantizer_registers
    )


def _gemm_features(
    problem: MXFP8Problem,
    config: object,
    device: DeviceFingerprint | Mapping[str, object] | None,
    *,
    materialized_quant: bool,
    split_reduction: int = 1,
) -> dict[str, float]:
    profile = _device_dict(device)
    m_tiles = (problem.m + config.tile_m - 1) // config.tile_m
    n_tiles = (problem.n + config.tile_n - 1) // config.tile_n
    natural_ctas = m_tiles * n_tiles
    total_work_ctas = natural_ctas * split_reduction
    grid_ctas = (
        total_work_ctas + config.tiles_per_cta - 1
    ) // config.tiles_per_cta
    if config.persistent_waves:
        sm_count = max(
            1, int(profile_value(profile, "multiprocessor_count", 1) or 1)
        )
        grid_ctas = max(
            grid_ctas,
            min(total_work_ctas, sm_count * config.persistent_waves),
        )
    work_tiles_per_cta = (
        (total_work_ctas + grid_ctas - 1) // grid_ctas
        if config.persistent_waves
        else config.tiles_per_cta
    )
    same_a_edges = sum(
        1
        for edge in range(1, natural_ctas)
        if edge % n_tiles and edge % config.tiles_per_cta
    )
    same_b_edges = sum(
        1
        for edge in range(1, natural_ctas)
        if edge % m_tiles and edge % config.tiles_per_cta
    )
    geometry = geometry_features(
        m=problem.m,
        n=problem.n,
        k=problem.k,
        tile_m=config.tile_m,
        tile_n=config.tile_n,
        tile_k=config.tile_k,
        profile=profile,
        grid_ctas=grid_ctas,
    )
    registers = _specialized_register_budget(
        consumer_warps=config.num_mma_warps,
        consumer_registers=config.consumer_registers,
        producer_warps=1,
        producer_registers=config.producer_registers,
    )
    geometry.update(
        launch_resource_features(
            profile=profile,
            grid_ctas=grid_ctas,
            threads_per_cta=config.num_threads,
            smem_bytes_per_cta=_gemm_launch_smem_bytes(config),
            register_budget_per_cta=registers,
            register_limit_per_thread=config.maxrregcount,
        )
    )
    geometry.update(
        traffic_features(
            m=problem.m,
            n=problem.n,
            k=problem.k,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            input_element_bytes=2,
            output_element_bytes=2,
            profile=profile,
            materialized_quant=materialized_quant,
            quantized_element_bits=config.native_operand_bits,
            scale_vector_size=config.scale_vector_size,
        )
    )
    geometry.update(
        tile_flops=float(2 * config.tile_m * config.tile_n * problem.k),
        mma_k_tiles_per_cta=float((problem.k + config.tile_k - 1) // config.tile_k),
        mma_warp_issues_per_k_tile=float(config.num_mma_warps),
        work_tiles_per_cta=float(work_tiles_per_cta),
        persistent_waves=float(config.persistent_waves),
        balanced_persistent_grid=float(bool(config.persistent_waves)),
        epilogue_stages=float(config.epilogue_stages),
        epilogue_smem_bytes=float(
            config.epilogue_stages * config.tile_m * config.tile_n * 2
            if config.epilogue == "tma"
            else 0
        ),
        epilogue_async_overlap_tiles=float(
            min(config.epilogue_stages, work_tiles_per_cta)
            if config.epilogue == "tma"
            else 0
        ),
        split_work_ctas=float(total_work_ctas),
        final_cta_active_fraction=(
            total_work_ctas / (grid_ctas * work_tiles_per_cta)
        ),
        same_a_locality=float(
            config.tile_locality in ("same_a", "serpentine_a")
        ),
        same_b_locality=float(
            config.tile_locality in ("same_b", "serpentine_b")
        ),
        serpentine_locality=float(
            config.tile_locality in ("serpentine_a", "serpentine_b")
        ),
        consecutive_a_reuse_edges=float(
            same_a_edges
            if config.tile_locality in ("same_a", "serpentine_a")
            else 0
        ),
        consecutive_b_reuse_edges=float(
            same_b_edges
            if config.tile_locality in ("same_b", "serpentine_b")
            else 0
        ),
        consecutive_operand_reuse=float(
            same_a_edges
            if config.tile_locality in ("same_a", "serpentine_a")
            else (
                same_b_edges
                if config.tile_locality in ("same_b", "serpentine_b")
                else 0
            )
        ),
    )
    return geometry


def _quant_features(
    rows: int,
    k: int,
    config: object,
    device: DeviceFingerprint | Mapping[str, object] | None,
    *,
    transposed: bool = False,
) -> dict[str, float]:
    profile = _device_dict(device)
    sm_count = max(1, int(profile_value(profile, "multiprocessor_count", 1) or 1))
    if transposed:
        task_groups = rows // config.transposed_tile_rows * (
            k // config.transposed_tile_k
        )
        natural_ctas = task_groups
    else:
        warp_tasks = rows * (k // 32) // config.quant_vec
        task_groups = (warp_tasks + config.num_warps - 1) // config.num_warps
        natural_ctas = task_groups
    grid_ctas = min(natural_ctas, sm_count * config.persistent_waves)
    threads = config.num_warps * 32
    smem = 0
    if transposed:
        # One logical [row,K] tile is backed by physical [K,row] SMEM with a
        # configurable padding column. The kernel uses BF16 elements.
        smem = (
            config.transposed_tile_k
            * (config.transposed_tile_rows + config.transposed_smem_padding)
            * 2
            + config.transposed_tile_rows
            * (config.transposed_tile_k // 32)
        )
    values = {
        "rows": float(rows),
        "task_groups": float(task_groups),
        "natural_ctas": float(natural_ctas),
        "grid_ctas": float(grid_ctas),
        "values_quantized": float(rows * k),
        "scale_blocks": float(rows * (k // 32)),
        "values_per_warp_task": float(config.quant_vec * 32),
        "transposed_source": float(transposed),
        "transposed_tile_values": float(
            config.transposed_tile_rows * config.transposed_tile_k
            if transposed
            else 0
        ),
        "transposed_scale_store_bytes": float(
            4
            if transposed and config.native_scale_store == "packed"
            else (1 if transposed else 0)
        ),
        "transposed_async_load": float(
            transposed and config.transposed_load_engine == "cp_async"
        ),
    }
    values.update(
        launch_resource_features(
            profile=profile,
            grid_ctas=max(1, grid_ctas),
            threads_per_cta=threads,
            smem_bytes_per_cta=smem,
            register_budget_per_cta=threads * config.maxrregcount,
            register_limit_per_thread=config.maxrregcount,
        )
    )
    return values


def _nvfp4_quant_features(
    rows: int,
    k: int,
    config: object,
    device: DeviceFingerprint | Mapping[str, object] | None,
) -> dict[str, float]:
    profile = _device_dict(device)
    sm_count = max(1, int(profile_value(profile, "multiprocessor_count", 1) or 1))
    scale_blocks = rows * (k // 16)
    task_groups = scale_blocks // config.blocks_per_warp
    natural_ctas = (task_groups + config.num_warps - 1) // config.num_warps
    grid_ctas = min(natural_ctas, sm_count * config.persistent_waves)
    threads = config.num_warps * 32
    values = {
        "rows": float(rows),
        "task_groups": float(task_groups),
        "natural_ctas": float(natural_ctas),
        "grid_ctas": float(grid_ctas),
        "values_quantized": float(rows * k),
        "scale_blocks": float(scale_blocks),
        "values_per_lane": float(config.values_per_lane),
        "threads_per_scale": float(config.threads_per_scale),
        "blocks_per_warp": float(config.blocks_per_warp),
        "packed_output_bytes": float(rows * k / 2),
        "scale_output_bytes": float(scale_blocks),
    }
    values.update(
        launch_resource_features(
            profile=profile,
            grid_ctas=max(1, grid_ctas),
            threads_per_cta=threads,
            smem_bytes_per_cta=0,
            register_budget_per_cta=threads * config.maxrregcount,
            register_limit_per_thread=config.maxrregcount,
        )
    )
    return values


def _prefix(values: Mapping[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def _apply_fused_cluster_reuse_features(
    values: dict[str, float],
    *,
    problem: MXFP8Problem,
    config: MXFP8FwdConfig,
    natural_output_ctas: int,
    profile: Mapping[str, object] | None,
) -> None:
    """Account for native-tile DSMEM publication and GMEM load elision."""

    cluster_size = config.cluster_size if config.cluster_reuse != "none" else 1
    peer_ctas = natural_output_ctas * (cluster_size - 1) / cluster_size
    reuse_rows = (
        config.tile_m
        if config.cluster_reuse == "a"
        else config.tile_n if config.cluster_reuse == "b" else 0
    )
    # Cooperative scalar/cp.async peers never fetch or quantize the shared
    # BF16 operand. TMA currently still stages BF16 in every CTA, although only
    # rank zero performs quantization.
    bf16_saved = (
        peer_ctas * reuse_rows * problem.k * 2
        if config.cluster_reuse != "none" and config.load_engine != "tma"
        else 0.0
    )
    native_operand_bytes = (
        reuse_rows * problem.k * config.native_operand_bits / 8
    )
    native_scale_bytes = reuse_rows * (
        (problem.k + config.scale_vector_size - 1)
        // config.scale_vector_size
    )
    native_bytes = (
        peer_ctas * (native_operand_bytes + native_scale_bytes)
        if config.cluster_reuse != "none"
        else 0.0
    )
    if bf16_saved:
        values["estimated_operand_read_bytes"] = max(
            0.0, values["estimated_operand_read_bytes"] - bf16_saved
        )
        values["estimated_total_memory_bytes"] = max(
            1.0, values["estimated_total_memory_bytes"] - bf16_saved
        )
        values["arithmetic_intensity_flops_per_byte"] = (
            values["nominal_flops"] / values["estimated_total_memory_bytes"]
        )
        bandwidth = float(
            profile_value(profile, "measured_dram_bandwidth_gbps", 0)
            or profile_value(profile, "theoretical_memory_bandwidth_gbps", 0)
            or 0
        )
        if bandwidth:
            values["memory_roofline_ms"] = values[
                "estimated_total_memory_bytes"
            ] / (bandwidth * 1.0e6)
    values.update(
        cluster_operand_reuse=float(config.cluster_reuse != "none"),
        cluster_operand_reuse_a=float(config.cluster_reuse == "a"),
        cluster_operand_reuse_b=float(config.cluster_reuse == "b"),
        cluster_operand_reuse_size=float(cluster_size),
        cluster_operand_bf16_bytes_saved=float(bf16_saved),
        cluster_native_quant_bytes_saved=float(native_bytes),
        cluster_dsmem_publication_bytes=float(native_bytes),
    )


def make_mxfp8_fwd_adapter(
    problem: MXFP8Problem,
    evaluator: Callable[[MXFP8FwdConfig], TrialOutcome],
    *,
    initial: MXFP8FwdConfig = DEFAULT_MXFP8_FWD_CONFIG,
    axes: Mapping[str, Iterable[object]] = FWD_SEARCH_SPACE,
    device: DeviceFingerprint | Mapping[str, object] | None = None,
    regime: str = "hot",
    tags: Mapping[str, object] | None = None,
    _family: str = "mxfp8_fused_fwd",
    _revision: int = MXFP8_FWD_KERNEL_REVISION,
    _allowed_axes: Mapping[str, Iterable[object]] = FWD_SEARCH_SPACE,
    _normalizer: Callable[..., MXFP8FwdConfig] = normalize_fwd_config,
    _deserialize: Callable[[dict[str, object]], MXFP8FwdConfig] = fwd_config_from_dict,
) -> DiscreteKernelAdapter[MXFP8FwdConfig]:
    axis_values = {name: tuple(values) for name, values in axes.items()}
    unknown = set(axis_values).difference(_allowed_axes)
    if unknown:
        raise ValueError(f"unknown fused-forward tuning axes: {sorted(unknown)}")

    def rejection(config: MXFP8FwdConfig) -> tuple[str, str] | None:
        reason = config.architecture_rejection(problem)
        if reason is not None:
            return "architecture_rejected", reason
        profile = _device_dict(device)
        smem_limit = int(
            profile_value(profile, "shared_memory_per_block_optin", 0)
            or profile_value(profile, "shared_memory_per_block", 0)
            or 0
        )
        if smem_limit and _fused_smem_bytes(config) > smem_limit:
            return (
                "architecture_rejected",
                f"candidate requires {_fused_smem_bytes(config)} bytes of CTA SMEM, "
                f"device limit is {smem_limit}",
            )
        _, _, work_tiles_per_cta = _fused_grid_geometry(problem, config, device)
        if (
            config.persistent
            and work_tiles_per_cta
            > MAX_FUSED_PERSISTENT_CONSTEXPR_WORK_TILES
        ):
            return (
                "implementation_rejected",
                f"persistent constexpr work has {work_tiles_per_cta} tiles per "
                "CTA, exceeding the compiler-safety cap of "
                f"{MAX_FUSED_PERSISTENT_CONSTEXPR_WORK_TILES}",
            )
        reason = config.implementation_rejection(problem)
        return None if reason is None else ("implementation_rejected", reason)

    def derived(config: MXFP8FwdConfig) -> Mapping[str, float]:
        profile = _device_dict(device)
        natural_ctas, grid_ctas, _ = _fused_grid_geometry(
            problem, config, device
        )
        values = geometry_features(
            m=problem.m,
            n=problem.n,
            k=problem.k,
            tile_m=config.tile_m,
            tile_n=config.tile_n,
            tile_k=config.tile_k,
            profile=profile,
            grid_ctas=grid_ctas,
        )
        if config.schedule == "cooperative":
            register_budget = config.num_threads * config.maxrregcount
        else:
            register_budget = _specialized_register_budget(
                consumer_warps=config.num_mma_warps,
                consumer_registers=config.consumer_registers,
                producer_warps=config.producer_warps,
                producer_registers=config.producer_registers,
                quantizer_warps=(
                    config.quantizer_warps if config.schedule == "three_role" else 0
                ),
                quantizer_registers=config.quantizer_registers,
            )
        delayed_enabled = bool(getattr(config, "collect_amax", False))
        telemetry_slots = (
            1
            if getattr(config, "telemetry_layout", "per_cta")
            == "scalar_atomic"
            else grid_ctas
        )
        history_len = int(getattr(config, "amax_history_len", 1))
        history_reads = (
            1
            if getattr(config, "amax_history_algo", "most_recent")
            == "most_recent"
            else history_len
        )
        owner_only = (
            getattr(config, "telemetry_ownership", "all")
            == "operand_owner"
        )
        m_tiles = (problem.m + config.tile_m - 1) // config.tile_m
        n_tiles = (problem.n + config.tile_n - 1) // config.tile_n
        values.update(
            launch_resource_features(
                profile=profile,
                grid_ctas=grid_ctas,
                threads_per_cta=config.num_threads,
                smem_bytes_per_cta=_fused_smem_bytes(config),
                register_budget_per_cta=register_budget,
                register_limit_per_thread=config.maxrregcount,
            )
        )
        values.update(
            traffic_features(
                m=problem.m,
                n=problem.n,
                k=problem.k,
                tile_m=config.tile_m,
                tile_n=config.tile_n,
                input_element_bytes=2,
                output_element_bytes=2,
                profile=profile,
                materialized_quant=False,
            )
        )
        _apply_fused_cluster_reuse_features(
            values,
            problem=problem,
            config=config,
            natural_output_ctas=natural_ctas,
            profile=profile,
        )
        values.update(
            work_tiles_per_cta=natural_ctas / max(1, grid_ctas),
            epilogue_stages=float(config.epilogue_stages),
            epilogue_smem_bytes=float(
                config.epilogue_stages * config.tile_m * config.tile_n * 2
                if config.epilogue == "tma"
                else 0
            ),
            epilogue_async_overlap_tiles=float(
                min(
                    config.epilogue_stages,
                    natural_ctas / max(1, grid_ctas),
                )
                if config.epilogue == "tma"
                else 0
            ),
            bf16_values_quantized_per_output_cta=float(
                (config.tile_m + config.tile_n) * problem.k
            ),
            tile_flops=float(2 * config.tile_m * config.tile_n * problem.k),
            mma_k_tiles_per_output_cta=float(
                (problem.k + config.tile_k - 1) // config.tile_k
            ),
            pipeline_buffer_bytes=float(_fused_smem_bytes(config)),
            delayed_telemetry_slots=float(
                telemetry_slots if delayed_enabled else 0
            ),
            delayed_telemetry_state_bytes=float(
                telemetry_slots * history_len * 2 * 4
                if delayed_enabled else 0
            ),
            delayed_telemetry_l2_read_bytes=float(
                grid_ctas * telemetry_slots * history_reads * 2 * 4
                if delayed_enabled else 0
            ),
            delayed_telemetry_memsets=float(
                2
                if delayed_enabled
                and getattr(config, "telemetry_layout", "per_cta")
                == "scalar_atomic"
                else 0
            ),
            delayed_x_observer_fraction=float(
                1.0 / n_tiles if delayed_enabled and owner_only else 1.0
                if delayed_enabled else 0.0
            ),
            delayed_weight_observer_fraction=float(
                1.0 / m_tiles if delayed_enabled and owner_only else 1.0
                if delayed_enabled else 0.0
            ),
            delayed_atomic_contention_slots=float(
                telemetry_slots if delayed_enabled else 0
            ),
            delayed_scale_prepare_launches=0.0,
            total_kernel_launches=1.0,
        )
        return values

    return DiscreteKernelAdapter(
        context=_context(
            _family,
            _revision,
            problem,
            device,
            regime,
            tags,
        ),
        initial_config=initial,
        axes=axis_values,
        config_id_fn=fwd_config_id,
        serialize_fn=fwd_config_to_dict,
        deserialize_fn=lambda value: _deserialize(dict(value)),
        update_fn=lambda config, coordinate, value: _normalizer(
            config, **{coordinate: value}
        ),
        evaluator=evaluator,
        rejection_fn=rejection,
        extra_features_fn=derived,
    )


def make_nvfp4_fwd_adapter(
    problem,
    evaluator: Callable[[object], TrialOutcome],
    *,
    initial=None,
    axes=None,
    device: DeviceFingerprint | Mapping[str, object] | None = None,
    regime: str = "hot",
    tags: Mapping[str, object] | None = None,
):
    from ..configs.nvfp4 import (
        DEFAULT_NVFP4_FWD_CONFIG,
        NVFP4_FWD_SEARCH_SPACE,
        NVFP4_KERNEL_REVISION,
        normalize_nvfp4_fwd_config,
    )

    selected_axes = NVFP4_FWD_SEARCH_SPACE if axes is None else axes
    selected_initial = initial or replace(
        DEFAULT_NVFP4_FWD_CONFIG, collect_amax=True
    )

    def deserialize(values: dict[str, object]):
        return normalize_nvfp4_fwd_config(**values)

    return make_mxfp8_fwd_adapter(
        problem,
        evaluator,
        initial=selected_initial,
        axes=selected_axes,
        device=device,
        regime=regime,
        tags=tags,
        _family="nvfp4_fused_fwd",
        _revision=NVFP4_KERNEL_REVISION,
        _allowed_axes=NVFP4_FWD_SEARCH_SPACE,
        _normalizer=normalize_nvfp4_fwd_config,
        _deserialize=deserialize,
    )


def make_mxfp8_prequant_adapter(
    problem: MXFP8Problem,
    evaluator: Callable[[object], TrialOutcome],
    *,
    initial: object | None = None,
    axes: Mapping[str, Iterable[Mapping[str, object]]] | None = None,
    device: DeviceFingerprint | Mapping[str, object] | None = None,
    regime: str = "hot",
    tags: Mapping[str, object] | None = None,
) -> DiscreteKernelAdapter[object]:
    from ..fp8 import DEFAULT_MXFP8_PREQUANT_CONFIG
    from ..prequant_autotune import (
        KERNEL_REVISION,
        PREQUANT_SEARCH_SPACE,
        prequant_config_from_dict,
        prequant_config_id,
        prequant_config_to_dict,
        update_prequant_config,
    )

    initial_config = DEFAULT_MXFP8_PREQUANT_CONFIG if initial is None else initial
    selected_axes = PREQUANT_SEARCH_SPACE if axes is None else axes
    axis_values = {name: tuple(values) for name, values in selected_axes.items()}
    unknown = set(axis_values).difference(PREQUANT_SEARCH_SPACE)
    if unknown:
        raise ValueError(f"unknown prequant-forward tuning axes: {sorted(unknown)}")

    def rejection(config: object) -> tuple[str, str] | None:
        reason = config.rejection(problem)  # type: ignore[attr-defined]
        if reason is None:
            gemm = config.gemm  # type: ignore[attr-defined]
            reason = _gemm_smem_rejection(gemm, device)
        return None if reason is None else ("implementation_rejected", reason)

    def derived(config: object) -> Mapping[str, float]:
        gemm = config.gemm  # type: ignore[attr-defined]
        quant_x = config.quant  # type: ignore[attr-defined]
        quant_w = config.resolved_weight_quant()  # type: ignore[attr-defined]
        values = _gemm_features(
            problem, gemm, device, materialized_quant=True
        )
        values.update(_prefix(_quant_features(problem.m, problem.k, quant_x, device), "quant_x_"))
        values.update(_prefix(_quant_features(problem.n, problem.k, quant_w, device), "quant_w_"))
        values["quant_launch_count"] = 1.0 if config.quant_launches == "dual" else 2.0  # type: ignore[attr-defined]
        values["total_kernel_launches"] = values["quant_launch_count"] + 1.0
        return values

    return DiscreteKernelAdapter(
        context=_context(
            "mxfp8_prequant_fwd", KERNEL_REVISION, problem, device, regime, tags
        ),
        initial_config=initial_config,
        axes=axis_values,
        config_id_fn=prequant_config_id,
        serialize_fn=prequant_config_to_dict,
        deserialize_fn=prequant_config_from_dict,
        update_fn=lambda config, _coordinate, value: update_prequant_config(config, value),
        evaluator=evaluator,
        rejection_fn=rejection,
        extra_features_fn=derived,
    )


def make_mxfp8_weight_prequant_adapter(
    problem: MXFP8Problem,
    evaluator: Callable[[object], TrialOutcome],
    *,
    initial: object | None = None,
    axes: Mapping[str, Iterable[Mapping[str, object]]] | None = None,
    device: DeviceFingerprint | Mapping[str, object] | None = None,
    regime: str = "hot",
    tags: Mapping[str, object] | None = None,
) -> DiscreteKernelAdapter[object]:
    """Tune only work executed per call for BF16 X and an AOT-packed W."""

    from ..configs import MXFP8WeightPrequantConfig
    from ..inference_autotune import (
        INFERENCE_KERNEL_REVISION,
        MXFP8_WEIGHT_PREQUANT_SEARCH_SPACE,
        update_weight_prequant_config,
        weight_prequant_config_from_dict,
        weight_prequant_config_id,
        weight_prequant_config_to_dict,
    )

    initial_config = MXFP8WeightPrequantConfig() if initial is None else initial
    selected_axes = MXFP8_WEIGHT_PREQUANT_SEARCH_SPACE if axes is None else axes
    axis_values = {name: tuple(values) for name, values in selected_axes.items()}
    unknown = set(axis_values).difference(MXFP8_WEIGHT_PREQUANT_SEARCH_SPACE)
    if unknown:
        raise ValueError(f"unknown AOT-weight tuning axes: {sorted(unknown)}")

    def rejection(config: object) -> tuple[str, str] | None:
        reason = config.rejection(problem)  # type: ignore[attr-defined]
        if reason is None:
            gemm = config.gemm  # type: ignore[attr-defined]
            reason = _gemm_smem_rejection(gemm, device)
        return None if reason is None else ("implementation_rejected", reason)

    def derived(config: object) -> Mapping[str, float]:
        gemm = config.gemm  # type: ignore[attr-defined]
        quant_x = config.quant_x  # type: ignore[attr-defined]
        values = _gemm_features(problem, gemm, device, materialized_quant=True)
        values.update(
            _prefix(
                _quant_features(problem.m, problem.k, quant_x, device),
                "quant_x_",
            )
        )
        x_bf16 = 2 * problem.m * problem.k
        qx = problem.m * problem.k + problem.m * (problem.k // 32)
        qw = problem.n * problem.k + problem.n * (problem.k // 32)
        out = 2 * problem.m * problem.n
        values.update(
            operand_state_weight_prequantized=1.0,
            quant_launch_count=1.0,
            total_kernel_launches=2.0,
            untimed_weight_packing=1.0,
            estimated_total_memory_bytes=float(x_bf16 + qx + qx + qw + out),
            quantized_materialization_bytes=float(qx),
        )
        return values

    state_tags = {**dict(tags or {}), "operand_state": "weight_prequantized"}
    return DiscreteKernelAdapter(
        context=_context(
            "mxfp8_weight_prequant_fwd",
            INFERENCE_KERNEL_REVISION,
            problem,
            device,
            regime,
            state_tags,
        ),
        initial_config=initial_config,
        axes=axis_values,
        config_id_fn=weight_prequant_config_id,
        serialize_fn=weight_prequant_config_to_dict,
        deserialize_fn=weight_prequant_config_from_dict,
        update_fn=lambda config, _coordinate, value: update_weight_prequant_config(
            config, value
        ),
        evaluator=evaluator,
        rejection_fn=rejection,
        extra_features_fn=derived,
    )


def make_mxfp8_fully_prequant_adapter(
    problem: MXFP8Problem,
    evaluator: Callable[[object], TrialOutcome],
    *,
    initial: object | None = None,
    axes: Mapping[str, Iterable[Mapping[str, object]]] | None = None,
    device: DeviceFingerprint | Mapping[str, object] | None = None,
    regime: str = "hot",
    tags: Mapping[str, object] | None = None,
) -> DiscreteKernelAdapter[object]:
    """Tune GEMM-only execution when X and W are both already packed."""

    from ..configs import MXFP8FullyPrequantConfig
    from ..inference_autotune import (
        INFERENCE_KERNEL_REVISION,
        MXFP8_FULLY_PREQUANT_SEARCH_SPACE,
        fully_prequant_config_from_dict,
        fully_prequant_config_id,
        fully_prequant_config_to_dict,
        update_fully_prequant_config,
    )

    initial_config = MXFP8FullyPrequantConfig() if initial is None else initial
    selected_axes = MXFP8_FULLY_PREQUANT_SEARCH_SPACE if axes is None else axes
    axis_values = {name: tuple(values) for name, values in selected_axes.items()}
    unknown = set(axis_values).difference(MXFP8_FULLY_PREQUANT_SEARCH_SPACE)
    if unknown:
        raise ValueError(f"unknown fully-packed tuning axes: {sorted(unknown)}")

    def rejection(config: object) -> tuple[str, str] | None:
        reason = config.rejection(problem)  # type: ignore[attr-defined]
        if reason is None:
            gemm = config.gemm  # type: ignore[attr-defined]
            reason = _gemm_smem_rejection(gemm, device)
        return None if reason is None else ("implementation_rejected", reason)

    def derived(config: object) -> Mapping[str, float]:
        gemm = config.gemm  # type: ignore[attr-defined]
        values = _gemm_features(problem, gemm, device, materialized_quant=True)
        qx = problem.m * problem.k + problem.m * (problem.k // 32)
        qw = problem.n * problem.k + problem.n * (problem.k // 32)
        out = 2 * problem.m * problem.n
        values.update(
            operand_state_fully_prequantized=1.0,
            quant_launch_count=0.0,
            total_kernel_launches=1.0,
            untimed_activation_packing=1.0,
            untimed_weight_packing=1.0,
            estimated_total_memory_bytes=float(qx + qw + out),
            quantized_materialization_bytes=0.0,
        )
        return values

    state_tags = {**dict(tags or {}), "operand_state": "fully_prequantized"}
    return DiscreteKernelAdapter(
        context=_context(
            "mxfp8_fully_prequant_fwd",
            INFERENCE_KERNEL_REVISION,
            problem,
            device,
            regime,
            state_tags,
        ),
        initial_config=initial_config,
        axes=axis_values,
        config_id_fn=fully_prequant_config_id,
        serialize_fn=fully_prequant_config_to_dict,
        deserialize_fn=fully_prequant_config_from_dict,
        update_fn=lambda config, _coordinate, value: update_fully_prequant_config(
            config, value
        ),
        evaluator=evaluator,
        rejection_fn=rejection,
        extra_features_fn=derived,
    )


def make_nvfp4_weight_prequant_adapter(
    problem,
    evaluator: Callable[[object], TrialOutcome],
    *,
    initial: object | None = None,
    axes: Mapping[str, Iterable[Mapping[str, object]]] | None = None,
    device: DeviceFingerprint | Mapping[str, object] | None = None,
    regime: str = "hot",
    tags: Mapping[str, object] | None = None,
) -> DiscreteKernelAdapter[object]:
    """Tune BF16-X quantization plus GEMM with an AOT NVFP4 weight."""

    from ..configs.nvfp4 import NVFP4WeightPrequantConfig
    from ..nvfp4_inference_autotune import (
        NVFP4_INFERENCE_KERNEL_REVISION,
        NVFP4_WEIGHT_PREQUANT_SEARCH_SPACE,
        update_weight_prequant_config,
        weight_prequant_config_from_dict,
        weight_prequant_config_id,
        weight_prequant_config_to_dict,
    )

    initial_config = NVFP4WeightPrequantConfig() if initial is None else initial
    selected_axes = NVFP4_WEIGHT_PREQUANT_SEARCH_SPACE if axes is None else axes
    axis_values = {name: tuple(values) for name, values in selected_axes.items()}
    unknown = set(axis_values).difference(NVFP4_WEIGHT_PREQUANT_SEARCH_SPACE)
    if unknown:
        raise ValueError(f"unknown NVFP4 AOT-weight tuning axes: {sorted(unknown)}")

    def rejection(config: object) -> tuple[str, str] | None:
        reason = config.rejection(problem)  # type: ignore[attr-defined]
        if reason is None:
            reason = _gemm_smem_rejection(config.gemm, device)  # type: ignore[attr-defined]
        return None if reason is None else ("implementation_rejected", reason)

    def derived(config: object) -> Mapping[str, float]:
        gemm = config.gemm  # type: ignore[attr-defined]
        quant_x = config.quant_x  # type: ignore[attr-defined]
        values = _gemm_features(problem, gemm, device, materialized_quant=True)
        values.update(
            _prefix(
                _nvfp4_quant_features(problem.m, problem.k, quant_x, device),
                "quant_x_",
            )
        )
        x_bf16 = 2 * problem.m * problem.k
        qx = problem.m * problem.k / 2 + problem.m * (problem.k // 16)
        qw = problem.n * problem.k / 2 + problem.n * (problem.k // 16)
        out = 2 * problem.m * problem.n
        values.update(
            operand_state_weight_prequantized=1.0,
            quant_launch_count=1.0,
            total_kernel_launches=2.0,
            untimed_weight_packing=1.0,
            estimated_total_memory_bytes=float(x_bf16 + 2 * qx + qw + out),
            quantized_materialization_bytes=float(qx),
        )
        return values

    state_tags = {**dict(tags or {}), "operand_state": "weight_prequantized"}
    return DiscreteKernelAdapter(
        context=_context(
            "nvfp4_weight_prequant_fwd",
            NVFP4_INFERENCE_KERNEL_REVISION,
            problem,
            device,
            regime,
            state_tags,
        ),
        initial_config=initial_config,
        axes=axis_values,
        config_id_fn=weight_prequant_config_id,
        serialize_fn=weight_prequant_config_to_dict,
        deserialize_fn=weight_prequant_config_from_dict,
        update_fn=lambda config, _coordinate, value: update_weight_prequant_config(
            config, value
        ),
        evaluator=evaluator,
        rejection_fn=rejection,
        extra_features_fn=derived,
    )


def make_nvfp4_dynamic_adapter(
    problem,
    evaluator: Callable[[object], TrialOutcome],
    *,
    initial: object | None = None,
    axes: Mapping[str, Iterable[Mapping[str, object]]] | None = None,
    device: DeviceFingerprint | Mapping[str, object] | None = None,
    regime: str = "hot",
    tags: Mapping[str, object] | None = None,
    _family: str = "nvfp4_dynamic_fwd",
    _revision: int | None = None,
) -> DiscreteKernelAdapter[object]:
    """Jointly tune both dynamic quantizers and the materialized NVFP4 GEMM."""

    from ..configs.nvfp4 import NVFP4DynamicConfig
    from ..nvfp4_inference_autotune import (
        NVFP4_DYNAMIC_KERNEL_REVISION,
        NVFP4_DYNAMIC_SEARCH_SPACE,
        dynamic_config_from_dict,
        dynamic_config_id,
        dynamic_config_to_dict,
        preferred_dynamic_config,
        update_dynamic_config,
    )

    initial_config = preferred_dynamic_config(problem) if initial is None else initial
    revision = NVFP4_DYNAMIC_KERNEL_REVISION if _revision is None else _revision
    selected_axes = NVFP4_DYNAMIC_SEARCH_SPACE if axes is None else axes
    axis_values = {name: tuple(values) for name, values in selected_axes.items()}
    unknown = set(axis_values).difference(NVFP4_DYNAMIC_SEARCH_SPACE)
    if unknown:
        raise ValueError(f"unknown dynamic NVFP4 tuning axes: {sorted(unknown)}")

    def rejection(config: object) -> tuple[str, str] | None:
        reason = config.rejection(problem)  # type: ignore[attr-defined]
        if reason is None:
            reason = _gemm_smem_rejection(config.gemm, device)  # type: ignore[attr-defined]
        return None if reason is None else ("implementation_rejected", reason)

    def derived(config: object) -> Mapping[str, float]:
        gemm = config.gemm  # type: ignore[attr-defined]
        quant = config.quant  # type: ignore[attr-defined]
        values = _gemm_features(problem, gemm, device, materialized_quant=True)
        values.update(
            _prefix(
                _nvfp4_quant_features(problem.m, problem.k, quant, device),
                "quant_x_",
            )
        )
        values.update(
            _prefix(
                _nvfp4_quant_features(problem.n, problem.k, quant, device),
                "quant_w_",
            )
        )
        qx = problem.m * problem.k / 2 + problem.m * (problem.k // 16)
        qw = problem.n * problem.k / 2 + problem.n * (problem.k // 16)
        values.update(
            operand_state_dynamic=1.0,
            native_scale_transport=(
                1.0 if quant.scale_layout == "mma128" else 0.0
            ),
            quant_launch_count=(
                1.0 if config.quant_launches == "dual" else 2.0  # type: ignore[attr-defined]
            ),
            quant_launch_concurrency=(
                2.0 if config.quant_launches == "concurrent" else 1.0  # type: ignore[attr-defined]
            ),
            total_kernel_launches=(
                2.0 if config.quant_launches == "dual" else 3.0  # type: ignore[attr-defined]
            ),
            quantized_materialization_bytes=float(qx + qw),
        )
        return values

    state_tags = {
        **dict(tags or {}),
        "operand_state": (
            "delayed_materialized" if _family == "nvfp4_delayed_fwd" else "dynamic"
        ),
    }
    return DiscreteKernelAdapter(
        context=_context(
            _family,
            revision,
            problem,
            device,
            regime,
            state_tags,
        ),
        initial_config=initial_config,
        axes=axis_values,
        config_id_fn=dynamic_config_id,
        serialize_fn=dynamic_config_to_dict,
        deserialize_fn=dynamic_config_from_dict,
        update_fn=lambda config, _coordinate, value: update_dynamic_config(
            config, value
        ),
        evaluator=evaluator,
        rejection_fn=rejection,
        extra_features_fn=derived,
    )


def make_nvfp4_delayed_adapter(
    problem,
    evaluator: Callable[[object], TrialOutcome],
    *,
    initial: object | None = None,
    axes: Mapping[str, Iterable[Mapping[str, object]]] | None = None,
    device: DeviceFingerprint | Mapping[str, object] | None = None,
    regime: str = "hot",
    tags: Mapping[str, object] | None = None,
) -> DiscreteKernelAdapter[object]:
    """Tune the one-launch delayed observer/quantizer plus native GEMM."""

    from ..nvfp4_inference_autotune import (
        NVFP4_DELAYED_KERNEL_REVISION,
        NVFP4_DELAYED_SEARCH_SPACE,
        preferred_dynamic_config,
    )

    selected_initial = preferred_dynamic_config(problem) if initial is None else initial
    selected_initial = replace(selected_initial, quant_launches="dual")
    return make_nvfp4_dynamic_adapter(
        problem,
        evaluator,
        initial=selected_initial,
        axes=NVFP4_DELAYED_SEARCH_SPACE if axes is None else axes,
        device=device,
        regime=regime,
        tags=tags,
        _family="nvfp4_delayed_fwd",
        _revision=NVFP4_DELAYED_KERNEL_REVISION,
    )


def make_nvfp4_fully_prequant_adapter(
    problem,
    evaluator: Callable[[object], TrialOutcome],
    *,
    initial: object | None = None,
    axes: Mapping[str, Iterable[Mapping[str, object]]] | None = None,
    device: DeviceFingerprint | Mapping[str, object] | None = None,
    regime: str = "hot",
    tags: Mapping[str, object] | None = None,
) -> DiscreteKernelAdapter[object]:
    """Tune GEMM-only execution for two TorchAO-packed NVFP4 operands."""

    from ..configs.nvfp4 import NVFP4FullyPrequantConfig
    from ..nvfp4_inference_autotune import (
        NVFP4_FULLY_PREQUANT_SEARCH_SPACE,
        NVFP4_INFERENCE_KERNEL_REVISION,
        fully_prequant_config_from_dict,
        fully_prequant_config_id,
        fully_prequant_config_to_dict,
        update_fully_prequant_config,
    )

    initial_config = NVFP4FullyPrequantConfig() if initial is None else initial
    selected_axes = NVFP4_FULLY_PREQUANT_SEARCH_SPACE if axes is None else axes
    axis_values = {name: tuple(values) for name, values in selected_axes.items()}
    unknown = set(axis_values).difference(NVFP4_FULLY_PREQUANT_SEARCH_SPACE)
    if unknown:
        raise ValueError(f"unknown NVFP4 fully-packed tuning axes: {sorted(unknown)}")

    def rejection(config: object) -> tuple[str, str] | None:
        reason = config.rejection(problem)  # type: ignore[attr-defined]
        if reason is None:
            reason = _gemm_smem_rejection(config.gemm, device)  # type: ignore[attr-defined]
        return None if reason is None else ("implementation_rejected", reason)

    def derived(config: object) -> Mapping[str, float]:
        gemm = config.gemm  # type: ignore[attr-defined]
        values = _gemm_features(problem, gemm, device, materialized_quant=True)
        qx = problem.m * problem.k / 2 + problem.m * (problem.k // 16)
        qw = problem.n * problem.k / 2 + problem.n * (problem.k // 16)
        out = 2 * problem.m * problem.n
        values.update(
            operand_state_fully_prequantized=1.0,
            quant_launch_count=0.0,
            total_kernel_launches=1.0,
            untimed_activation_packing=1.0,
            untimed_weight_packing=1.0,
            estimated_total_memory_bytes=float(qx + qw + out),
            quantized_materialization_bytes=0.0,
        )
        return values

    state_tags = {**dict(tags or {}), "operand_state": "fully_prequantized"}
    return DiscreteKernelAdapter(
        context=_context(
            "nvfp4_fully_prequant_fwd",
            NVFP4_INFERENCE_KERNEL_REVISION,
            problem,
            device,
            regime,
            state_tags,
        ),
        initial_config=initial_config,
        axes=axis_values,
        config_id_fn=fully_prequant_config_id,
        serialize_fn=fully_prequant_config_to_dict,
        deserialize_fn=fully_prequant_config_from_dict,
        update_fn=lambda config, _coordinate, value: update_fully_prequant_config(
            config, value
        ),
        evaluator=evaluator,
        rejection_fn=rejection,
        extra_features_fn=derived,
    )


def make_mxfp8_bwd_adapter(
    problem: MXFP8Problem,
    evaluator: Callable[[object], TrialOutcome],
    *,
    initial: object | None = None,
    axes: Mapping[str, Iterable[Mapping[str, object]]] | None = None,
    device: DeviceFingerprint | Mapping[str, object] | None = None,
    regime: str = "hot",
    tags: Mapping[str, object] | None = None,
) -> DiscreteKernelAdapter[object]:
    from ..bwd_autotune import (
        BWD_SEARCH_SPACE,
        KERNEL_REVISION,
        bwd_config_from_dict,
        bwd_config_id,
        bwd_config_to_dict,
        update_bwd_config,
    )
    from ..kernels.mxfp8_bwd import DEFAULT_MXFP8_BWD_CONFIG

    initial_config = DEFAULT_MXFP8_BWD_CONFIG if initial is None else initial
    selected_axes = BWD_SEARCH_SPACE if axes is None else axes
    axis_values = {name: tuple(values) for name, values in selected_axes.items()}
    unknown = set(axis_values).difference(BWD_SEARCH_SPACE)
    if unknown:
        raise ValueError(f"unknown backward tuning axes: {sorted(unknown)}")

    # A random walk usually changes only dX or dW. Cache the unchanged half's
    # derived geometry/traffic vector so a large learned-search pool does not
    # recompute both matmuls for every candidate.
    matmul_feature_cache: dict[tuple[str, object], dict[str, float]] = {}
    matmul_feature_cache_limit = 4096

    def remember_matmul_features(
        key: tuple[str, object],
        values: Mapping[str, float],
    ) -> None:
        if len(matmul_feature_cache) >= matmul_feature_cache_limit:
            # Clearing is deterministic and bounds what can otherwise become a
            # multi-hour campaign cache. Hot random-walk components repopulate
            # immediately.
            matmul_feature_cache.clear()
        matmul_feature_cache[key] = dict(values)

    def update(config: object, coordinate: str, value: object) -> object:
        current = config
        parts = coordinate.split("_")
        if len(parts) >= 3 and parts[0] in ("dx", "dw") and parts[1] == "b":
            matmul = current.dx if parts[0] == "dx" else current.dw  # type: ignore[attr-defined]
            if (
                matmul.quant_launches == "dual"
                and matmul.a_orientation == matmul.b_orientation
            ):
                current = update_bwd_config(
                    current, {parts[0]: {"quant_launches": "separate"}}
                )
        return update_bwd_config(current, value)  # type: ignore[arg-type]

    def rejection(config: object) -> tuple[str, str] | None:
        reason = config.implementation_rejection(problem)  # type: ignore[attr-defined]
        if reason is None:
            for name, matmul in (("dX", config.dx), ("dW", config.dw)):  # type: ignore[attr-defined]
                if matmul.backend == "fused":
                    profile = _device_dict(device)
                    smem_limit = int(
                        profile_value(profile, "shared_memory_per_block_optin", 0)
                        or profile_value(profile, "shared_memory_per_block", 0)
                        or 0
                    )
                    required = _fused_smem_bytes(matmul.fused)
                    reason = (
                        f"fused kernel requires {required} bytes of CTA SMEM, "
                        f"device limit is {smem_limit}"
                        if smem_limit and required > smem_limit
                        else None
                    )
                else:
                    reason = _gemm_smem_rejection(matmul.gemm, device)
                if reason is not None:
                    reason = f"{name}: {reason}"
                    break
        return None if reason is None else ("implementation_rejected", reason)

    def derived(config: object) -> Mapping[str, float]:
        dx_matmul = config.dx  # type: ignore[attr-defined]
        dw_matmul = config.dw  # type: ignore[attr-defined]
        dx_problem = MXFP8Problem(problem.m, problem.k, problem.n)
        dw_problem = MXFP8Problem(problem.n, problem.k, problem.m)
        values: dict[str, float] = {}
        for name, matmul, matmul_problem in (
            ("dx", dx_matmul, dx_problem),
            ("dw", dw_matmul, dw_problem),
        ):
            cache_key = (name, matmul)
            cached = matmul_feature_cache.get(cache_key)
            if cached is not None:
                values.update(cached)
                continue
            keys_before = set(values)
            if matmul.backend == "fused":
                fused = matmul.fused
                profile = _device_dict(device)
                natural_output_ctas = (
                    (matmul_problem.m + fused.tile_m - 1) // fused.tile_m
                ) * ((matmul_problem.n + fused.tile_n - 1) // fused.tile_n)
                natural_ctas = natural_output_ctas * (
                    matmul.split_reduction
                    if matmul.reduction != "full_fp32"
                    else 1
                )
                grid_ctas = natural_ctas
                if fused.persistent:
                    sm_count = max(
                        1,
                        int(profile_value(profile, "multiprocessor_count", 1) or 1),
                    )
                    if matmul.reduction != "full_fp32":
                        per_split_grid = min(
                            natural_output_ctas,
                            max(
                                1,
                                sm_count
                                * fused.persistent_waves
                                // matmul.split_reduction,
                            ),
                        )
                        while (
                            per_split_grid > 1
                            and natural_output_ctas % per_split_grid
                        ):
                            per_split_grid -= 1
                        grid_ctas = matmul.split_reduction * per_split_grid
                    else:
                        grid_ctas = min(
                            natural_ctas, sm_count * fused.persistent_waves
                        )
                        while grid_ctas > 1 and natural_ctas % grid_ctas:
                            grid_ctas -= 1
                matmul_values = geometry_features(
                    m=matmul_problem.m,
                    n=matmul_problem.n,
                    k=matmul_problem.k,
                    tile_m=fused.tile_m,
                    tile_n=fused.tile_n,
                    tile_k=fused.tile_k,
                    profile=profile,
                    grid_ctas=grid_ctas,
                )
                matmul_values.update(
                    traffic_features(
                        m=matmul_problem.m,
                        n=matmul_problem.n,
                        k=matmul_problem.k,
                        tile_m=fused.tile_m,
                        tile_n=fused.tile_n,
                        input_element_bytes=2,
                        output_element_bytes=2,
                        profile=profile,
                        materialized_quant=False,
                    )
                )
                matmul_values.update(
                    backend_fused=1.0,
                    total_kernel_launches=(
                        2.0
                        if matmul.reduction == "split_fp32_workspace"
                        else (
                            3.0
                            if matmul.reduction == "split_fp32_atomic"
                            else 1.0
                        )
                    ),
                    quantized_materialization_bytes=0.0,
                    split_work_ctas=float(natural_ctas),
                    work_tiles_per_cta=natural_ctas / max(1, grid_ctas),
                    pipeline_buffer_bytes=float(_fused_smem_bytes(fused)),
                )
                _apply_fused_cluster_reuse_features(
                    matmul_values,
                    problem=matmul_problem,
                    config=fused,
                    natural_output_ctas=natural_output_ctas,
                    profile=profile,
                )
                values.update(_prefix(matmul_values, f"{name}_"))
                values[f"{name}_split_reduction"] = float(
                    matmul.split_reduction
                )
                values[f"{name}_reduction_tile"] = float(matmul.reduction_tile)
                values[f"{name}_workspace_fp32_bytes"] = float(
                    matmul_problem.m
                    * matmul_problem.n
                    * (
                        matmul.split_reduction
                        if matmul.reduction == "split_fp32_workspace"
                        else (1 if matmul.reduction == "split_fp32_atomic" else 0)
                    )
                    * 4
                )
                values[f"{name}_cluster_size"] = float(
                    matmul.split_reduction
                    if matmul.reduction == "cluster_fp32"
                    else 1
                )
                values[f"{name}_cluster_dsmem_reduction"] = float(
                    matmul.reduction == "cluster_fp32"
                )
                if matmul.reduction == "cluster_fp32":
                    mma_threads = fused.num_mma_warps * 32
                    accum_per_thread = (
                        fused.tile_m * fused.tile_n // mma_threads
                    )
                    scratch_per_thread = (
                        fused.tile_m
                        * fused.tile_k
                        * fused.mxfp8_stages
                        // 4
                        // mma_threads
                    )
                    chunk_elems = max(
                        1, min(accum_per_thread, scratch_per_thread)
                    )
                    chunks = (
                        accum_per_thread + chunk_elems - 1
                    ) // chunk_elems
                    values[f"{name}_cluster_reduction_chunks"] = float(chunks)
                    values[f"{name}_cluster_barrier_phases"] = float(
                        3 * chunks
                    )
                    values[f"{name}_cluster_dsmem_atomic_bytes"] = float(
                        matmul_problem.m
                        * matmul_problem.n
                        * matmul.split_reduction
                        * 4
                    )
                else:
                    values[f"{name}_cluster_reduction_chunks"] = 0.0
                    values[f"{name}_cluster_barrier_phases"] = 0.0
                    values[f"{name}_cluster_dsmem_atomic_bytes"] = 0.0
                values[f"{name}_reduction_threads"] = float(
                    matmul.reduction_threads
                )
                values[f"{name}_reduction_vector"] = float(
                    matmul.reduction_vector
                )
                values[f"{name}_reduction_waves"] = float(
                    matmul.reduction_waves
                )
                values[f"{name}_persistent_split"] = float(
                    fused.persistent and matmul.reduction != "full_fp32"
                )
                values[f"{name}_persistent_split_grid_ctas_per_slice"] = float(
                    grid_ctas / matmul.split_reduction
                    if fused.persistent and matmul.reduction != "full_fp32"
                    else 0
                )
                values[f"{name}_persistent_split_pipeline_tail_count"] = float(
                    1 if fused.persistent and matmul.reduction != "full_fp32" else 0
                )
                values[f"{name}_persistent_split_tiles_per_pipeline_tail"] = float(
                    natural_ctas / max(1, grid_ctas)
                    if fused.persistent and matmul.reduction != "full_fp32"
                    else 0
                )
                transpose_operands = int(matmul.a_orientation == "transpose") + int(
                    matmul.b_orientation == "transpose"
                )
                values[f"{name}_logical_transpose_operands"] = float(
                    transpose_operands
                )
                values[f"{name}_oriented_cpasync"] = float(
                    fused.load_engine == "cpasync" and transpose_operands > 0
                )
                values[f"{name}_oriented_cpasync_ldmatrix_operands"] = float(
                    transpose_operands
                    if fused.load_engine == "cpasync"
                    and fused.quant_load_bits == 128
                    else 0
                )
                remember_matmul_features(
                    cache_key,
                    {
                        key: value
                        for key, value in values.items()
                        if key not in keys_before
                    },
                )
                continue

            values.update(
                _prefix(
                    _gemm_features(
                        matmul_problem,
                        matmul.gemm,
                        device,
                        materialized_quant=True,
                        split_reduction=matmul.split_reduction,
                    ),
                    f"{name}_",
                )
            )
            quant_b = matmul.resolved_quant_b()
            values.update(
                _prefix(
                    _quant_features(
                        matmul_problem.m,
                        matmul_problem.k,
                        matmul.quant_a,
                        device,
                        transposed=matmul.a_orientation == "transpose",
                    ),
                    f"{name}_quant_a_",
                )
            )
            values.update(
                _prefix(
                    _quant_features(
                        matmul_problem.n,
                        matmul_problem.k,
                        quant_b,
                        device,
                        transposed=matmul.b_orientation == "transpose",
                    ),
                    f"{name}_quant_b_",
                )
            )
            values[f"{name}_quant_launch_count"] = (
                1.0 if matmul.quant_launches == "dual" else 2.0
            )
            values[f"{name}_backend_fused"] = 0.0
            values[f"{name}_total_kernel_launches"] = (
                values[f"{name}_quant_launch_count"]
                + (
                    2.0
                    if matmul.reduction == "split_fp32_workspace"
                    else (
                        3.0
                        if matmul.reduction == "split_fp32_atomic"
                        else 1.0
                    )
                )
            )
            values[f"{name}_split_reduction"] = float(matmul.split_reduction)
            values[f"{name}_reduction_tile"] = float(matmul.reduction_tile)
            values[f"{name}_workspace_fp32_bytes"] = float(
                0
                if matmul.reduction in ("full_fp32", "cluster_fp32")
                else matmul_problem.m
                * matmul_problem.n
                * (
                    matmul.split_reduction
                    if matmul.reduction == "split_fp32_workspace"
                    else 1
                )
                * 4
            )
            values[f"{name}_cluster_size"] = float(
                matmul.split_reduction
                if matmul.reduction == "cluster_fp32"
                else 1
            )
            values[f"{name}_cluster_dsmem_reduction"] = float(
                matmul.reduction == "cluster_fp32"
            )
            if matmul.reduction == "cluster_fp32":
                mma_threads = matmul.gemm.num_mma_warps * 32
                accum_per_thread = (
                    matmul.gemm.tile_m * matmul.gemm.tile_n // mma_threads
                )
                scratch_per_thread = (
                    matmul.gemm.tile_m
                    * matmul.gemm.tile_k
                    * matmul.gemm.stages
                    // 4
                    // mma_threads
                )
                chunk_elems = max(
                    1, min(accum_per_thread, scratch_per_thread)
                )
                chunks = (
                    accum_per_thread + chunk_elems - 1
                ) // chunk_elems
                values[f"{name}_cluster_reduction_chunks"] = float(chunks)
                values[f"{name}_cluster_barrier_phases"] = float(3 * chunks)
                values[f"{name}_cluster_dsmem_atomic_bytes"] = float(
                    matmul_problem.m
                    * matmul_problem.n
                    * matmul.split_reduction
                    * 4
                )
            else:
                values[f"{name}_cluster_reduction_chunks"] = 0.0
                values[f"{name}_cluster_barrier_phases"] = 0.0
                values[f"{name}_cluster_dsmem_atomic_bytes"] = 0.0
            values[f"{name}_reduction_threads"] = float(
                matmul.reduction_threads
            )
            values[f"{name}_reduction_vector"] = float(
                matmul.reduction_vector
            )
            values[f"{name}_reduction_waves"] = float(
                matmul.reduction_waves
            )
            values[f"{name}_persistent_split"] = 0.0
            values[f"{name}_persistent_split_grid_ctas_per_slice"] = 0.0
            values[f"{name}_persistent_split_pipeline_tail_count"] = 0.0
            values[f"{name}_persistent_split_tiles_per_pipeline_tail"] = 0.0
            values[f"{name}_logical_transpose_operands"] = float(
                int(matmul.a_orientation == "transpose")
                + int(matmul.b_orientation == "transpose")
            )
            values[f"{name}_oriented_cpasync"] = 0.0
            values[f"{name}_oriented_cpasync_ldmatrix_operands"] = 0.0
            remember_matmul_features(
                cache_key,
                {
                    key: value
                    for key, value in values.items()
                    if key not in keys_before
                },
            )
        quant_schedule = config.quant_schedule  # type: ignore[attr-defined]
        is_quad = quant_schedule in ("quad", "shared_g_quad")
        is_shared_g = quant_schedule == "shared_g_quad"
        values["backward_quad_quant"] = float(is_quad)
        values["backward_shared_g_quant"] = float(is_shared_g)
        values["backward_standalone_quant_launches"] = float(
            1
            if is_quad
            else sum(
                0
                if matmul.backend == "fused"
                else (1 if matmul.quant_launches == "dual" else 2)
                for matmul in (dx_matmul, dw_matmul)
            )
        )
        values["backward_total_kernel_launches"] = float(
            values["dx_total_kernel_launches"]
            + values["dw_total_kernel_launches"]
            - (1 if is_quad else 0)
        )
        values["backward_grad_output_bf16_read_bytes"] = float(
            problem.m * problem.n * 2 * (1 if is_shared_g else 2)
        )
        values["backward_shared_g_bf16_bytes_saved"] = float(
            problem.m * problem.n * 2 if is_shared_g else 0
        )
        shared_tile = (
            dx_matmul.quant_a.transposed_tile_rows if is_shared_g else 0
        )
        values["backward_shared_g_tile"] = float(shared_tile)
        values["backward_shared_g_tile_bytes"] = float(
            shared_tile * shared_tile * 2
        )
        values["backward_shared_g_tile_count"] = float(
            (problem.m // shared_tile) * (problem.n // shared_tile)
            if shared_tile
            else 0
        )
        values["dw_reduction_length"] = problem.m
        values["dx_reduction_length"] = problem.n
        values["combined_nominal_flops"] = float(
            2 * problem.m * problem.k * problem.n
            + 2 * problem.n * problem.k * problem.m
        )
        return values

    return DiscreteKernelAdapter(
        context=_context("mxfp8_bwd", KERNEL_REVISION, problem, device, regime, tags),
        initial_config=initial_config,
        axes=axis_values,
        config_id_fn=bwd_config_id,
        serialize_fn=bwd_config_to_dict,
        deserialize_fn=bwd_config_from_dict,
        update_fn=update,
        evaluator=evaluator,
        rejection_fn=rejection,
        extra_features_fn=derived,
    )


__all__ = [
    "make_mxfp8_bwd_adapter",
    "make_mxfp8_fully_prequant_adapter",
    "make_mxfp8_fwd_adapter",
    "make_nvfp4_fwd_adapter",
    "make_mxfp8_prequant_adapter",
    "make_mxfp8_weight_prequant_adapter",
    "make_nvfp4_fully_prequant_adapter",
    "make_nvfp4_dynamic_adapter",
    "make_nvfp4_delayed_adapter",
    "make_nvfp4_weight_prequant_adapter",
]
