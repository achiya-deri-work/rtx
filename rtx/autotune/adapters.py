"""Adapters connecting every current MXFP8 tuning family to the shared engine."""

from __future__ import annotations

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
    fwd_config_from_dict,
    fwd_config_id,
    fwd_config_to_dict,
    normalize_fwd_config,
)


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
    ) * config.tile_k
    scales = _sm120_scale_bytes(
        config.tile_m, config.tile_n, config.tile_k, config.mxfp8_stages
    )
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
    return operands + scales + bf16 + epilogue


def _gemm_smem_bytes(config: object) -> int:
    q_bytes = config.stages * (config.tile_m + config.tile_n) * config.tile_k
    scale_bytes = _sm120_scale_bytes(
        config.tile_m, config.tile_n, config.tile_k, config.stages
    )
    out_bytes = config.tile_m * config.tile_n * 2 if config.epilogue == "tma" else 0
    return q_bytes + scale_bytes + out_bytes


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
) -> dict[str, float]:
    profile = _device_dict(device)
    geometry = geometry_features(
        m=problem.m,
        n=problem.n,
        k=problem.k,
        tile_m=config.tile_m,
        tile_n=config.tile_n,
        tile_k=config.tile_k,
        profile=profile,
    )
    grid_ctas = int(geometry["grid_ctas"])
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
            smem_bytes_per_cta=_gemm_smem_bytes(config),
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
        )
    )
    geometry.update(
        tile_flops=float(2 * config.tile_m * config.tile_n * problem.k),
        mma_k_tiles_per_cta=float((problem.k + config.tile_k - 1) // config.tile_k),
        mma_warp_issues_per_k_tile=float(config.num_mma_warps),
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
        task_groups = rows // config.transposed_tile_rows * (k // 32)
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
        smem = 32 * (config.transposed_tile_rows + config.transposed_smem_padding) * 2
    values = {
        "rows": float(rows),
        "task_groups": float(task_groups),
        "natural_ctas": float(natural_ctas),
        "grid_ctas": float(grid_ctas),
        "values_quantized": float(rows * k),
        "scale_blocks": float(rows * (k // 32)),
        "values_per_warp_task": float(config.quant_vec * 32),
        "transposed_source": float(transposed),
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


def _prefix(values: Mapping[str, float], prefix: str) -> dict[str, float]:
    return {f"{prefix}{key}": value for key, value in values.items()}


def make_mxfp8_fwd_adapter(
    problem: MXFP8Problem,
    evaluator: Callable[[MXFP8FwdConfig], TrialOutcome],
    *,
    initial: MXFP8FwdConfig = DEFAULT_MXFP8_FWD_CONFIG,
    axes: Mapping[str, Iterable[object]] = FWD_SEARCH_SPACE,
    device: DeviceFingerprint | Mapping[str, object] | None = None,
    regime: str = "hot",
    tags: Mapping[str, object] | None = None,
) -> DiscreteKernelAdapter[MXFP8FwdConfig]:
    axis_values = {name: tuple(values) for name, values in axes.items()}
    unknown = set(axis_values).difference(FWD_SEARCH_SPACE)
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
        reason = config.implementation_rejection(problem)
        return None if reason is None else ("implementation_rejected", reason)

    def derived(config: MXFP8FwdConfig) -> Mapping[str, float]:
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
        values.update(
            work_tiles_per_cta=natural_ctas / max(1, grid_ctas),
            bf16_values_quantized_per_output_cta=float(
                (config.tile_m + config.tile_n) * problem.k
            ),
            tile_flops=float(2 * config.tile_m * config.tile_n * problem.k),
            mma_k_tiles_per_output_cta=float(
                (problem.k + config.tile_k - 1) // config.tile_k
            ),
            pipeline_buffer_bytes=float(_fused_smem_bytes(config)),
        )
        return values

    return DiscreteKernelAdapter(
        context=_context(
            "mxfp8_fused_fwd",
            MXFP8_FWD_KERNEL_REVISION,
            problem,
            device,
            regime,
            tags,
        ),
        initial_config=initial,
        axes=axis_values,
        config_id_fn=fwd_config_id,
        serialize_fn=fwd_config_to_dict,
        deserialize_fn=lambda value: fwd_config_from_dict(dict(value)),
        update_fn=lambda config, coordinate, value: normalize_fwd_config(
            config, **{coordinate: value}
        ),
        evaluator=evaluator,
        rejection_fn=rejection,
        extra_features_fn=derived,
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
            profile = _device_dict(device)
            smem_limit = int(
                profile_value(profile, "shared_memory_per_block_optin", 0)
                or profile_value(profile, "shared_memory_per_block", 0)
                or 0
            )
            gemm = config.gemm  # type: ignore[attr-defined]
            if smem_limit and _gemm_smem_bytes(gemm) > smem_limit:
                reason = (
                    f"GEMM requires {_gemm_smem_bytes(gemm)} bytes of CTA SMEM, "
                    f"device limit is {smem_limit}"
                )
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
            profile = _device_dict(device)
            smem_limit = int(
                profile_value(profile, "shared_memory_per_block_optin", 0)
                or profile_value(profile, "shared_memory_per_block", 0)
                or 0
            )
            gemm = config.gemm  # type: ignore[attr-defined]
            if smem_limit and _gemm_smem_bytes(gemm) > smem_limit:
                reason = (
                    f"GEMM requires {_gemm_smem_bytes(gemm)} bytes of CTA SMEM, "
                    f"device limit is {smem_limit}"
                )
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
            profile = _device_dict(device)
            smem_limit = int(
                profile_value(profile, "shared_memory_per_block_optin", 0)
                or profile_value(profile, "shared_memory_per_block", 0)
                or 0
            )
            gemm = config.gemm  # type: ignore[attr-defined]
            if smem_limit and _gemm_smem_bytes(gemm) > smem_limit:
                reason = (
                    f"GEMM requires {_gemm_smem_bytes(gemm)} bytes of CTA SMEM, "
                    f"device limit is {smem_limit}"
                )
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
        return None if reason is None else ("implementation_rejected", reason)

    def derived(config: object) -> Mapping[str, float]:
        dx_matmul = config.dx  # type: ignore[attr-defined]
        dw_matmul = config.dw  # type: ignore[attr-defined]
        dx_problem = MXFP8Problem(problem.m, problem.k, problem.n)
        dw_problem = MXFP8Problem(problem.n, problem.k, problem.m)
        values = _prefix(
            _gemm_features(
                dx_problem, dx_matmul.gemm, device, materialized_quant=True
            ),
            "dx_",
        )
        values.update(
            _prefix(
                _gemm_features(
                    dw_problem, dw_matmul.gemm, device, materialized_quant=True
                ),
                "dw_",
            )
        )
        for name, matmul, matmul_problem in (
            ("dx", dx_matmul, dx_problem),
            ("dw", dw_matmul, dw_problem),
        ):
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
            values[f"{name}_split_reduction"] = float(matmul.split_reduction)
            values[f"{name}_reduction_tile"] = float(matmul.reduction_tile)
            values[f"{name}_workspace_fp32_bytes"] = float(
                0
                if matmul.reduction == "full_fp32"
                else matmul_problem.m
                * matmul_problem.n
                * matmul.split_reduction
                * 4
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
    "make_mxfp8_prequant_adapter",
    "make_mxfp8_weight_prequant_adapter",
]
