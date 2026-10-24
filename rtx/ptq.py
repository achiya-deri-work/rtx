"""TorchAO-style model conversion for RTX training and PTQ frontends."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
import math
from numbers import Real
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from .types import AutotuneMode, LinearBackend, NVFP4Backend, NVFP4ScalingMode

if TYPE_CHECKING:
    from .configs import (
        MXFP8QuantConfig,
        NVFP4DynamicConfig,
        NVFP4FullyPrequantConfig,
        NVFP4QuantConfig,
        NVFP4WeightPrequantConfig,
    )
    from .configs import NVFP4ScaleConfig
    from .fp8 import MXFP8PrequantConfig
    from .fp8_bwd import MXFP8BwdConfig
    from .kernels.mxfp8 import MXFP8FwdConfig


ModuleFilter = Callable[[nn.Module, str], bool]
ModuleReference = tuple[nn.Module | None, str | None, nn.Module, str]


@dataclass(frozen=True, slots=True)
class MXFP8TrainingConfig:
    """Policy used when swapping BF16 linears into dynamic MXFP8 training."""

    forward_config: "MXFP8FwdConfig | None" = None
    dynamic_config: "MXFP8PrequantConfig | None" = None
    backward_config: "MXFP8BwdConfig | None" = None
    backend: LinearBackend = "auto"
    autotune: AutotuneMode | bool | None = None
    tuning_policy: object | None = None
    autotune_cache_dir: Path | str | None = None


@dataclass(frozen=True, slots=True)
class NVFP4TrainingConfig:
    """Policy used when swapping BF16 linears into NVFP4/MXFP8 training."""

    scale_config: "NVFP4ScaleConfig | None" = None
    backward_config: "MXFP8BwdConfig | None" = None
    scaling: NVFP4ScalingMode | None = None
    scale_region_rows: int | None = None
    x_scale_region_rows: int | None = None
    weight_scale_region_rows: int | None = None
    backend: NVFP4Backend = "auto"
    autotune: AutotuneMode | bool | None = None
    tuning_policy: object | None = None
    autotune_cache_dir: Path | str | None = None
    dynamic_config: "NVFP4DynamicConfig | None" = None


@dataclass(frozen=True, slots=True)
class MXFP8WeightOnlyConfig:
    """PTQ policy for packed MXFP8 weights and dynamic MXFP8 activations.

    ``weight-only`` describes checkpoint residency: selected BF16 weights are
    quantized once and the BF16 master copy is removed. At execution time a
    BF16 activation is still dynamically quantized to MXFP8 before the GEMM.
    """

    dynamic_config: "MXFP8PrequantConfig | None" = None
    autotune: AutotuneMode | bool | None = None
    tuning_policy: object | None = None
    autotune_cache_dir: Path | str | None = None


@dataclass(frozen=True, slots=True)
class NVFP4CalibrationConfig:
    """Explicit offline weight-scale calibration, never a training policy.

    Local-Hessian requires caller-collected uncentered second moments. A
    nonzero shrinkage blends H toward mean(diag(H))*I; defaults are unchanged.
    Activation quantization remains the selected native packed-weight policy.
    """

    method: Literal["max", "four_over_six", "mse", "local_hessian"] = "mse"
    hessian_shrinkage: float = 0.0
    rows_per_chunk: int = 64

    def __post_init__(self) -> None:
        if self.method not in ("max", "four_over_six", "mse", "local_hessian"):
            raise ValueError(f"unknown NVFP4 calibration method: {self.method!r}")
        alpha = self.hessian_shrinkage
        if (not isinstance(alpha, Real) or isinstance(alpha, bool)
                or not math.isfinite(alpha) or not 0 <= alpha <= 1):
            raise ValueError("hessian_shrinkage must be a finite real number in [0,1]")
        if alpha != 0 and self.method != "local_hessian":
            raise ValueError("nonzero hessian_shrinkage requires local_hessian")
        if (not isinstance(self.rows_per_chunk, int) or isinstance(self.rows_per_chunk, bool)
                or self.rows_per_chunk <= 0):
            raise ValueError("rows_per_chunk must be a positive integer")


@dataclass(frozen=True, slots=True)
class NVFP4WeightOnlyConfig:
    """PTQ policy for packed NVFP4 weights and dynamic BF16 activations.

    ``weight_calibration`` opts into offline CPU scale search. It replaces,
    rather than composes with, ``quant_config`` weight quantization. It never
    changes the activation quantizer or adds work to the runtime forward.
    """

    scaling: Literal["current", "block"] = "current"
    quant_config: "NVFP4QuantConfig | None" = None
    weight_prequant_config: "NVFP4WeightPrequantConfig | None" = None
    fully_prequant_config: "NVFP4FullyPrequantConfig | None" = None
    autotune: AutotuneMode | bool | None = None
    tuning_policy: object | None = None
    autotune_cache_dir: Path | str | None = None
    weight_calibration: NVFP4CalibrationConfig | None = None

    def __post_init__(self) -> None:
        if self.scaling not in ("current", "block"):
            raise ValueError("packed NVFP4 PTQ scaling must be current or block")
        if self.weight_calibration is not None:
            if not isinstance(self.weight_calibration, NVFP4CalibrationConfig):
                raise TypeError("weight_calibration must be NVFP4CalibrationConfig or None")
            if self.quant_config is not None:
                raise ValueError("quant_config conflicts with offline weight_calibration")


def _default_linear_filter(module: nn.Module, fqn: str) -> bool:
    del fqn
    # This intentionally includes TorchAO Float8Linear, which is an nn.Linear
    # subclass retaining the original high-precision Parameter.
    return isinstance(module, nn.Linear)


def _module_references(root: nn.Module) -> list[ModuleReference]:
    """Return every acyclic module alias path, including shared descendants."""

    references: list[ModuleReference] = [(None, None, root, "")]

    def visit(parent: nn.Module, parent_fqn: str, ancestry: frozenset[int]) -> None:
        for name, child in tuple(parent._modules.items()):
            if child is None:
                continue
            fqn = f"{parent_fqn}.{name}" if parent_fqn else name
            references.append((parent, name, child, fqn))
            if id(child) not in ancestry:
                visit(child, fqn, ancestry | {id(child)})

    visit(root, "", frozenset({id(root)}))
    return references


def _selected_linears(
    model: nn.Module,
    filter_fn: ModuleFilter | None,
    *,
    require_cuda: bool,
) -> list[tuple[nn.Linear, list[ModuleReference]]]:
    selected_filter = filter_fn or _default_linear_filter
    grouped: dict[int, list[ModuleReference]] = defaultdict(list)
    for reference in _module_references(model):
        grouped[id(reference[2])].append(reference)

    selected: list[tuple[nn.Linear, list[ModuleReference]]] = []
    for aliases in grouped.values():
        decisions = [
            bool(selected_filter(module, fqn))
            for _, _, module, fqn in aliases
        ]
        if any(decisions) and not all(decisions):
            names = [fqn or "<root>" for _, _, _, fqn in aliases]
            raise ValueError(
                "filter selected only some aliases of one shared module: "
                + ", ".join(names)
            )
        if not any(decisions):
            continue
        module = aliases[0][2]
        if not isinstance(module, nn.Linear):
            raise TypeError(
                "RTX conversion can only replace nn.Linear modules; "
                f"filter selected {type(module).__name__}"
            )
        name = aliases[0][3] or "<root>"
        if module.bias is not None:
            raise NotImplementedError(
                f"RTX linear layers do not support bias; selected {name} has bias"
            )
        if module.weight.dtype is not torch.bfloat16:
            raise TypeError(
                f"RTX conversion requires a BF16 weight; {name} has "
                f"{module.weight.dtype}"
            )
        if require_cuda and module.weight.device.type != "cuda":
            raise ValueError(
                f"RTX PTQ requires CUDA weights; {name} is on {module.weight.device}"
            )
        selected.append((module, aliases))
    return selected


def _install_replacements(
    model: nn.Module,
    selected: list[tuple[nn.Linear, list[ModuleReference]]],
    replacements: dict[int, nn.Module],
) -> nn.Module:
    replacement_root = replacements.get(id(model), model)
    for module, aliases in selected:
        replacement = replacements[id(module)]
        for parent, name, _child, _fqn in aliases:
            if parent is not None:
                assert name is not None
                parent._modules[name] = replacement
    return replacement_root


def _release_async_packing_sources(tensors: list[torch.Tensor]) -> None:
    """Finish one-shot PTQ and drop launch-lifetime BF16 source references."""

    devices = {tensor.device for tensor in tensors if tensor.device.type == "cuda"}
    for device in devices:
        torch.cuda.current_stream(device).synchronize()
    for tensor in tensors:
        current: torch.Tensor | None = tensor
        visited: set[int] = set()
        while isinstance(current, torch.Tensor) and id(current) not in visited:
            visited.add(id(current))
            if hasattr(current, "_base_inputs"):
                del current._base_inputs
            current = getattr(current, "_base", None)


def _prepare_model(
    model: nn.Module, device: torch.device | str | int | None
) -> None:
    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be an nn.Module, got {type(model).__name__}")
    if device is not None:
        model.to(device=device)


def _nvfp4_calibration_plan(
    selected: list[tuple[nn.Linear, list[ModuleReference]]],
    config: NVFP4WeightOnlyConfig,
    moments: Mapping[str, torch.Tensor] | None,
    overrides: Mapping[str, NVFP4CalibrationConfig] | None,
) -> dict[int, tuple[NVFP4CalibrationConfig | None, torch.Tensor | None]]:
    """Resolve all FQNs and shared aliases before building replacements."""
    for name, mapping in (("calibration_moments", moments), ("calibration_overrides", overrides)):
        if mapping is not None and not isinstance(mapping, Mapping):
            raise TypeError(f"{name} must be a mapping from selected module FQNs")
    moments = {} if moments is None else moments
    overrides = {} if overrides is None else overrides
    known = {reference[3] for _, aliases in selected for reference in aliases}
    for name, mapping in (("calibration_moments", moments), ("calibration_overrides", overrides)):
        unknown = set(mapping) - known
        if unknown:
            raise ValueError(f"{name} contains unknown or unselected FQNs: {sorted(map(repr, unknown))}")
    for fqn, override in overrides.items():
        if not isinstance(override, NVFP4CalibrationConfig):
            raise TypeError(f"calibration_overrides[{fqn!r}] must be NVFP4CalibrationConfig")
    plan: dict[int, tuple[NVFP4CalibrationConfig | None, torch.Tensor | None]] = {}
    for module, aliases in selected:
        names = [reference[3] for reference in aliases]
        policies = [overrides.get(name, config.weight_calibration) for name in names]
        if any(policy != policies[0] for policy in policies[1:]):
            raise ValueError(f"conflicting calibration policies for shared aliases: {names}")
        policy = policies[0]
        if policy is not None and config.quant_config is not None:
            raise ValueError("quant_config conflicts with offline weight calibration")
        supplied = []
        for name in names:
            if name not in moments:
                continue
            value = moments[name]
            if (not isinstance(value, torch.Tensor) or value.layout != torch.strided
                    or not value.is_floating_point()):
                raise TypeError(f"calibration_moments[{name!r}] must be a strided floating tensor")
            expected = ((module.in_features + 15) // 16, 16, 16)
            if tuple(value.shape) != expected:
                raise ValueError(f"calibration_moments[{name!r}] must have shape {expected}")
            cpu = value.detach().to(device="cpu", dtype=torch.float32)
            if not bool(cpu.isfinite().all()):
                raise ValueError(f"calibration_moments[{name!r}] must be finite in FP32")
            supplied.append(cpu)
        if supplied and any(not torch.equal(value, supplied[0]) for value in supplied[1:]):
            raise ValueError(f"conflicting calibration moments for shared aliases: {names}")
        second_moments = supplied[0] if supplied else None
        if policy is not None and policy.method == "local_hessian" and second_moments is None:
            raise ValueError(f"local_hessian requires calibration_moments for {names}")
        plan[id(module)] = policy, second_moments
    return plan


def convert_to_mxfp8_training(
    model: nn.Module,
    *,
    module_filter_fn: ModuleFilter | None = None,
    config: MXFP8TrainingConfig | None = None,
    device: torch.device | str | int | None = None,
) -> nn.Module:
    """Swap selected BF16 linears into dynamic MXFP8 training modules.

    Original Parameter objects are retained, preserving optimizer references,
    weight sharing, state-dict keys, initialization, and ``requires_grad``.
    """

    _prepare_model(model, device)
    selected = _selected_linears(model, module_filter_fn, require_cuda=False)
    policy = config or MXFP8TrainingConfig()
    from .fp8 import MXFP8Linear

    replacements: dict[int, nn.Module] = {}
    for module, _aliases in selected:
        replacement = MXFP8Linear(
            module.in_features,
            module.out_features,
            bias=False,
            device=module.weight.device,
            dtype=torch.bfloat16,
            forward_config=policy.forward_config,
            dynamic_config=policy.dynamic_config,
            backward_config=policy.backward_config,
            backend=policy.backend,
            autotune=policy.autotune,
            tuning_policy=policy.tuning_policy,
            autotune_cache_dir=policy.autotune_cache_dir,
        )
        replacement.weight = module.weight
        replacement.train(module.training)
        replacements[id(module)] = replacement
    return _install_replacements(model, selected, replacements)


def convert_to_nvfp4_training(
    model: nn.Module,
    *,
    module_filter_fn: ModuleFilter | None = None,
    config: NVFP4TrainingConfig | None = None,
    device: torch.device | str | int | None = None,
) -> nn.Module:
    """Swap selected BF16 linears into NVFP4-forward/MXFP8-backward modules."""

    _prepare_model(model, device)
    selected = _selected_linears(model, module_filter_fn, require_cuda=False)
    policy = config or NVFP4TrainingConfig()
    from .fp4 import NVFP4Linear

    replacements: dict[int, nn.Module] = {}
    for module, _aliases in selected:
        replacement = NVFP4Linear(
            module.in_features,
            module.out_features,
            bias=False,
            device=module.weight.device,
            dtype=torch.bfloat16,
            scale_config=policy.scale_config,
            backward_config=policy.backward_config,
            scaling=policy.scaling,
            scale_region_rows=policy.scale_region_rows,
            x_scale_region_rows=policy.x_scale_region_rows,
            weight_scale_region_rows=policy.weight_scale_region_rows,
            backend=policy.backend,
            autotune=policy.autotune,
            tuning_policy=policy.tuning_policy,
            autotune_cache_dir=policy.autotune_cache_dir,
            dynamic_config=policy.dynamic_config,
        )
        replacement.weight = module.weight
        replacement.train(module.training)
        replacements[id(module)] = replacement
    return _install_replacements(model, selected, replacements)


def quantize_(
    model: nn.Module,
    config: MXFP8WeightOnlyConfig | NVFP4WeightOnlyConfig,
    filter_fn: ModuleFilter | None = None,
    device: torch.device | str | int | None = None,
    *,
    calibration_moments: Mapping[str, torch.Tensor] | None = None,
    calibration_overrides: Mapping[str, NVFP4CalibrationConfig] | None = None,
) -> nn.Module:
    """PTQ selected no-bias BF16 linears using a format policy.

    The interface follows TorchAO's ``quantize_`` convention. Child modules
    are replaced in place and the possibly replaced root is returned. Every
    selected module is validated and every replacement is built before the
    module tree is mutated.

    NVFP4 offline calibration is opt-in through ``config.weight_calibration``
    and/or per-FQN ``calibration_overrides``. Root linear FQN is ``""``.
    ``calibration_moments`` supplies uncentered [ceil(K/16),16,16] moments;
    only local-Hessian consumes these, but every supplied entry is validated.
    Shared aliases must resolve to one policy and equal supplied moments; one
    moment entry under any shared alias suffices. Collection and statistical
    independence are caller-owned. Block scaling fixes the outer scale to one.
    These controls do not apply to MXFP8 or dynamic training conversion.
    """

    _prepare_model(model, device)
    if not isinstance(config, (MXFP8WeightOnlyConfig, NVFP4WeightOnlyConfig)):
        raise TypeError(
            "rtx.quantize_ requires MXFP8WeightOnlyConfig or "
            f"NVFP4WeightOnlyConfig; got {type(config).__name__}"
        )
    if isinstance(config, MXFP8WeightOnlyConfig) and (
        calibration_moments is not None or calibration_overrides is not None
    ):
        raise TypeError("offline calibration arguments require NVFP4WeightOnlyConfig")
    if (isinstance(config, NVFP4WeightOnlyConfig) and calibration_moments is not None
            and config.weight_calibration is None and not calibration_overrides):
        raise ValueError("calibration_moments requires weight_calibration or calibration_overrides")
    selected = _selected_linears(model, filter_fn, require_cuda=True)
    calibration_plan = (_nvfp4_calibration_plan(selected, config, calibration_moments, calibration_overrides)
                        if isinstance(config, NVFP4WeightOnlyConfig) else {})
    replacements: dict[int, nn.Module] = {}
    packing_outputs: list[torch.Tensor] = []

    if isinstance(config, MXFP8WeightOnlyConfig):
        from .fp8 import (
            DEFAULT_MXFP8_INFERENCE_CONFIG,
            MXFP8Linear,
            quantize_mxfp8,
        )

        selected_dynamic = config.dynamic_config or DEFAULT_MXFP8_INFERENCE_CONFIG
        for module, _aliases in selected:
            packed = quantize_mxfp8(
                module.weight.detach(),
                config=selected_dynamic.resolved_weight_quant(),
            )
            packing_outputs.extend((packed.qdata, packed.scale))
            replacements[id(module)] = MXFP8Linear(
                module.in_features,
                module.out_features,
                bias=False,
                device=module.weight.device,
                dtype=torch.bfloat16,
                dynamic_config=config.dynamic_config,
                autotune=config.autotune,
                tuning_policy=config.tuning_policy,
                autotune_cache_dir=config.autotune_cache_dir,
                packed_weight=packed,
            )
    else:
        from .fp4 import NVFP4Linear, quantize_nvfp4

        for module, _aliases in selected:
            tensor_scale = (
                torch.ones((), dtype=torch.float32, device=module.weight.device)
                if config.scaling == "block"
                else None
            )
            calibration, moments = calibration_plan[id(module)]
            if calibration is None:
                packed = quantize_nvfp4(
                    module.weight.detach(),
                    tensor_scale=tensor_scale,
                    config=config.quant_config,
                )
            else:
                from .nvfp4_calibration import calibrate_nvfp4_weight

                packed = calibrate_nvfp4_weight(
                    module.weight.detach(), calibration.method,
                    second_moments=moments if calibration.method == "local_hessian" else None,
                    outer_scale=tensor_scale,
                    rows_per_chunk=calibration.rows_per_chunk,
                    hessian_shrinkage=calibration.hessian_shrinkage,
                )
            packing_outputs.extend((packed.qdata, packed.scale))
            replacements[id(module)] = NVFP4Linear(
                module.in_features,
                module.out_features,
                bias=False,
                device=module.weight.device,
                dtype=torch.bfloat16,
                scaling=config.scaling,
                autotune=config.autotune,
                tuning_policy=config.tuning_policy,
                autotune_cache_dir=config.autotune_cache_dir,
                weight_prequant_config=config.weight_prequant_config,
                fully_prequant_config=config.fully_prequant_config,
                packed_weight=packed,
            )
    # Standalone quantizers remain asynchronous and retain their inputs. PTQ
    # is a one-shot ownership transition: wait once after the complete batch
    # and ensure packed buffers do not keep the BF16 masters resident.
    _release_async_packing_sources(packing_outputs)
    return _install_replacements(model, selected, replacements)


__all__ = [
    "MXFP8TrainingConfig",
    "MXFP8WeightOnlyConfig",
    "ModuleFilter",
    "NVFP4TrainingConfig",
    "NVFP4CalibrationConfig",
    "NVFP4WeightOnlyConfig",
    "convert_to_mxfp8_training",
    "convert_to_nvfp4_training",
    "quantize_",
]
