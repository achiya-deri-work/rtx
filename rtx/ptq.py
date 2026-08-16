"""TorchAO-style model conversion for RTX training and PTQ frontends."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
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
class NVFP4WeightOnlyConfig:
    """PTQ policy for packed NVFP4 weights and dynamic BF16 activations."""

    scaling: Literal["current", "block"] = "current"
    quant_config: "NVFP4QuantConfig | None" = None
    weight_prequant_config: "NVFP4WeightPrequantConfig | None" = None
    fully_prequant_config: "NVFP4FullyPrequantConfig | None" = None
    autotune: AutotuneMode | bool | None = None
    tuning_policy: object | None = None
    autotune_cache_dir: Path | str | None = None

    def __post_init__(self) -> None:
        if self.scaling not in ("current", "block"):
            raise ValueError("packed NVFP4 PTQ scaling must be current or block")


def _default_linear_filter(module: nn.Module, fqn: str) -> bool:
    del fqn
    # This intentionally includes TorchAO Float8Linear, which is an nn.Linear
    # subclass retaining the original high-precision Parameter.
    return isinstance(module, nn.Linear)


def _module_references(root: nn.Module) -> list[ModuleReference]:
    """Return every module reference, including aliases, without recursing twice."""

    references: list[ModuleReference] = [(None, None, root, "")]
    visited = {id(root)}

    def visit(parent: nn.Module, parent_fqn: str) -> None:
        for name, child in tuple(parent._modules.items()):
            if child is None:
                continue
            fqn = f"{parent_fqn}.{name}" if parent_fqn else name
            references.append((parent, name, child, fqn))
            if id(child) not in visited:
                visited.add(id(child))
                visit(child, fqn)

    visit(root, "")
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
) -> nn.Module:
    """PTQ selected no-bias BF16 linears using a format policy.

    The interface follows TorchAO's ``quantize_`` convention. Child modules
    are replaced in place and the possibly replaced root is returned. Every
    selected module is validated and every replacement is built before the
    module tree is mutated.
    """

    _prepare_model(model, device)
    if not isinstance(config, (MXFP8WeightOnlyConfig, NVFP4WeightOnlyConfig)):
        raise TypeError(
            "rtx.quantize_ requires MXFP8WeightOnlyConfig or "
            f"NVFP4WeightOnlyConfig; got {type(config).__name__}"
        )
    selected = _selected_linears(model, filter_fn, require_cuda=True)
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
            packed = quantize_nvfp4(
                module.weight.detach(),
                tensor_scale=tensor_scale,
                config=config.quant_config,
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
    "NVFP4WeightOnlyConfig",
    "convert_to_mxfp8_training",
    "convert_to_nvfp4_training",
    "quantize_",
]
