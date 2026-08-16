"""Pretrained DINOv3 ViT-S/16 regression matrix for RTX linear frontends."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import hashlib
import json
import os
from pathlib import Path
import sys
import time
from typing import Literal

import torch
from torch import nn

import rtx


REPO_ROOT = Path(__file__).resolve().parents[1]
DINOV3_ROOT = REPO_ROOT / "fp8dinov3"
DEFAULT_CHECKPOINT = Path(
    os.environ.get(
        "RTX_DINOV3_CHECKPOINT",
        DINOV3_ROOT
        / "weights"
        / "dinov3_vits16_pretrain_lvd1689m-08c60483.pth",
    )
)
EXPECTED_SHA256 = "08c60483bc63c04f533611e34bf70b120eedb7240f469bc16e9e20bf344b941d"
EXPECTED_PARAMETERS = 21_601_152
EXPECTED_LINEAR_COUNT = 48

Variant = Literal[
    "bf16",
    "bf16_bias_adapter",
    "torchao_rowwise",
    "mxfp8_training",
    "mxfp8_ptq",
    "nvfp4_training_delayed",
    "nvfp4_training_current",
    "nvfp4_training_jit",
    "nvfp4_training_block",
    "nvfp4_ptq_current",
    "nvfp4_ptq_block",
]

DEFAULT_VARIANTS: tuple[Variant, ...] = (
    "bf16",
    "bf16_bias_adapter",
    "torchao_rowwise",
    "mxfp8_training",
    "mxfp8_ptq",
    "nvfp4_training_delayed",
    "nvfp4_training_current",
    "nvfp4_training_jit",
    "nvfp4_training_block",
    "nvfp4_ptq_current",
    "nvfp4_ptq_block",
)


def checkpoint_sha256(path: Path = DEFAULT_CHECKPOINT) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _import_dinov3():
    if not DINOV3_ROOT.exists():
        raise FileNotFoundError(f"local DINOv3 source is missing: {DINOV3_ROOT}")
    root = str(DINOV3_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from dinov3.hub.backbones import dinov3_vits16

    return dinov3_vits16


def load_dinov3_vits16(
    *,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.bfloat16,
) -> nn.Module:
    """Construct locally and strictly load the official pretrained state."""

    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"DINOv3 ViT-S/16 checkpoint is missing: {checkpoint}"
        )
    constructor = _import_dinov3()
    model = constructor(pretrained=False)
    state = torch.load(
        checkpoint,
        map_location="cpu",
        weights_only=True,
        mmap=True,
    )
    model.load_state_dict(state, strict=True)
    model.to(device=device, dtype=dtype)
    model.eval()
    return model


class DinoBiasAdapter(nn.Module):
    """Preserve DINO's bias outside an intentionally bias-free linear core."""

    def __init__(self, core: nn.Module, source: nn.Linear) -> None:
        super().__init__()
        self.in_features = source.in_features
        self.out_features = source.out_features
        self.core = core
        self.bias = source.bias
        bias_mask = getattr(source, "bias_mask", None)
        if bias_mask is None:
            self.register_buffer("bias_mask", None, persistent=False)
        else:
            self.register_buffer("bias_mask", bias_mask)
        self.train(source.training)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        output = self.core(value)
        if self.bias is None:
            return output
        bias = self.bias
        if self.bias_mask is not None:
            bias = bias * self.bias_mask.to(dtype=bias.dtype)
        return output + bias


def _biasless_source(source: nn.Linear) -> nn.Linear:
    linear = nn.Linear(
        source.in_features,
        source.out_features,
        bias=False,
        device=source.weight.device,
        dtype=source.weight.dtype,
    )
    linear.weight = source.weight
    linear.train(source.training)
    return linear


def _is_backbone_linear(name: str, module: nn.Module) -> bool:
    return name.startswith("blocks.") and isinstance(module, nn.Linear)


def _convert_core(source: nn.Linear, variant: Variant) -> nn.Module:
    linear = _biasless_source(source)
    if variant == "bf16_bias_adapter":
        return linear
    if variant == "mxfp8_training":
        return rtx.convert_to_mxfp8_training(
            linear,
            config=rtx.MXFP8TrainingConfig(autotune="off"),
        )
    if variant == "mxfp8_ptq":
        return rtx.quantize_(
            linear,
            rtx.MXFP8WeightOnlyConfig(autotune="off"),
        )
    if variant.startswith("nvfp4_training_"):
        scaling = {
            "nvfp4_training_delayed": "delayed",
            "nvfp4_training_current": "current",
            "nvfp4_training_jit": "jit_row_region",
            "nvfp4_training_block": "block",
        }[variant]
        return rtx.convert_to_nvfp4_training(
            linear,
            config=rtx.NVFP4TrainingConfig(
                scaling=scaling,
                backend="auto",
                autotune="off",
            ),
        )
    if variant in ("nvfp4_ptq_current", "nvfp4_ptq_block"):
        scaling = "current" if variant.endswith("current") else "block"
        return rtx.quantize_(
            linear,
            rtx.NVFP4WeightOnlyConfig(
                scaling=scaling,
                autotune="off",
            ),
        )
    raise ValueError(f"variant {variant!r} does not use an RTX core")


def convert_dinov3_variant(model: nn.Module, variant: Variant) -> nn.Module:
    """Convert transformer linears while retaining DINO bias semantics."""

    if variant == "bf16":
        return model
    if variant == "torchao_rowwise":
        root = str(DINOV3_ROOT)
        if root not in sys.path:
            sys.path.insert(0, root)
        from dinov3.layers.fp8_linear import convert_linears_to_fp8

        return convert_linears_to_fp8(model, filter=r"^blocks\.")

    targets = [
        (name, module)
        for name, module in model.named_modules()
        if _is_backbone_linear(name, module)
    ]
    replacements: list[tuple[nn.Module, str, nn.Module]] = []
    for name, source in targets:
        assert isinstance(source, nn.Linear)
        parent_name, child_name = name.rsplit(".", 1)
        parent = model.get_submodule(parent_name)
        core = _convert_core(source, variant)
        replacements.append((parent, child_name, DinoBiasAdapter(core, source)))
    for parent, child_name, replacement in replacements:
        parent._modules[child_name] = replacement
    return model


def deterministic_images(
    batch_size: int,
    image_size: int,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 3407,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    random = torch.randn(
        batch_size,
        3,
        image_size,
        image_size,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    # A deterministic spatial component prevents an all-noise regression from
    # hiding structured drift in patch projections.
    axis = torch.linspace(-1, 1, image_size, device=device, dtype=dtype)
    pattern = axis[:, None] + axis[None, :]
    random[:, 0].add_(pattern)
    random[:, 1].add_(pattern.T)
    return random


def tensor_metrics(actual: torch.Tensor, reference: torch.Tensor) -> dict[str, float | bool]:
    actual_f = actual.float()
    reference_f = reference.float()
    difference = actual_f - reference_f
    denominator = reference_f.square().sum().sqrt().clamp_min(1.0e-30)
    flat_actual = actual_f.flatten(1)
    flat_reference = reference_f.flatten(1)
    cosine = torch.nn.functional.cosine_similarity(
        flat_actual, flat_reference, dim=1
    )
    return {
        "finite": bool(torch.isfinite(actual_f).all()),
        "relative_l2": float(difference.square().sum().sqrt() / denominator),
        "rms_error": float(difference.square().mean().sqrt()),
        "max_abs_error": float(difference.abs().max()),
        "mean_cosine": float(cosine.mean()),
        "minimum_cosine": float(cosine.min()),
    }


def _capture_blocks(model: nn.Module) -> tuple[dict[str, torch.Tensor], list[object]]:
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for index, block in enumerate(model.blocks):
        name = f"blocks.{index}"

        def capture(_module, _inputs, output, *, key=name):
            # NestedBlockChunk invokes a block through its list-valued path,
            # while an unchunked block may return a tensor directly.
            value = output
            while isinstance(value, (tuple, list)):
                if len(value) != 1:
                    raise TypeError(
                        f"{key} returned {len(value)} values; expected one"
                    )
                value = value[0]
            if not isinstance(value, torch.Tensor):
                raise TypeError(
                    f"{key} returned {type(value).__name__}, expected Tensor"
                )
            captured[key] = value.detach().clone()

        handles.append(block.register_forward_hook(capture))
    return captured, handles


@torch.no_grad()
def run_variant(
    variant: Variant,
    images: torch.Tensor,
    *,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    compile_model: bool = False,
    capture_blocks: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor], dict[str, object]]:
    model = load_dinov3_vits16(
        checkpoint=checkpoint,
        device=images.device,
        dtype=images.dtype,
    )
    model = convert_dinov3_variant(model, variant).eval()
    blocks: dict[str, torch.Tensor] = {}
    handles: list[object] = []
    if capture_blocks:
        blocks, handles = _capture_blocks(model)
    function = (
        torch.compile(model, fullgraph=True, dynamic=False)
        if compile_model
        else model
    )
    started = time.perf_counter()
    output = function(images)
    torch.cuda.synchronize(images.device)
    elapsed_s = time.perf_counter() - started
    for handle in handles:
        handle.remove()
    metadata = {
        "variant": variant,
        "compiled": compile_model,
        "elapsed_first_call_s": elapsed_s,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "converted_linears": sum(
            isinstance(module, DinoBiasAdapter) for module in model.modules()
        ),
        "packed_bytes": sum(
            buffer.numel() * buffer.element_size()
            for module in model.modules()
            if getattr(module, "weight_mode", None) == "prequantized"
            for buffer in module.buffers(recurse=False)
        ),
    }
    return output.detach(), blocks, metadata


def run_regression(
    *,
    variants: Sequence[Variant] = DEFAULT_VARIANTS,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    batch_size: int = 2,
    image_size: int = 224,
    device: torch.device | str = "cuda:0",
    compile_model: bool = False,
    capture_blocks: bool = True,
) -> dict[str, object]:
    device = torch.device(device)
    images = deterministic_images(batch_size, image_size, device=device)
    reference, reference_blocks, reference_meta = run_variant(
        "bf16",
        images,
        checkpoint=checkpoint,
        compile_model=compile_model,
        capture_blocks=capture_blocks,
    )
    adapted_reference, adapted_blocks, adapted_meta = run_variant(
        "bf16_bias_adapter",
        images,
        checkpoint=checkpoint,
        compile_model=compile_model,
        capture_blocks=capture_blocks,
    )
    results: dict[str, object] = {
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": checkpoint_sha256(checkpoint),
        "batch_size": batch_size,
        "image_size": image_size,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "compute_capability": list(torch.cuda.get_device_capability(device)),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "reference": reference_meta,
        "bias_adapter_reference": {
            **adapted_meta,
            "versus_original": tensor_metrics(adapted_reference, reference),
        },
        "variants": {},
    }
    for variant in variants:
        if variant in ("bf16", "bf16_bias_adapter"):
            continue
        output, blocks, metadata = run_variant(
            variant,
            images,
            checkpoint=checkpoint,
            compile_model=compile_model,
            capture_blocks=capture_blocks,
        )
        metadata["final_vs_original"] = tensor_metrics(output, reference)
        metadata["final_vs_bias_adapter"] = tensor_metrics(
            output, adapted_reference
        )
        if capture_blocks:
            metadata["blocks_vs_original"] = {
                name: tensor_metrics(blocks[name], reference_blocks[name])
                for name in reference_blocks
            }
            metadata["blocks_vs_bias_adapter"] = {
                name: tensor_metrics(blocks[name], adapted_blocks[name])
                for name in adapted_blocks
            }
        results["variants"][variant] = metadata
        torch.cuda.empty_cache()
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--no-blocks", action="store_true")
    parser.add_argument("--variants", nargs="*", choices=DEFAULT_VARIANTS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("autotune_reports/dinov3_vits16_regression.json"),
    )
    args = parser.parse_args()
    report = run_regression(
        variants=tuple(args.variants or DEFAULT_VARIANTS),
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=args.device,
        compile_model=args.compile,
        capture_blocks=not args.no_blocks,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
