"""Numerical DINOv3 study for W4A16 and W4A8 candidate formats.

This is deliberately an emulation harness, not a performance benchmark. It
dequantizes packed weights/activations before BF16 ``F.linear`` so format error
can be evaluated independently of whether SM120 has a matching MMA operation.
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import json
from pathlib import Path
import sys
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torchao.prototype.mx_formats.mx_tensor import MXTensor

import rtx

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.dinov3_regression import (
    DEFAULT_CHECKPOINT,
    DinoBiasAdapter,
    _is_backbone_linear,
    deterministic_images,
    load_dinov3_vits16,
    tensor_metrics,
)


MixedMode = Literal[
    "nvfp4_w4a16",
    "nvfp4_w4a8",
    "mxf4_w4a16",
    "mxf4_w4a8",
]

DEFAULT_MODES: tuple[MixedMode, ...] = (
    "nvfp4_w4a16",
    "nvfp4_w4a8",
    "mxf4_w4a16",
    "mxf4_w4a8",
)


def mode_metadata(mode: MixedMode) -> dict[str, object]:
    mxf4 = mode.startswith("mxf4_")
    activation_bits = 8 if mode.endswith("a8") else 16
    return {
        "mode": mode,
        "weight_format": "MXF4_E2M1_E8M0_block32" if mxf4 else "NVFP4_E2M1_E4M3_block16",
        "activation_format": "MXFP8_E4M3_E8M0_block32" if activation_bits == 8 else "BF16",
        # SM120 MmaMXF8F6F4Op supports E2M1 x E4M3 with UE8M0
        # block-32 scales. The other combinations require composition.
        "direct_sm120_mma": bool(mxf4 and activation_bits == 8),
        "emulated": True,
        "performance_result": False,
    }


class EmulatedMixedLinear(nn.Module):
    """Bias-free format emulator used only by the DINO numeric study."""

    def __init__(self, weight: torch.Tensor, mode: MixedMode) -> None:
        super().__init__()
        self.in_features = int(weight.shape[1])
        self.out_features = int(weight.shape[0])
        self.mode = mode
        if mode.startswith("nvfp4_"):
            quantized_weight = rtx.quantize_nvfp4(weight)
        else:
            quantized_weight = MXTensor.to_mx(
                weight,
                torch.float4_e2m1fn_x2,
                block_size=32,
            )
        self.register_buffer(
            "dequantized_weight",
            quantized_weight.dequantize(torch.bfloat16),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.mode.endswith("a8"):
            shape = value.shape
            value = rtx.quantize_mxfp8(
                value.reshape(-1, shape[-1])
            ).dequantize(torch.bfloat16).reshape(shape)
        return F.linear(value, self.dequantized_weight)


def convert_dinov3_mixed(model: nn.Module, mode: MixedMode) -> nn.Module:
    targets = [
        (name, module)
        for name, module in model.named_modules()
        if _is_backbone_linear(name, module)
    ]
    replacements = []
    for name, source in targets:
        assert isinstance(source, nn.Linear)
        parent_name, child_name = name.rsplit(".", 1)
        replacement = DinoBiasAdapter(
            EmulatedMixedLinear(source.weight, mode),
            source,
        )
        replacements.append((model.get_submodule(parent_name), child_name, replacement))
    for parent, child_name, replacement in replacements:
        parent._modules[child_name] = replacement
    return model


@torch.no_grad()
def run_mixed_regression(
    *,
    modes: Sequence[MixedMode] = DEFAULT_MODES,
    checkpoint: Path = DEFAULT_CHECKPOINT,
    batch_size: int = 2,
    image_size: int = 224,
    device: torch.device | str = "cuda:0",
) -> dict[str, object]:
    device = torch.device(device)
    images = deterministic_images(batch_size, image_size, device=device)
    reference_model = load_dinov3_vits16(
        checkpoint=checkpoint,
        device=device,
    )
    from benchmarks.dinov3_regression import convert_dinov3_variant

    reference = convert_dinov3_variant(
        reference_model, "bf16_bias_adapter"
    ).eval()(images)
    results = {}
    for mode in modes:
        model = convert_dinov3_mixed(
            load_dinov3_vits16(checkpoint=checkpoint, device=device),
            mode,
        ).eval()
        output = model(images)
        results[mode] = {
            **mode_metadata(mode),
            "converted_linears": sum(
                isinstance(module, EmulatedMixedLinear)
                for module in model.modules()
            ),
            "versus_bf16_bias_adapter": tensor_metrics(output, reference),
        }
    return {
        "checkpoint": str(checkpoint),
        "batch_size": batch_size,
        "image_size": image_size,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--modes", nargs="*", choices=DEFAULT_MODES)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("autotune_reports/dinov3_mixed_precision.json"),
    )
    args = parser.parse_args()
    report = run_mixed_regression(
        modes=tuple(args.modes or DEFAULT_MODES),
        checkpoint=args.checkpoint,
        batch_size=args.batch_size,
        image_size=args.image_size,
        device=args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
