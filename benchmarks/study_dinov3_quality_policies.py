"""Controlled, multi-seed DINO policy study; never promotes runtime defaults.

Run as ``python -m benchmarks.study_dinov3_quality_policies --output PATH``.
The generated structured/noise images measure feature fidelity, not task
accuracy. All transformer linears are converted, with original biases/masks
retained. Each JSONL row is flushed independently to preserve partial evidence.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
import gc
import json
from pathlib import Path

import torch
from torch import nn

import rtx
from rtx.configs import NVFP4QuantConfig, NVFP4WeightPrequantConfig
from rtx.fp8 import DEFAULT_MXFP8_INFERENCE_CONFIG
from benchmarks.dinov3_regression import (
    DinoBiasAdapter, _biasless_source, checkpoint_sha256,
    convert_dinov3_variant, deterministic_images, load_dinov3_vits16,
    tensor_metrics,
)


@dataclass(frozen=True)
class SplitAccumulationStudy:
    """Research-only FP32 split accumulation; unchanged infinity quantization."""
    reduction_tiles: int


class SplitAccumulationLinear(nn.Module):
    """Reuse native backward split/reduction kernels, not an exported API."""
    def __init__(self, source, config):
        super().__init__()
        self.config = config
        self.register_buffer('packed_weight', rtx.quantize_mxfp8(source.weight.detach()))
        self.in_features = source.in_features
        self.out_features = source.out_features

    def forward(self, x):
        from rtx.kernels.mxfp8 import MXFP8Problem
        from rtx.kernels.mxfp8_gemm import compile_mxfp8_gemm
        from rtx.kernels.mxfp8_reduce import compile_mxfp8_workspace_reduce
        packed = rtx.quantize_mxfp8(x.reshape(-1, self.in_features))
        weight = self.packed_weight
        m = x.numel() // self.in_features
        problem = MXFP8Problem(m, self.out_features, self.in_features)
        gemm = DEFAULT_MXFP8_INFERENCE_CONFIG.gemm
        reduction_tile = self.config.reduction_tiles * gemm.tile_k
        split = (self.in_features + reduction_tile - 1) // reduction_tile
        output = torch.empty((m, self.out_features), device=x.device, dtype=x.dtype)
        if split == 1:
            compile_mxfp8_gemm(problem, gemm)(packed.qdata, weight.qdata,
                                            packed.scale, weight.scale, output)
        else:
            # The existing reducer accepts power-of-two split counts. Zero
            # padding is diagnostic-only and is NOT a proposed hot launcher.
            padded_split = 1 << (split - 1).bit_length()
            workspace = torch.zeros(padded_split * m * self.out_features,
                                    device=x.device, dtype=torch.float32)
            compile_mxfp8_gemm(problem, gemm, split_reduction=split, reduction_tile=reduction_tile)(
                packed.qdata, weight.qdata, packed.scale, weight.scale,
                workspace[:split * m * self.out_features])
            compile_mxfp8_workspace_reduce(m, self.out_features, padded_split,
                algorithm="serial", threads=256, vector=1, persistent_waves=1)(workspace, output)
        return output.reshape(*x.shape[:-1], self.out_features)


def policies():
    result = {}
    base = DEFAULT_MXFP8_INFERENCE_CONFIG
    for x_rounding in ("infinity", "floor"):
        for w_rounding in ("infinity", "floor"):
            result[f"mx_x_{x_rounding}_w_{w_rounding}"] = rtx.MXFP8WeightOnlyConfig(
                dynamic_config=replace(
                    base,
                    quant=replace(base.quant, scale_rounding=x_rounding),
                    weight_quant=replace(base.resolved_weight_quant(), scale_rounding=w_rounding),
                ),
                autotune="off",
            )
    for scaling in ("block", "current"):
        for selection in ("amax", "four_over_six", "mse5", "lsq1"):
            quant = NVFP4QuantConfig(block_scale_selection=selection)
            result[f"nv_{scaling}_{selection}"] = rtx.NVFP4WeightOnlyConfig(
                scaling=scaling, quant_config=quant,
                weight_prequant_config=NVFP4WeightPrequantConfig(quant_x=quant),
                autotune="off",
            )
    return result


def convert(model, policy):
    targets = [(n, m) for n, m in model.named_modules()
               if n.startswith("blocks.") and isinstance(m, nn.Linear)]
    for name, source in targets:
        core = (SplitAccumulationLinear(_biasless_source(source), policy)
                if isinstance(policy, SplitAccumulationStudy)
                else rtx.quantize_(_biasless_source(source), policy))
        parent, child = name.rsplit(".", 1)
        model.get_submodule(parent)._modules[child] = DinoBiasAdapter(core, source)
    return model


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[4403, 4410, 4511, 4512])
    parser.add_argument("--policies", nargs="+", choices=list(policies()))
    parser.add_argument("--mx-reduction-tiles", type=int, nargs="+", default=[],
                        help="Append native FP32 split-accumulation experiments (positive tile counts)")
    args = parser.parse_args()
    if any(n < 1 for n in args.mx_reduction_tiles):
        parser.error("MX reduction tile counts must be positive")
    torch.backends.cuda.matmul.allow_tf32 = False
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Exclusive creation prevents silently replacing a previous experiment.
    with args.output.open("x") as stream:
        metadata = dict(torch=torch.__version__, gpu=torch.cuda.get_device_name(),
                        checkpoint_sha256=checkpoint_sha256(), seeds=args.seeds,
                        scope="synthetic-image feature fidelity, not task accuracy or timing")
        stream.write(json.dumps({"metadata": metadata}) + "\n")
        stream.flush()
        images = {seed: deterministic_images(2, 224, device="cuda", seed=seed)
                  for seed in args.seeds}
        baseline = convert_dinov3_variant(load_dinov3_vits16(device="cuda"), "bf16_bias_adapter")
        expected = {seed: baseline.forward_features(value) for seed, value in images.items()}
        del baseline
        selected = policies()
        names = list(args.policies or selected)
        for tiles in args.mx_reduction_tiles:
            name = f"mx_native_split_tiles_{tiles}"
            selected[name] = SplitAccumulationStudy(tiles)
            names.append(name)
        for name in names:
            policy = selected[name]
            model = convert(load_dinov3_vits16(device="cuda"), policy)
            for seed, value in images.items():
                actual = model.forward_features(value)
                row = dict(policy=name, config=asdict(policy), seed=seed, outputs={
                    key: tensor_metrics(actual[key], expected[seed][key])
                    for key in ("x_norm_clstoken", "x_norm_patchtokens", "x_storage_tokens")
                })
                stream.write(json.dumps(row) + "\n")
                stream.flush()
                print(json.dumps(row), flush=True)
            del model, actual
            gc.collect()


if __name__ == "__main__":
    main()
