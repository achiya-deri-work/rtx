"""Offline pretrained DINOv3/CIFAR-10 policy study, not training qualification.

Native variants measure actual RTX forward implementations. Optional emulated
W-only/X-only interventions isolate format error; their times are NOT kernel
performance. No classifier is trained: task accuracy is cosine 1-NN against a
fixed BF16 gallery drawn exclusively from the CIFAR training split.

Use ``--variants bf16 ptq:max ptq:mse ptq:local_hessian ptq:four_over_six``
for offline native packed-weight comparison with identical dynamic-X policy.
Calibration uses separate train images; ``--test-count 500`` expands heldout
evaluation without changing the conservative default dataset size.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from benchmarks.dinov3_regression import (
    DEFAULT_CHECKPOINT, DinoBiasAdapter, checkpoint_sha256,
    convert_dinov3_variant, load_dinov3_vits16,
)
from benchmarks.study_nvfp4_precision import fp4_round, metrics

POLICIES = ("amax", "lsq1", "mse5", "four_over_six")
OUTER_MODES = ("exact", "exact_four_over_six")
PTQ_POLICIES = ("max", "mse", "local_hessian", "four_over_six", "shrink50", "shrink90", "selected")
SELECTION_CANDIDATES = (("max", 0.), ("mse", 0.), ("local_hessian", 0.), ("local_hessian", .5), ("local_hessian", .9))


def quantize_reference(value, rows, policy, outer_mode="exact"):
    """Format intervention, not bitexact native-kernel oracle.

    Uses FP32 division (native uses a rounded reciprocal) and BF16 decoded
    operands followed by BF16 GEMM, rather than packed native FP4 MMA.
    """
    if policy not in POLICIES:
        raise ValueError(policy)
    if outer_mode not in OUTER_MODES:
        raise ValueError(outer_mode)
    shape = value.shape
    flat = value.reshape(-1, shape[-1]).float()
    m, k = flat.shape
    padded = F.pad(flat, (0, (-k) % 16, 0, (-m) % rows))
    regions = padded.reshape(-1, rows, padded.shape[-1])
    amax = regions.abs().amax((1, 2), keepdim=True)
    # Explicit headroom changes the outer independently of the block policy.
    # All policies, including amax, therefore remain matched-outer controls.
    outer_divisor = 1536 if outer_mode == "exact_four_over_six" else 2688
    outer = torch.where(amax == 0, 1., (amax / outer_divisor).clamp_min(2. ** -126))
    outer = outer.expand_as(regions).reshape(-1, 16)[:, :1]
    blocks = padded.reshape(-1, 16)
    raw = blocks.abs().amax(1, keepdim=True) / (6 * outer)

    def decode(scale):
        scale = scale.clamp(2. ** -6, 448).to(torch.float8_e4m3fn).float() * outer
        return fp4_round(blocks / scale) * scale, scale

    best, scale = decode(raw)
    error = (best - blocks).square().sum(1, keepdim=True)
    factors = (.75, .875, 1.125, 1.25) if policy == "mse5" else ((1.5,) if policy == "four_over_six" else ())
    candidates = [raw * factor for factor in factors]
    if policy == "lsq1":
        codes = fp4_round(blocks / scale)
        fit = (blocks * codes).sum(1, keepdim=True) / codes.square().sum(1, keepdim=True).clamp_min(1e-20)
        candidates.append(fit / outer)
    for candidate in candidates:
        decoded, _ = decode(candidate)
        candidate_error = (decoded - blocks).square().sum(1, keepdim=True)
        best = torch.where(candidate_error < error, decoded, best)
        error = torch.minimum(candidate_error, error)
    return best.reshape_as(padded)[:m, :k].reshape(shape).to(value.dtype)


def select_indices(labels, count, seed, exclude=()):
    """Seeded stratified prefix, independent of images or model outputs."""
    excluded = set(exclude)
    if count < 1 or count > len(labels) - len(excluded):
        raise ValueError("sample count outside dataset")
    generator = np.random.default_rng(seed)
    pools = [generator.permutation([i for i in np.flatnonzero(np.asarray(labels) == c) if i not in excluded]).tolist() for c in range(10)]
    selected = []
    while len(selected) < count:
        for pool in pools:
            if pool and len(selected) < count:
                selected.append(pool.pop())
    return selected


def select_window_indices(labels, count, seed, exclude=(), offset=0):
    if offset < 0 or count < 1:
        raise ValueError("nonnegative offset and positive count required")
    prefix = select_indices(labels, count + offset, seed, exclude)
    return prefix[offset:], prefix[:offset]


def load_images(root, train, count, seed, image_size, exclude=(), offset=0, excluded_prefix=None):
    from torchvision.datasets import CIFAR10
    dataset = CIFAR10(root=str(root), train=train, download=False)
    indices, skipped = select_window_indices(dataset.targets, count, seed, exclude, offset)
    if excluded_prefix is not None:
        excluded_prefix.extend(skipped)
    tensors = []
    for index in indices:
        image = Image.fromarray(dataset.data[index]).resize((image_size, image_size), Image.Resampling.BICUBIC)
        tensors.append(torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float() / 255)
    images = torch.stack(tensors)
    images = (images - images.new_tensor([.485, .456, .406])[None, :, None, None]) / images.new_tensor([.229, .224, .225])[None, :, None, None]
    labels = torch.tensor([dataset.targets[i] for i in indices])
    return images, labels, indices


class EmulatedLinear(nn.Module):
    def __init__(self, source, policy, operand, x_rows, w_rows, outer_mode="exact"):
        super().__init__()
        self.policy, self.operand, self.x_rows = policy, operand, x_rows
        self.outer_mode = outer_mode
        quantized = quantize_reference(source.weight, w_rows, policy, outer_mode)
        self.weight_metrics = metrics(quantized, source.weight)
        self.register_buffer("weight", quantized if operand in ("w", "both") else source.weight.detach())
        self.activation_metrics = None

    def forward(self, x):
        if self.operand in ("x", "both"):
            q = quantize_reference(x, self.x_rows, self.policy, self.outer_mode)
            if self.activation_metrics is None:
                self.activation_metrics = metrics(q, x)
            x = q
        return F.linear(x, self.weight)


def convert(model, variant, batch_size, image_size, x_rows, w_rows, outer_mode="exact", calibration=None, calibration_validation=None):
    if variant == "bf16":
        return model, {}
    if variant == "mxfp8":
        return convert_dinov3_variant(model, "mxfp8_training"), {}
    import rtx
    from rtx.nvfp4_inference_autotune import preferred_jit_row_region_config
    pieces = variant.split(":")
    policy = pieces[1]
    metadata = {}
    for name, source in list(model.named_modules()):
        if not name.startswith("blocks.") or not isinstance(source, nn.Linear):
            continue
        if pieces[0] == "emulated":
            core = EmulatedLinear(source, policy, pieces[2], x_rows, w_rows, outer_mode)
        elif pieces[0] == "ptq":
            from rtx.nvfp4_calibration import calibrate_nvfp4_weight
            if calibration is None or name not in calibration:
                raise ValueError(f"missing train-only calibration for {name}")
            # Offline packing only. Runtime is the production packed-W,
            # dynamic BF16-X API, identical across all PTQ weight policies.
            outer = None
            if outer_mode == "exact_four_over_six":
                maximum = source.weight.detach().float().abs().max().cpu()
                outer = torch.where(maximum == 0, 1., (maximum / 1536).clamp_min(2. ** -126))
            started = time.perf_counter()
            selection = None
            if policy == "selected":
                if calibration_validation is None or name not in calibration_validation:
                    raise ValueError(f"missing independent train-validation activations for {name}")
                if "selected_weight" not in calibration[name]:
                    raise ValueError("selected weights must be frozen on train-validation before heldout evaluation")
                packed = move_native_packed(calibration[name]["selected_weight"], source.weight.device)
                selection = calibration[name]["selection"]
            else:
                shrinkage = {"shrink50": .5, "shrink90": .9}.get(policy, 0.)
                pack_policy = "local_hessian" if policy.startswith("shrink") else policy
                packed = calibrate_nvfp4_weight(source.weight, policy=pack_policy, second_moments=calibration[name]["second_moments"], outer_scale=outer, hessian_shrinkage=shrinkage)
            packed_config = rtx.NVFP4WeightPrequantConfig(quant_x=rtx.NVFP4QuantConfig(block_scale_selection="amax", scale_reciprocal="direct"))
            core = rtx.NVFP4Linear(source.in_features, source.out_features, device=source.weight.device, packed_weight=packed, scaling="current", weight_prequant_config=packed_config, autotune="off")
            metadata[name] = {"offline_pack_elapsed_s": time.perf_counter() - started, "weight_policy": policy, "weight_outer_mode": outer_mode, "activation_policy": "native current tensor amax/2688 exact outer; native amax block16", "weight_prequant_config": asdict(packed_config), "runtime": "native dynamic BF16-X / packed NVFP4-W", "calibration": calibration[name]["metadata"]}
            if selection is not None:
                metadata[name]["selection"] = {**selection, "validation": calibration_validation[name]["metadata"]}
        else:
            problem = rtx.NVFP4Problem(batch_size * ((image_size // 16) ** 2 + 5), source.out_features, source.in_features)
            cfg = preferred_jit_row_region_config(problem)
            cfg = replace(cfg, x_scale_region_rows=x_rows, weight_scale_region_rows=w_rows, tensor_scale_mode=outer_mode, quant=replace(cfg.quant, block_scale_selection=policy))
            if cfg.rejection(problem):
                cfg = replace(cfg, region_ownership="cta")
            if cfg.rejection(problem):
                cfg = replace(cfg, gemm=replace(cfg.gemm, regional_scale_epilogue="direct", regional_epilogue_values=1))
            if cfg.rejection(problem):
                raise ValueError(f"{name}: {cfg.rejection(problem)}")
            core = rtx.NVFP4Linear(source.in_features, source.out_features, device=source.weight.device, scaling="jit_row_region", x_scale_region_rows=x_rows, weight_scale_region_rows=w_rows, scale_config=rtx.NVFP4ScaleConfig(tensor_scale_mode=outer_mode), dynamic_config=cfg, autotune="off")
            core.weight = source.weight
            metadata[name] = asdict(cfg)
        parent, child = name.rsplit(".", 1)
        model.get_submodule(parent)._modules[child] = DinoBiasAdapter(core, source)
    return model, metadata


@torch.inference_mode()
def extract(model, images, batch_size, device):
    cls, patches = [], []
    for start in range(0, len(images), batch_size):
        batch = images[start:start + batch_size].to(device=device, dtype=torch.bfloat16)
        count = len(batch)
        if count < batch_size:
            batch = torch.cat([batch, batch[-1:].expand(batch_size - count, -1, -1, -1)])
        output = model.forward_features(batch)
        if not bool(torch.isfinite(output["x_norm_clstoken"]).all()) or not bool(torch.isfinite(output["x_norm_patchtokens"]).all()):
            raise RuntimeError("nonfinite DINO features; task metrics and timing are invalid")
        cls.append(output["x_norm_clstoken"][:count].float().cpu())
        patches.append(output["x_norm_patchtokens"][:count].float().cpu())
    return torch.cat(cls), torch.cat(patches)


def task_metrics(features, gallery, gallery_labels, labels, reference_predictions=None):
    if not bool(features.isfinite().all()) or not bool(gallery.isfinite().all()):
        raise ValueError("nonfinite features cannot define nearest-neighbor accuracy")
    similarity = F.normalize(features.float(), dim=-1) @ F.normalize(gallery.float(), dim=-1).T
    predictions = gallery_labels[similarity.argmax(1)]
    result = {"accuracy": float((predictions == labels).float().mean()), "correct": int((predictions == labels).sum()), "count": len(labels), "predictions": predictions.tolist()}
    if reference_predictions is not None:
        baseline = torch.tensor(reference_predictions)
        result["bf16_prediction_agreement"] = float((predictions == baseline).float().mean())
        result["bf16_correct_candidate_wrong"] = int(((baseline == labels) & (predictions != labels)).sum())
        result["bf16_wrong_candidate_correct"] = int(((baseline != labels) & (predictions == labels)).sum())
    return result


def feature_metrics(candidate, reference):
    cosine = F.cosine_similarity(candidate.float(), reference.float(), dim=-1)
    return {**metrics(candidate, reference), "mean_cosine": float(cosine.mean()), "minimum_cosine": float(cosine.min())}


@torch.inference_mode()
def capture_calibration_inputs(model, images, batch_size, device, max_tokens, expected_modules=48, retain_activations=False, retain_quantized=False):
    """Bounded, image-balanced BF16 capture from every unconverted block linear.

    Every selected image contributes evenly spaced token rows, not merely a
    prefix of the first batch. Hooks are removed before any candidate executes.
    Moments are uncentered E[x_b x_b^T], never centered covariance, and contain
    within-block16 correlations only (not a full Hessian).
    """
    from rtx.nvfp4_calibration import local_second_moments
    if batch_size < 1 or len(images) < 2 or max_tokens < len(images):
        raise ValueError("calibration needs at least two images, positive batch size, and token budget >= image count")
    modules = {name: module for name, module in model.named_modules() if name.startswith("blocks.") and isinstance(module, nn.Linear)}
    if len(modules) != expected_modules:
        raise ValueError(f"expected {expected_modules} unconverted block linears, found {len(modules)}")
    chunks = {name: [] for name in modules}
    quantized_chunks = {name: [] for name in modules}
    image_counts = {name: 0 for name in modules}
    handles = []
    batch_start, batch_count = 0, 0
    quota, remainder = divmod(max_tokens, len(images))

    def hook(name):
        def capture(_module, inputs):
            value = inputs[0]
            if not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16:
                raise TypeError("calibration capture requires BF16 activations")
            if value.shape[-1] != modules[name].in_features or value.numel() % (batch_count * value.shape[-1]):
                raise ValueError("cannot resolve per-image token rows")
            per_image = value.detach().reshape(batch_count, -1, value.shape[-1])
            quantized_per_image = None
            if retain_quantized:
                import rtx
                # Quantize the full original batch BEFORE sampling so the
                # current outer maximum equals the deployment policy.
                packed = rtx.quantize_nvfp4(value, config=rtx.NVFP4QuantConfig(block_scale_selection="amax", scale_reciprocal="direct"))
                quantized_per_image = decode_native_packed(packed, device=value.device).reshape_as(per_image)
            for offset, tokens in enumerate(per_image):
                count = min(len(tokens), quota + (batch_start + offset < remainder))
                indices = torch.linspace(0, len(tokens) - 1, count, device=value.device).long()
                chunks[name].append(tokens[indices].to(device="cpu", dtype=torch.bfloat16).clone())
                if quantized_per_image is not None:
                    quantized_chunks[name].append(quantized_per_image[offset, indices].to(device="cpu", dtype=torch.float32).clone())
                image_counts[name] += 1
        return capture

    try:
        for name, module in modules.items():
            handles.append(module.register_forward_pre_hook(hook(name)))
        for batch_start in range(0, len(images), batch_size):
            batch = images[batch_start:batch_start + batch_size].to(device=device, dtype=torch.bfloat16)
            batch_count = len(batch)
            # No repeated padding images enter the calibration statistics.
            model.forward_features(batch)
    finally:
        for handle in handles:
            handle.remove()
    result = {}
    for name in modules:
        if image_counts[name] != len(images):
            raise RuntimeError(f"{name}: expected one call per batch covering all calibration images")
        activations = torch.cat(chunks[name])
        if not bool(activations.isfinite().all()):
            raise ValueError(f"{name}: nonfinite calibration activations")
        moments = local_second_moments(activations)
        result[name] = {"second_moments": moments, "metadata": {"images": image_counts[name], "sampled_rows": len(activations), "row_budget": max_tokens, "in_features": activations.shape[-1], "sampling": "deterministic evenly spaced tokens per image; image-balanced quota", "activation_sha256": hashlib.sha256(activations.view(torch.uint8).numpy().tobytes()).hexdigest(), "second_moments_sha256": hashlib.sha256(moments.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest(), "moment": "uncentered E[x_block16 x_block16^T]; no inter-block terms", "no_padding": True}}
        if retain_activations:
            result[name]["activations"] = activations
        if retain_quantized:
            quantized = torch.cat(quantized_chunks[name])
            result[name]["quantized_activations"] = quantized
            result[name]["metadata"]["quantized_activation_sha256"] = hashlib.sha256(quantized.view(torch.uint8).numpy().tobytes()).hexdigest()
            result[name]["metadata"]["quantization"] = "native amax/direct block16, exact current amax/2688 outer of full original batch BEFORE row sampling; no image padding"
    return result


def decode_native_packed(value, device="cpu"):
    """Offline LUT decoder of row-major native storage (never model runtime)."""
    from rtx.formats.nvfp4 import nvfp4_matrix_shape, nvfp4_scale_layout, nvfp4_tensor_scale
    if nvfp4_scale_layout(value) != "row_major":
        raise ValueError("selection requires row-major native scales")
    rows, k = nvfp4_matrix_shape(value)
    packed = value.qdata.detach().to(device).view(torch.uint8).reshape(rows, -1)
    codes = torch.stack((packed & 15, packed >> 4), -1).flatten(-2).long()
    lut = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6., -0., -.5, -1., -1.5, -2., -3., -4., -6.], device=device)
    decoded = lut[codes].reshape(packed.shape[0], -1, 16)
    decoded = decoded * value.scale.detach().to(device).float().reshape(rows, -1)[..., None] * nvfp4_tensor_scale(value).detach().to(device)
    return decoded.flatten(1)[:, :k]


def move_native_packed(value, device):
    """Move explicit native buffers, avoiding TorchAO subclass storage moves."""
    from rtx.formats.nvfp4 import make_nvfp4_tensor, nvfp4_scale_layout, nvfp4_tensor_scale
    shape = tuple(getattr(value, "_rtx_logical_shape", value.shape))
    return make_nvfp4_tensor(value.qdata.detach().to(device), value.scale.detach().to(device), nvfp4_tensor_scale(value).detach().to(device), shape, nvfp4_scale_layout(value))


def save_frozen_selection(path, calibration, *, checkpoint_hash, fit_ids, validation_ids, outer_mode):
    """Exclusive, buffer-only audit artifact; no TorchAO subclasses pickled."""
    from rtx.formats.nvfp4 import nvfp4_scale_layout, nvfp4_tensor_scale
    layers = {}
    for name, data in calibration.items():
        packed = data["selected_weight"]
        layers[name] = {"qdata": packed.qdata.detach().cpu().clone(), "block_scales": packed.scale.detach().cpu().clone(), "tensor_scale": nvfp4_tensor_scale(packed).detach().cpu().clone(), "logical_shape": tuple(getattr(packed, "_rtx_logical_shape", packed.shape)), "scale_layout": nvfp4_scale_layout(packed), "selection": data["selection"], "fit_metadata": data["metadata"]}
    artifact = {"schema": "rtx.dinov3.native_nvfp4_frozen_selection.v1", "checkpoint_sha256": checkpoint_hash, "fit_train_indices": fit_ids, "validation_train_indices": validation_ids, "weight_outer_mode": outer_mode, "frozen_before_any_heldout_forward": True, "candidates_prespecified": SELECTION_CANDIDATES, "layers": layers}
    with path.open("xb") as stream:
        torch.save(artifact, stream)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "module_count": len(layers)}


@torch.inference_mode()
def select_calibrated_weight(weight, second_moments, validation_activations, *, outer_scale=None, decoded_x=None):
    """Prespecified per-layer choice using independent TRAIN-validation SSE.

    Native activation quantization is performed once, identically for every
    candidate. The full-layer score includes inter-block interactions. It is
    not the block-local fitting objective or a test-set/model-level search.
    ``decoded_x`` permits an independent CPU-only scoring test.
    """
    import rtx
    from rtx.nvfp4_calibration import calibrate_nvfp4_weight
    x = validation_activations.detach().cpu().float()
    w = weight.detach().cpu().float()
    supplied_decoded_x = decoded_x is not None
    if decoded_x is None:
        qx = rtx.quantize_nvfp4(validation_activations.to(device=weight.device, dtype=torch.bfloat16), config=rtx.NVFP4QuantConfig(block_scale_selection="amax", scale_reciprocal="direct"))
        decoded_x = decode_native_packed(qx)
    else:
        decoded_x = decoded_x.detach().cpu().float()
    if x.ndim != 2 or x.shape != decoded_x.shape or x.shape[1] != w.shape[1] or not bool(x.isfinite().all()) or not bool(decoded_x.isfinite().all()):
        raise ValueError("finite matching validation X and decoded X matrices required")
    reference = x @ w.T
    if not bool(reference.isfinite().all()):
        raise ValueError("nonfinite validation reference")
    scores, best, best_score = [], None, float("inf")
    for policy, shrinkage in SELECTION_CANDIDATES:
        candidate = calibrate_nvfp4_weight(weight, policy=policy, second_moments=second_moments, outer_scale=outer_scale, hessian_shrinkage=shrinkage)
        output = decoded_x @ decode_native_packed(candidate).T
        score = float((output.double() - reference.double()).square().sum())
        if not bool(output.isfinite().all()) or not np.isfinite(score):
            raise ValueError("nonfinite candidate validation output/score")
        scores.append({"policy": policy, "hessian_shrinkage": shrinkage, "full_layer_output_sse": score})
        if score < best_score:
            best, best_score, selected = candidate, score, scores[-1]
    return best, {"candidates": scores, "selected": selected, "tie_break": "prespecified candidate order", "objective": "full-layer FP32 qX@qW vs BF16-source X@W FP32; SSE accumulated FP64", "validation_split": "CIFAR train, disjoint fit and gallery", "test_access": False, "activation_quantization": "provided full-original-batch native quantized/dequantized samples" if supplied_decoded_x else "native quantization of bounded samples"}


@torch.inference_mode()
def capture_first_inputs(model, module_name, images, batch_size, device, max_tokens):
    """Eager BF16-only diagnostic forward; hook never survives this call."""
    module = model.get_submodule(module_name)
    if not isinstance(module, nn.Linear):
        raise TypeError("capture target must be an unconverted BF16 nn.Linear")
    captured = {}

    def capture(_module, inputs):
        if captured:
            return
        value = inputs[0]
        if not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16:
            raise TypeError("capture requires BF16 tensor inputs")
        flat = value.detach().reshape(-1, value.shape[-1])
        captured["tensor"] = flat[:max_tokens].to(device="cpu", dtype=torch.bfloat16).clone()
        captured["metadata"] = {"original_shape": list(value.shape), "original_flat_rows": len(flat), "captured_rows": min(len(flat), max_tokens), "truncated": len(flat) > max_tokens}

    handle = module.register_forward_pre_hook(capture)
    try:
        # No artificial padding: only distinct examples from this split.
        batch = images[:batch_size].to(device=device, dtype=torch.bfloat16)
        model.forward_features(batch)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError(f"capture target {module_name} was not called")
    return captured["tensor"], captured["metadata"]


def save_linear_capture(model, args, gallery_images, images, gallery_ids, ids):
    module = model.get_submodule(args.capture_linear)
    if not isinstance(module, nn.Linear):
        raise TypeError("capture target must be a BF16 nn.Linear")
    calibration, cal_meta = capture_first_inputs(model, args.capture_linear, gallery_images, args.batch_size, args.device, args.capture_max_tokens)
    heldout, held_meta = capture_first_inputs(model, args.capture_linear, images, args.batch_size, args.device, args.capture_max_tokens)
    full_weight = module.weight.detach()
    rows = args.capture_weight_rows or full_weight.shape[0]
    weight = full_weight[:rows].to(device="cpu", dtype=torch.bfloat16).clone()
    provenance = {
        "source": "offline CIFAR-10 / frozen pretrained DINOv3 BF16 eager diagnostic forwards",
        "dataset": str(args.dataset.resolve()), "module": args.capture_linear,
        "checkpoint": str(args.checkpoint.resolve()), "checkpoint_sha256": checkpoint_sha256(args.checkpoint),
        "calibration_split": "train", "heldout_split": "test",
        "calibration_image_ids": gallery_ids[:args.batch_size], "heldout_image_ids": ids[:args.batch_size],
        "calibration_activations": cal_meta, "heldout_activations": held_meta,
        "weight": {"original_shape": list(full_weight.shape), "captured_shape": list(weight.shape), "truncated": weight.shape != full_weight.shape, "original_absmax": float(full_weight.float().abs().max())},
        "image_size": args.image_size, "seed": args.seed,
        "no_padding": True, "no_training_or_candidate_policy_selection": True,
        "scope": "first disjoint train/test image batches only; token rows within images are correlated; not entire-dataset activation capture",
    }
    args.capture_output.parent.mkdir(parents=True, exist_ok=True)
    with args.capture_output.open("xb") as stream:
        torch.save({"weight": weight, "calibration_activations": calibration, "heldout_activations": heldout, "provenance": provenance}, stream)
    return {"path": str(args.capture_output), "sha256": hashlib.sha256(args.capture_output.read_bytes()).hexdigest(), "provenance": provenance}


def benchmark(model, batch, warmup, repeats, compiled=False):
    for _ in range(warmup):
        model.forward_features(batch)
    torch.cuda.synchronize(batch.device)
    times = []
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        model.forward_features(batch)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end))
    return {"median_ms": statistics.median(times), "samples_ms": times, "mode": ("compiled fullgraph/static" if compiled else "eager") + " CUDA events; full backbone; no transfer/preprocess; not external CUDA-graph replay"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=Path("/home/achiya/vs_code_main/cifar_example/data"))
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--gallery-count", type=int, default=100)
    p.add_argument("--test-count", type=int, default=100)
    p.add_argument("--test-offset", type=int, default=0, help="Exclude this many images from deterministic stratified test prefix")
    p.add_argument("--calibration-count", type=int, default=32, help="Disjoint train images; only loaded for ptq variants")
    p.add_argument("--calibration-max-tokens", type=int, default=1024, help="Per-linear image-balanced token row budget, >= calibration-count")
    p.add_argument("--calibration-validation-count", type=int, default=32)
    p.add_argument("--calibration-validation-max-tokens", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=20261023)
    p.add_argument("--x-rows", type=int, default=8)
    p.add_argument("--w-rows", type=int, default=8)
    p.add_argument("--outer-mode", choices=OUTER_MODES, default="exact", help="NVFP4 outer: amax/2688 or amax/1536 headroom; PTQ changes W only and always keeps X exact amax/2688")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--timing-repeats", type=int, default=10)
    p.add_argument("--compile", action="store_true", help="Compile forward_features fullgraph/static independently per variant")
    p.add_argument("--capture-linear", help="BF16 nn.Linear module name, e.g. blocks.0.mlp.fc1; first train/test batches only")
    p.add_argument("--capture-output", type=Path, help="Exclusive-create .pt tensor capture, requires --capture-linear")
    p.add_argument("--capture-max-tokens", type=int, default=1024)
    p.add_argument("--capture-weight-rows", type=int, default=0, help="Optional output-row prefix (e.g. 64); 0 keeps full weight/outer-scale context")
    p.add_argument("--variants", nargs="+", default=["bf16", "mxfp8"] + [f"native:{s}" for s in POLICIES])
    args = p.parse_args()
    if min(args.batch_size, args.x_rows, args.w_rows, args.timing_repeats, args.image_size, args.warmup) < 1 or args.image_size % 16:
        p.error("positive dimensions/repeats and image size divisible by 16 required")
    if bool(args.capture_linear) != bool(args.capture_output):
        p.error("--capture-linear and --capture-output must be supplied together")
    if args.capture_max_tokens < 1 or args.capture_weight_rows < 0:
        p.error("capture tokens must be positive and weight rows nonnegative")
    if args.capture_output and args.capture_output.exists():
        p.error("capture output already exists; refusing overwrite")
    variants = list(dict.fromkeys(["bf16"] + args.variants))
    needs_calibration = any(v.startswith("ptq:") for v in variants)
    needs_selection = "ptq:selected" in variants
    selection_output = args.output.with_suffix(".selection.pt")
    if needs_selection and selection_output.exists():
        p.error("frozen selection output already exists; refusing overwrite")
    if args.test_offset < 0:
        p.error("test offset must be nonnegative")
    if needs_selection and (args.calibration_validation_count < 2 or args.calibration_validation_max_tokens < args.calibration_validation_count):
        p.error("PTQ selection requires >=2 independent train-validation images and token budget >= image count")
    if needs_calibration and (args.calibration_count < 2 or args.calibration_max_tokens < args.calibration_count):
        p.error("PTQ calibration requires >=2 train images and token budget >= image count")
    for variant in variants:
        if variant not in ("bf16", "mxfp8") and variant not in [f"ptq:{s}" for s in PTQ_POLICIES] + [f"native:{s}" for s in POLICIES] + [f"emulated:{s}:{o}" for s in POLICIES for o in ("w", "x", "both")]:
            p.error(f"unknown variant {variant}")
    gallery_images, gallery_labels, gallery_ids = load_images(args.dataset, True, args.gallery_count, args.seed, args.image_size)
    excluded_test_ids = []
    images, labels, ids = load_images(args.dataset, False, args.test_count, args.seed + 1, args.image_size, offset=args.test_offset, excluded_prefix=excluded_test_ids)
    calibration_images, calibration_ids = None, []
    validation_images, validation_ids = None, []
    if needs_calibration:
        calibration_images, _, calibration_ids = load_images(args.dataset, True, args.calibration_count, args.seed + 2, args.image_size, exclude=gallery_ids)
    if needs_selection:
        validation_images, _, validation_ids = load_images(args.dataset, True, args.calibration_validation_count, args.seed + 3, args.image_size, exclude=gallery_ids + calibration_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    with args.output.open("x") as stream, torch.inference_mode():
        def save(record):
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps(record, allow_nan=False), flush=True)
        save({"kind": "dataset_labels", "heldout_test_indices": ids, "heldout_labels": labels.tolist(), "test_offset": args.test_offset, "excluded_test_prefix_indices": excluded_test_ids, "gallery_train_indices": gallery_ids, "gallery_labels": gallery_labels.tolist(), "calibration_train_indices": calibration_ids, "calibration_validation_train_indices": validation_ids})
        save({"kind": "environment", "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "checkpoint_sha256": checkpoint_sha256(args.checkpoint), "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(args.device), "gallery_train_indices": gallery_ids, "heldout_test_indices": ids, "image_tensor_sha256": hashlib.sha256(images.numpy().tobytes()).hexdigest(), "scope": "pretrained frozen image backbone; CIFAR heldout cosine 1-NN with fixed BF16 train gallery; no training/backward qualification; no ImageNet claim", "preprocessing": "PIL bicubic direct resize, RGB, ImageNet mean/std; no random augmentation"})
        calibration, calibration_validation = None, None
        if needs_calibration:
            started = time.perf_counter()
            calibration_model = load_dinov3_vits16(checkpoint=args.checkpoint, device=args.device)
            calibration = capture_calibration_inputs(calibration_model, calibration_images, args.batch_size, args.device, args.calibration_max_tokens)
            if needs_selection:
                validation_started = time.perf_counter()
                calibration_validation = capture_calibration_inputs(calibration_model, validation_images, args.batch_size, args.device, args.calibration_validation_max_tokens, retain_activations=True, retain_quantized=True)
                torch.cuda.synchronize(args.device)
                save({"kind": "calibration_validation", "elapsed_s": time.perf_counter() - validation_started, "split": "train", "image_ids": validation_ids, "fit_and_gallery_disjoint": not bool(set(validation_ids) & set(gallery_ids + calibration_ids)), "test_used": False, "module_count": len(calibration_validation), "modules": {name: data["metadata"] for name, data in calibration_validation.items()}, "candidates_prespecified": SELECTION_CANDIDATES})
                # Freeze all per-layer choices before ANY heldout model forward,
                # including the BF16 reference and compilation checks.
                for index, (name, data) in enumerate(calibration.items()):
                    selection_started = time.perf_counter()
                    weight = calibration_model.get_submodule(name).weight.detach().cpu()
                    outer = None
                    if args.outer_mode == "exact_four_over_six":
                        maximum = weight.float().abs().max()
                        outer = torch.where(maximum == 0, 1., (maximum / 1536).clamp_min(2. ** -126))
                    data["selected_weight"], data["selection"] = select_calibrated_weight(weight, data["second_moments"], calibration_validation[name]["activations"], outer_scale=outer, decoded_x=calibration_validation[name]["quantized_activations"])
                    data["selection"]["validation_metadata"] = calibration_validation[name]["metadata"]
                    save({"kind": "layer_policy_selection", "module": name, "selection": data["selection"], "validation_train_indices": validation_ids, "validation": calibration_validation[name]["metadata"], "elapsed_s": time.perf_counter() - selection_started, "frozen_before_any_heldout_forward": True})
                    print(f"selected {index + 1}/{len(calibration)} {name}: {data['selection']['selected']}", file=sys.stderr, flush=True)
                save({"kind": "frozen_selection_artifact", **save_frozen_selection(selection_output, calibration, checkpoint_hash=checkpoint_sha256(args.checkpoint), fit_ids=calibration_ids, validation_ids=validation_ids, outer_mode=args.outer_mode)})
            torch.cuda.synchronize(args.device)
            save({"kind": "calibration", "elapsed_s_including_bf16_model_load_capture_and_moments": time.perf_counter() - started, "checkpoint_sha256": checkpoint_sha256(args.checkpoint), "split": "train", "image_ids": calibration_ids, "gallery_disjoint": not bool(set(calibration_ids) & set(gallery_ids)), "heldout_used": False, "policy_selection_on_heldout": False, "module_count": len(calibration), "modules": {name: data["metadata"] for name, data in calibration.items()}, "scope": "frozen matched BF16 checkpoint activations; offline local block16 uncentered second moments; no runtime hooks/calibration; not full GPTQ"})
            del calibration_model
            torch.cuda.empty_cache()
        for variant in variants:
            torch.compiler.reset()
            model = load_dinov3_vits16(checkpoint=args.checkpoint, device=args.device)
            conversion_started = time.perf_counter()
            model, configs = convert(model, variant, args.batch_size, args.image_size, args.x_rows, args.w_rows, args.outer_mode, calibration, calibration_validation)
            torch.cuda.synchronize(args.device)
            conversion_elapsed = time.perf_counter() - conversion_started
            model.eval()
            if variant == "bf16" and args.capture_linear:
                # Both hooks removed before compilation, gallery extraction,
                # heldout evaluation, candidate modes, and steady-state timing.
                save({"kind": "linear_capture", **save_linear_capture(model, args, gallery_images, images, gallery_ids, ids)})
            compile_check = None
            if args.compile:
                if variant.startswith("emulated:"):
                    raise ValueError("emulated diagnostic metrics deliberately synchronize; compile only native variants")
                check_images = images[:args.batch_size]
                eager_cls, eager_patch = extract(model, check_images, args.batch_size, args.device)
                model.forward_features = torch.compile(model.forward_features, fullgraph=True, dynamic=False)
                compiled_cls, compiled_patch = extract(model, check_images, args.batch_size, args.device)
                compile_check = {"cls": feature_metrics(compiled_cls, eager_cls), "patch": feature_metrics(compiled_patch, eager_patch)}
            if variant == "bf16":
                gallery, _ = extract(model, gallery_images, args.batch_size, args.device)
            started = time.perf_counter()
            cls, patch = extract(model, images, args.batch_size, args.device)
            task = task_metrics(cls, gallery, gallery_labels, labels, None if variant == "bf16" else ref_task["predictions"])
            if variant == "bf16":
                reference_cls, reference_patch, ref_task = cls, patch, task
            batch = images[:args.batch_size].to(device=args.device, dtype=torch.bfloat16)
            if len(batch) < args.batch_size:
                batch = torch.cat([batch, batch[-1:].expand(args.batch_size - len(batch), -1, -1, -1)])
            timing = None if variant.startswith("emulated:") else benchmark(model, batch, args.warmup, args.timing_repeats, args.compile)
            diagnostics = {name: {"weight": module.weight_metrics, "first_batch_activation": module.activation_metrics} for name, module in model.named_modules() if isinstance(module, EmulatedLinear)}
            if variant.startswith("ptq:"):
                save({"kind": "ptq_setup", "variant": variant, "conversion_elapsed_s_not_inference": conversion_elapsed, "calibration_train_indices": calibration_ids, "activation_policy": "native current exact amax/2688 outer and amax block16; identical for all PTQ candidates", "weight_outer_mode": args.outer_mode, "runtime": "production packed NVFP4-W / dynamic BF16-X; no emulation"})
            save({"kind": "result", "variant": variant, "outer_mode": args.outer_mode if variant not in ("bf16", "mxfp8") else None, "outer_divisor": (1536 if args.outer_mode == "exact_four_over_six" else 2688) if variant not in ("bf16", "mxfp8") else None, "emulated": variant.startswith("emulated:"), "emulation_contract": "FP32 division, BF16 decoded operands/GEMM; format intervention, not bitexact native oracle" if variant.startswith("emulated:") else None, "cls": feature_metrics(cls, reference_cls), "patch": feature_metrics(patch, reference_patch), "task": task, "timing": timing, "compiled_vs_eager_first_batch": compile_check, "diagnostic_elapsed_s_not_benchmark": time.perf_counter() - started, "configs": configs, "tensor_quantization_diagnostics": diagnostics})
            del model
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
