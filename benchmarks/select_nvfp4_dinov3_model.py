"""Label-free, train-only whole-model selection for native DINOv3 W4A4.

One prespecified block-coordinate pass, then an independent compiled guard
against global max/MSE controls. No training, test-set policy fitting, decoded
runtime operands, or claims of general PTQ optimality.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch
from torch import nn
from rtx.autotune.accuracy import coordinate_search, select_guard

from benchmarks.dinov3_regression import DEFAULT_CHECKPOINT, DinoBiasAdapter, checkpoint_sha256, load_dinov3_vits16
from benchmarks.evaluate_nvfp4_dinov3_cifar import (
    benchmark, capture_calibration_inputs, extract, feature_metrics, load_images,
    move_native_packed, task_metrics,
)

CANDIDATES = (("max", "max", 0.), ("mse", "mse", 0.),
              ("lh0", "local_hessian", 0.), ("lh50", "local_hessian", .5),
              ("lh90", "local_hessian", .9))
PACKED_BUFFERS = ("weight_data", "weight_block_scales", "weight_tensor_scale")


def normalized_feature_score(candidate, reference):
    """Equal-weight CLS/patch SSE, normalized by each teacher feature energy."""
    if len(candidate) != 2 or len(reference) != 2:
        raise ValueError("CLS and patch features required")
    scores = []
    for value, teacher in zip(candidate, reference):
        if value.shape != teacher.shape or not bool(value.isfinite().all()) or not bool(teacher.isfinite().all()):
            raise ValueError("finite matching feature shapes required")
        teacher = teacher.double()
        scores.append(float((value.double() - teacher).square().sum() / teacher.square().sum().clamp_min(1e-30)))
    return .5 * sum(scores)


@torch.inference_mode()
def copy_packed_buffers(core, buffers):
    """Validate all metadata before in-place copies; module identity is stable."""
    if set(buffers) != set(PACKED_BUFFERS):
        raise ValueError("packed buffer names mismatch")
    for name in PACKED_BUFFERS:
        source, target = buffers[name], getattr(core, name)
        if source.shape != target.shape or source.dtype != target.dtype:
            raise ValueError(f"{name}: packed buffer shape/dtype mismatch")
    for name in PACKED_BUFFERS:
        target = getattr(core, name)
        target.copy_(buffers[name].to(device=target.device))


def packed_buffers(packed):
    from rtx.formats.nvfp4 import nvfp4_tensor_scale
    return {"weight_data": packed.qdata.detach().cpu().clone(),
            "weight_block_scales": packed.scale.detach().cpu().clone(),
            "weight_tensor_scale": nvfp4_tensor_scale(packed).detach().cpu().clone()}


def apply_state(cores, packed, state):
    for name, core in cores.items():
        block = ".".join(name.split(".")[:2])
        copy_packed_buffers(core, packed[name][state[block]])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=Path("/home/achiya/vs_code_main/cifar_example/data"))
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gallery-count", type=int, default=100)
    parser.add_argument("--fit-count", type=int, default=64)
    parser.add_argument("--fit-max-tokens", type=int, default=4096)
    parser.add_argument("--search-count", type=int, default=32)
    parser.add_argument("--guard-count", type=int, default=32)
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument("--test-offset", type=int, default=1500)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timing-repeats", type=int, default=10)
    args = parser.parse_args()
    if min(args.batch_size, args.gallery_count, args.fit_count, args.search_count, args.guard_count, args.test_count, args.warmup, args.timing_repeats) < 1 or args.fit_count < 2 or args.fit_max_tokens < args.fit_count or args.image_size < 16 or args.image_size % 16 or args.test_offset < 0:
        parser.error("positive counts, >=2 fit images, fit token budget >= image count, nonnegative offset, image size divisible by16 required")
    artifact_path = args.output.with_suffix(".selection.pt")
    if args.output.exists() or artifact_path.exists():
        parser.error("output/artifact exists; refusing overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    gallery_images, gallery_labels, gallery_ids = load_images(args.dataset, True, args.gallery_count, args.seed, args.image_size)
    fit_images, _, fit_ids = load_images(args.dataset, True, args.fit_count, args.seed + 2, args.image_size, exclude=gallery_ids)
    search_images, _, search_ids = load_images(args.dataset, True, args.search_count, args.seed + 3, args.image_size, exclude=gallery_ids + fit_ids)
    guard_images, _, guard_ids = load_images(args.dataset, True, args.guard_count, args.seed + 4, args.image_size, exclude=gallery_ids + fit_ids + search_ids)
    checkpoint_hash = checkpoint_sha256(args.checkpoint)
    with args.output.open("x") as stream, torch.inference_mode():
        def save(record):
            stream.write(json.dumps(record, allow_nan=False) + "\n")
            stream.flush()
            print(json.dumps(record, allow_nan=False), flush=True)
        environment = {"kind": "environment", "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "checkpoint_sha256": checkpoint_hash, "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "torch": torch.__version__, "cuda": torch.version.cuda, "device": torch.cuda.get_device_name(args.device), "gallery_train_indices": gallery_ids, "fit_train_indices": fit_ids, "search_train_indices": search_ids, "guard_train_indices": guard_ids, "scope": "label-free train-only whole-model native W4A4 block-coordinate selection; no training qualification", "activation_policy": "native current exact amax/2688 outer; native amax/direct block16; identical all candidates"}
        save(environment)
        started = time.perf_counter()
        teacher = load_dinov3_vits16(checkpoint=args.checkpoint, device=args.device)
        fit = capture_calibration_inputs(teacher, fit_images, args.batch_size, args.device, args.fit_max_tokens)
        search_reference = extract(teacher, search_images, args.batch_size, args.device)
        from rtx.nvfp4_calibration import calibrate_nvfp4_weight
        import rtx
        packed, shapes = {}, {}
        for index, (name, data) in enumerate(fit.items()):
            weight = teacher.get_submodule(name).weight.detach().cpu()
            shapes[name] = tuple(weight.shape)
            packed[name] = {}
            for label, policy, alpha in CANDIDATES:
                value = calibrate_nvfp4_weight(weight, policy, second_moments=data["second_moments"], hessian_shrinkage=alpha)
                packed[name][label] = packed_buffers(value)
            print(f"packed {index + 1}/{len(fit)} {name}", file=sys.stderr, flush=True)
        save({"kind": "fit", "elapsed_s_including_teacher_capture_and_all_candidate_packing": time.perf_counter() - started, "module_count": len(fit), "modules": {name: data["metadata"] for name, data in fit.items()}, "candidates": CANDIDATES})
        model = load_dinov3_vits16(checkpoint=args.checkpoint, device=args.device)
        cores = {}
        from rtx.formats.nvfp4 import make_nvfp4_tensor
        for name, options in packed.items():
            source = model.get_submodule(name)
            buffers = options["max"]
            value = make_nvfp4_tensor(buffers["weight_data"], buffers["weight_block_scales"], buffers["weight_tensor_scale"], shapes[name])
            value = move_native_packed(value, args.device)
            core = rtx.NVFP4Linear(source.in_features, source.out_features, packed_weight=value, scaling="current", weight_prequant_config=rtx.NVFP4WeightPrequantConfig(quant_x=rtx.NVFP4QuantConfig(block_scale_selection="amax", scale_reciprocal="direct")), autotune="off")
            parent, child = name.rsplit(".", 1)
            model.get_submodule(parent)._modules[child] = DinoBiasAdapter(core, source)
            cores[name] = core
        model.eval()
        blocks = list(dict.fromkeys(".".join(name.split(".")[:2]) for name in packed))
        if len(blocks) != 12 or len(cores) != 48 or any(sum(name.startswith(block + ".") for name in cores) != 4 for block in blocks):
            raise ValueError("expected12blocks each with4linears")
        def evaluate(state):
            apply_state(cores, packed, state)
            return normalized_feature_score(extract(model, search_images, args.batch_size, args.device), search_reference)
        def progress(record):
            save({"kind": "search_candidate", **record})
        selected_state, search_score, search_records = coordinate_search(blocks, [value[0] for value in CANDIDATES], evaluate, progress=progress)
        states = {"max": {block: "max" for block in blocks}, "mse": {block: "mse" for block in blocks}, "selected": selected_state}
        torch.compiler.reset()
        teacher.forward_features = torch.compile(teacher.forward_features, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
        guard_reference = extract(teacher, guard_images, args.batch_size, args.device)
        torch.compiler.reset()
        model.forward_features = torch.compile(model.forward_features, fullgraph=True, dynamic=False, options={"triton.cudagraphs": False})
        guard_scores = {}
        for label, state in states.items():
            apply_state(cores, packed, state)
            guard_scores[label] = normalized_feature_score(extract(model, guard_images, args.batch_size, args.device), guard_reference)
            save({"kind": "guard_candidate", "candidate": label, "score": guard_scores[label], "state": state, "mode": "compiled fullgraph/static candidate and BF16 teacher"})
        choice = select_guard(guard_scores)
        frozen_state = states[choice]
        artifact = {"schema": "rtx.dinov3.whole_model_selection.v1", "environment": environment, "candidate_buffers": packed, "logical_shapes": shapes, "scale_layout": "row_major", "fit_metadata": {name: data["metadata"] for name, data in fit.items()}, "search_records": search_records, "search_state": selected_state, "search_score": search_score, "guard_scores": guard_scores, "guard_choice": choice, "frozen_state": frozen_state, "frozen_before_any_heldout_access": True, "normalization": "0.5*(CLS SSE/teacher CLS energy + patch SSE/teacher patch energy); float64 reduction"}
        with artifact_path.open("xb") as artifact_stream:
            torch.save(artifact, artifact_stream)
        save({"kind": "frozen_selection_artifact", "path": str(artifact_path), "sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(), "guard_scores": guard_scores, "guard_choice": choice, "frozen_state": frozen_state, "search_state": selected_state, "frozen_before_any_heldout_access": True})
        # First test data access occurs only after policy and guard are frozen.
        excluded = []
        images, labels, ids = load_images(args.dataset, False, args.test_count, args.seed + 1, args.image_size, offset=args.test_offset, excluded_prefix=excluded)
        save({"kind": "dataset_labels", "heldout_test_indices": ids, "heldout_labels": labels.tolist(), "gallery_train_indices": gallery_ids, "gallery_labels": gallery_labels.tolist(), "test_offset": args.test_offset, "excluded_test_prefix_indices": excluded})
        gallery, _ = extract(teacher, gallery_images, args.batch_size, args.device)
        reference = extract(teacher, images, args.batch_size, args.device)
        ref_task = task_metrics(reference[0], gallery, gallery_labels, labels)
        batch = images[:args.batch_size].to(device=args.device, dtype=torch.bfloat16)
        if len(batch) < args.batch_size:
            batch = torch.cat((batch, batch[-1:].expand(args.batch_size - len(batch), -1, -1, -1)))
        save({"kind": "result", "variant": "bf16", "cls": feature_metrics(reference[0], reference[0]), "patch": feature_metrics(reference[1], reference[1]), "task": ref_task, "timing": benchmark(teacher, batch, args.warmup, args.timing_repeats, True)})
        evaluated = {}
        for label, state in (("max", states["max"]), ("mse", states["mse"]), ("coordinate", selected_state), ("selected", frozen_state)):
            key = tuple(state.items())
            alias = evaluated.get(key)
            if alias is None:
                apply_state(cores, packed, state)
                candidate = extract(model, images, args.batch_size, args.device)
                result = {"cls": feature_metrics(candidate[0], reference[0]), "patch": feature_metrics(candidate[1], reference[1]), "task": task_metrics(candidate[0], gallery, gallery_labels, labels, ref_task["predictions"]), "timing": benchmark(model, batch, args.warmup, args.timing_repeats, True)}
                evaluated[key] = (label, result)
            else:
                _, result = alias
            save({"kind": "result", "variant": f"ptq:{label}", "emulated": False, **result, "state": state, "guard_choice": choice if label == "selected" else None, "selection_stage": "raw train-search coordinate state" if label == "coordinate" else ("final compiled train-guard winner" if label == "selected" else "global baseline"), "alias_of": f"ptq:{alias[0]}" if alias else None, "alias_timing_reused": alias is not None, "compile_options": {"triton.cudagraphs": False}})


if __name__ == "__main__":
    main()
