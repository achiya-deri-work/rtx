"""Localize DINO compile drift without treating it as a performance result.

Three distinct checks: original full-model eager/compiled; selected modules on
identical captured eager inputs; and a prefix trace returning block boundaries.
Returning intermediates changes graph liveness/fusion, so trace results are
localization evidence, NOT proof of equivalence for the original full graph.
No tensor used as a reference aliases compiled CUDA-graph output buffers.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils._pytree import tree_flatten, tree_map

from benchmarks.dinov3_regression import DEFAULT_CHECKPOINT, checkpoint_sha256, load_dinov3_vits16
from benchmarks.evaluate_nvfp4_dinov3_cifar import OUTER_MODES, convert, feature_metrics, load_images


def clone_tree(value, cpu=False):
    return tree_map(lambda x: x.detach().to("cpu").clone() if cpu and isinstance(x, torch.Tensor) else x.detach().clone() if isinstance(x, torch.Tensor) else x, value)


def compare_trees(candidate, reference):
    c, cs = tree_flatten(candidate)
    r, rs = tree_flatten(reference)
    if cs != rs:
        raise ValueError("different output tree structures")
    records = []
    for index, (actual, expected) in enumerate(zip(c, r)):
        if isinstance(expected, torch.Tensor):
            if not isinstance(actual, torch.Tensor) or actual.shape != expected.shape:
                raise ValueError("different output tensor shapes")
            if expected.is_floating_point():
                if not bool(actual.isfinite().all()) or not bool(expected.isfinite().all()):
                    raise ValueError("nonfinite output")
                records.append({"leaf": index, "shape": list(expected.shape), "dtype": str(expected.dtype), "exact_equal": bool(torch.equal(actual, expected)), **feature_metrics(actual, expected)})
            else:
                records.append({"leaf": index, "exact_equal": bool(torch.equal(actual, expected))})
    return records


class PrefixTrace(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):
        # Faithfully follows forward_features_list's single-image-size path.
        x, (height, width) = self.model.prepare_tokens_with_masks(images, None)
        outputs = [x]
        for block in self.model.blocks:
            rope = self.model.rope_embed(H=height, W=width)
            x = block([x], [rope])[0]
            outputs.append(x)
        outputs.extend(self.model.rope_embed(H=height, W=width))
        return tuple(outputs)


@torch.inference_mode()
def diagnose(args, save):
    if args.emulate_precision_casts:
        # Deliberate numerical control, never silently enabled in benchmark.
        # Preserves eager BF16 cast boundaries that Inductor otherwise elides.
        torch._inductor.config.emulate_precision_casts = True
    images, _, ids = load_images(args.dataset, False, args.batch_size, args.seed + 1, args.image_size)
    images = images.to(device=args.device, dtype=torch.bfloat16)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    save({"kind": "environment", "torch": torch.__version__, "cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(args.device), "checkpoint_sha256": checkpoint_sha256(args.checkpoint), "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "heldout_indices": ids, "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}, "scope": "numerical localization; no timings; tree references cloned onto CPU immediately; prefix trace changes graph observability"})
    for variant in args.variants:
        torch.compiler.reset()
        model = load_dinov3_vits16(checkpoint=args.checkpoint, device=args.device).eval()
        model, _ = convert(model, variant, args.batch_size, args.image_size, args.x_rows, args.w_rows, args.outer_mode)
        if any(module.training for module in model.modules()):
            raise RuntimeError("training submodule in eval diagnostic")
        captures = {}
        handles = []
        for name in args.modules:
            module = model.get_submodule(name)

            def hook(_module, inputs, kwargs, output, name=name):
                if name not in captures:
                    captures[name] = (clone_tree(inputs), clone_tree(kwargs), clone_tree(output, cpu=True))

            handles.append(module.register_forward_hook(hook, with_kwargs=True))
        try:
            eager = clone_tree(model.forward_features(images), cpu=True)
        finally:
            for handle in handles:
                handle.remove()
        eager_repeat = clone_tree(model.forward_features(images), cpu=True)
        save({"kind": "full_eager_repeat", "variant": variant, "outputs": compare_trees(eager_repeat, eager)})
        # Capture output clones before another invocation can recycle storage.
        compiled = torch.compile(model.forward_features, fullgraph=True, dynamic=False)
        compiled_output = clone_tree(compiled(images), cpu=True)
        save({"kind": "original_fullgraph", "variant": variant, "outputs": compare_trees(compiled_output, eager)})
        compiled_repeat = clone_tree(compiled(images), cpu=True)
        save({"kind": "full_compiled_repeat", "variant": variant, "outputs": compare_trees(compiled_repeat, compiled_output)})
        del compiled
        for name in args.modules:
            inputs, kwargs, reference = captures[name]
            torch.compiler.reset()
            module = model.get_submodule(name)
            isolated_eager = clone_tree(module(*clone_tree(inputs), **clone_tree(kwargs)), cpu=True)
            isolated = torch.compile(module, fullgraph=True, dynamic=False)
            isolated_compiled = clone_tree(isolated(*clone_tree(inputs), **clone_tree(kwargs)), cpu=True)
            save({"kind": "same_input_module", "variant": variant, "module": name, "captured_vs_isolated_eager": compare_trees(isolated_eager, reference), "compiled_vs_isolated_eager": compare_trees(isolated_compiled, isolated_eager)})
            del isolated
        del captures
        torch.compiler.reset()
        trace = PrefixTrace(model).eval()
        eager_trace = clone_tree(trace(images), cpu=True)
        compiled_trace = torch.compile(trace, fullgraph=True, dynamic=False)
        actual_trace = clone_tree(compiled_trace(images), cpu=True)
        names = ["prepared_tokens"] + [f"blocks.{index}" for index in range(len(model.blocks))] + ["rope.sin", "rope.cos"]
        for name, candidate, reference in zip(names, actual_trace, eager_trace):
            save({"kind": "observable_prefix_trace", "variant": variant, "boundary": name, "outputs": compare_trees(candidate, reference)})
        del compiled_trace, trace, model
        torch.cuda.empty_cache()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--variants", nargs="+", default=["bf16", "native:amax", "native:four_over_six"])
    p.add_argument("--outer-mode", choices=OUTER_MODES, default="exact")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=20261023)
    p.add_argument("--x-rows", type=int, default=8)
    p.add_argument("--w-rows", type=int, default=8)
    p.add_argument("--emulate-precision-casts", action="store_true", help="Explicit Inductor control preserving eager low-precision downcast/upcast boundaries")
    p.add_argument("--modules", nargs="+", default=["patch_embed", "rope_embed", "blocks.0.norm1", "blocks.0.attn.qkv", "blocks.0.attn.proj", "blocks.0.norm2", "blocks.0.mlp.fc1", "blocks.0.mlp.act", "blocks.0.mlp.fc2", "blocks.1.norm1", "blocks.1.attn.qkv", "blocks.11.mlp.fc2", "norm"])
    args = p.parse_args()
    if min(args.batch_size, args.image_size, args.x_rows, args.w_rows) < 1 or args.image_size % 16:
        p.error("positive dimensions; image size divisible by 16 required")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        def save(record):
            text = json.dumps(record, allow_nan=False)
            stream.write(text + "\n")
            stream.flush()
            print(text, flush=True)
        diagnose(args, save)


if __name__ == "__main__":
    main()
