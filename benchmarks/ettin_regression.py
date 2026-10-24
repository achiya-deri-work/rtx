"""Pinned Ettin masked-language-model fidelity smoke study, not a task benchmark.

Optional dependencies: transformers==4.57.6, huggingface-hub, safetensors.
Checkpoint downloads use the external Hugging Face cache, never the repository.
Run: python -m benchmarks.ettin_regression --output PATH
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import replace
import gc
import hashlib
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

import rtx

MODEL_ID = "jhu-clsp/ettin-encoder-32m"
REVISION = "1b8ba06455dd44f80fc9c1ca9e22806157a57379"
TEXTS = (
    "The astronomer measured the distance between the planet and its nearest star.",
    "A small library opened near the railway station on Monday morning.",
    "Water freezes when its temperature falls below zero degrees Celsius at ordinary pressure.",
    "The programmer checked the array bounds before reading the final element.",
    "The committee postponed its decision until every member had reviewed the evidence.",
    "A patient gardener planted tomatoes and watered the seedlings every evening.",
    "The experiment compared two treatments while keeping the other conditions constant.",
    "She returned the borrowed book and asked for another historical novel.",
    "The compiler can remove a calculation when its result is never used.",
    "Rain continued through the night, leaving the narrow streets quiet and wet.",
    "The invoice included the original price, a discount, and the applicable tax.",
    "Although the first explanation seemed plausible, the measurements contradicted it.",
    "A triangle has three sides, and the angles of a Euclidean triangle sum to a straight angle.",
    "The musician adjusted the tuning before the orchestra began its final rehearsal.",
    "The database transaction failed, so none of its changes became visible to readers.",
    "The train arrived late because engineers were repairing a bridge outside the city.",
)
MODES = ("bf16", "mxfp8_dynamic", "mxfp8_ptq", "nvfp4_jit", "nvfp4_delayed",
         "nvfp4_current", "nvfp4_block", "nvfp4_block_four_over_six", "nvfp4_ptq", "mxfp8_mlp")
CORPUS_ID = "Salesforce/wikitext"
CORPUS_REVISION = "b08601e04326c79dfdd32d625aee71d232d685c3"


def corpus_texts(split, count, offset=0):
    """Deterministic public corpus selection, no test-label-based filtering."""
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    if split not in ("test", "validation", "train") or count <= 0 or offset < 0:
        raise ValueError("invalid corpus split/count")
    path = hf_hub_download(CORPUS_ID, f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
                          revision=CORPUS_REVISION, repo_type="dataset")
    texts = [s.strip() for s in pq.read_table(path, columns=["text"])["text"].to_pylist()
             if len(s.strip()) >= 80 and not s.lstrip().startswith("=")]
    if offset + count > len(texts):
        raise ValueError("requested more paragraphs than available")
    return tuple(texts[offset:offset + count])


def quality_summary(rows, mode):
    """Coarse prospective regression guards, not task-accuracy certification."""
    tokens = sum(r['masked_tokens'] for r in rows)
    if tokens <= 0:
        raise ValueError("quality assessment requires masked targets")
    weighted = lambda key: sum(r[key]*r['masked_tokens'] for r in rows)/tokens
    fp4 = mode.startswith('nvfp4')
    limits = dict(max_nll_increase=.5 if fp4 else .1,
                  min_mean_hidden_cosine=.85 if fp4 else .97,
                  minimum_masked_tokens=1000)
    increase = weighted('masked_nll') - weighted('reference_nll')
    cosine = sum(r['hidden_mean_cosine'] for r in rows)/len(rows)
    finite = all(r['finite'] for r in rows)
    passed = (finite and tokens >= limits['minimum_masked_tokens']
              and increase <= limits['max_nll_increase']
              and cosine >= limits['min_mean_hidden_cosine'])
    import numpy as np
    rng = np.random.default_rng(4802)
    indices = rng.integers(0, len(rows), (5000, len(rows)))
    weights = np.array([r['masked_tokens'] for r in rows])
    differences = np.array([r['masked_nll']-r['reference_nll'] for r in rows])
    estimates = (weights[indices]*differences[indices]).sum(1)/weights[indices].sum(1)
    interval = np.quantile(estimates, [.025, .975]).tolist()
    return dict(mode=mode, passed=passed, finite=finite, masked_tokens=tokens,
                nll_increase=increase, mean_hidden_cosine=cosine, limits=limits,
                nll_increase_ci95=interval, bootstrap_unit="four-paragraph batch",
                bootstrap_replicates=5000, bootstrap_seed=4802)


class EttinRecomputedMLP(nn.Module):
    """Research adapter retaining original Wi/Wo parameter names and objects."""
    def __init__(self, source):
        super().__init__()
        if source.config.hidden_activation != "gelu" or source.drop.p != 0:
            raise ValueError("adapter requires exact GELU and zero dropout")
        if source.Wi.bias is not None or source.Wo.bias is not None:
            raise ValueError("adapter requires bias-free projections")
        self.Wi, self.Wo = source.Wi, source.Wo

    def forward(self, x):
        # ModernBERT activates the first half of Wi. RTX gate/up ordering agrees.
        return rtx.mxfp8_mlp(x, self.Wi.weight, self.Wo.weight, activation="gelu")


def backbone_linear(module, name):
    return name.startswith("model.layers.") and isinstance(module, nn.Linear)


def convert(model, mode):
    if mode == "bf16":
        return model
    if mode == "mxfp8_mlp":
        for layer in model.model.layers:
            layer.mlp = EttinRecomputedMLP(layer.mlp)
        return rtx.convert_to_mxfp8_training(model, config=rtx.MXFP8TrainingConfig(autotune="off"),
            module_filter_fn=lambda module, name: backbone_linear(module, name) and ".attn." in name)
    if mode == "mxfp8_dynamic":
        return rtx.convert_to_mxfp8_training(model, config=rtx.MXFP8TrainingConfig(autotune="off"),
                                            module_filter_fn=backbone_linear)
    if mode == "mxfp8_ptq":
        return rtx.quantize_(model, rtx.MXFP8WeightOnlyConfig(autotune="off"),
                             filter_fn=backbone_linear)
    if mode == "nvfp4_ptq":
        return rtx.quantize_(model, rtx.NVFP4WeightOnlyConfig(autotune="off"),
                             filter_fn=backbone_linear)
    scaling = {"nvfp4_jit": "jit_row_region", "nvfp4_delayed": "delayed",
               "nvfp4_current": "current", "nvfp4_block": "block",
               "nvfp4_block_four_over_six": "block"}[mode]
    config = rtx.NVFP4TrainingConfig(scaling=scaling, autotune="off")
    if mode.endswith("four_over_six"):
        from rtx.configs import NVFP4DynamicConfig, NVFP4QuantConfig
        config = replace(config, backend="materialized", dynamic_config=NVFP4DynamicConfig(
            quant=NVFP4QuantConfig(block_scale_selection="four_over_six")))
    return rtx.convert_to_nvfp4_training(model, config=config, module_filter_fn=backbone_linear)


def batches(tokenizer, seed, texts=TEXTS, batch_size=4, max_length=None):
    generator = torch.Generator().manual_seed(seed)
    if batch_size <= 0 or (max_length is not None and max_length < 3):
        raise ValueError("positive batch size and max_length >= 3 required")
    options = (dict(padding="max_length", truncation=True, max_length=max_length)
               if max_length else dict(padding=True))
    for start in range(0, len(texts), batch_size):
        encoded = tokenizer(list(texts[start:start + batch_size]), return_tensors="pt",
                            return_special_tokens_mask=True, **options)
        eligible = encoded.pop("special_tokens_mask").eq(0) & encoded["attention_mask"].bool()
        chosen = (torch.rand(eligible.shape, generator=generator) < .15) & eligible
        for row in range(len(chosen)):
            if not chosen[row].any():
                if not eligible[row].any():
                    raise ValueError("text must contain at least one non-special token")
                chosen[row, eligible[row].nonzero()[0, 0]] = True
        labels = encoded["input_ids"].clone()
        labels[~chosen] = -100
        encoded["input_ids"][chosen] = tokenizer.mask_token_id
        yield encoded, labels


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[4700, 4701, 4702])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--compile", action="store_true", help="Compile the complete model fullgraph/static")
    parser.add_argument("--corpus", choices=("fixture", "wikitext"), default="fixture")
    parser.add_argument("--paragraphs", type=int, default=128)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--paragraph-offset", type=int, default=0)
    parser.add_argument("--require-quality", action="store_true",
                        help="Enforce documented coarse model regression guards")
    parser.add_argument("--summarize", type=Path, help="Summarize existing JSONL on CPU; no model loading")
    args = parser.parse_args()
    if args.summarize:
        grouped = {}
        for line in args.summarize.read_text().splitlines():
            record = json.loads(line)
            if 'mode' in record and 'masked_tokens' in record:
                grouped.setdefault(record['mode'], []).append(record)
        if not grouped:
            parser.error("no inference records found")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open('x') as stream:
            json.dump({mode: quality_summary(rows, mode) for mode, rows in grouped.items()}, stream, indent=2)
        return
    from transformers import AutoModelForMaskedLM, AutoTokenizer, __version__ as hf_version
    from rtx.autotune.dataset import _source_snapshot
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    texts = TEXTS if args.corpus == "fixture" else corpus_texts("test", args.paragraphs, args.paragraph_offset)
    original = AutoModelForMaskedLM.from_pretrained(MODEL_ID, revision=REVISION,
        dtype=torch.bfloat16, attn_implementation="sdpa", reference_compile=False).cuda().eval()
    inventory = [(n, list(m.weight.shape)) for n, m in original.named_modules() if backbone_linear(m, n)]
    assert len(inventory) == 40
    inputs = [(seed, index, x, y) for seed in args.seeds
              for index, (x, y) in enumerate(batches(tokenizer, seed, texts, max_length=args.max_length))]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    references = {}
    with args.output.open("x") as stream:
        def emit(row):
            stream.write(json.dumps(row) + "\n"); stream.flush()
        emit(dict(metadata=dict(model=MODEL_ID, revision=REVISION, torch=torch.__version__,
            transformers=hf_version, gpu=torch.cuda.get_device_name(), source=_source_snapshot(),
            text_sha256=hashlib.sha256("\n".join(texts).encode()).hexdigest(),
            corpus=args.corpus, corpus_revision=CORPUS_REVISION if args.corpus == "wikitext" else None,
            paragraphs=len(texts), max_length=args.max_length,
            paragraph_offset=args.paragraph_offset, require_quality=args.require_quality,
            seeds=args.seeds, converted_inventory=inventory,
            compiled=args.compile,
            scope="fixed-text MLM fidelity smoke; not downstream accuracy or training convergence")))
        for mode in dict.fromkeys(["bf16", *args.modes]):
            rows = []
            model = convert(copy.deepcopy(original), mode)
            if args.compile:
                torch._dynamo.reset()
                model = torch.compile(model, fullgraph=True, dynamic=False)
            for seed, index, cpu_inputs, labels in inputs:
                x = {k: v.cuda() for k, v in cpu_inputs.items()}
                masked = labels.ne(-100)
                result = model(**x, output_hidden_states=True)
                logits = result.logits[masked.cuda()].float().cpu()
                hidden = result.hidden_states[-1][x["attention_mask"].bool()].float().cpu()
                targets = labels[masked]
                if mode == "bf16":
                    references[seed, index] = logits, hidden
                ref_logits, ref_hidden = references[seed, index]
                logp, ref_logp = logits.log_softmax(-1), ref_logits.log_softmax(-1)
                row = dict(mode=mode, seed=seed, batch=index, shape=list(labels.shape),
                    masked_tokens=len(targets), finite=bool(logits.isfinite().all() and hidden.isfinite().all()),
                    masked_nll=float(F.cross_entropy(logits, targets)),
                    reference_nll=float(F.cross_entropy(ref_logits, targets)),
                    reference_kl=float((ref_logp.exp() * (ref_logp-logp)).sum(-1).mean()),
                    top1_agreement=float((logits.argmax(-1)==ref_logits.argmax(-1)).float().mean()),
                    masked_accuracy=float((logits.argmax(-1)==targets).float().mean()),
                    hidden_relative_l2=float((hidden-ref_hidden).norm()/ref_hidden.norm().clamp_min(1e-30)),
                    hidden_mean_cosine=float(F.cosine_similarity(hidden, ref_hidden).mean()))
                emit(row)
                rows.append(row)
                print(json.dumps(row), flush=True)
            assessment = quality_summary(rows, mode)
            emit(dict(quality_summary=assessment))
            if args.require_quality and not assessment['passed']:
                raise AssertionError(f"Ettin model regression guard failed: {assessment}")
            del model, result
            gc.collect()


if __name__ == "__main__":
    main()
