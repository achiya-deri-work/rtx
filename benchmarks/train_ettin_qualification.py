"""Matched short Ettin optimizer trajectories and initial gradient comparison.

Not a convergence benchmark. Uses BF16 parameters, gradients and AdamW moments;
cross entropy/reductions are FP32. All policies start from the pinned checkpoint.
"""
from __future__ import annotations

import argparse
import copy
import gc
import json
from pathlib import Path

import torch
from torch import nn
import torch.nn.functional as F

from benchmarks.ettin_regression import MODEL_ID, REVISION, CORPUS_REVISION, batches, corpus_texts, convert

TRAIN_MODES = ("bf16", "mxfp8_dynamic", "nvfp4_jit", "nvfp4_delayed",
               "nvfp4_current", "nvfp4_block", "nvfp4_block_four_over_six", "mxfp8_mlp")


class MaskedLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, labels):
        logits = self.model(input_ids=input_ids, attention_mask=attention_mask).logits
        return F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), labels.reshape(-1))


def gradient_metrics(model, reference=None):
    gradients, metrics = {}, {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None or not bool(parameter.grad.isfinite().all()):
            raise AssertionError(f"missing/nonfinite gradient: {name}")
        value = parameter.grad.detach().float().cpu()
        gradients[name] = value
        if reference is not None:
            ref = reference[name]
            delta = value-ref
            metrics[name] = dict(relative_l2=float(delta.norm()/ref.norm().clamp_min(1e-30)),
                cosine=float(F.cosine_similarity(value.reshape(1, -1), ref.reshape(1, -1))),
                max_abs=float(delta.abs().max()), reference_norm=float(ref.norm()))
    if reference is not None and gradients.keys() != reference.keys():
        raise AssertionError("parameter/gradient coverage differs from BF16")
    return gradients, metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--length", type=int, default=128)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--modes", nargs="+", choices=TRAIN_MODES, default=list(TRAIN_MODES))
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("steps must be positive")
    from transformers import AutoModelForMaskedLM, AutoTokenizer, __version__ as hf_version
    from rtx.autotune.dataset import _source_snapshot
    torch.set_num_threads(4)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    original = AutoModelForMaskedLM.from_pretrained(MODEL_ID, revision=REVISION,
        dtype=torch.bfloat16, attn_implementation="sdpa", reference_compile=False).cuda().eval()
    train_batches = list(batches(tokenizer, 4810, corpus_texts("train", 16), max_length=args.length))
    validation_batches = list(batches(tokenizer, 4811, corpus_texts("validation", 16), max_length=args.length))
    def device_batch(batch):
        inputs, labels = batch
        return {**{k: v.cuda() for k, v in inputs.items()}, "labels": labels.cuda()}
    train_batches = [device_batch(b) for b in train_batches]
    validation_batches = [device_batch(b) for b in validation_batches]
    reference = None
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        def emit(row):
            stream.write(json.dumps(row)+"\n"); stream.flush()
        emit(dict(metadata=dict(model=MODEL_ID, revision=REVISION, corpus_revision=CORPUS_REVISION,
            train_seed=4810, validation_seed=4811, torch=torch.__version__, transformers=hf_version,
            source=_source_snapshot(), compiled=args.compile, steps=args.steps, length=args.length,
            optimizer="AdamW", learning_rate=1e-5, weight_decay=.01,
            scope="short lifecycle/trajectory qualification, not convergence")))
        for mode in dict.fromkeys(["bf16", *args.modes]):
            torch.manual_seed(4812)
            model = convert(copy.deepcopy(original), mode)
            objective = MaskedLoss(model)
            if args.compile:
                torch._dynamo.reset()
                objective = torch.compile(objective, fullgraph=True, dynamic=False)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=.01)
            def validation():
                # Stateful delayed evaluation must not contaminate the training
                # sequence: run on a separate copy with matching current state.
                evaluation = MaskedLoss(copy.deepcopy(model)).eval()
                total = 0.; count = 0
                with torch.no_grad():
                    for batch in validation_batches:
                        n = int(batch['labels'].ne(-100).sum())
                        loss = evaluation(**batch)
                        if not bool(loss.isfinite()):
                            raise AssertionError("nonfinite validation loss")
                        total += float(loss)*n; count += n
                return total/count
            before = validation()
            losses = []
            for step in range(args.steps):
                optimizer.zero_grad(set_to_none=True)
                batch = train_batches[step % len(train_batches)]
                loss = objective(**batch)
                if not bool(loss.isfinite()):
                    raise AssertionError("nonfinite training loss")
                loss.backward()
                if step == 0:
                    gradients, metrics = gradient_metrics(model, reference)
                    if mode == "bf16":
                        reference = gradients
                    emit(dict(mode=mode, gradients=metrics, gradient_count=len(gradients)))
                    del gradients
                else:
                    for name, p in model.named_parameters():
                        if p.requires_grad and (p.grad is None or not bool(p.grad.isfinite().all())):
                            raise AssertionError(f"invalid gradient at step {step}: {name}")
                optimizer.step()
                for parameter, state in optimizer.state.items():
                    if parameter.dtype != torch.bfloat16:
                        raise AssertionError("parameter dtype changed")
                    for key in ("exp_avg", "exp_avg_sq"):
                        if state[key].dtype != torch.bfloat16 or not bool(state[key].isfinite().all()):
                            raise AssertionError("invalid BF16 optimizer state")
                losses.append(float(loss.detach()))
                emit(dict(mode=mode, step=step, loss=losses[-1]))
                print(f"{mode} step={step} loss={losses[-1]:.5f}", flush=True)
            emit(dict(mode=mode, validation_before=before, validation_after=validation(), losses=losses))
            del optimizer, objective, model
            gc.collect()


if __name__ == "__main__":
    main()
