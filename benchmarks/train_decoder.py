"""Resumable byte-level TinyStories convergence training for RTX decoders.

This is intentionally a comparison harness rather than a general trainer. It
feeds identical initialization and step-indexed corpus windows to BF16,
MXFP8, and NVFP4 models, while preserving raw per-step evidence in append-only
JSONL files and atomically replacing checkpoints.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
from typing import Iterable
from urllib.request import urlopen

import numpy as np
import pyarrow.parquet as pq
import torch

from rtx.models import DecoderConfig, DecoderOnlyTransformer, LinearSpec, causal_lm_loss


TINYSTORIES_TRAIN_SHARD = (
    "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/"
    "data/train-00000-of-00004-2d5a1467fff1081b.parquet?download=true"
)
BYTE_VOCAB_SIZE = 257
DOCUMENT_SEPARATOR = 256
FORMAT_VERSION = 1
_STOP_REQUESTED = False


def _request_stop(signum: int, _frame: object) -> None:
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    print(f"STOP signal={signum}; checkpointing after the current step", flush=True)


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    os.replace(temporary, path)


def _atomic_torch_save(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=path.parent, prefix=f".{path.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "wb", dir=destination.parent, prefix=f".{destination.name}.", delete=False
    ) as handle:
        temporary = Path(handle.name)
        with urlopen(url, timeout=60) as response:
            expected = int(response.headers.get("Content-Length", 0))
            copied = 0
            started = time.monotonic()
            while chunk := response.read(8 << 20):
                handle.write(chunk)
                copied += len(chunk)
                elapsed = max(time.monotonic() - started, 1.0e-6)
                total = f"/{expected / (1 << 20):.1f}" if expected else ""
                print(
                    f"DATA {copied / (1 << 20):.1f}{total} MiB "
                    f"{copied / elapsed / (1 << 20):.1f} MiB/s",
                    flush=True,
                )
    os.replace(temporary, destination)


def _token_cache_path(parquet_path: Path) -> Path:
    return parquet_path.with_suffix(".byte_tokens.npy")


def _prepare_corpus(parquet_path: Path, *, url: str) -> tuple[Path, np.ndarray]:
    if not parquet_path.exists():
        print(f"DOWNLOAD {url} -> {parquet_path}", flush=True)
        _download(url, parquet_path)
    token_path = _token_cache_path(parquet_path)
    if (
        not token_path.exists()
        or token_path.stat().st_mtime < parquet_path.stat().st_mtime
    ):
        print(f"TOKENIZE byte-level corpus -> {token_path}", flush=True)
        table = pq.read_table(parquet_path, columns=["text"])
        encoded: list[np.ndarray] = []
        total = 0
        for index, text in enumerate(table.column("text").to_pylist()):
            values = np.frombuffer(str(text).encode("utf-8"), dtype=np.uint8).astype(
                np.uint16
            )
            encoded.extend((values, np.asarray([DOCUMENT_SEPARATOR], dtype=np.uint16)))
            total += values.size + 1
            if index and index % 100_000 == 0:
                print(f"TOKENIZE stories={index:,} tokens={total:,}", flush=True)
        tokens = np.concatenate(encoded)
        with tempfile.NamedTemporaryFile(
            "wb", dir=token_path.parent, prefix=f".{token_path.name}.", delete=False
        ) as handle:
            temporary = Path(handle.name)
            np.save(handle, tokens, allow_pickle=False)
        os.replace(temporary, token_path)
    tokens = np.load(token_path, mmap_mode="r")
    if tokens.dtype != np.uint16 or tokens.ndim != 1:
        raise RuntimeError(f"invalid token cache {token_path}")
    return token_path, tokens


def _batch(
    tokens: np.ndarray,
    *,
    step: int,
    seed: int,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    start: int,
    stop: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    width = sequence_length + 1
    if stop - start <= width:
        raise ValueError("corpus split is shorter than one training sequence")
    # A step-local RNG makes resume and cross-precision batches independent of
    # all model/compiler RNG consumption.
    generator = np.random.default_rng(seed + step * 0x9E3779B1)
    offsets = generator.integers(start, stop - width, size=batch_size)
    host = np.empty((batch_size, width), dtype=np.int64)
    for row, offset in enumerate(offsets):
        host[row] = tokens[int(offset) : int(offset) + width]
    values = torch.from_numpy(host).to(device=device, non_blocking=True)
    return values[:, :-1], values[:, 1:]


def _learning_rate(
    step: int, *, base: float, warmup: int, total: int, minimum_ratio: float
) -> float:
    if warmup and step <= warmup:
        return base * step / warmup
    progress = (step - warmup) / max(total - warmup, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return base * (minimum_ratio + (1.0 - minimum_ratio) * cosine)


def _git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _model_config(args: argparse.Namespace, precision: str) -> DecoderConfig:
    return DecoderConfig(
        vocab_size=BYTE_VOCAB_SIZE,
        max_seq_len=args.sequence_length,
        num_layers=args.layers,
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.heads,
        gradient_checkpointing=args.gradient_checkpointing,
        dtype=torch.bfloat16,
        linear=LinearSpec(
            precision=precision,
            autotune=args.autotune,
            mxfp8_backend=args.mxfp8_backend,
            nvfp4_scaling=args.nvfp4_scaling,
            nvfp4_backend=args.nvfp4_backend,
        ),
    )


@torch.no_grad()
def _validation_loss(
    model: DecoderOnlyTransformer,
    tokens: np.ndarray,
    *,
    seed: int,
    batches: int,
    batch_size: int,
    sequence_length: int,
    device: torch.device,
    start: int,
    stop: int,
) -> float:
    model.eval()
    total = 0.0
    for index in range(batches):
        x, targets = _batch(
            tokens,
            step=10_000_000 + index,
            seed=seed,
            batch_size=batch_size,
            sequence_length=sequence_length,
            device=device,
            start=start,
            stop=stop,
        )
        total += float(causal_lm_loss(model(x), targets))
    model.train()
    return total / batches


def _append_jsonl(path: Path, records: Iterable[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _checkpoint(
    path: Path,
    *,
    precision: str,
    step: int,
    model: DecoderOnlyTransformer,
    optimizer: torch.optim.Optimizer,
    config: DecoderConfig,
    corpus_hash: str,
) -> None:
    _atomic_torch_save(
        path,
        {
            "format_version": FORMAT_VERSION,
            "precision": precision,
            "step": step,
            "model_config": asdict(config),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "corpus_sha256": corpus_hash,
        },
    )


def _train_precision(
    args: argparse.Namespace,
    precision: str,
    tokens: np.ndarray,
    initial_state: dict[str, torch.Tensor],
    corpus_hash: str,
) -> None:
    device = torch.device(args.device)
    config = _model_config(args, precision)
    torch.manual_seed(args.seed)
    model = DecoderOnlyTransformer(config, device=device)
    model.load_state_dict(initial_state, strict=True)
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
        fused=True,
    )
    output = args.output / precision
    checkpoint_path = output / "checkpoint.pt"
    start_step = 0
    if args.resume and checkpoint_path.exists():
        saved = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if saved.get("precision") != precision:
            raise RuntimeError(f"checkpoint precision mismatch in {checkpoint_path}")
        if saved.get("corpus_sha256") != corpus_hash:
            raise RuntimeError(f"checkpoint corpus mismatch in {checkpoint_path}")
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        start_step = int(saved["step"])
        print(f"RESUME {precision} step={start_step}", flush=True)
    if start_step >= args.steps:
        print(f"DONE {precision} already reached step={start_step}", flush=True)
        return

    train_stop = int(tokens.size * (1.0 - args.validation_fraction))
    validation_start = train_stop
    executable = (
        torch.compile(
            model,
            fullgraph=True,
            dynamic=False,
            options={"triton.cudagraphs": False},
        )
        if args.compile
        else model
    )
    metrics_path = output / "metrics.jsonl"
    interval_started = time.perf_counter()
    interval_tokens = 0
    recent_losses: list[torch.Tensor] = []
    print(
        f"START {precision} step={start_step}->{args.steps} "
        f"params={model.num_parameters():,} compile={args.compile}",
        flush=True,
    )
    for step in range(start_step + 1, args.steps + 1):
        learning_rate = _learning_rate(
            step,
            base=args.learning_rate,
            warmup=args.warmup_steps,
            total=args.steps,
            minimum_ratio=args.minimum_lr_ratio,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.zero_grad(set_to_none=True)
        accumulated: torch.Tensor | None = None
        for microbatch in range(args.gradient_accumulation):
            x, targets = _batch(
                tokens,
                step=(step - 1) * args.gradient_accumulation + microbatch,
                seed=args.seed,
                batch_size=args.batch_size,
                sequence_length=args.sequence_length,
                device=device,
                start=0,
                stop=train_stop,
            )
            loss = causal_lm_loss(executable(x), targets)
            (loss / args.gradient_accumulation).backward()
            detached_loss = loss.detach()
            accumulated = (
                detached_loss
                if accumulated is None
                else accumulated + detached_loss
            )
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.max_grad_norm, foreach=True
        )
        optimizer.step()
        assert accumulated is not None
        recent_losses.append(accumulated / args.gradient_accumulation)
        should_log = (
            step == 1
            or step % args.log_interval == 0
            or step == args.steps
        )
        interval_tokens += (
            args.batch_size * args.sequence_length * args.gradient_accumulation
        )

        if should_log:
            torch.cuda.synchronize(device)
            now = time.perf_counter()
            elapsed = now - interval_started
            validation = None
            if step % args.validation_interval == 0 or step == args.steps:
                validation = _validation_loss(
                    model,
                    tokens,
                    seed=args.seed,
                    batches=args.validation_batches,
                    batch_size=args.batch_size,
                    sequence_length=args.sequence_length,
                    device=device,
                    start=validation_start,
                    stop=tokens.size,
                )
            record = {
                "record_type": "training_step",
                "precision": precision,
                "step": step,
                "train_loss": float(torch.stack(recent_losses).mean()),
                "validation_loss": validation,
                "learning_rate": learning_rate,
                "grad_norm": float(grad_norm),
                "tokens_per_second": interval_tokens / max(elapsed, 1.0e-9),
                "tokens_seen": (
                    step
                    * args.batch_size
                    * args.sequence_length
                    * args.gradient_accumulation
                ),
                "elapsed_seconds": elapsed,
                "timestamp": time.time(),
            }
            _append_jsonl(metrics_path, (record,))
            print(
                f"STEP {precision} {step}/{args.steps} "
                f"loss={record['train_loss']:.5f} "
                f"val={validation if validation is not None else '-'} "
                f"lr={learning_rate:.3e} grad={record['grad_norm']:.3f} "
                f"tok/s={record['tokens_per_second']:.0f}",
                flush=True,
            )
            interval_started = time.perf_counter()
            interval_tokens = 0
            recent_losses.clear()
        if (
            step % args.checkpoint_interval == 0
            or step == args.steps
            or _STOP_REQUESTED
        ):
            checkpoint_started = time.perf_counter()
            _checkpoint(
                checkpoint_path,
                precision=precision,
                step=step,
                model=model,
                optimizer=optimizer,
                config=config,
                corpus_hash=corpus_hash,
            )
            # Checkpoint I/O is operational overhead, not model throughput.
            interval_started += time.perf_counter() - checkpoint_started
            print(f"CHECKPOINT {precision} step={step} {checkpoint_path}", flush=True)
        if _STOP_REQUESTED:
            break
    del executable, optimizer, model
    torch.compiler.reset()
    torch.cuda.empty_cache()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--precision",
        nargs="+",
        choices=("bf16", "mxfp8", "nvfp4"),
        default=("bf16", "mxfp8", "nvfp4"),
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("training_data/tinystories-train-00000.parquet"),
    )
    parser.add_argument("--data-url", default=TINYSTORIES_TRAIN_SHARD)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("training_results/tinystories_decoder_v1"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--steps", type=int, default=1_000)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--sequence-length", type=int, default=512)
    parser.add_argument("--gradient-accumulation", type=int, default=1)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--hidden-size", type=int, default=768)
    parser.add_argument("--intermediate-size", type=int, default=1_536)
    parser.add_argument("--heads", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=3.0e-4)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.1)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1.0e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--validation-batches", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--checkpoint-interval", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument(
        "--autotune",
        choices=("off", "cache", "coordinate"),
        default="cache",
    )
    parser.add_argument(
        "--mxfp8-backend",
        choices=("auto", "fused", "materialized"),
        default="auto",
    )
    parser.add_argument(
        "--nvfp4-scaling",
        choices=("delayed", "current", "regional", "block"),
        default="delayed",
    )
    parser.add_argument(
        "--nvfp4-backend",
        choices=("auto", "fused", "materialized"),
        default="auto",
    )
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    positive = (
        "steps",
        "batch_size",
        "sequence_length",
        "gradient_accumulation",
        "layers",
        "hidden_size",
        "intermediate_size",
        "heads",
        "learning_rate",
        "validation_batches",
        "log_interval",
        "validation_interval",
        "checkpoint_interval",
    )
    if any(getattr(args, name) <= 0 for name in positive):
        parser.error("training dimensions, rates, and intervals must be positive")
    if not 0.0 < args.validation_fraction < 0.5:
        parser.error("validation-fraction must be between zero and 0.5")
    if args.warmup_steps < 0 or args.warmup_steps >= args.steps:
        parser.error("warmup-steps must be nonnegative and smaller than steps")
    if args.hidden_size % args.heads:
        parser.error("hidden-size must be divisible by heads")
    return args


def main() -> None:
    args = _parse_args()
    if (
        not torch.cuda.is_available()
        or torch.cuda.get_device_capability(args.device)[0] != 12
    ):
        raise SystemExit("decoder convergence training requires an SM120/SM121 GPU")
    signal.signal(signal.SIGINT, _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)
    args.output.mkdir(parents=True, exist_ok=True)
    token_path, tokens = _prepare_corpus(args.data, url=args.data_url)
    corpus_hash = _sha256(token_path)
    base_config = _model_config(args, "bf16")
    torch.manual_seed(args.seed)
    initial = DecoderOnlyTransformer(base_config, device="cpu").state_dict()
    initial_state = {name: value.clone() for name, value in initial.items()}
    base_model_config = asdict(base_config)
    base_model_config["dtype"] = str(base_config.dtype)
    manifest = {
        "format_version": FORMAT_VERSION,
        "git_commit": _git_commit(),
        "command": sys.argv,
        "device": torch.cuda.get_device_name(args.device),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "corpus": str(token_path.resolve()),
        "corpus_sha256": corpus_hash,
        "tokens": int(tokens.size),
        "precisions": list(args.precision),
        "base_model_config": base_model_config,
    }
    _atomic_json(args.output / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, default=str), flush=True)
    for precision in args.precision:
        _train_precision(args, precision, tokens, initial_state, corpus_hash)
        if _STOP_REQUESTED:
            break


if __name__ == "__main__":
    main()
