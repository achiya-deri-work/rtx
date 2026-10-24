"""Explicit, streaming offline calibration collection for NVFP4 PTQ.

Collection does not quantize a model, select runtime schedules, call the model,
or change its training/gradient mode. Callers own representative input choice.
"""
from __future__ import annotations

from collections.abc import Callable

import torch
from torch import nn

from .nvfp4_calibration import NVFP4SecondMomentCollector
from .ptq import _selected_linears


class NVFP4CalibrationCollector:
    """Single-use context collecting uncentered block16 activation moments.

    Selected modules must be bias-free BF16 ``nn.Linear`` instances. A shared
    module gets one hook and one accumulator, keyed by its first fully qualified
    name (``""`` for a root linear); filters must agree across all aliases.

    Each invocation contributes at most ``max_tokens_per_batch`` evenly spaced
    rows from flattened ``[..., K]`` BF16 input. This is a per-call bound, not
    a dataset-size limit: detached CPU FP64 sums stream over all invocations,
    without retaining activations. The caller controls representative images,
    sequences, masks, batches, and train/eval mode; padded rows are not removed.

    Example::

        with NVFP4CalibrationCollector(model, max_tokens_per_batch=512) as cal:
            for inputs in representative_train_batches:
                model(inputs)
        moments = cal.second_moments()

    Read results only after a successful context exit. Any escaping exception
    or collection error invalidates all partial results. Instances cannot be
    re-entered or reused. No gradients are recorded by the hooks themselves;
    the surrounding forward's gradient mode is left untouched.
    """

    def __init__(self, model: nn.Module, *, filter_fn: Callable[[nn.Module, str], bool] | None = None, max_tokens_per_batch: int = 1024):
        if not isinstance(model, nn.Module):
            raise TypeError("model must be an nn.Module")
        if isinstance(max_tokens_per_batch, bool) or not isinstance(max_tokens_per_batch, int) or max_tokens_per_batch < 1:
            raise ValueError("max_tokens_per_batch must be a positive integer")
        self.model = model
        self.filter_fn = filter_fn
        self.max_tokens_per_batch = max_tokens_per_batch
        self._selected = _selected_linears(model, filter_fn, require_cuda=False)
        self._collectors = {aliases[0][3]: NVFP4SecondMomentCollector(module.in_features) for module, aliases in self._selected}
        self._handles = []
        self._state = "new"

    def __enter__(self):
        if self._state != "new":
            raise RuntimeError("calibration collector is single-use; re-entry/reuse is not allowed")
        self._state = "active"
        try:
            for module, aliases in self._selected:
                name = aliases[0][3]
                self._handles.append(module.register_forward_pre_hook(self._hook(name, module.in_features), with_kwargs=True))
        except BaseException:
            self._state = "invalid"
            self._remove_hooks()
            raise
        return self

    def _hook(self, name, in_features):
        @torch.no_grad()
        def collect(_module, args, kwargs):
            try:
                if self._state != "active":
                    raise RuntimeError("calibration collection is invalid")
                value = args[0] if args else kwargs.get("input")
                if not isinstance(value, torch.Tensor) or value.dtype != torch.bfloat16 or value.layout != torch.strided:
                    raise TypeError("calibration requires dense BF16 tensor inputs")
                if value.ndim < 1 or value.shape[-1] != in_features or value.numel() == 0:
                    raise ValueError("calibration requires nonempty [..., in_features] inputs")
                rows = value.detach().reshape(-1, in_features)
                count = min(len(rows), self.max_tokens_per_batch)
                indices = torch.linspace(0, len(rows) - 1, count, device=value.device).long()
                self._collectors[name].update(rows[indices])
            except BaseException:
                self._state = "invalid"
                raise
        return collect

    def _remove_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def __exit__(self, exc_type, exc_value, traceback):
        self._remove_hooks()
        if exc_type is not None or self._state == "invalid":
            self._state = "invalid"
        else:
            self._state = "complete"
        return False

    def _check_results(self):
        if self._state == "active":
            raise RuntimeError("calibration results are unavailable while context is active")
        if self._state != "complete":
            raise RuntimeError("calibration results require a successfully completed context; collector is unused or invalid")
        unobserved = [name or "<root>" for name, collector in self._collectors.items() if collector.num_samples == 0]
        if unobserved:
            self._state = "invalid"
            raise RuntimeError("selected linears were unobserved: " + ", ".join(unobserved))

    def second_moments(self) -> dict[str, torch.Tensor]:
        """Return fresh CPU FP32 ``[ceil(K/16),16,16]`` uncentered moments."""
        self._check_results()
        return {name: collector.second_moments() for name, collector in self._collectors.items()}

    @property
    def sample_counts(self) -> dict[str, int]:
        """Sampled token-row totals, keyed by the same canonical names."""
        self._check_results()
        return {name: collector.num_samples for name, collector in self._collectors.items()}


__all__ = ["NVFP4CalibrationCollector"]
