"""Optimizer-visible logical projections over one physical weight allocation."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import copy
import math

import torch
from torch import nn


def _strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    stride, result = 1, []
    for size in reversed(shape):
        result.append(stride)
        stride *= size
    return tuple(reversed(result))


class _VirtualWeight(torch.autograd.Function):
    """Alias the allocation, but give every logical leaf its own gradient."""

    @staticmethod
    def forward(ctx, shape, *parameters):
        ctx.shapes = tuple(p.shape for p in parameters)
        # Original Parameters keep their identities/version counters after
        # packing. Check every leaf's version, not just the first alias.
        ctx.save_for_backward(*parameters)
        return parameters[0].as_strided(shape, _strides(shape))

    @staticmethod
    def backward(ctx, gradient):
        _ = ctx.saved_tensors  # Enforce saved-Parameter version checks.
        # A consumer normally supplies a contiguous packed gradient. Reshape
        # also handles noncontiguous upstream gradients correctly when needed.
        flat, offset, result = gradient.reshape(-1), 0, []
        for shape in ctx.shapes:
            size = shape.numel()
            result.append(flat.narrow(0, offset, size).view(shape))
            offset += size
        return None, *result


class VirtualLayer(nn.Module):
    """Pack bias-free projections without merging their optimizer parameters.

    Args:
        projections: Ordered name-to-module mapping. Each module must own one
            contiguous 2D floating-point ``weight`` Parameter and no bias or
            other parameters. Existing weights, initialization and Parameter
            identities are preserved; construction packs their data once.
        packed_shape: Physical row-major tensor shape exposed by ``weight``.
            Defaults to concatenated output rows when input widths agree, or
            a flat allocation otherwise. For equal-sized experts, use
            ``(num_experts, out_features, in_features)``.

    ``weight`` is a differentiable Tensor alias, **not** an extra Parameter.
    Optimizers see only the original, independently updated 2D matrices.
    There is no forward weight copy, concatenation, pointer table or GPU op.
    Use it with RTX's functional kernels, or call this module for a Torch
    dense linear (2D packing) / expert-major batched matmul (3D packing).

    Construct before optimizers/compilation/capture. Move the owner, not
    individual projections. Ordinary optimizer updates and ``load_state_dict``
    preserve packing. After deliberately replacing a child/weight, call
    ``repack_`` outside live autograd graphs and rebuild any optimizer whose
    Parameters changed. Do not replace Parameter ``.data`` or ``set_`` during
    compiled execution. Distributed sharding, tensor subclasses and concurrent
    mutation are not supported by this storage contract.
    """

    def __init__(
        self,
        projections: Mapping[str, nn.Module],
        *,
        packed_shape: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if not projections:
            raise ValueError("VirtualLayer requires at least one projection")
        self.projections = nn.ModuleDict(projections)
        self._names = tuple(self.projections)
        parameters = self._logical_parameters()
        self._check_parameters(parameters)
        if packed_shape is None:
            widths = {p.shape[1] for p in parameters}
            packed_shape = ((sum(p.shape[0] for p in parameters), parameters[0].shape[1])
                            if len(widths) == 1 else (sum(p.numel() for p in parameters),))
        shape = tuple(packed_shape)
        if not shape or any(type(size) is not int or size <= 0 for size in shape):
            raise ValueError("packed_shape must contain positive integer extents")
        if math.prod(shape) != sum(p.numel() for p in parameters):
            raise ValueError("packed_shape must contain exactly the logical weight elements")
        self.packed_shape = shape
        self._shapes = tuple(tuple(p.shape) for p in parameters)
        self._offsets = tuple(sum(p.numel() for p in parameters[:i]) for i in range(len(parameters)))
        # Plain references, not extra registered parameters/buffers. The leaves
        # own the allocation. State dicts contain only logical projection keys.
        self._owned_parameters: tuple[nn.Parameter, ...] = ()
        self.repack_()
        self.register_load_state_dict_post_hook(_restore_packing_after_load)

    def _logical_parameters(self) -> tuple[nn.Parameter, ...]:
        if tuple(self.projections) != self._names:
            raise RuntimeError("Changing projection names/order requires constructing a new VirtualLayer")
        return tuple(self.projections[name].weight for name in self._names)

    @staticmethod
    def _check_parameters(parameters):
        if any(type(p) is not nn.Parameter or p.ndim != 2 or not p.is_floating_point()
               or min(p.shape) <= 0 or not p.is_contiguous() for p in parameters):
            raise ValueError("Each projection needs an ordinary contiguous positive 2D floating Parameter")
        if len({id(p) for p in parameters}) != len(parameters):
            raise ValueError("Tied weight Parameters cannot be separate virtual projections")
        if len({(p.device, p.dtype) for p in parameters}) != 1:
            raise ValueError("All projections must have the same device and dtype")
        if parameters[0].device.type != 'meta':
            intervals = {}
            for p in parameters:
                key = p.untyped_storage().data_ptr()
                intervals.setdefault(key, []).append((p.storage_offset(), p.storage_offset()+p.numel()))
            for ranges in intervals.values():
                ranges.sort()
                if any(left[1] > right[0] for left,right in zip(ranges,ranges[1:])):
                    raise ValueError("Overlapping/tied logical weights cannot be packed independently")

    def _check_modules(self):
        if tuple(self.projections) != self._names:
            raise RuntimeError("Changing projection names/order requires constructing a new VirtualLayer")
        for name in self._names:
            module = self.projections[name]
            if getattr(module, 'bias', None) is not None:
                raise ValueError("VirtualLayer projections must be bias-free")
            named = tuple(module.named_parameters())
            if len(named) != 1 or named[0][0] != 'weight' or named[0][1] is not module.weight:
                raise ValueError("Each projection must own only its weight Parameter")

    @torch.no_grad()
    def repack_(self) -> VirtualLayer:
        """Restore storage after deliberate replacement; setup-only, not forward."""
        if torch.compiler.is_compiling():
            raise RuntimeError("Repack outside compilation and graph capture")
        self._check_modules()
        parameters = self._logical_parameters()
        self._check_parameters(parameters)
        if tuple(tuple(p.shape) for p in parameters) != self._shapes:
            raise ValueError("Logical shapes changed; construct a new VirtualLayer")
        first = parameters[0]
        if first.is_cuda and torch.cuda.is_current_stream_capturing():
            raise RuntimeError("Repack outside compilation and graph capture")
        base = torch.empty(self.packed_shape, device=first.device, dtype=first.dtype)
        flat = base.view(-1)
        # Complete every read before rebinding any source Parameter.
        for p, offset in zip(parameters, self._offsets):
            flat.narrow(0, offset, p.numel()).copy_(p.detach().reshape(-1))
        for p, offset in zip(parameters, self._offsets):
            p.data = flat.narrow(0, offset, p.numel()).view(p.shape)
        self._owned_parameters = parameters
        return self

    def validate(self) -> None:
        """Check physical layout without synchronization or reading tensor data."""
        parameters = self._logical_parameters()
        first = parameters[0]
        for p, owner, shape, offset in zip(parameters, self._owned_parameters, self._shapes, self._offsets):
            if (p is not owner or tuple(p.shape) != shape or p.device != first.device
                    or p.dtype != first.dtype or not p.is_contiguous() or p.storage_offset() != offset):
                raise RuntimeError("Virtual packing changed; move the owner or call repack_() before compilation")
        if first.device.type != 'meta':
            address = first.untyped_storage().data_ptr()
            if any(p.untyped_storage().data_ptr() != address for p in parameters):
                raise RuntimeError("Virtual storage was replaced; call repack_() before compilation")

    @property
    def weight(self) -> torch.Tensor:
        """Zero-copy packed weight, with gradients to every logical projection."""
        parameters = self._logical_parameters()
        # Identity/shape guards are traceable. Full pointer checks run only in
        # eager/setup: Dynamo must never read data_ptr or allocate packed data.
        for p, owner, shape in zip(parameters, self._owned_parameters, self._shapes):
            if (p is not owner or tuple(p.shape) != shape or not p.is_contiguous()
                    or p.device != parameters[0].device
                    or p.dtype != parameters[0].dtype):
                raise RuntimeError("Virtual packing changed; call repack_() outside compilation")
        if not torch.compiler.is_compiling():
            self.validate()
        return _VirtualWeight.apply(self.packed_shape, *parameters)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Torch reference execution; RTX kernels consume ``self.weight``."""
        if len(self.packed_shape) == 2:
            return torch.nn.functional.linear(x, self.weight)
        if len(self.packed_shape) == 3:
            if x.ndim != 3 or x.shape[0] != self.packed_shape[0]:
                raise ValueError("Batched virtual execution expects [experts, rows, in_features]")
            return torch.bmm(x, self.weight.transpose(1, 2))
        raise ValueError("Use weight directly for a virtual packing other than 2D/3D")

    def __getitem__(self, name: str) -> nn.Module:
        return self.projections[name]

    def _apply(self, fn, recurse=True):
        # Transform the complete allocation once. Standard Module._apply then
        # retains PyTorch's Parameter identity/swap/meta and gradient semantics.
        self.validate()
        parameters = self._logical_parameters()
        with torch.no_grad():
            base = fn(parameters[0].detach().as_strided(self.packed_shape, _strides(self.packed_shape)))
        if not base.is_contiguous() or tuple(base.shape) != self.packed_shape:
            raise ValueError("VirtualLayer conversion must preserve contiguous packed shape")
        flat = base.reshape(-1)
        converted = {id(p):flat.narrow(0, offset, p.numel()).view(p.shape)
                     for p, offset in zip(parameters, self._offsets)}
        def transform(tensor):
            return converted[id(tensor)] if id(tensor) in converted else fn(tensor)
        result = super()._apply(transform, recurse=recurse)
        self._owned_parameters = self._logical_parameters()
        self.validate()
        return result

    def __deepcopy__(self, memo):
        # Parameter.__deepcopy__ clones leaves independently and loses aliases.
        # Restore packing on the new owner, without touching the original.
        result = type(self).__new__(type(self))
        memo[id(self)] = result
        for key, value in self.__dict__.items():
            setattr(result, key, copy.deepcopy(value, memo))
        result.repack_()
        return result

    def extra_repr(self) -> str:
        return f"packed_shape={self.packed_shape}, logical_projections={len(self._names)}"


def _restore_packing_after_load(module, incompatible_keys):
    # Copy loading retains aliases. assign=True / meta materialization may
    # replace children independently, so rebuild only when validation fails.
    try:
        module.validate()
    except RuntimeError:
        module.repack_()
