"""Experimental batching of same-shaped FP32 masters for the G2DETR benchmark.

Call after stock Muon has allocated momentum and after installing persistent
gradient buffers. This is not a general-purpose replacement optimizer.
"""
from collections import defaultdict

import torch
from torch.optim._muon import _adjust_lr, _single_tensor_muon, _zeropower_via_newtonschulz


def _batched_zeropower(grad, coefficients, steps, eps):
    """Preserve both addmm epilogues; batch only independent matrices."""
    a, b, c = coefficients
    tall = grad.shape[-2] > grad.shape[-1]
    x = grad.bfloat16()
    if tall:
        x = x.mT
    x = x / torch.linalg.vector_norm(x, dim=(-1, -2), keepdim=True).clamp(min=eps)
    for _ in range(steps):
        gram = torch.bmm(x, x.mT)
        delta = torch.baddbmm(gram, gram, gram, beta=b, alpha=c)
        x = torch.baddbmm(x, delta, x, beta=a)
    return x.mT if tall else x


def make_grouped_muon_step(opt, *, max_group=32, orthogonalize="baddbmm"):
    if type(opt) is not torch.optim.Muon or max_group < 1:
        raise ValueError("Expected stock Muon with a positive batch bound")
    if orthogonalize not in ("vmap", "baddbmm"):
        raise ValueError("Unknown orthogonalization implementation")
    plans = []
    for group in opt.param_groups:
        buckets = defaultdict(list)
        for p in group["params"]:
            if p.grad is None or p.dtype != torch.float32 or p.ndim != 2:
                raise ValueError("Expected persistent gradients for all FP32 matrix masters")
            b = opt.state[p]["momentum_buffer"]
            if b.dtype != p.dtype or p.grad.dtype != p.dtype or p.grad.is_sparse:
                raise ValueError("Expected FP32 dense momentum and gradient")
            buckets[(p.device, p.shape, p.stride())].append((p, p.grad, b))
        options = {key: group[key] for key in (
            "lr", "momentum", "weight_decay", "nesterov", "ns_coefficients",
            "ns_steps", "eps", "adjust_lr_fn")}
        for bucket in buckets.values():
            for i in range(0, len(bucket), max_group):
                plans.append((*zip(*bucket[i:i+max_group]), options))

    @torch.no_grad()
    def step():
        for ps, gs, bs, options in plans:
            if len(ps) == 1:
                _single_tensor_muon(list(ps), list(gs), list(bs), has_complex=False, **options)
                continue

            updated, grad, momentum = torch.stack(ps), torch.stack(gs), torch.stack(bs)
            mu = options["momentum"]
            momentum.lerp_(grad, 1 - mu)
            direction = grad.lerp(momentum, mu) if options["nesterov"] else momentum
            def one(g):
                # Only orthogonalization sees a mapped axis. Each norm is
                # still over one original 2-D matrix, just as in stock Muon.
                return _zeropower_via_newtonschulz(g, options["ns_coefficients"],
                                                   options["ns_steps"], options["eps"])
            if orthogonalize == "vmap":
                direction = torch.vmap(one)(direction)
            else:
                direction = _batched_zeropower(direction, options["ns_coefficients"],
                                               options["ns_steps"], options["eps"])
            lr = options["lr"]
            adjusted = _adjust_lr(lr, options["adjust_lr_fn"], ps[0].shape)
            updated.mul_(1 - lr * options["weight_decay"])
            updated.add_(direction, alpha=-adjusted)
            torch._foreach_copy_(ps, updated.unbind())
            torch._foreach_copy_(bs, momentum.unbind())

    return step
