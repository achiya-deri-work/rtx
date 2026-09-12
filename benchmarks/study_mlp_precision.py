"""Numerical recipe experiments, not production kernels or timing benchmarks."""

import argparse
import json
from pathlib import Path

import torch

from benchmarks.compare_mlp_recipes import ao_quantize
from benchmarks.stress_mxfp8_mlp_numerics import metrics


POLICIES = {
    "mxfp8": ("mx", "mx"),
    "bf16_gate": ("bf16", "mx"),
    "corrected_gate": ("corrected", "mx"),
    "bf16_hidden_mixed_down": ("mx", "hidden"),
    "bf16_down": ("mx", "bf16"),
    "bf16_gate_and_hidden": ("bf16", "hidden"),
    "bf16_gate_and_down": ("bf16", "bf16"),
}


def product(a, b, mode):
    """a @ b.T; hidden mode protects a, still quantizes b."""
    if mode == "bf16":
        return a @ b.T
    if mode == "hidden":
        return (a.float() @ ao_quantize(b).T).bfloat16()
    aq, bq = ao_quantize(a), ao_quantize(b)
    result = aq @ bq.T
    if mode == "corrected":
        result = result + ao_quantize(a.float() - aq) @ bq.T
        result = result + aq @ ao_quantize(b.float() - bq).T
    return result.bfloat16()


class RecipeLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, mode):
        ctx.save_for_backward(x, w)
        ctx.mode = mode
        return product(x, w, "bf16" if mode=="bf16_fwd" else mode)

    @staticmethod
    def backward(ctx, dy):
        x, w = ctx.saved_tensors
        if ctx.mode == "hidden":
            # Quantized gradient/weight for dX, but preserve hidden for dW.
            dx = product(dy, w.T, "mx")
            dw = (ao_quantize(dy.T) @ x.float()).bfloat16()
        else:
            mode = "mx" if ctx.mode=="bf16_fwd" else ctx.mode
            dx = product(dy, w.T, mode)
            dw = product(dy.T, x.T, mode)
        return dx, dw, None


class CombinedGateReference(torch.autograd.Function):
    """BF16 gate forward with a single independently quantized gate/up backward."""
    @staticmethod
    def forward(ctx,x,gate,up):
        ctx.save_for_backward(x,gate,up)
        return x@gate.T,product(x,up,"mx")

    @staticmethod
    def backward(ctx,dg,du):
        x,gate,up=ctx.saved_tensors
        dp=torch.cat((dg,du),-1)
        weights=torch.cat((gate,up),0)
        dx=product(dp,weights.T,"mx")
        dw=product(dp.T,x.T,"mx")
        dgate,dup=dw.chunk(2,0)
        return dx,dgate,dup


def recipe(x, gate, up, down, activation, gate_mode, down_mode, up_mode="mx"):
    if gate_mode=="bf16_fwd":
        g,u=CombinedGateReference.apply(x,gate,up)
        g,u=g.float(),u.float()
    else:
        g = RecipeLinear.apply(x, gate, gate_mode).float()
        u = RecipeLinear.apply(x, up, up_mode).float()
    h = (getattr(torch.nn.functional, activation)(g) * u).bfloat16()
    return RecipeLinear.apply(h, down, down_mode)


def evaluate(inputs, dy, activation, gate_mode, down_mode, up_mode="mx"):
    leaves = tuple(t.detach().requires_grad_() for t in inputs)
    y = recipe(*leaves, activation, gate_mode, down_mode, up_mode)
    return (y, *torch.autograd.grad(y, leaves, dy))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=False)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    manifest = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "policies": POLICIES,
        "oracle": "BF16 GEMMs, FP32 activation/product, BF16 linear/hidden boundaries",
        "note": "TorchAO RCEIL conversion and reconstructed FP32 GEMMs. No timing claims.",
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    count = 0
    failures = 0
    with (out / "cases.jsonl").open("w") as stream:
        for m, k, h in ((17,35,97), (512,768,1536)):
            for distribution in ("uniform_half", "init_002"):
                for activation in ("silu", "gelu", "relu"):
                    for seed in (17,31,59):
                        torch.manual_seed(seed)
                        inputs = [(torch.rand(*s, device="cuda")-.5).bfloat16()
                                  for s in ((m,k),(h,k),(h,k),(k,h))]
                        if distribution == "init_002":
                            for w in inputs[1:]:
                                torch.nn.init.trunc_normal_(w, std=.02, a=-.04, b=.04)
                        dy = (torch.rand(m,k,device="cuda")-.5).bfloat16()
                        oracle = evaluate(inputs, dy, activation, "bf16", "bf16", "bf16")
                        record = {"shape": [m,k,h,k], "distribution": distribution,
                                  "activation": activation, "seed": seed, "policies": {}}
                        for name, (gate_mode, down_mode) in POLICIES.items():
                            values = evaluate(inputs, dy, activation, gate_mode, down_mode)
                            errors = {n: metrics(a,b) for n,a,b in zip(
                                ("y","dx","dgate","dup","ddown"), values, oracle)}
                            errors["residual_y"] = metrics(inputs[0]+values[0], inputs[0]+oracle[0])
                            errors["residual_dx"] = metrics(dy+values[1], dy+oracle[1])
                            record["policies"][name] = errors
                        finite = all(v["finite"] for p in record["policies"].values() for v in p.values())
                        failures += not finite
                        count += 1
                        stream.write(json.dumps(record, allow_nan=False)+"\n")
                        stream.flush()
                        print(count, record["shape"], distribution, activation, seed, finite, flush=True)
    (out / "summary.json").write_text(json.dumps({"cases":count,"nonfinite_cases":failures}, indent=2))
    raise SystemExit(bool(failures))


if __name__ == "__main__":
    main()
