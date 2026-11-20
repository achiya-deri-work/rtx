"""Same-device, same-input numerical audit of old and new MXFP8 MLP policies.

Run this identical script in isolated source snapshots, previous first. It
records reference errors separately from direct version/policy differences.
It does not tune, install winners, or time performance.
"""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import time

import torch


def metrics(a, b):
    a, b = a.detach().double(), b.detach().double()
    bad_a, bad_b = int((~a.isfinite()).sum()), int((~b.isfinite()).sum())
    if bad_a or bad_b:
        return dict(finite=False, candidate_nonfinite=bad_a, reference_nonfinite=bad_b)
    d = a - b
    rms, rr = float(d.square().mean().sqrt()), float(b.square().mean().sqrt())
    return dict(finite=True, max_abs=float(d.abs().max()), rms=rms,
                reference_rms=rr, relative_l2=rms / max(rr, 1e-30),
                normalized_max=float(d.abs().max()) / max(float(b.abs().max()), 1e-30),
                cosine=float(torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0, eps=1e-30)),
                unequal=int((a != b).sum()), elements=a.numel())


def quantized(t):
    # Independent PyTorch E4M3/E8M0 conversion; integer exponent selection avoids
    # logarithm boundary ambiguity. No production quantizer or GEMM is called.
    width = t.shape[-1]
    blocks = torch.nn.functional.pad(t.float(), (0, (-width) % 32)).reshape(t.shape[0], -1, 32)
    bits = blocks.abs().amax(-1, keepdim=True).view(torch.int32)
    exponent = ((bits >> 23) & 255) - 127 - 8 + ((bits & 0x7fffff) > 0x600000)
    scale = ((exponent.clamp(-127, 128) + 127) << 23).view(torch.float32).clamp_min(2.**-126)
    return ((blocks / scale).clamp(-448, 448).to(torch.float8_e4m3fn).float() * scale).flatten(1)[:, :width]


def recipe(values, dy, activation):
    x, w, down = values
    g, u = (quantized(x) @ quantized(w).T).chunk(2, -1)
    a = getattr(torch.nn.functional, activation)(g)
    hidden = quantized(a * u).bfloat16()
    y = (hidden.float() @ quantized(down).T).bfloat16()
    dh = (quantized(dy) @ quantized(down.T).T).bfloat16().float()
    dd = (quantized(dy.T) @ quantized(hidden.T).T).bfloat16()
    if activation == 'silu':
        s = torch.sigmoid(g)
        derivative = s * (1 + g * (1 - s))
    else:
        derivative = .5 * (1 + torch.erf(g / 2**.5)) + g * torch.exp(-g*g/2) / (2*torch.pi)**.5
    dp = torch.cat((dh * u * derivative, dh * a), -1).bfloat16()
    dx = (quantized(dp) @ quantized(w.T).T).bfloat16()
    dw = (quantized(dp.T) @ quantized(x.T).T).bfloat16()
    return y, dx, dw, dd


def dense_reference(values, dy, activation, dtype):
    values = tuple(t.detach().to(dtype).requires_grad_() for t in values)
    x, w, down = values
    g, u = (x @ w.T).chunk(2, -1)
    y = (getattr(torch.nn.functional, activation)(g) * u) @ down.T
    return tuple(t.detach() for t in (y, *torch.autograd.grad(y, values, dy.to(dtype))))


def cases():
    for k in (128, 256, 384):
        yield 192, k, 'silu', 'normal', 91001
        yield 40000, k, 'silu', 'normal', 91002
        yield 40000, k, 'gelu', 'uniform', 91003
        yield 40960, k, 'gelu', 'normal', 91004
        yield 512, k, 'silu', 'outliers', 91005
        yield 40000, k, 'silu', 'normal', 91006


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--version', choices=('previous', 'current'), required=True)
    parser.add_argument('--configs', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--previous-output', type=Path)
    parser.add_argument('--limit', type=int, default=100)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    from rtx.mlp import mxfp8_mlp
    from rtx._bwd_search import split_leg
    from benchmarks.qualify_mxfp8_long_mlp import initial_schedule
    def load_policy(path, shape):
        config = initial_schedule(json.loads(path.read_text()), shape)
        if shape[0] < 32768:
            # Long-sequence split counts are illegal on tiny oracle cases.
            # Use the identical unsplit reduction for both source revisions.
            for role in ('gate_backward', 'down_backward'):
                bwd = getattr(config, role)
                config = replace(config, **{role:replace(bwd, dw=split_leg(bwd.dw, shape[0], 1))})
        return config
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    import cutlass
    root = Path.cwd()
    manifest = dict(version=args.version, torch=str(torch.__version__), cuda=torch.version.cuda,
                    cutlass=cutlass.__version__, device=str(torch.cuda.get_device_properties(0)),
                    input_recipe='CPU generator FP32 then BF16; weights std .02, upstream std .1',
                    reference_floor=1e-30, implementation_gate='finite and (relative_l2<.025 or max_abs<=1e-12)',
                    source_sha256={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                                   for p in sorted((root/'rtx').rglob('*.py'))})
    (args.output/'environment.json').write_text(json.dumps(manifest, indent=2))
    names = ('y', 'dx', 'dgate_up', 'ddown')
    started = time.monotonic()
    failures = 0
    for index, (m, k, act, distribution, seed) in enumerate(cases()):
        if index >= args.limit:
            break
        shape = (m, k, 2*k, k)
        generator = torch.Generator().manual_seed(seed)
        cpu = [torch.randn(s, generator=generator) * scale for s, scale in
               (((m,k),.2), ((4*k,k),.02), ((k,2*k),.02), ((m,k),.1))]
        if distribution == 'uniform':
            cpu[0] = torch.rand((m,k), generator=generator) - .5
        if distribution == 'outliers':
            cpu[0][:,::32] *= 32
            cpu[1][:,::32] *= 8
        cpu = [t.bfloat16() for t in cpu]
        digest = hashlib.sha256(b''.join(t.view(torch.uint8).numpy().tobytes() for t in cpu)).hexdigest()
        values = tuple(t.cuda().requires_grad_() for t in cpu[:3])
        dy = cpu[3].cuda().T.contiguous().T  # noncontiguous upstream
        previous = load_policy(args.configs/f'previous-w{k}.json', shape)
        policies = [('inherited', previous)]
        if args.version == 'current':
            policies.extend((('fused_none', replace(previous, gradient_quantization='recompute_both')),
                             ('fused_xor', replace(previous, gradient_quantization='recompute_both',
                                recompute=replace(previous.recompute, gemm=replace(previous.recompute.gemm, glu_scratch_swizzle='xor_16b')))),
                             ('selected', load_policy(args.configs/f'current-w{k}.json', shape))))
        with torch.no_grad():
            operational = recipe(values, dy, act)
        fp32 = dense_reference(values, dy, act, torch.float32)
        bf16 = dense_reference(values, dy, act, torch.bfloat16)
        prior = None
        if args.previous_output:
            prior = torch.load(args.previous_output/f'outputs-{index:02}.pt', weights_only=True)
            assert prior['input_sha256'] == digest, 'inputs drifted between sources'
            prior = tuple(t.cuda() for t in prior['outputs'])
        inherited = None
        compiled = distribution == 'normal' and m == 40000 and seed == 91002
        for label, config in policies:
            fn = lambda *v: mxfp8_mlp(*v, activation=act, config=config)
            if compiled:
                torch._dynamo.reset()
                fn = torch.compile(fn, fullgraph=True, dynamic=False, options={'triton.cudagraphs':False})
            y = fn(*values)
            actual = tuple(t.detach().clone() for t in (y, *torch.autograd.grad(y, values, dy)))
            result = dict(index=index, shape=shape, activation=act, distribution=distribution,
                          seed=seed, input_sha256=digest, policy=label, compiled=compiled,
                          config=config.to_dict(), elapsed_s=time.monotonic()-started)
            for key, expected in [('recipe', operational), ('fp32', fp32), ('bf16', bf16),
                                  ('previous_source', prior), ('same_source_inherited', inherited)]:
                if expected is not None:
                    result[key] = {name:metrics(a,b) for name,a,b in zip(names, actual, expected)}
            result['passed'] = all(v['finite'] and (v['relative_l2'] < .025 or v['max_abs'] <= 1e-12)
                                   for v in result['recipe'].values())
            failures += not result['passed']
            if label == 'inherited':
                inherited = actual
                if args.version == 'previous':
                    torch.save(dict(input_sha256=digest,outputs=[t.cpu() for t in actual]),args.output/f'outputs-{index:02}.pt')
            with (args.output/'cases.jsonl').open('a') as stream:
                stream.write(json.dumps(result,allow_nan=False)+'\n')
            print(json.dumps(dict(index=index,shape=shape,activation=act,distribution=distribution,
                                  policy=label,passed=result['passed'],elapsed_s=round(result['elapsed_s'],1),
                                  recipe={n:round(v['relative_l2'],6) for n,v in result['recipe'].items()},
                                  previous={n:round(v['relative_l2'],6) for n,v in result.get('previous_source',{}).items()})),flush=True)
        del values, dy, operational, fp32, bf16, prior, inherited, actual, y, fn
    (args.output/'summary.json').write_text(json.dumps(dict(failures=failures,elapsed_s=time.monotonic()-started),indent=2))
    raise SystemExit(bool(failures))


if __name__ == '__main__':
    main()
