"""Public linear accuracy/latency frontier with matched CUDA-graph timing.

Portable fixed configs, not tuned claims. Each invocation handles one shape so
allocators and compilation history do not carry across shapes. CUDA graph replay
removes Python launch gaps equally for BF16, MXFP8 and NVFP4. Gradients use the
same upstream tensor, and timing includes both requested parameter/input grads.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import gc
import hashlib
import json
from pathlib import Path
import random
import statistics

import torch
import rtx
from benchmarks.study_nvfp4_precision import metrics
from rtx.nvfp4_inference_autotune import preferred_jit_row_region_config, dynamic_config_from_dict


def capture(fn, stream, unroll):
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            result = fn()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        for _ in range(unroll):
            result = fn()
    return graph, result


def measure(graph, calls):
    start, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    start.record()
    for _ in range(calls):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / calls


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shape', default='512,768,768')
    p.add_argument('--output', required=True, type=Path)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--rounds', type=int, default=15)
    p.add_argument('--calls', type=int, default=20)
    p.add_argument('--graph-unroll', type=int, default=4)
    p.add_argument('--variants', nargs='+', help='Optional named subset, always includes BF16/MXFP8')
    p.add_argument('--config', type=Path, help='Explicit adaptive-study winner JSON to add as explicit-POLICY')
    p.add_argument('--mx-trials', type=int, default=0, help='Optional MXFP8 forward tuning budget; backward stays matched')
    p.add_argument('--allow-shape-transfer',action='store_true',
                   help='Explicitly test a saved schedule on another shape; still checks kernel legality')
    args = p.parse_args()
    m,n,k = map(int, args.shape.split(','))
    if min(m,n,k,args.rounds,args.calls,args.graph_unroll) < 1:
        p.error('shape, rounds, calls and graph-unroll must be positive')
    env = rtx.validate_runtime_environment()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(args.seed)
    x = torch.randn(m,k,device='cuda',dtype=torch.bfloat16).requires_grad_()
    weight = torch.randn(n,k,device='cuda',dtype=torch.bfloat16) * .02
    grad = torch.randn(m,n,device='cuda',dtype=torch.bfloat16)
    refs = [x.detach().float() @ weight.float().T, grad.float() @ weight.float(),
            grad.float().T @ x.detach().float()]
    variants = [('bf16', {}), ('mxfp8', {})]
    variants += [(f'jit-{xr}x{wr}-{mode}', dict(scaling='jit_row_region',
        x_scale_region_rows=xr, weight_scale_region_rows=wr,
        scale_config=rtx.NVFP4ScaleConfig(tensor_scale_mode=mode)))
        for xr,wr in ((1,1),(5,4),(8,4),(8,8),(32,32)) for mode in ('power2','exact')]
    variants += [(policy, dict(scaling=policy)) for policy in ('delayed','current','block')]
    variants += [(f'jit-8x8-exact-{selection}',dict(scaling='jit_row_region',
        x_scale_region_rows=8,weight_scale_region_rows=8,
        scale_config=rtx.NVFP4ScaleConfig(tensor_scale_mode='exact'),
        block_scale_selection=selection)) for selection in ('mse5','lsq1','four_over_six')]
    if args.config:
        document=json.loads(args.config.read_text())
        cfg=dynamic_config_from_dict(document['config'])
        if document['shape'] != [m,n,k] and not args.allow_shape_transfer:
            p.error('explicit winner shape does not match; use --allow-shape-transfer for a transfer experiment')
        variants.append((f'explicit-{cfg.quant.block_scale_selection}',dict(
            scaling='jit_row_region', x_scale_region_rows=cfg.x_scale_region_rows,
            weight_scale_region_rows=cfg.weight_scale_region_rows,
            scale_config=rtx.NVFP4ScaleConfig(tensor_scale_mode=cfg.tensor_scale_mode),
            block_scale_selection=cfg.quant.block_scale_selection,_explicit_config=cfg)))
    if args.variants:
        allowed = set(args.variants) | {'bf16','mxfp8'}
        if allowed - {name for name,_ in variants}:
            p.error('unknown variant')
        variants = [(name,kw) for name,kw in variants if name in allowed]
    # Randomized execution order reduces systematic heating bias; report it.
    random.Random(args.seed).shuffle(variants)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    failed = False
    with args.output.open('x') as out:
        def save(record):
            out.write(json.dumps(record, allow_nan=False)+'\n'); out.flush()
            print(json.dumps(record, allow_nan=False), flush=True)
        save(dict(kind='environment', **env, gpu=torch.cuda.get_device_name(),
                  shape=[m,n,k],seed=args.seed, mode='cuda_graph',
                  autotune={'nvfp4':'explicit' if args.config else 'off',
                            'mxfp8_forward_trials':args.mx_trials,'backward':'fixed'},
                  graph_unroll=args.graph_unroll,
                  mxfp8_forward_trials=args.mx_trials,
                  transferred_config_shape=(document['shape'] if args.config else None),
                  harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope=('explicit NVFP4 config; budgeted MXFP8 forward; fixed shared backward'
                         if args.config or args.mx_trials else
                         'fixed portable schedules; no tuned speedup claim')))
        for name, kwargs in variants:
            try:
                if name == 'bf16':
                    model = torch.nn.Linear(k,n,bias=False,device='cuda',dtype=torch.bfloat16)
                elif name == 'mxfp8':
                    from rtx.autotune import AutotunePolicy
                    model = rtx.MXFP8Linear(k,n,device='cuda',
                        autotune=('balanced' if args.mx_trials else 'off'),
                        tuning_policy=(AutotunePolicy(max_trials=args.mx_trials,max_seconds=180)
                                       if args.mx_trials else None),
                        autotune_cache_dir=args.output.parent/'mx-cache',
                        backward_config=rtx.DEFAULT_MXFP8_BWD_CONFIG)
                else:
                    if kwargs.get('scaling') == 'jit_row_region':
                        problem = rtx.NVFP4Problem(m,n,k)
                        cfg = replace(kwargs.pop('_explicit_config',preferred_jit_row_region_config(problem)),
                            x_scale_region_rows=kwargs['x_scale_region_rows'],
                            weight_scale_region_rows=kwargs['weight_scale_region_rows'],
                            tensor_scale_mode=kwargs['scale_config'].tensor_scale_mode)
                        cfg = replace(cfg,quant=replace(cfg.quant,
                            block_scale_selection=kwargs.get('block_scale_selection','amax')))
                        if cfg.rejection(problem):
                            cfg = replace(cfg, region_ownership='cta')
                        if cfg.rejection(problem):
                            cfg = replace(cfg, gemm=replace(cfg.gemm,
                                regional_scale_epilogue='direct',regional_epilogue_values=1))
                        if cfg.rejection(problem):
                            raise ValueError(cfg.rejection(problem))
                        kwargs = dict(kwargs, dynamic_config=cfg)
                        kwargs.pop('block_scale_selection',None)
                    model = rtx.NVFP4Linear(k,n,device='cuda',autotune='off',
                        backward_config=rtx.DEFAULT_MXFP8_BWD_CONFIG,**kwargs)
                with torch.no_grad(): model.weight.copy_(weight)
                # Bootstrap delayed state, then check all outputs before timing.
                model(x)
                y = model(x)
                dx,dw = torch.autograd.grad(y,(x,model.weight),grad)
                checks = dict(zip(('y','dx','dw'),[metrics(a,b) for a,b in zip((y,dx,dw),refs)]))
                if not all(c['finite'] for c in checks.values()):
                    raise RuntimeError('nonfinite forward/backward')
                del y,dx,dw
                gc.collect()
                def step():
                    y = model(x)
                    return y,torch.autograd.grad(y,(x,model.weight),grad)
                stream = torch.cuda.Stream()
                gf,yf = capture(lambda: model(x), stream, args.graph_unroll)
                gt,yt = capture(step, stream, args.graph_unroll)
                # Check captured outputs too, not only eager correctness.
                gf.replay(); gt.replay(); torch.cuda.synchronize()
                graph_checks = dict(y=metrics(yt[0],refs[0]),
                    dx=metrics(yt[1][0],refs[1]),dw=metrics(yt[1][1],refs[2]))
                if not all(c['finite'] for c in graph_checks.values()):
                    raise RuntimeError('nonfinite captured forward/backward')
                times = {'forward':[], 'forward_backward':[]}
                for i in range(args.rounds):
                    order = [('forward',gf),('forward_backward',gt)]
                    if i%2: order.reverse()
                    for key,g in order:
                        times[key].append(measure(g,args.calls)/args.graph_unroll)
                save(dict(kind='result',name=name,numerics=checks,graph_numerics=graph_checks,
                    dynamic_config=(asdict(kwargs['dynamic_config']) if 'dynamic_config' in kwargs else None),
                    timings={key:dict(median_us=statistics.median(v),min_us=min(v),
                                     samples_us=v) for key,v in times.items()}))
                del gf,gt,yf,yt,model,step,stream,order,g
                gc.collect()
                rtx.clear_runtime_caches(device=x.device)
                torch.cuda.empty_cache()
            except Exception as exc:
                save(dict(kind='error',name=name,error=f'{type(exc).__name__}: {exc}'))
                failed = True
                # CUDA faults can poison later candidates; do not keep timing.
                break
    if failed: raise SystemExit(1)


if __name__ == '__main__': main()
