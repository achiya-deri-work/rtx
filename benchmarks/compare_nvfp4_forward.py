"""Fresh-process dynamic-BF16 forward comparison; optional isolated Quartet-II.

No backward, prequantized weights, or scale-layout conversion is excluded from
the invoked forward. Policies intentionally differ and are recorded explicitly.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import torch
import rtx
from benchmarks.evaluate_quartet2 import build
from benchmarks.profile_preparation_phases import measure
from benchmarks.study_nvfp4_precision import metrics
from rtx.autotune.measurement import runtime_timing_metadata
from rtx.nvfp4_inference_autotune import dynamic_config_from_dict


def load_config(path, shape, scaling='jit_row_region'):
    value=json.loads(Path(path).read_text())
    if tuple(value['shape'])!=tuple(shape):
        raise ValueError('forward winner shape mismatch')
    config=dynamic_config_from_dict(value['config'])
    if scaling=='jit_row_region' and (config.x_scale_region_rows<=0 or config.weight_scale_region_rows<=0):
        raise ValueError('JIT-region winner required')
    if scaling=='current' and (config.x_scale_region_rows or config.weight_scale_region_rows):
        raise ValueError('tensor-current winner required')
    if scaling not in ('current','jit_row_region'):
        raise ValueError('unsupported winner policy')
    if value.get('scaling',scaling)!=scaling:
        raise ValueError('winner scaling metadata mismatch')
    return config


def replay_check(fn, x, weight, eager_fn=None):
    """Both operands must remain dynamic after compilation and graph capture."""
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3): fn()
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):
        output=fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph.replay()
    torch.cuda.synchronize()
    original_x=x.detach().clone(); original_w=weight.detach().clone()
    baseline=output.detach().clone()
    with torch.no_grad():
        x.mul_(1.5)
        weight.neg_()
    expected=(eager_fn or fn)().detach().clone()
    torch.testing.assert_close(fn(),expected,rtol=0,atol=0)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output,expected,rtol=0,atol=0)
    if torch.equal(output,baseline):
        raise RuntimeError('changed operands produced stale graph output')
    with torch.no_grad():
        x.copy_(original_x); weight.copy_(original_w)
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(output,baseline,rtol=0,atol=0)
    return {'changed_inputs_exact':True,'restored_inputs_exact':True}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('rtx_jit','rtx_current','rtx_four_six','quartet'),required=True)
    p.add_argument('--shape',required=True)
    p.add_argument('--forward-config',type=Path)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seed',type=int,default=17)
    p.add_argument('--distribution',choices=('normal','sparse','outliers'),default='normal')
    args=p.parse_args()
    if args.output.exists(): p.error('output already exists')
    shape=tuple(map(int,args.shape.split(',')))
    if len(shape)!=3 or any(d<=0 or d%128 for d in shape):
        p.error('positive M,N,K divisible by 128 required')
    m,n,k=shape
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16).requires_grad_()
    w=torch.randn(n,k,device='cuda',dtype=torch.bfloat16)*.02
    with torch.no_grad():
        if args.distribution=='sparse':
            x.mul_((torch.rand_like(x,dtype=torch.float32)<.1).to(x.dtype)*10**.5)
        elif args.distribution=='outliers':
            x.mul_(torch.where(torch.rand_like(x,dtype=torch.float32)<.01,16.,1.).to(x.dtype))
    config=None
    if args.mode in ('rtx_jit','rtx_current'):
        if args.forward_config is None: p.error('explicit winner required')
        scaling='current' if args.mode=='rtx_current' else 'jit_row_region'
        config=load_config(args.forward_config,shape,scaling)
        layer=rtx.NVFP4Linear(k,n,device='cuda',autotune='off',scaling=scaling,
            dynamic_config=config,scale_config=rtx.NVFP4ScaleConfig(tensor_scale_mode=config.tensor_scale_mode))
    else:
        layer=build(args.mode,k,n)
    with torch.no_grad(): layer.weight.copy_(w)
    with torch.no_grad(): reference=x.float()@w.float().T
    eager=layer(x).detach()
    compiled=torch.compile(layer,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    def run(): return compiled(x)
    output=run()
    torch.testing.assert_close(output,eager,rtol=0,atol=0)
    error=metrics(output.detach(),reference)
    if not error['finite'] or error['normalized_rms']>.2:
        raise RuntimeError(f'forward numerical gate failed: {error}')
    del output,eager,reference
    replay=replay_check(run,x,layer.weight,lambda:layer(x))
    timing=measure(run,time_based_warmup=True)
    quality=runtime_timing_metadata(timing['samples_ms'])
    provenance={}
    if args.mode=='quartet':
        import quartet2.linear
        source=Path(quartet2.linear.__file__)
        provenance={'linear_source':str(source),'linear_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
                    'backend':'FlashInfer/cuDNN','upstream_revision':'5f2a47e8f464b17ef4b1cac78db8d14edd1f3f7d',
                    'adapter':'isolated current-stream/GPU-seed research patch'}
    result=dict(mode=args.mode,shape=shape,seed=args.seed,distribution=args.distribution,metrics=error,timing=timing,quality=quality,
        replay=replay,config=asdict(config) if config else None,provenance=provenance,
        torch=torch.__version__,cuda=torch.version.cuda,device=str(torch.cuda.get_device_properties(0)),
        contract='Compiled fullgraph/static grad-enabled forward only; both BF16 operands dynamic; quantization included; no backward; graph8x10 samples15 warm200ms; TF32 off.',
        schedule='Explicit tuned config' if config else 'Fixed implementation defaults; vendor plan selection enabled')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as f: json.dump(result,f,indent=2)
    print(json.dumps({'mode':args.mode,'shape':shape,'us':timing['median_ms']*1000,
                      'nrms':error['normalized_rms'],'stationary':quality['sampling']['collection']['stationary']}),flush=True)


if __name__=='__main__': main()
