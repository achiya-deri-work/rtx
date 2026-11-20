"""Exercise MoE as a client application. No private RTX APIs/configuration.

Uses regular nn.Module, optimizer, state_dict, compile and inference workflows.
Wall times describe this small host-fed app, not standalone kernel performance.
"""
import argparse
import collections
import copy
import json
from pathlib import Path
import time
import traceback

import torch
from torch import nn
import rtx
from examples.balanced_moe import TinyTokenClassifier

INFERENCE_CONTEXT=torch.inference_mode


def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')


def journals(root):
    return {str(p.relative_to(root)):sum(bool(l.strip()) for l in p.read_text().splitlines())
            for p in root.rglob('observations.jsonl')}


def timed(call):
    torch.cuda.synchronize();start=time.perf_counter();value=call();torch.cuda.synchronize()
    return value,time.perf_counter()-start


def parameter_types(model):
    return dict(collections.Counter(str(p.dtype) for p in model.parameters()))


def train(root):
    torch.manual_seed(20261120)
    cache=root/'cache'
    model=TinyTokenClassifier(device='cuda',autotune_cache_dir=cache)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-3)
    compiled=torch.compile(model,fullgraph=True,dynamic=False)
    tokens_cpu=torch.randint(256,(2,64));tokens=tokens_cpu.cuda();targets=tokens%32
    report=dict(stage='train',torch=torch.__version__,rtx=rtx.__version__,gpu=torch.cuda.get_device_name(),
                repr=repr(model.moe),parameter_dtypes=parameter_types(model),steps=[])
    before={n:p.detach().clone() for n,p in model.named_parameters() if n.startswith('moe.')}
    for step in range(8):
        def iteration():
            optimizer.zero_grad(set_to_none=True)
            output=compiled(tokens)
            loss=nn.functional.cross_entropy(output.flatten(0,1),targets.flatten())
            loss.backward();optimizer.step()
            return loss.detach()
        loss,wall=timed(iteration)
        missing=[n for n,p in model.named_parameters() if p.grad is None]
        finite=all(bool(p.grad.isfinite().all()) for p in model.parameters() if p.grad is not None)
        if missing or not finite:raise AssertionError(('Gradient check',missing,finite))
        report['steps'].append(dict(step=step,loss=loss.item(),wall_s=wall))
        print('TRAIN',report['steps'][-1],flush=True);write(root/'train.json',report)
    report['moe_changed_parameters']=[n for n,p in model.named_parameters() if n in before and not torch.equal(p,before[n])]
    report['optimizer_state_dtypes']=dict(collections.Counter(str(v.dtype) for state in optimizer.state.values()
        for key,v in state.items() if key!='step' and isinstance(v,torch.Tensor)))
    model.eval()
    with INFERENCE_CONTEXT():
        expected,cold=timed(lambda:compiled(tokens))
        repeated,hot=timed(lambda:compiled(tokens))
        torch.testing.assert_close(expected,repeated,rtol=0,atol=0)
    report['first_inference_wall_s']=cold;report['second_inference_wall_s']=hot
    report['journals']=journals(cache)
    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),tokens=tokens_cpu,
                    expected_inference=expected.cpu()),root/'checkpoint.pt')
    report['checkpoint_bytes']=(root/'checkpoint.pt').stat().st_size
    report['complete']=True;write(root/'train.json',report)


def restore(root):
    checkpoint=torch.load(root/'checkpoint.pt',map_location='cpu',weights_only=True)
    cache=root/'cache';before=journals(cache)
    # Ordinary CPU construction/load followed by device movement, no repack.
    model=TinyTokenClassifier(autotune_cache_dir=cache)
    model.load_state_dict(checkpoint['model']);model=model.cuda()
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-3)
    optimizer.load_state_dict(checkpoint['optimizer'])
    compiled=torch.compile(model,fullgraph=True,dynamic=False)
    tokens=checkpoint['tokens'].cuda();model.eval()
    with INFERENCE_CONTEXT():
        output,wall=timed(lambda:compiled(tokens))
        torch.testing.assert_close(output.cpu(),checkpoint['expected_inference'],rtol=2e-3,atol=2e-5)
    model.train();optimizer.zero_grad(set_to_none=True)
    loss=nn.functional.cross_entropy(compiled(tokens).flatten(0,1),(tokens%32).flatten())
    loss.backward();optimizer.step()
    after=journals(cache)
    if before!=after:raise AssertionError('Fresh process performed additional tuning')
    write(root/'restore.json',dict(complete=True,first_restored_inference_wall_s=wall,
        resumed_loss=loss.item(),cache_journals_unchanged=True,parameter_dtypes=parameter_types(model),
        inference_max_abs=float((output.cpu()-checkpoint['expected_inference']).abs().max())))
    print('RESTORE PASS: model/optimizer reload, matching inference, train resume, no retune',flush=True)


def edges(root):
    results=[]
    def probe(name,call):
        try:
            detail=call();results.append(dict(name=name,status='ok',detail=detail))
        except Exception as error:
            results.append(dict(name=name,status='error',error=f'{type(error).__name__}: {error}',
                                traceback=traceback.format_exc(limit=5)))
            if any(word in str(error).lower() for word in ('illegal memory access','device-side assert','illegal instruction')):
                write(root/'edges.json',results)
                raise
        print('PROBE',name,results[-1]['status'],flush=True);write(root/'edges.json',results)
    def make():return rtx.BalancedMoE(128,16,192,top_k=2,shared_hidden_features=128,autotune='off').cuda()
    def inference():
        model=make().eval();fn=torch.compile(model,fullgraph=True,dynamic=False)
        x=torch.randn(2,64,128,device='cuda',dtype=torch.bfloat16)
        with torch.inference_mode():y=fn(x)
        return dict(shape=list(y.shape),dtype=str(y.dtype))
    probe('CPU constructor -> cuda -> compiled inference',inference)
    def packed():
        model=make();fn=torch.compile(model,fullgraph=True,dynamic=False)
        x=torch.randn(128,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        cu=torch.tensor([0,48,48,128],device='cuda',dtype=torch.int32)
        y=fn(x,cu_seqlens=cu);y.float().square().mean().backward()
        assert x.grad is not None and bool(x.grad.isfinite().all())
        return dict(shape=list(y.shape),empty_sequence=True)
    probe('compiled packed varlen including empty sample',packed)
    def frozen():
        model=make().requires_grad_(False);fn=torch.compile(model,fullgraph=True,dynamic=False)
        x=torch.randn(2,64,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        fn(x).float().square().mean().backward()
        assert x.grad is not None and bool(x.grad.isfinite().all())
        assert all(p.grad is None for p in model.parameters())
        return 'input gradients pass; parameter gradients remain absent'
    probe('frozen MoE with upstream input gradients',frozen)
    def ema():
        model=make().eval();other=copy.deepcopy(model)
        a={p.data_ptr() for p in model.parameters()};b={p.data_ptr() for p in other.parameters()}
        assert not a&b
        x=torch.randn(2,64,128,device='cuda',dtype=torch.bfloat16)
        with torch.no_grad():torch.testing.assert_close(model(x),other(x),rtol=0,atol=0)
        return 'deepcopy has independent storage and identical output'
    probe('deepcopy for EMA model',ema)
    def meta():
        state={n:p.cpu() for n,p in make().state_dict().items()}
        model=rtx.BalancedMoE(128,16,192,top_k=2,shared_hidden_features=128,device='meta',autotune='off')
        model.load_state_dict(state,assign=True);model=model.cuda()
        x=torch.randn(2,64,128,device='cuda',dtype=torch.bfloat16)
        return dict(shape=list(model(x).shape))
    probe('meta construction -> assign state_dict -> cuda',meta)
    def cast():
        model=make();before=str(model.router.logit_scale.dtype);model.bfloat16()
        x=torch.randn(2,64,128,device='cuda',dtype=torch.bfloat16)
        y=model(x)
        return dict(scalar_before=before,scalar_after=str(model.router.logit_scale.dtype),shape=list(y.shape))
    probe('whole-layer bfloat16 conversion precision audit',cast)
    def amp():
        model=make();x=torch.randn(2,64,128,device='cuda',dtype=torch.float32)
        with torch.autocast('cuda',dtype=torch.bfloat16):return str(model(x).dtype)
    probe('FP32 input under BF16 autocast',amp)
    def amp_bf16():
        model=make();x=torch.randn(2,64,128,device='cuda',dtype=torch.bfloat16)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            scores=model.router.log_affinities(x.flatten(0,1))
            print('AMP affinity dtype',scores.dtype,flush=True)
            y=model(x)
        return dict(output_dtype=str(y.dtype),affinity_dtype=str(scores.dtype))
    probe('BF16 input under BF16 autocast',amp_bf16)
    def strided():
        model=make();x=torch.randn(64,2,128,device='cuda',dtype=torch.bfloat16).transpose(0,1)
        return str(model(x).shape)
    probe('sequence-first transpose without contiguous copy',strided)
    def fresh_shape():
        model=make();fn=torch.compile(model,fullgraph=True,dynamic=False)
        for length in (64,80,64):
            x=torch.randn(2,length,128,device='cuda',dtype=torch.bfloat16,requires_grad=True)
            fn(x).float().square().mean().backward();model.zero_grad(set_to_none=True)
        return 'multiple shapes forward/backward pass'
    probe('two token lengths through one compiled layer',fresh_shape)


def main():
    global INFERENCE_CONTEXT
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('stage',choices=('train','restore','edges'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--no-grad-inference',action='store_true',help='Compare ordinary no_grad with cold inference_mode')
    args=p.parse_args()
    if args.no_grad_inference:INFERENCE_CONTEXT=torch.no_grad
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    try:{'train':train,'restore':restore,'edges':edges}[args.stage](root)
    except Exception as error:
        write(root/(args.stage+'-failure.json'),dict(error=f'{type(error).__name__}: {error}',traceback=traceback.format_exc()))
        raise
    write(root/(args.stage+'-complete.json'),dict(complete=True))
    print('STAGE_DONE',args.stage,flush=True)


if __name__=='__main__':main()
