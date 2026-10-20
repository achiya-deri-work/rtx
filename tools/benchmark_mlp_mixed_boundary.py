"""Mixed-width attention/MLP checkpoint surrogate, never an AFRL qualification."""
import argparse
import gc
import hashlib
import json
from pathlib import Path
import statistics
import sys

import torch
from torch.utils.checkpoint import checkpoint
import rtx
from rtx.autotune.replacement_benchmark import source_manifest, error_metrics


class Mixed(torch.nn.Module):
    def __init__(self, image_config, backbone_config, tokenizer, boundary, depth):
        super().__init__()
        self.tokenizer, self.boundary = tokenizer, boundary
        self.image = rtx.MXFP8MLP(256, 768, 256, device='cuda', config=image_config)
        self.layers = torch.nn.ModuleList([
            rtx.MXFP8MLP(256,1536,256,device='cuda',config=backbone_config) for _ in range(depth)])
        for layer in (self.image, *self.layers):
            layer._first_hit_spec = json.dumps({'mode':'fixed','initial':layer.config.to_dict()},sort_keys=True)

    def encode(self, x):
        if self.tokenizer == 'mxfp8':
            return self.image(x)
        gu, down = tuple(self.image.parameters())
        gate, up = torch.nn.functional.linear(x,gu).chunk(2,dim=-1)
        hidden = (torch.nn.functional.silu(gate.float()) * up.float()).to(x.dtype)
        return torch.nn.functional.linear(hidden,down)

    def backbone(self, x):
        # Explicit synthetic connection: select 11296 of 20480 image tokens.
        # AFRL instead has masking, recurrent tokens and teacher branches.
        x = x.reshape(-1,256)[:11296].reshape(16,706,256).contiguous()
        for layer in self.layers:
            z = torch.nn.functional.layer_norm(x,(256,))
            q = z.reshape(16,706,4,64).transpose(1,2)
            attended = torch.nn.functional.scaled_dot_product_attention(q,q,q)
            x = x + attended.transpose(1,2).reshape_as(x)
            x = x + layer(torch.nn.functional.layer_norm(x,(256,)))
        return x

    def plain(self, x):
        return self.backbone(self.encode(x))

    def forward(self, x):
        if self.boundary == 'joint':
            return checkpoint(self.plain,x,use_reentrant=False)
        return checkpoint(self.backbone,checkpoint(self.encode,x,use_reentrant=False),use_reentrant=False)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner-index',type=Path,required=True)
    p.add_argument('--tokenizer',choices=('bf16','mxfp8'),required=True)
    p.add_argument('--boundary',choices=('joint','split'),required=True)
    p.add_argument('--depth',type=int,default=12)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.depth < 1:
        raise ValueError('depth must be positive')
    source=source_manifest()
    index=json.loads(args.owner_index.read_text())
    configs={tuple(r['shape']):rtx.MXFP8MLPConfig.from_dict(r['config']) for r in index if r['phase']=='train'}
    torch.manual_seed(317)
    model=Mixed(configs[20480,256,768,256],configs[11296,256,1536,256],args.tokenizer,args.boundary,args.depth)
    parameters=tuple(model.parameters()); ids=tuple(map(id,parameters))
    options={'triton.cudagraphs':False,'emulate_precision_casts':True}
    compiled=torch.compile(model,fullgraph=True,dynamic=False,options=options)
    reference=torch.compile(model.plain,fullgraph=True,dynamic=False,options=options)
    x=torch.empty((32,640,256),device='cuda',dtype=torch.bfloat16,requires_grad=True)
    dy=torch.empty((16,706,256),device='cuda',dtype=torch.bfloat16)
    def pair(fn, external):
        x.grad=None
        for weight in parameters:
            weight.grad=None
        # External reentrant checkpoint stays outside Dynamo/AOT as in owner.
        y=checkpoint(fn,x,use_reentrant=True) if external else fn(x)
        y.backward(dy)
        return y,x.grad,*(weight.grad for weight in parameters)
    checks=[]
    for seed in (317,318):
        torch.manual_seed(seed)
        with torch.no_grad():
            x.normal_(0,.2);dy.normal_(0,.1)
            for weight in parameters:
                weight.normal_(0,.02)
        expected=pair(reference,False)
        actual=pair(compiled,True)
        assert len(actual)==len(expected)==2+len(parameters)
        metrics=[error_metrics(a,b) for a,b in zip(actual,expected)]
        if not all(m['finite'] and m['normalized_rms']<=.025 for m in metrics):
            args.output.parent.mkdir(parents=True,exist_ok=True)
            with args.output.open('x') as stream:
                json.dump(dict(status='numerical_failure',metrics=metrics,seed=seed,command=sys.argv,
                    source_hashes=source,torch=str(torch.__version__)),stream,indent=2)
            raise AssertionError(metrics)
        checks.append(dict(seed=seed,metrics=metrics))
        del actual,expected
    for _ in range(3):
        pair(compiled,True)
    x.grad=None
    for weight in parameters:
        weight.grad=None
    torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache()
    resident=torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    samples=[];live=[]
    for _ in range(7):
        start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(3):
            pair(compiled,True)
        end.record();end.synchronize()
        samples.append(start.elapsed_time(end)/3)
        x.grad=None
        for weight in parameters:
            weight.grad=None
        torch.cuda.synchronize()
        live.append(torch.cuda.memory_allocated())
    assert len(set(live))==1,live
    assert tuple(map(id,model.parameters()))==ids
    assert source==source_manifest()
    report=dict(status='ok',command=sys.argv,torch=str(torch.__version__),cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),compile_options=options,source_hashes=source,
        tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        owner_index_sha256=hashlib.sha256(args.owner_index.read_bytes()).hexdigest(),
        tokenizer=args.tokenizer,boundary=args.boundary,depth=args.depth,
        configs={','.join(map(str,k)):v.to_dict() for k,v in configs.items()},
        checks=checks,parameters_preserved=True,live_bytes=live,resident_bytes=resident,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(),samples_ms=samples,
        median_ms=statistics.median(samples),workspace=rtx.workspace_report(),
        application_qualified=False,scope='synthetic token selection, SDPA and mixed-width MLP; no recurrent/teacher/optimizer state',
        oracle='compiled uncheckpointed same-recipe graph; not independent quantization or convergence oracle')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps({k:report[k] for k in ('status','median_ms','resident_bytes','peak_allocated_bytes')}),flush=True)


if __name__=='__main__':
    main()
