"""Validate public phase bindings on actual input rank before application use."""
import argparse
import json
from pathlib import Path

import torch
import rtx
from rtx.autotune.mlp import reference
from rtx.autotune.replacement_benchmark import error_metrics, require_recipe_metrics, source_manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--train-request',type=Path,required=True)
    parser.add_argument('--forward-request',type=Path,required=True)
    parser.add_argument('--input-shape',default='32,640,256')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--checkpoint',action='store_true',
        help='qualify a non-reentrant checkpoint around the bound forward, without changing compiler options')
    parser.add_argument('--fixed',action='store_true',
        help='research-only: validate the fixed-config functional boundary with the qualified phase plans')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources=source_manifest()
    candidate=rtx.MLPCandidate.load(args.train_request/'candidate.json')
    inference=rtx.MLPCandidate.load(args.forward_request/'candidate.json')
    if candidate.phase!='train' or inference.phase!='forward':
        raise ValueError('provide train and forward requests in their matching roles')
    shape=tuple(map(int,args.input_shape.split(',')))
    m,k,h,o=candidate.shape
    torch.manual_seed(999)
    module=rtx.MXFP8MLP(k,h,o,device='cuda',activation=candidate.activation,config=candidate.config)
    ids=tuple(id(p) for p in module.parameters())
    binding=rtx.bind_qualified_mlp_plans_(module,[(args.train_request,shape),(args.forward_request,shape)])
    assert ids==tuple(id(p) for p in module.parameters())
    target=module
    if args.fixed:
        def target(x):
            if tuple(x.shape)!=shape:
                raise ValueError('fixed research target only qualifies the exact requested shape')
            phase_config=candidate.config if torch.is_grad_enabled() else inference.config
            return rtx.mxfp8_mlp(x,module.gate_up_weight,module.down_weight,
                                activation=candidate.activation,config=phase_config)
    if args.checkpoint:
        from torch.utils.checkpoint import checkpoint
        uncheckpointed=target
        def target(x):
            return checkpoint(uncheckpointed,x,use_reentrant=False)
    compiled=torch.compile(target,fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
    cases=[]
    for seed in (999,1000):
        torch.manual_seed(seed)
        x=(torch.randn(*shape,device='cuda',dtype=torch.bfloat16)*.2).requires_grad_()
        dy=torch.randn(*shape[:-1],o,device='cuda',dtype=torch.bfloat16)*.1
        with torch.no_grad():
            for w in module.parameters():
                w.normal_(0,.02)
        y=compiled(x)
        actual=(y,*torch.autograd.grad(y,(x,*module.parameters()),dy))
        with torch.no_grad():
            expected=reference(x.reshape(m,k),*module.parameters(),dy.reshape(m,o),candidate.activation)
            forward=compiled(x)
        metrics={name:error_metrics(a.reshape_as(b),b) for name,a,b in
            zip(('output','dx','dgate_up','ddown'),actual,expected)}
        require_recipe_metrics(metrics,('output','dx','dgate_up','ddown'))
        forward_metrics={'output':error_metrics(forward.reshape(m,o),expected[0])}
        require_recipe_metrics(forward_metrics,('output',))
        cases.append(dict(seed=seed,train=metrics,forward=forward_metrics))
    assert ids==tuple(id(p) for p in module.parameters())
    report=dict(status='ok',input_shape=shape,torch=str(torch.__version__),cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),binding=binding,cases=cases,parameters_preserved=True,
        source_hashes_before=sources,source_hashes_after=source_manifest(),
        checkpoint='non_reentrant' if args.checkpoint else 'none',
        boundary='fixed_functional' if args.fixed else 'selected_bound',
        application_qualified=False,scope='actual-rank compiled train/no-grad independent recipe checks')
    if report['source_hashes_before']!=report['source_hashes_after']:
        raise RuntimeError('source changed during qualification')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps(dict(status=report['status'],input_shape=shape,parameters_preserved=True)))


if __name__=='__main__':
    main()
