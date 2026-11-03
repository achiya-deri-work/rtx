"""Matched default-compiled SiTU MLP timing, numerical and tuning evidence."""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint

import rtx
from rtx.configs.nvfp4_mlp import NVFP4MLPConfig
from rtx._nvfp4_mlp_reference import reference,situ
from qualify_mxfp8_row_alignment import measure


def errors(actual,expected):
    rows=[]
    for a,r in zip(actual,expected):
        delta=a.detach().float()-r.float()
        rms=float(delta.square().mean().sqrt())
        rr=float(r.float().square().mean().sqrt())
        rows.append(dict(finite=bool(a.isfinite().all()),rms=rms,reference_rms=rr,
                         relative_l2=rms/max(rr,1e-12),max_abs=float(delta.abs().max())))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape',action='append',help='M,K,H,O; repeat; aligned targets first')
    parser.add_argument('--trials',type=int,default=8)
    parser.add_argument('--seconds',type=float,default=120)
    parser.add_argument('--torch-max-autotune',action=argparse.BooleanOptionalAction,default=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    shapes=args.shape or ['512,768,1536,768','8192,768,1536,768','8128,768,1536,768']
    torch.manual_seed(31103);torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    options={'max_autotune':args.torch_max_autotune,'max_autotune_gemm':args.torch_max_autotune,
             'epilogue_fusion':True,'triton.cudagraphs':False}
    from rtx.autotune.nvfp4_mlp import source_identity
    report=dict(torch=str(torch.__version__),cuda=torch.version.cuda,
                gpu=str(torch.cuda.get_device_properties(0)),seed=31103,
                mode='default',compile='fullgraph_static',source=source_identity(),inductor_options=options,rows=[],tuning=[])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for shape in shapes:
        m,k,h,o=map(int,shape.split(','))
        torch._dynamo.reset()
        layer=rtx.NVFP4MLP(k,h,o,device='cuda')
        x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16,requires_grad=True)
        gu,down=layer.gate_up_weight,layer.down_weight
        dy=torch.randn(m,o,device='cuda',dtype=torch.bfloat16)*.1
        oracle=reference(x,gu,down,dy,layer.config)
        if args.trials:
            layer.autotune_(x,tuning_policy=rtx.AutotunePolicy(max_trials=args.trials,max_seconds=args.seconds,
                warmup=3,samples=7,calls_per_sample=10,confirmation_repeats=1),
                output_dir=args.output.parent/'tuning',progress=print)
            result=layer.tuning_result
            report['tuning'].append(dict(shape=[m,k,h,o],trials=result.evaluated_trials,
                elapsed_s=result.elapsed_s,median_ms=result.median_ms,context_id=result.context_id,
                store_path=result.store_path))
        config=layer.policy_for(x)
        nvfp4_actual=None
        for kind in ('nvfp4_situ','bf16_reentrant_checkpoint'):
            torch._dynamo.reset()
            if kind=='nvfp4_situ':
                fn=torch.compile(layer,fullgraph=True,dynamic=False,options=options)
            else:
                def body(a,b,d):
                    g,u=torch.nn.functional.linear(a,b).chunk(2,-1)
                    hidden=situ(g.float(),u.float(),config)[0].bfloat16()
                    return torch.nn.functional.linear(hidden,d)
                fn=torch.compile(lambda a:checkpoint(body,a,gu,down,use_reentrant=True),fullgraph=True,dynamic=False,options=options)
            def pair():
                y=fn(x)
                grads=torch.autograd.grad(y,(x,gu,down),dy)
                return y.detach(),*(g.detach() for g in grads)
            actual,train=measure(pair)
            with torch.no_grad():
                _,forward=measure(lambda:fn(x))
            row=dict(shape=[m,k,h,o],kind=kind,forward=forward,training=train,
                     quantization_aware_error=errors(actual,oracle),config=config.to_dict())
            if kind=='nvfp4_situ':
                assert all(v['finite'] and v['relative_l2']<.03 for v in row['quantization_aware_error']),row
                nvfp4_actual=tuple(t.detach().clone() for t in actual)
            else:
                report['rows'][-1]['bf16_reference_error']=errors(nvfp4_actual,actual)
            report['rows'].append(row)
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print('RESULT',shape,kind,'fwd_ms',forward['median_ms'],'train_ms',train['median_ms'],flush=True)
        del x,gu,down,layer,dy,oracle,actual,fn,nvfp4_actual
        rtx.clear_runtime_caches()
    print('PASS',flush=True)


if __name__=='__main__':
    main()
