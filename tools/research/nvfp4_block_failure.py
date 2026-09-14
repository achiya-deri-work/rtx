"""Localize the frozen first-hit block-scaling numerical failure; no timing."""
import argparse
import json
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--winner',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    output=Path(args.output)
    if output.exists():raise ValueError('do not overwrite numerical evidence')
    import torch
    import rtx
    from torchao.prototype.mx_formats.nvfp4_tensor import NVFP4Tensor
    from rtx.nvfp4_inference_autotune import dynamic_config_from_dict
    from rtx.runtime import validate_runtime_environment
    winner=json.loads(Path(args.winner).read_text())
    m,n,k=(winner['problem'][name] for name in ('m','n','k'))
    torch.manual_seed(2718)
    base=rtx.NVFP4Linear(k,n,device='cuda',backend='materialized',scaling='block',autotune='off').eval()
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.1
    records=[]
    def measure(label,actual,reference):
        delta=actual.float()-reference.float()
        rms=float(delta.square().mean().sqrt());ref_rms=float(reference.float().square().mean().sqrt())
        result=dict(label=label,finite=bool(actual.isfinite().all()),rms=rms,reference_rms=ref_rms,
                    normalized_rms=rms/max(ref_rms,1e-12),max_abs=float(delta.abs().max()))
        records.append(result);print(json.dumps(result),flush=True)
        assert result['finite']
    with torch.inference_mode():
        ref=(x.float()@base.weight.float().T).bfloat16()
        one=torch.ones((),device='cuda',dtype=torch.float32)
        qx=NVFP4Tensor.to_nvfp4(x,block_size=16,per_tensor_scale=one,use_triton_kernel=False).dequantize(torch.float32)
        qw=NVFP4Tensor.to_nvfp4(base.weight,block_size=16,per_tensor_scale=one,use_triton_kernel=False).dequantize(torch.float32)
        quant_ref=(qx@qw.T).bfloat16()
        measure('torchao_block_vs_bf16_reference',quant_ref,ref)
        measure('torchao_block_X',qx,x)
        measure('torchao_block_W',qw,base.weight)
        for scaling in ('block','current','jit_row_region','delayed'):
            for selected in ((False,True) if scaling=='block' else (False,)):
                torch.compiler.reset()
                kwargs=dict(dynamic_config=dynamic_config_from_dict(winner['config'])) if selected else {}
                layer=rtx.NVFP4Linear(k,n,device='cuda',backend='materialized',scaling=scaling,
                                    autotune='off',**kwargs).eval()
                layer.weight.copy_(base.weight)
                actual=torch.compile(layer,fullgraph=True,dynamic=False)(x)
                label=scaling+('_selected' if selected else '_default')
                measure(label,actual,ref)
                if scaling=='block':
                    measure(label+'_vs_torchao_block',actual,quant_ref)
                    torch.testing.assert_close(actual,quant_ref,rtol=0,atol=0)
                else:
                    assert records[-1]['normalized_rms'] < .20, records[-1]
    from .source_snapshot import write_snapshot
    output.parent.mkdir(parents=True,exist_ok=True)
    sources=write_snapshot(Path(__file__).resolve().parents[2],output.with_suffix('.sources.zip'))
    from .matched_gpu import write_new
    write_new(output,dict(shape=[m,n,k],seed=2718,records=records,source_snapshot=sources,
        environment=validate_runtime_environment(),device=torch.cuda.get_device_name(),
        winner=winner,weight_rms=float(base.weight.detach().float().square().mean().sqrt()),
        limitation='Numerical localization only; no performance or policy promotion.'))


if __name__=='__main__':main()
