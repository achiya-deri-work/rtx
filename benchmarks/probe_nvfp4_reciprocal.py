"""Localize quantizer reciprocal drift; diagnostic correctness, not timing."""
import argparse
from dataclasses import replace
import torch
from rtx.configs.nvfp4 import NVFP4QuantConfig, nvfp4_storage_k
from rtx.fp4 import compile_nvfp4_quant, _current_tensor_scale
from rtx.autotune.first_hit_study import atomic_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    records=[]
    cases=[(m,k,seed) for m,k in ((128,128),(257,1537),(1024,3072)) for seed in (0,1)]
    # Exact shape/seed pairs from the 5090 weight-prequant failure journal.
    cases += [(127,1536,3017910726),(1009,257,2063734097),(513,145,1616336479)]
    def measure_case(m,k,seed):
            generator=torch.Generator(device='cuda').manual_seed(seed)
            x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16,generator=generator)
            ts=_current_tensor_scale(x)
            outputs={}
            for mode in ('direct','e4m3_lut','rcp_approx'):
                config=replace(NVFP4QuantConfig(),scale_reciprocal=mode)
                q=torch.empty((m,nvfp4_storage_k(k)//2),device='cuda',dtype=torch.uint8)
                s=torch.empty((m,nvfp4_storage_k(k)//16),device='cuda',dtype=torch.float8_e4m3fn)
                compile_nvfp4_quant(m,k,config)(x,q,s,ts)
                torch.cuda.synchronize()
                # Independent nibble decode and FP32 reconstruction; no GEMM.
                code=torch.stack((q & 15,q >> 4),-1).flatten(1)
                levels=torch.tensor([0,.5,1,1.5,2,3,4,6],device='cuda')
                decoded=levels[(code & 7).long()]*torch.where(code<8,1.,-1.)
                decoded*=s.float().repeat_interleave(16,dim=1)*ts
                outputs[mode]=(q,s,decoded[:,:k])
            for mode in ('e4m3_lut','rcp_approx'):
                q,s,y=outputs[mode]; q0,s0,y0=outputs['direct']
                delta=y-y0
                row=dict(shape=[m,k],seed=seed,mode=mode,
                    packed_byte_mismatches=int((q!=q0).sum()),
                    scale_mismatches=int((s.float()!=s0.float()).sum()),
                    changed_values=int((y!=y0).sum()),
                    finite=bool(y.isfinite().all()),max_abs=float(delta.abs().max()),
                    rms_error=float(delta.square().mean().sqrt()),
                    reference_rms=float(y0.square().mean().sqrt()),
                    relative_l2=float(delta.norm()/y0.norm().clamp_min(1e-20)))
                records.append(row);print(row,flush=True)
    for m,k,seed in cases:
        measure_case(m,k,seed)
    atomic_json(args.output,dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(),
        oracle='Independent FP4 nibble decode; candidate versus direct quantizer, not BF16 equivalence',
        records=records))


if __name__=='__main__':main()
