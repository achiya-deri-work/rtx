"""CPU numerical experiment, NOT production-kernel parity or timing evidence.

Independent TorchAO FP4 encoder/decoder; native E4M3 scales retain RTX's
minimum-normal floor. Compare unchanged amax, bounded 4/6, and 4/6 with the
official outer headroom A/(6*256). The latter is not a complete reproduction
of official subnormal handling, CUDA reciprocals, reduction order, or training.
Official policy source:
https://github.com/mit-han-lab/fouroversix/blob/dadfad6901d473a734fe71e0b082e70ee993e23a/src/fouroversix/quantize/pytorch/reference.py
"""

import argparse
import json
import math

import torch
from torchao.prototype.mx_formats.kernels import f32_to_f4_unpacked, f4_unpacked_to_f32


def quantize(x, region_rows, policy):
    """Return decoded FP32 tensor plus operand error and selection statistics."""
    m,k=x.shape
    padded=torch.nn.functional.pad(x.float(),(0,(-k)%16,0,(-m)%region_rows))
    regions=padded.reshape(-1,region_rows,padded.shape[-1])
    amax=regions.abs().amax(dim=(1,2),keepdim=True)
    ceiling=256. if policy=='headroom_4over6' else 448.
    outer=torch.where(amax==0,torch.ones_like(amax),amax/(6*ceiling)).unsqueeze(-1)
    blocks=regions.reshape(regions.shape[0],region_rows,-1,16)
    maximum=blocks.abs().amax(-1,keepdim=True)
    candidates=[]
    for target in ((6.,) if policy=='amax' else (6.,4.)):
        raw=maximum/(target*outer)
        scale=raw.clamp(2**-6,448).to(torch.float8_e4m3fn).float()
        codes=f32_to_f4_unpacked(blocks*((1/outer)/scale))
        decoded=f4_unpacked_to_f32(codes)*(scale*outer)
        error=(blocks-decoded).square().sum(-1,keepdim=True)
        candidates.append((decoded,error,raw))
    baseline=candidates[0]
    use_four=torch.zeros_like(baseline[1],dtype=torch.bool)
    decoded=baseline[0]
    if len(candidates)==2:
        use_four=candidates[1][1]<baseline[1]
        decoded=torch.where(use_four,candidates[1][0],decoded)
    decoded=decoded.reshape_as(padded)[:m,:k]
    error=(decoded-x.float()).square().sum()
    return decoded,dict(sse=float(error),energy=float(x.float().square().sum()),
        finite=bool(decoded.isfinite().all()),selected_four_fraction=float(use_four.float().mean()),
        candidate_four_clipped_fraction=(float((candidates[1][2]>448).float().mean()) if len(candidates)==2 else 0.))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds',type=int,default=3)
    parser.add_argument('--shape',type=int,nargs=3,default=(128,128,256),metavar=('M','N','K'))
    parser.add_argument('--threads',type=int,default=4)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    m,n,k=args.shape
    print(json.dumps(dict(kind='environment',torch=torch.__version__,device='cpu',shape=args.shape,
        seeds=args.seeds,oracle='TorchAO native FP4 encode/decode, FP32 CPU GEMM',
        scope='numerical simulation only; no kernel timing, CUDA parity, or training qualification')))
    for seed in range(args.seeds):
        for distribution in ('normal','uniform','outliers','row_dynamic'):
            generator=torch.Generator().manual_seed(seed)
            x=torch.randn(m,k,generator=generator)
            w=torch.randn(n,k,generator=generator)/math.sqrt(k)
            if distribution=='uniform':
                x=torch.rand(m,k,generator=generator)-.5
                w=(torch.rand(n,k,generator=generator)-.5)*math.sqrt(12/k)
            if distribution=='outliers':
                x[:,::97]*=20
                w[:,::113]*=10
            if distribution=='row_dynamic':
                x*=torch.exp2(torch.linspace(-6,6,m))[:,None]
                w*=torch.exp2(torch.linspace(-3,3,n))[:,None]
            x,w=x.bfloat16(),w.bfloat16()
            reference=x.float()@w.float().T
            reference_energy=reference.square().sum()
            for region_rows in (1,8):
                for policy in ('amax','bounded_4over6','headroom_4over6'):
                    qx,xstats=quantize(x,region_rows,policy)
                    qw,wstats=quantize(w,region_rows,policy)
                    output=qx@qw.T
                    output_sse=(output-reference).square().sum()
                    print(json.dumps(dict(kind='result',seed=seed,distribution=distribution,
                        x_region_rows=region_rows,w_region_rows=region_rows,policy=policy,
                        x=xstats,w=wstats,output=dict(finite=bool(output.isfinite().all()),
                            sse=float(output_sse),energy=float(reference_energy),
                            normalized_rms=float((output_sse/reference_energy.clamp_min(1e-20)).sqrt()))),allow_nan=False))


if __name__=='__main__':
    main()
