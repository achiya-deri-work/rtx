"""Byte-level policy comparison; diagnostic evidence, never a legality override."""
import argparse
from dataclasses import asdict,replace
import json
import math
from pathlib import Path

import torch

from rtx.autotune.replacement_benchmark import source_manifest
from rtx.kernels.mxfp8_quant import MXFP8QuantConfig,compile_mxfp8_quant,compile_mxfp8_transposed_quant


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--quick',action='store_true',help='selected exponents only; not the full scale sweep')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    sources=source_manifest()
    bits=torch.arange(65536,dtype=torch.int32,device='cuda')
    finite_bits=bits[(bits&0x7f80)!=0x7f80]
    values=finite_bits.to(torch.int16).view(torch.bfloat16)
    rows=values.numel()
    src=values[:,None].expand(rows,32).clone()
    baseline=MXFP8QuantConfig(quant_vec=4,load_bits=32,quant_store_bits=32,
        num_warps=8,persistent_waves=4,quant_math='fp32',quant_amax='fp32')
    candidate=replace(baseline,quant_math='bf16x2',quant_amax='bf16_bits')
    runners={}
    for axis in ('row','transposed'):
        compile_fn=compile_mxfp8_quant if axis=='row' else compile_mxfp8_transposed_quant
        runners[axis]=[(compile_fn(rows,32,c),torch.empty_like(src,dtype=torch.float8_e4m3fn),
            torch.empty((rows,1),device='cuda',dtype=torch.float8_e8m0fnu)) for c in (baseline,candidate)]
    exponents=(-126,-64,-8,0,8,64,127) if args.quick else range(-126,128)
    cases=[('uniform',None)]+[(f'anchor_2pow_{e}',math.ldexp(1.,e)) for e in exponents]
    cases.append(('anchor_max_finite',torch.finfo(torch.bfloat16).max))
    records=[]
    observed_codes=set()
    for name,anchor in cases:
        src[:,-1].copy_(values if anchor is None else torch.full_like(values,anchor))
        physical=src.t().contiguous()
        for axis in runners:
            for fn,q,s in runners[axis]:
                fn(src if axis=='row' else physical,q,s)
            torch.cuda.synchronize()
            qa,sa=runners[axis][0][1:]
            qb,sb=runners[axis][1][1:]
            mismatch=(qa.view(torch.uint8)!=qb.view(torch.uint8))
            scale_mismatch=(sa.view(torch.uint8)!=sb.view(torch.uint8))
            count=int(mismatch.sum())
            scale_count=int(scale_mismatch.sum())
            observed_codes.update(sa.view(torch.uint8).unique().cpu().tolist())
            row=dict(case=name,axis=axis,quantized_byte_mismatches=count,scale_byte_mismatches=scale_count)
            if count:
                indices=mismatch.nonzero()[:8]
                row['examples']=[dict(input_bits=int(src[i,j].view(torch.int16))&65535,
                    baseline_byte=int(qa[i,j].view(torch.uint8)),candidate_byte=int(qb[i,j].view(torch.uint8)),
                    baseline_scale=int(sa[i,0].view(torch.uint8)),candidate_scale=int(sb[i,0].view(torch.uint8)))
                    for i,j in indices.cpu().tolist()]
            records.append(row)
        if name=='uniform' or len(records)%32==0:
            print(json.dumps(dict(case=name,completed=len(records),mismatches=sum(r['quantized_byte_mismatches']+r['scale_byte_mismatches'] for r in records))),flush=True)
    assert sources==source_manifest(),'sources changed during audit'
    result=dict(torch=str(torch.__version__),cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),
        finite_bf16_patterns=rows,block_size=32,full_exponent_sweep=not args.quick,
        policies={'baseline':asdict(baseline),'candidate':asdict(candidate)},
        observed_scale_codes=sorted(observed_codes),records=records,source_hashes=sources,
        passed=all(r['quantized_byte_mismatches']==r['scale_byte_mismatches']==0 for r in records),
        limitations=['finite inputs only; NaN/Inf behavior is not covered',
            'all finite BF16 values with uniform blocks and chosen maxima; not all 32-value combinations',
            'standalone quantizers, not the fused recompute implementation',
            'no kernel guard, numerical policy or runtime winner changed'])
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result,stream,indent=2)
    print(json.dumps(dict(passed=result['passed'],cases=len(records),scale_codes=len(observed_codes))),flush=True)
    if not result['passed']:
        raise SystemExit(1)


if __name__=='__main__':
    main()
