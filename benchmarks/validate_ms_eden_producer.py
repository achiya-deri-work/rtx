"""Focused independent Hadamard/EDEN producer and zero-replay checks."""
import argparse
import json
from pathlib import Path

import torch


def validate(shape=(128,256), threads=(128,512)):
    from benchmarks.reference_ms_eden import prepare_extended
    from benchmarks.compare_nvfp4_packed_gemm import decode
    from rtx.kernels.nvfp4_posthoc import compile_hadamard_preparation,PosthocScaleConfig
    torch.manual_seed(824)
    original=torch.randn(*shape,device="cuda").bfloat16()
    signs=(torch.randint(2,(128,),device="cuda")*2-1).float()
    h=torch.ones(1,1,device="cuda",dtype=torch.float64)
    for _ in range(7):
        h=torch.cat((torch.cat((h,h),1),torch.cat((h,-h),1)),0)
    norm=torch.tensor(128**-.5,device="cuda",dtype=torch.float32).double()
    results=[]
    for transpose in (False,True):
        logical=original.T if transpose else original
        rotated=((logical.double().reshape(-1,128)*signs.double())@h*norm).float().reshape(logical.shape)
        reference=prepare_extended(rotated,mantissa_bits=4)
        for mode in ("direct","coalesced","coalesced_wide"):
            for count_threads in threads:
                # The high grid cap intentionally creates idle cooperative CTAs.
                source=original.clone()
                runner,count=compile_hadamard_preparation(*shape,PosthocScaleConfig(count_threads,128),4,transpose,"row_major",mode)
                packed=torch.empty(source.numel()//2,device="cuda",dtype=torch.uint8)
                scales=torch.empty(source.numel()//16,device="cuda",dtype=torch.bfloat16)
                partials=torch.full((count,),float("nan"),device="cuda")
                def run():
                    runner(source.flatten(),signs,packed,scales,partials)
                run()
                actual=decode(packed.reshape(logical.shape[0],logical.shape[1]//2),
                    torch.ones_like(scales).reshape(logical.shape[0],-1),1.)
                torch.testing.assert_close(actual.reshape(reference.values.shape),reference.values,rtol=0,atol=0)
                expected=reference.corrected_scales.flatten()
                different=scales!=expected
                if different.any():
                    bits=(scales.view(torch.int16).int()-expected.view(torch.int16).int()).abs()
                    assert (bits[different]==1).all()
                    midpoint=(scales.float()+expected.float())*.5
                    fp32=reference.fp32_corrected_scales.flatten()
                    assert ((midpoint-fp32).abs()[different] <=
                            (8*torch.finfo(torch.float32).eps*fp32.abs())[different]).all()
                assert partials.isfinite().all()
                torch.testing.assert_close(partials.amax(),scales.float().amax(),rtol=0,atol=0)
                stream=torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    run()
                stream.synchronize()
                graph=torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph,stream=stream):
                    run()
                source.zero_(); partials.fill_(float("nan"))
                graph.replay()
                assert (partials==0).all() and (packed==0).all() and (scales==0).all()
                results.append(dict(shape=list(shape),transpose=transpose,load=mode,
                    threads=count_threads,partial_count=count,packed_exact=True,
                    corrected_scale_midpoint_gate=True,zero_graph_replay=True))
                print("PASS",results[-1],flush=True)
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape",default="128,256")
    parser.add_argument("--output",required=True)
    args=parser.parse_args()
    result=validate(tuple(map(int,args.shape.split(","))))
    with Path(args.output).open("x") as handle:
        json.dump(result,handle,indent=2)


if __name__=="__main__":
    main()
