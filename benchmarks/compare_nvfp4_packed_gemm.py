"""Optional Quartet/FlashInfer versus RTX GEMM diagnostic on identical FP4 bytes.

Requires the isolated Quartet research installation; never a runtime dependency.
"""
import argparse
from dataclasses import asdict, replace
import json
from pathlib import Path

import torch
from benchmarks.profile_preparation_phases import measure


def decode(packed, scales, tensor_scale):
    grid = torch.tensor([0,.5,1,1.5,2,3,4,6,0,-.5,-1,-1.5,-2,-3,-4,-6],device=packed.device)
    data = packed.to(torch.int64)
    indices = torch.stack((data & 15, data >> 4),dim=-1).flatten(-2)
    return grid[indices] * scales.float().repeat_interleave(16,dim=1) * tensor_scale


def native_scales(scales):
    """Bit-preserving row-major to SM120 mma128 scale layout conversion."""
    rows,blocks=scales.shape
    if rows % 128 or blocks % 8:
        raise ValueError("mma128 scale conversion requires 128 rows and 8 scale blocks")
    return (scales.view(torch.uint8).reshape(rows//128,4,32,blocks//8,2,4)
            .permute(0,3,4,2,1,5).contiguous().reshape(rows//128,blocks//8,1024)
            .view(scales.dtype))


def row_major_scales(scales, rows, k, layout):
    """Independent inverse layout used only by decoded-operand references."""
    if layout == "row_major":
        return scales.reshape(rows, k // 16)
    if layout != "mma128" or rows % 128 or k % 128:
        raise ValueError("unsupported native scale layout/shape")
    return (scales.view(torch.uint8).reshape(rows//128, k//128, 2, 32, 4, 4)
            .permute(0,4,3,1,2,5).contiguous().reshape(rows,k//16).view(scales.dtype))


def gemm_config_from_record(record, shape):
    """Accept a schedule or a shape-checked forward research winner."""
    from rtx.configs.nvfp4 import NVFP4GemmConfig
    if 'shape' in record and tuple(record['shape']) != tuple(shape):
        raise ValueError('packed diagnostic winner shape mismatch')
    raw = record.get('config', record)
    if raw.get('l2_fetch_granularity') is not None:
        raise ValueError('this diagnostic requires default L2 fetch policy')
    return NVFP4GemmConfig(**raw.get('gemm', raw))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape",default="2048,768,1536")
    p.add_argument("--output",required=True)
    p.add_argument("--config-json",help="Fixed NVFP4 GEMM schedule; operands stay identical")
    p.add_argument("--native-scales",action="store_true")
    args = p.parse_args()
    from quartet2.quant import quant_fp4, NVFP4QuantMode
    from quartet2.linear import unblock
    from flashinfer import mm_fp4
    from rtx.fp4 import compile_nvfp4_gemm, _packed_fp4_view
    from rtx.configs.nvfp4 import NVFP4GemmConfig, NVFP4Problem
    m,n,k = map(int,args.shape.split(","))
    torch.manual_seed(20261023)
    x = torch.randn(m,k,device="cuda",dtype=torch.bfloat16)*.25
    w = torch.randn(n,k,device="cuda",dtype=torch.bfloat16)*.02
    qx = quant_fp4(x,scale_override=1.,mode=NVFP4QuantMode.FOUR_SIX)
    qw = quant_fp4(w,scale_override=1.,mode=NVFP4QuantMode.FOUR_SIX)
    sx = unblock(qx.micro_scales,m,k).contiguous()
    sw = unblock(qw.micro_scales,n,k).contiguous()
    alpha = (qx.tensor_scale * qw.tensor_scale).reshape(1)
    a,b = _packed_fp4_view(qx.fp4),_packed_fp4_view(qw.fp4)
    out_cute = torch.empty(m,n,device="cuda",dtype=torch.bfloat16)
    out_vendor = torch.empty_like(out_cute)
    raw = json.loads(Path(args.config_json).read_text()) if args.config_json else {}
    config = gemm_config_from_record(raw, (m,n,k))
    if args.native_scales:
        config=replace(config,scale_layout="mma128",scale_role="tma")
    if raw.get("l2_fetch_granularity") is not None:
        raise ValueError("this diagnostic requires default L2 fetch policy")
    sx_kernel,sw_kernel=(native_scales(sx),native_scales(sw)) if config.scale_layout=="mma128" else (sx,sw)
    runner = compile_nvfp4_gemm(NVFP4Problem(m,n,k),config)
    def cute_run():
        runner(a,b,sx_kernel,sw_kernel,out_cute,alpha)
    def vendor_run():
        mm_fp4(qx.fp4,qw.fp4.T,qx.micro_scales.view(-1,k//16),
               qw.micro_scales.view(-1,k//16).T,alpha,torch.bfloat16,out_vendor,
               block_size=16,use_8x4_sf_layout=False,backend="cudnn",use_nvfp4=True)
    reference = decode(qx.fp4,sx,qx.tensor_scale) @ decode(qw.fp4,sw,qw.tensor_scale).T
    metrics = {}
    for name,fn,out in (("rtx",cute_run,out_cute),("flashinfer_cudnn",vendor_run,out_vendor)):
        fn()
        torch.cuda.synchronize()
        error = out.float()-reference
        rms, rrms = error.square().mean().sqrt().item(),reference.square().mean().sqrt().item()
        metrics[name] = dict(finite=bool(out.isfinite().all()),rms=rms,reference_rms=rrms,
                             nrms=rms/max(rrms,1e-12),max_abs=error.abs().max().item())
        # Only accumulation and final BF16 conversion differ; no FP4 quantizer
        # error belongs in this gate since both consume identical packed data.
        if not metrics[name]["finite"] or rms > max(.005*rrms,1e-6):
            raise RuntimeError(f"packed GEMM reference failed: {name} {metrics[name]}")
    def convert_scales():
        sx.copy_(unblock(qx.micro_scales,m,k))
        sw.copy_(unblock(qw.micro_scales,n,k))
    result = dict(shape=[m,n,k],config=asdict(config),metrics=metrics,
        timings={"rtx_gemm":measure(cute_run),"flashinfer_cudnn_gemm":measure(vendor_run),
                 "blocked_to_row_major_scale_copy":measure(convert_scales)},
        torch=torch.__version__,cuda=torch.version.cuda,device=str(torch.cuda.get_device_properties(0)),
        seed=20261023,policy="identical_Quartet_FourOverSix_packed_operands",
        caveat="Fixed RTX schedule; scale layout conversion measured separately; caller owns tuning and confirmation.")
    if config.scale_layout=="mma128":
        result["timings"]["row_major_to_mma128_scale_copy"]=measure(
            lambda:(sx_kernel.copy_(native_scales(sx)),sw_kernel.copy_(native_scales(sw))))
    from rtx.autotune.measurement import runtime_timing_metadata
    result["timing_quality"] = {name:runtime_timing_metadata(value["samples_ms"])
                                for name,value in result["timings"].items()}
    path=Path(args.output)
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("x") as handle:
        json.dump(result,handle,indent=2)
    print(json.dumps({"metrics":metrics,"median_ms":{key:value["median_ms"] for key,value in result["timings"].items()}}),flush=True)


if __name__ == "__main__":
    main()
