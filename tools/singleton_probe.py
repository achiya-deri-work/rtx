"""Disposable-process quant/GEMM phase isolation for singleton-row failures."""
import argparse
from dataclasses import replace
import json
import torch
from tools.predictor_study import atomic


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--m',type=int,default=1);p.add_argument('--n',type=int,default=1280)
    p.add_argument('--k',type=int,default=896);p.add_argument('--epilogue',choices=('direct','tma'),default='direct')
    p.add_argument('--output',required=True);a=p.parse_args()
    from rtx.fp8 import _build_prequant_runner,_intern_prequant_config
    from rtx.kernels.mxfp8 import MXFP8Problem
    from rtx.prequant_experiments import _reference_prequant_config
    torch.manual_seed(20260919)
    x=torch.randn(a.m,a.k,device='cuda',dtype=torch.bfloat16)*.1
    w=torch.randn(a.n,a.k,device='cuda',dtype=torch.bfloat16)*.02
    config=_reference_prequant_config(MXFP8Problem(a.m,a.n,a.k))
    config=replace(config,gemm=replace(config.gemm,epilogue=a.epilogue))
    result=dict(shape=[a.m,a.n,a.k],epilogue=a.epilogue,status='error',phase='compile')
    try:
        runner=_build_prequant_runner(x,w,_intern_prequant_config(config))
        result['phase']='quant'
        if config.quant_launches=='dual':runner.quant_x(x,w,runner.qx,runner.qw,runner.sx,runner.sw)
        else:
            runner.quant_x(x,runner.qx,runner.sx);runner.quant_w(w,runner.qw,runner.sw)
        torch.cuda.synchronize();print('QUANT OK',flush=True)
        # Independent dequantization of the actual quantized operands isolates
        # GEMM correctness from expected FP8 quantization error versus BF16.
        def dq(q,s):
            scale=torch.exp2(s.view(torch.uint8).int().float()-127)
            return q.float()*scale.repeat_interleave(32,dim=-1)
        reference=dq(runner.qx,runner.sx)@dq(runner.qw,runner.sw).T
        out=torch.empty((a.m,a.n),device='cuda',dtype=torch.bfloat16)
        result['phase']='gemm';runner.gemm(runner.qx,runner.qw,runner.sx,runner.sw,out)
        torch.cuda.synchronize()
        err=out.float()-reference
        result.update(status='ok',max_abs=err.abs().max().item(),
            normalized_rms=(err.norm()/reference.norm().clamp_min(1e-20)).item(),finite=bool(out.isfinite().all()))
        if not result['finite'] or result['normalized_rms']>.005:raise AssertionError(result)
    except Exception:
        import traceback
        result['status']='error';result['error']=traceback.format_exc();raise
    finally:atomic(a.output,result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
