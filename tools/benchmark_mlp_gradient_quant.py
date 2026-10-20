"""Research-only recompute-plus-row-quant A/B; not complete MLP timing."""
import argparse
import json
from pathlib import Path
import statistics

import torch

from rtx.autotune.replacement_benchmark import source_manifest
from rtx.kernels.mxfp8_glu_prequant import build_prequant_glu_runner
from rtx.kernels.mxfp8_quant import compile_mxfp8_quant
from rtx.kernels.mxfp8_quant import compile_mxfp8_transposed_quant
from rtx.mlp import MXFP8MLPConfig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',type=Path,required=True)
    p.add_argument('--shape',default='11712,256,768')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--hidden-transpose',action='store_true',
                   help='Compare fused gradient + separate hidden quantization against fusing both')
    a=p.parse_args()
    if a.output.exists():raise FileExistsError(a.output)
    sources=source_manifest()
    cfg=MXFP8MLPConfig.from_dict(json.loads(a.config.read_text())['challenger'])
    m,k,h=map(int,a.shape.split(','))
    if a.hidden_transpose and m%32:p.error('hidden transport baseline requires M divisible by 32')
    torch.manual_seed(947)
    x=torch.randn(m,k,device='cuda',dtype=torch.bfloat16)*.1
    w=torch.randn(2*h,k,device='cuda',dtype=torch.bfloat16)*.02
    dh=torch.randn(m,h,device='cuda',dtype=torch.bfloat16)*.1
    def buffers(fused=False):
        values=(torch.empty(2*m,h,device='cuda',dtype=torch.bfloat16),
                dh if fused and a.hidden_transpose else torch.empty(m,h,device='cuda',dtype=torch.bfloat16),
                torch.empty(2*m,h,device='cuda',dtype=torch.float8_e4m3fn),
                torch.empty(2*m,h//32,device='cuda',dtype=torch.float8_e8m0fnu))
        if a.hidden_transpose:
            values+= (torch.empty(h,m,device='cuda',dtype=torch.float8_e4m3fn),
                      torch.empty(h,m//32,device='cuda',dtype=torch.float8_e8m0fnu))
        return values
    plain=buffers();fused=buffers(True)
    common=dict(config=cfg.recompute,activation='silu',output_format='mxfp8',
                derivative=True,emit_hidden=True,separate_hidden=True)
    if a.hidden_transpose:
        base=build_prequant_glu_runner(x,w,plain[0],((dh,plain[1]),plain[2],plain[3]),
                                      emit_gradient_quant=True,**common)
        quant=compile_mxfp8_transposed_quant(h,m,cfg.down_backward.dw.resolved_quant_b())
        # This launcher accepts physical [M,H] and constructs its logical
        # [H,M] view in CuTe; no Python transpose argument is needed.
        logical_hidden=plain[1]
        candidate=build_prequant_glu_runner(x,w,fused[0],
            (((dh,fused[1]),fused[2],fused[3]),fused[4],fused[5]),
            emit_gradient_quant=True,emit_hidden_quant=True,**common)
        def baseline():
            base();quant(logical_hidden,plain[4],plain[5])
    else:
        base=build_prequant_glu_runner(x,w,plain[0],(dh,plain[1]),**common)
        quant=compile_mxfp8_quant(2*m,h,cfg.gate_backward.dx.quant_a)
        candidate=build_prequant_glu_runner(x,w,fused[0],((dh,fused[1]),fused[2],fused[3]),
                                          emit_gradient_quant=True,**common)
        def baseline():
            base();quant(plain[0],plain[2],plain[3])
    fns={'baseline':baseline,'fused':candidate}
    for _ in range(5):baseline();candidate()
    torch.cuda.synchronize()
    checks=[]
    for generation in range(3):
        x.add_(.001);w.add_(.0001);dh.mul_(.99)
        baseline();candidate();torch.cuda.synchronize()
        equal=[torch.equal(u.view(torch.uint8),v.view(torch.uint8))
               for i,(u,v) in enumerate(zip(plain,fused)) if not (a.hidden_transpose and i==1)]
        assert all(equal),equal
        checks.append(equal)
    samples={'baseline':[],'fused':[]}
    for round_index in range(6):
        for name in (('baseline','fused') if round_index%2==0 else ('fused','baseline')):
            timings=[]
            for _ in range(15):
                start,end=(torch.cuda.Event(enable_timing=True) for _ in range(2))
                start.record()
                for _ in range(20):fns[name]()
                end.record();end.synchronize()
                timings.append(start.elapsed_time(end)/20)
            samples[name].append(timings)
    assert sources==source_manifest(),'sources changed'
    report=dict(scope='recompute plus operand quantization only; not MLP F+B',
                hidden_transpose=a.hidden_transpose,
                shape=[m,k,h],torch=str(torch.__version__),cuda=torch.version.cuda,
                config=cfg.to_dict(),rtx_sources=sources,checks=checks,samples_ms=samples,
                median_ms={name:statistics.median(statistics.median(v) for v in rounds)
                           for name,rounds in samples.items()},installs_winner=False)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    with a.output.open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report['median_ms']),flush=True)


if __name__=='__main__':main()
