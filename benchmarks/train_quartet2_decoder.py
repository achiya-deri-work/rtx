"""Optional matched TinyStories decoder study of isolated Quartet-II patches.

Uses the existing RTX decoder recipe, BF16 boundaries, and identical original
weights/data across modes. Only internal projections are replaced. Not a public
backend integration or a throughput benchmark.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from rtx.models import DecoderConfig, DecoderOnlyTransformer, LinearSpec, causal_lm_loss
from benchmarks.evaluate_quartet2 import build


RESEARCH_POLICIES={
    'research_mx_mx':('mxfp8','mxfp8'),
    'research_fp4_fp4':('ms_eden','ms_eden'),
    'research_mx_fp4':('mxfp8','ms_eden'),
    'research_fp4_mx':('ms_eden','mxfp8'),
}


class ResearchBackwardLinear(torch.nn.Module):
    """Same exact-outer NVFP4 forward; only original-source backward differs."""
    def __init__(self, module, mode):
        super().__init__()
        if module.bias is not None:
            raise ValueError('research projection is bias-free')
        self.weight=module.weight
        self.dx_policy,self.dw_policy=RESEARCH_POLICIES[mode]

    def forward(self,x):
        from rtx import DEFAULT_MXFP8_BWD_CONFIG
        from rtx.experimental_ms_eden import ms_eden_linear
        shape=x.shape
        out=ms_eden_linear(x.reshape(-1,shape[-1]).contiguous(),self.weight,
            source_policy='original_bf16',forward_scaling='jit_row_region',
            mxfp8_backward_config=DEFAULT_MXFP8_BWD_CONFIG if 'mxfp8' in (self.dx_policy,self.dw_policy) else None,
            forward_outer_mode='exact',dx_policy=self.dx_policy,dw_policy=self.dw_policy)
        return out.reshape(*shape[:-1],self.weight.shape[0])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokens', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--mode', choices=('bf16','mxfp8','rtx_nvfp4','rtx_four_six','quartet','quartet_hybrid','quartet_ste',
                                    'rtx_jit_four_six','rtx_jit_amax_exact',*RESEARCH_POLICIES), required=True)
    p.add_argument('--seed', type=int, default=17)
    p.add_argument('--width', type=int, default=256)
    p.add_argument('--steps', type=int, default=250)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--layers', type=int, default=2)
    a = p.parse_args()
    if a.width % 128 or min(a.steps,a.batch,a.width,a.layers) <= 0:
        p.error('positive dimensions and width multiple of 128 required')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    with a.output.open('x') as handle:
        def emit(**v):
            line=json.dumps(v,allow_nan=False)
            handle.write(line+'\n'); handle.flush(); print(line,flush=True)
        torch.manual_seed(a.seed)
        torch.backends.cuda.matmul.allow_tf32=False
        tokens=np.load(a.tokens,mmap_mode='r')[:10_000_000]
        split=int(len(tokens)*.9)
        config=DecoderConfig(vocab_size=257,max_seq_len=128,num_layers=a.layers,
            hidden_size=a.width,intermediate_size=2*a.width,num_attention_heads=a.width//64,
            dtype=torch.bfloat16,linear=LinearSpec(precision='bf16'))
        model=DecoderOnlyTransformer(config,device='cuda')
        initial_hash=hashlib.sha256()
        for name,param in model.named_parameters():
            initial_hash.update(name.encode())
            initial_hash.update(param.detach().view(torch.uint8).cpu().numpy().tobytes())
        replacements=0
        if a.mode != 'bf16':
            for name,module in list(model.named_modules()):
                if name.startswith('layers.') and isinstance(module,torch.nn.Linear):
                    replacement=(ResearchBackwardLinear(module,a.mode) if a.mode in RESEARCH_POLICIES
                                 else build(a.mode,module.in_features,module.out_features))
                    replacement.weight=module.weight
                    parent,leaf=name.rsplit('.',1)
                    setattr(model.get_submodule(parent),leaf,replacement)
                    replacements+=1
        emit(kind='environment',mode=a.mode,seed=a.seed,width=a.width,steps=a.steps,
             torch=torch.__version__,gpu=torch.cuda.get_device_name(),replaced=replacements,
             initial_weights_sha256=initial_hash.hexdigest(),
             corpus_prefix_sha256=hashlib.sha256(tokens.tobytes()).hexdigest(),
             token_count=len(tokens),split=split,optimizer='BF16 fused AdamW',
             recipe='QK norm; pre/post norm; 0.02 trunc normal; depth residual scaling',
             compile='fullgraph static; cudagraphs disabled',batch=a.batch,sequence=128,layers=a.layers)
        if a.mode in RESEARCH_POLICIES:
            emit(kind='research_policy',dx_policy=RESEARCH_POLICIES[a.mode][0],
                 dw_policy=RESEARCH_POLICIES[a.mode][1],source='original_bf16',
                 forward='NVFP4 jit_row_region exact FP32 outer scaling',
                 scope='short decoder trajectory, not convergence qualification')
        def batch(step,validation=False):
            rng=np.random.default_rng(a.seed+step*0x9E3779B1)
            lo,hi=(split,len(tokens)) if validation else (0,split)
            offsets=rng.integers(lo,hi-129,size=a.batch)
            host=np.stack([tokens[i:i+129] for i in offsets]).astype(np.int64)
            data=torch.from_numpy(host).cuda()
            return data[:,:128],data[:,1:]
        def loss_fn(x,y):
            return causal_lm_loss(model(x),y)
        compiled=torch.compile(loss_fn,fullgraph=True,dynamic=False,
                               options={'triton.cudagraphs':False})
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),
                                    weight_decay=.1,fused=True)
        try:
            if a.mode in RESEARCH_POLICIES:
                with torch.no_grad():
                    initial_logits=model(batch(0)[0]).detach().contiguous()
                    initial_forward_hash=hashlib.sha256(initial_logits.view(torch.uint8).cpu().numpy().tobytes()).hexdigest()
                emit(kind='initial_forward',sha256=initial_forward_hash)
                del initial_logits
            # Compile both phases before resetting stochastic state for training.
            torch.cuda.synchronize()
            compilation_start=time.monotonic()
            loss=compiled(*batch(0)); loss.backward(); optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            emit(kind='compile_warmup',seconds=time.monotonic()-compilation_start,
                 scope='first compiled loss plus backward and gradient clear; not pure compilation')
            torch.manual_seed(a.seed+100_000)
            start=time.monotonic()
            for step in range(a.steps):
                optimizer.zero_grad(set_to_none=True)
                loss=compiled(*batch(step))
                loss.backward()
                norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.,error_if_nonfinite=True)
                optimizer.step()
                emit(kind='train',step=step+1,loss=float(loss.detach()),grad_norm=float(norm),
                     elapsed_seconds=time.monotonic()-start)
            with torch.no_grad():
                vals=[float(compiled(*batch(10_000_000+i,True))) for i in range(16)]
            finite=all(bool(v.isfinite().all()) for v in model.parameters())
            emit(kind='result',validation_losses=vals,validation_mean=float(np.mean(vals)),
                 finite_parameters=finite,elapsed_seconds=time.monotonic()-start)
        except Exception as exc:
            emit(kind='error',error=f'{type(exc).__name__}: {exc}')
            raise


if __name__=='__main__':
    main()
