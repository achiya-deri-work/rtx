"""Bounded, matched TinyStories decoder study of recomputed MLP precision.

One fresh process per policy. Attention, norms, embeddings and optimizer are
identical. This is a short trajectory/integration test, not a convergence claim.
"""
import argparse
from dataclasses import replace
import json
import hashlib
from pathlib import Path
import statistics
import time

import numpy as np
import torch

from rtx import MXFP8MLP, MXFP8MLPConfig
from rtx.models import DecoderConfig, DecoderOnlyTransformer, causal_lm_loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--precision",choices=("bf16","mxfp8","bf16_gate"),required=True)
    p.add_argument("--activation",choices=("silu","gelu"),default="silu")
    p.add_argument("--steps",type=int,default=300)
    p.add_argument("--batch",type=int,default=8)
    p.add_argument("--sequence",type=int,default=128)
    p.add_argument("--seed",type=int,default=123)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--tokens",default="training_data/tinystories-train-00000.byte_tokens.npy")
    p.add_argument("--mlp-config",help="Explicit revisioned MLP schedule or tuning result")
    p.add_argument("--cuda-graph",action="store_true",help="Capture forward, backward and AdamW; copy new batches into static inputs")
    a=p.parse_args()
    if min(a.steps,a.batch,a.sequence)<=0:
        p.error("steps, batch and sequence must be positive")
    out=Path(a.output_dir);out.mkdir(parents=True,exist_ok=False)
    torch.manual_seed(a.seed)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction=False
    cfg=DecoderConfig(vocab_size=257,max_seq_len=a.sequence,num_layers=2,hidden_size=256,
                      intermediate_size=512,num_attention_heads=4)
    model=DecoderOnlyTransformer(cfg,device="cuda")
    schedule=MXFP8MLPConfig()
    if a.mlp_config:
        values=json.loads(Path(a.mlp_config).read_text())
        if "precision" in values and values["precision"]!=a.precision:
            raise ValueError("tuned numerical policy does not match the requested training precision")
        if "shape" in values and values["shape"]!=[a.batch*a.sequence,256,512,256]:
            raise ValueError("tuned shape does not match this decoder workload")
        if "activation" in values and values["activation"]!=a.activation:
            raise ValueError("tuned activation does not match the decoder")
        schedule=MXFP8MLPConfig.from_dict(values.get("mlp_preset",values.get("config",values)))
    for layer in model.layers:
        old=layer.mlp
        new=MXFP8MLP(256,512,precision=a.precision,activation=a.activation,config=schedule,device="meta")
        # Preserve the decoder's exact 0.02/depth-scaled initialization.
        new.gate_up_weight=torch.nn.Parameter(old.gate_up_proj.weight.detach().clone())
        new.down_weight=torch.nn.Parameter(old.down_proj.weight.detach().clone())
        layer.mlp=new
    initial={k:v.detach().clone() for k,v in model.state_dict().items()}
    tokens=np.load(a.tokens,mmap_mode="r")
    length=a.batch*(a.sequence+1)
    if len(tokens)<(a.steps+24)*length:
        raise ValueError("corpus is too short for disjoint study windows")
    def batch(step,evaluation=False):
        start=(len(tokens)-(step+1)*length) if evaluation else step*length
        data=torch.as_tensor(np.array(tokens[start:start+length],copy=True),device="cuda",dtype=torch.long)
        data=data.reshape(a.batch,a.sequence+1)
        return data[:,:-1].contiguous(),data[:,1:].contiguous()
    def loss_fn(ids,targets):
        return causal_lm_loss(model(ids),targets)
    fn=torch.compile(loss_fn,fullgraph=True,dynamic=False)
    def optimizer():
        return torch.optim.AdamW(model.parameters(),lr=3e-4,betas=(.9,.95),weight_decay=.1,
                                 fused=True,capturable=a.cuda_graph)
    opt=optimizer()
    compile_start=time.monotonic()
    for step in range(3):
        ids,targets=batch(step)
        opt.zero_grad(set_to_none=True);fn(ids,targets).backward();opt.step()
    torch.cuda.synchronize()
    compile_seconds=time.monotonic()-compile_start
    graph=None
    if a.cuda_graph:
        static_ids,static_targets=batch(0)
        stream=torch.cuda.Stream();stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                opt.zero_grad(set_to_none=True);fn(static_ids,static_targets).backward();opt.step()
        stream.synchronize()
        opt.zero_grad(set_to_none=True)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            graph_loss=fn(static_ids,static_targets)
            graph_loss.backward();opt.step()
        # Capture updates weights and optimizer state. Reset values in place:
        # replay must keep the same parameter/state/gradient storage addresses.
        torch.cuda.synchronize()
        for state in opt.state.values():
            for value in state.values():
                if isinstance(value,torch.Tensor):
                    value.zero_()
    else:
        opt=optimizer();opt.zero_grad(set_to_none=True)
    model.load_state_dict(initial);del initial
    torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
    def evaluate():
        # Same fixed held-out tail windows for every policy; eager no-grad is
        # outside throughput measurement and tests inference composition too.
        with torch.no_grad():
            return statistics.mean(float(loss_fn(*batch(i,True))) for i in range(8))
    before=evaluate()
    records=[]
    with (out/"steps.jsonl").open("x",buffering=1) as log:
        for step in range(a.steps):
            ids,targets=batch(step)
            if graph is None:
                opt.zero_grad(set_to_none=True)
            else:
                static_ids.copy_(ids);static_targets.copy_(targets)
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            if graph is None:
                loss=fn(ids,targets);loss.backward();opt.step()
            else:
                graph.replay();loss=graph_loss
            end.record();end.synchronize()
            # Diagnostics are outside the timed step. No clipping differences.
            norm=torch.stack([v.grad.float().square().sum() for v in model.parameters() if v.grad is not None]).sum().sqrt()
            value=float(loss.detach());grad_norm=float(norm)
            if not np.isfinite(value) or not np.isfinite(grad_norm):
                raise RuntimeError(f"nonfinite training at step {step}")
            record={"step":step,"loss":value,"grad_norm":grad_norm,"ms":start.elapsed_time(end),
                    "allocated_bytes":torch.cuda.memory_allocated()}
            records.append(record);log.write(json.dumps(record)+"\n")
            if step%25==0:
                print(f"{a.precision} step={step}/{a.steps} loss={value:.5f} grad={grad_norm:.4f} ms={record['ms']:.3f}",flush=True)
    after=evaluate()
    result={"args":vars(a),"torch":torch.__version__,"cuda":torch.version.cuda,
            "mlp_config":schedule.to_dict(),
            "data_sha256":hashlib.sha256(tokens[:(a.steps+3)*length].tobytes()+tokens[-8*length:].tobytes()).hexdigest(),
            "gpu":torch.cuda.get_device_name(),"compile_warmup_seconds":compile_seconds,
            "validation_initial":before,"validation_final":after,
            "train_final_25_loss":statistics.mean(r["loss"] for r in records[-25:]),
            "median_step_ms":statistics.median(r["ms"] for r in records[10:] or records),
            "peak_allocated_bytes":torch.cuda.max_memory_allocated(),
            "optimizer":"AdamW fused, BF16 parameters/gradients/moments; FP32 norm/loss reductions",
            "timing":"CUDA events, compiled forward/backward plus optimizer; "+("whole-step CUDA graph" if a.cuda_graph else "no whole-step CUDA graph"),
            "scope":"2-layer byte decoder; only MLP precision differs; not full convergence"}
    (out/"result.json").write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=="__main__":
    main()
