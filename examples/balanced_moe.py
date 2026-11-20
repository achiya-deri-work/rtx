"""Small public-API example, not a convergence or performance benchmark.

Run from this checkout: python -m examples.balanced_moe
The first forward may compile and autotune; subsequent shapes use the cache.
"""
import torch
from torch import nn
import rtx


class TinyTokenClassifier(nn.Module):
    def __init__(self,*,device=None,autotune='balanced',autotune_cache_dir=None):
        super().__init__()
        self.embedding=nn.Embedding(256,128,device=device,dtype=torch.bfloat16)
        self.norm=nn.RMSNorm(128,eps=1e-6,device=device,dtype=torch.float32)
        self.moe=rtx.BalancedMoE(128,16,192,top_k=2,shared_hidden_features=128,
            device=device,autotune=autotune,autotune_cache_dir=autotune_cache_dir)
        self.final_norm=nn.RMSNorm(128,eps=1e-6,device=device,dtype=torch.float32)
        self.head=nn.Linear(128,32,bias=False,device=device,dtype=torch.bfloat16)
        for weight in (self.embedding.weight,self.head.weight):
            nn.init.trunc_normal_(weight,std=.02,a=-.04,b=.04)

    def forward(self,tokens,*,cu_seqlens=None):
        x=self.embedding(tokens)
        normalized=self.norm(x.float()).to(torch.bfloat16)
        x=(x.float()+self.moe(normalized,cu_seqlens=cu_seqlens).float()).to(torch.bfloat16)
        return self.head(self.final_norm(x.float()).to(torch.bfloat16)).float()


def main():
    model=TinyTokenClassifier(device='cuda',autotune='quick')
    optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4)
    compiled=torch.compile(model,fullgraph=True,dynamic=False)
    tokens=torch.randint(256,(2,64),device='cuda')
    targets=tokens%32
    for step in range(5):
        optimizer.zero_grad(set_to_none=True)
        logits=compiled(tokens)
        loss=nn.functional.cross_entropy(logits.flatten(0,1),targets.flatten())
        loss.backward();optimizer.step()
        print(f'step={step} loss={loss.item():.4f}',flush=True)


if __name__=='__main__':main()
