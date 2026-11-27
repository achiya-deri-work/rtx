"""Independent Torch/FP64 operational oracle, used only during qualification.

This is not a production fallback and never runs in a hot native launcher.
Leading-axis quantization is per expert, including zero-filled final blocks.
FP64 matmuls keep validation independent of process-global TF32 settings.
"""
import torch


def quantize_decode(value):
    width=value.shape[-1]
    padded=torch.nn.functional.pad(value.float(),(0,(-width)%32))
    blocks=padded.reshape(*padded.shape[:-1],-1,32)
    largest=blocks.abs().amax(-1,keepdim=True)
    exponent=torch.ceil(torch.log2(largest.clamp_min(2.**-119)/448)).clamp(-127,127)
    scale=torch.exp2(exponent)
    return ((blocks/scale).to(torch.float8_e4m3fn).float()*scale).reshape_as(padded)[...,:width]


def _product(a,b):
    # The oracle is offline, but must fit alongside production buffers at
    # long packed lengths. Converting every expert to FP64 simultaneously
    # made correctness-gated tuning run out of memory before its first trial.
    # Chunk only the independent batch axis: each dot product and its FP64
    # reduction stay identical to the unchunked independent reference.
    out=torch.empty((*a.shape[:-1],b.shape[-2]),device=a.device,dtype=torch.float32)
    for expert in range(a.shape[0]):
        out[expert:expert+1]=torch.bmm(a[expert:expert+1].double(),
            b[expert:expert+1].double().transpose(1,2)).float()
    return out


@torch.no_grad()
def operational_reference(x,gu,down,ids,weights,inverse,dy=None,*,bf16_dw=False):
    n,d=x.shape
    e,c=ids.shape
    h=down.shape[2]
    slots=torch.arange(e*c,device=x.device).reshape(e,c)
    valid=inverse[ids].eq(slots[...,None]).any(-1)
    gathered=x[ids]*valid[...,None]
    projections=_product(quantize_decode(gathered),quantize_decode(gu))
    gate,up=projections[...,:h],projections[...,h:]
    sigmoid=1/(1+torch.exp(-gate))
    activation=gate/(1+torch.exp(-gate))
    hidden=quantize_decode(activation*up)
    outputs=_product(hidden,quantize_decode(down)).bfloat16()
    result=(outputs.flatten(0,1)[inverse].float()*weights.flatten()[inverse][...,None]).sum(1)
    if dy is None:
        return (result,)
    slot_dy=(dy[ids]*weights[...,None]*valid[...,None]).bfloat16()
    dweights=(dy[ids]*outputs.float()*valid[...,None]).sum(-1)
    dh=_product(quantize_decode(slot_dy),quantize_decode(down.transpose(1,2))).bfloat16()
    dp=torch.cat((dh.float()*up*sigmoid*(1+gate*(1-sigmoid)),
                  dh.float()*activation),-1).bfloat16()
    operand=(lambda a:a) if bf16_dw else quantize_decode
    ddown=_product(operand(slot_dy.transpose(1,2)),
                   operand(hidden.bfloat16().transpose(1,2))).bfloat16()
    dgu=_product(operand(dp.transpose(1,2)),operand(gathered.transpose(1,2))).bfloat16()
    dxg=_product(quantize_decode(dp),quantize_decode(gu.transpose(1,2))).bfloat16()
    dx=dxg.flatten(0,1)[inverse].float().sum(1).bfloat16()
    return result,dx,dgu,ddown,dweights
