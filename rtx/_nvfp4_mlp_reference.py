"""Independent Torch-only oracle; never called from production launchers."""
import torch
import torch.nn.functional as F


def _fp4(values):
    levels=values.new_tensor([0.,.5,1.,1.5,2.,3.,4.,6.])
    mid=(levels[:-1]+levels[1:])*.5
    index=torch.bucketize(values.abs().contiguous(),mid)
    tie=(index<7)&(values.abs()==mid[index.clamp_max(6)])&((index%2)==1)
    index=index+tie.to(index.dtype)
    return levels[index]*torch.sign(values)


def quant4(tensor,config,region_rows):
    """Reference E2M1/E4M3 quantization with independently computed FP32 scales."""
    rows,k=tensor.shape
    values=F.pad(tensor.float(),(0,(-k)%32))
    maxima=values.abs().amax(-1)
    groups=F.pad(maxima,(0,(-rows)%region_rows)).view(-1,region_rows).amax(-1)
    divisor=1536. if config.tensor_scale_mode=='exact_four_over_six' else 2688.
    outer=groups*(1./divisor)
    if config.tensor_scale_mode=='power2':
        outer=outer.clamp_min(2.**-126)
        # Exact exponent ceiling, independent of the kernel's bit arithmetic.
        outer=torch.exp2(torch.ceil(torch.log2(outer)))
    outer=torch.where(groups==0,torch.ones_like(outer),outer).repeat_interleave(region_rows)[:rows,None,None]
    blocks=values.view(rows,-1,16)
    amax=blocks.abs().amax(-1,keepdim=True)
    def encode(raw):
        return raw.clamp(2.**-6,448.).to(torch.float8_e4m3fn).float()
    def fit(scale):
        codes=_fp4(blocks*(1./(outer*scale)))
        decoded=codes*(outer*scale)
        return decoded,(decoded-blocks).square().sum(-1,keepdim=True),codes
    raw=amax/(6.*outer)
    scale=encode(raw)
    best,error,codes=fit(scale)
    policy=config.quant.block_scale_selection
    candidates=[]
    if policy=='four_over_six':
        candidates=[encode(amax/(4.*outer))]
    elif policy=='mse5':
        candidates=[encode(raw*f) for f in (.75,.875,1.125,1.25)]
    elif policy=='lsq1':
        numerator=(blocks*codes).sum(-1,keepdim=True)
        denominator=codes.square().sum(-1,keepdim=True)
        fitted=torch.where(denominator>0,numerator/denominator,outer*scale)
        candidates=[encode(fitted/outer)]
    for candidate in candidates:
        decoded,new_error,_=fit(candidate)
        improve=new_error<error
        best=torch.where(improve,decoded,best)
        error=torch.minimum(error,new_error)
    return best.view(rows,-1)[:,:k]


def quant8(tensor):
    rows,k=tensor.shape
    blocks=F.pad(tensor.float(),(0,(-k)%32)).view(rows,-1,32)
    scale=torch.exp2(torch.ceil(torch.log2((blocks.abs().amax(-1,keepdim=True)/448.).clamp_min(2.**-126))))
    return ((blocks/scale).to(torch.float8_e4m3fn).float()*scale).view(rows,-1)[:,:k]


def situ(gate,up,config):
    t=torch.tanh(gate/config.beta)
    s=torch.sigmoid(gate)
    v=torch.tanh(up/config.linear_beta)
    a=config.beta*t*s
    b=config.linear_beta*v
    return (a*b,((1-t*t)*s+config.beta*t*s*(1-s))*b,a*(1-v*v))


@torch.no_grad()
def reference(x,gu,down,dy,config):
    """Return output, dX, dGateUp, dDown using analytic SiTU and FP32 GEMMs."""
    x2=x.reshape(-1,x.shape[-1])
    c=config.gate_up
    projection=quant4(x2,c,c.x_scale_region_rows)@quant4(gu,c,c.weight_scale_region_rows).T
    gate,up=projection.chunk(2,-1)
    hidden,dg,du=situ(gate,up,config)
    if config.hidden_dtype=='mxfp8':
        hidden=quant8(hidden)
    hidden=hidden.bfloat16()
    c=config.down
    output=(quant4(hidden,c,c.x_scale_region_rows)@quant4(down,c,c.weight_scale_region_rows).T).bfloat16()
    dy=dy.reshape_as(output)
    dh=(quant8(dy)@quant8(down.T).T).bfloat16().float()
    dp=torch.cat((dh*dg,dh*du),-1).bfloat16()
    dx=(quant8(dp)@quant8(gu.T).T).bfloat16().reshape_as(x)
    dgu=(quant8(dp.T)@quant8(x2.T).T).bfloat16()
    ddown=(quant8(dy.T)@quant8(hidden.T).T).bfloat16()
    return output.reshape(*x.shape[:-1],down.shape[0]),dx,dgu,ddown
