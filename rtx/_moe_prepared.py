"""Launch-only quantize-once expert path; no eager CUDA numerical operations."""
from functools import lru_cache

import torch

from .configs.moe import DEFAULT_PREPARED_MOE_CONFIG
from .kernels.mxfp8 import MXFP8Problem
from .runtime import load_kernel_symbol


@lru_cache(maxsize=1)
def default_schedules():
    return DEFAULT_PREPARED_MOE_CONFIG.schedules()


DEFAULT_QUANT=DEFAULT_PREPARED_MOE_CONFIG.quant


def prepare(operands,quant=DEFAULT_QUANT,transpose_mode='staged',grid_mode='operand'):
    """Fuse unrelated expert/orientation quantization tasks into one launch."""
    compile_quant=load_kernel_symbol('mxfp8_moe_prepared','compile_expert_quant')
    specs=[];sources=[];packed=[];scales=[]
    for value,orientation,interleave in operands:
        e,rows,k=value.shape
        if orientation=='transpose':rows,k=k,rows
        width=(k+31)//32*32
        specs.append((rows,k,orientation,interleave))
        sources.append(value.flatten(0,1))
        packed.append(torch.empty((e,rows,width),device=value.device,dtype=torch.float8_e4m3fn))
        scales.append(torch.empty((e,rows,width//32),device=value.device,dtype=torch.float8_e8m0fnu))
    compile_quant(e,tuple(specs),quant,transpose_mode,grid_mode)(sources[0],tuple(sources),tuple(t.flatten(0,1) for t in packed),
                                      tuple(t.flatten(0,1) for t in scales))
    return tuple(zip(packed,scales))


def packed_gemm(a,b,out,config,kind='linear',aux=None,transport='tma'):
    compile_gemm=load_kernel_symbol('mxfp8_moe_prepared','compile_expert_packed_gemm')
    e,m,k=a[0].shape
    problem=MXFP8Problem(m,b[0].shape[1],k)
    extra=out if aux is None else aux
    compile_gemm(e,problem,config,kind,transport)(a[0].flatten(0,1),b[0].flatten(0,1),
        a[1].flatten(0,1),b[1].flatten(0,1),out.flatten(0,1),extra.flatten(0,1))


def projections(xg,gu,down,schedules,quant,grid_mode='operand'):
    e,c,d=xg.shape
    h=down.shape[2];width=(h+31)//32*32
    qx,qgu,qd=prepare(((xg,'row',False),(gu,'row',True),(down,'row',False)),quant,grid_mode=grid_mode)
    hidden=torch.empty((e,c,width),device=xg.device,dtype=torch.float8_e4m3fn)
    scales=torch.empty((e,c,width//32),device=xg.device,dtype=torch.float8_e8m0fnu)
    packed_gemm(qx,qgu,hidden,schedules['gate_up'],'glu',scales.view(torch.uint8))
    outputs=torch.empty((e,c,d),device=xg.device,dtype=xg.dtype)
    down_scales=scales
    if quant.scale_layout=='mma128':
        down_scales=torch.empty_like(scales)
        load_kernel_symbol('mxfp8_moe_prepared','compile_native_hidden_scales')(e,c,width)(
            scales.view(-1),down_scales.view(-1))
    packed_gemm((hidden,down_scales),qd,outputs,schedules['down'],transport=schedules.get('down_transport','tma'))
    return outputs,(hidden,scales),(qx,qgu)


def forward(x,gu,down,ids,weights,inverse,config,*,schedules=None,quant=DEFAULT_QUANT):
    from .moe import _movement
    schedules=default_schedules() if schedules is None else schedules
    e,c=ids.shape;d=x.shape[1]
    c=config.prepared.physical_capacity(c)
    gathered=torch.empty((e,c,d),device=x.device,dtype=x.dtype)
    _movement(x,ids,weights,inverse,gathered.flatten(0,1),config,'gather')
    outputs,_,_=projections(gathered,gu,down,schedules,quant,config.prepared.quant_grid)
    result=torch.empty(x.shape,device=x.device,dtype=torch.float32)
    _movement(outputs.flatten(0,1),ids,weights,inverse,result,config,'combine')
    return result


def backward(x,gu,down,ids,weights,inverse,dy,config,*,schedules=None,quant=DEFAULT_QUANT,transpose_mode='staged'):
    from .moe import _movement
    schedules=default_schedules() if schedules is None else schedules
    if config.prepared.backward_preparation=='joint':
        return _joint_backward(x,gu,down,ids,weights,inverse,dy,config,schedules,quant,transpose_mode)
    e,c=ids.shape;d,h=x.shape[1],down.shape[2]
    c=config.prepared.physical_capacity(c)
    gathered=torch.empty((e,c,d),device=x.device,dtype=x.dtype)
    _movement(x,ids,weights,inverse,gathered.flatten(0,1),config,'gather')
    outputs,qh,(qx,qgu)=projections(gathered,gu,down,schedules,quant,config.prepared.quant_grid)
    slot_dy=torch.empty_like(outputs);dweights=torch.empty_like(weights)
    _movement(dy,ids,weights,inverse,slot_dy.flatten(0,1),config,'slot_grad',
              extra=outputs.flatten(0,1),auxiliary=dweights)
    del outputs
    hidden=torch.empty((e,c,h),device=x.device,dtype=x.dtype)
    load_kernel_symbol('mxfp8_mlp','compile_dequant_hidden')(e*c,h)(
        qh[0].flatten(0,1),qh[1].flatten(0,1),hidden.flatten(0,1))
    del qh
    qdy,qdt,qdyt,qht=prepare(((slot_dy,'row',False),(down,'transpose',False),
                             (slot_dy,'transpose',False),(hidden,'transpose',False)),quant,transpose_mode,config.prepared.quant_grid)
    del slot_dy,hidden
    dh=torch.empty((e,c,h),device=x.device,dtype=x.dtype)
    packed_gemm(qdy,qdt,dh,schedules['down_dx'],transport=schedules.get('down_dx_transport','tma'))
    ddown=torch.empty_like(down)
    packed_gemm(qdyt,qht,ddown,schedules['down_dw'],transport=schedules.get('down_dw_transport','tma'))
    del qdy,qdt,qdyt,qht
    dp=torch.empty((e,c,2*h),device=x.device,dtype=x.dtype)
    packed_gemm(qx,qgu,dp,schedules['recompute'],'derivative',dh)
    del qx,qgu,dh
    qdp,qdpt,qgut,qxt=prepare(((dp,'row',False),(dp,'transpose',False),
                              (gu,'transpose',False),(gathered,'transpose',False)),quant,transpose_mode,config.prepared.quant_grid)
    del dp,gathered
    dxg=torch.empty((e,c,d),device=x.device,dtype=x.dtype)
    packed_gemm(qdp,qgut,dxg,schedules['gate_dx'],transport=schedules.get('gate_dx_transport','tma'))
    dgu=torch.empty_like(gu)
    packed_gemm(qdpt,qxt,dgu,schedules['gate_dw'],transport=schedules.get('gate_dw_transport','tma'))
    dx=torch.empty_like(x)
    _movement(dxg.flatten(0,1),ids,weights,inverse,dx,config,'dx')
    return dx,dgu,ddown,dweights


def _joint_backward(x,gu,down,ids,weights,inverse,dy,config,schedules,quant,transpose_mode):
    """Recompute gate/up once and emit all hidden/derivative orientations.

    Pad only expert-local physical token capacity to a 32-value scale block.
    The two leading-axis quantizers keep their own scales; a transpose is
    never mistaken for reusing feature-axis scales.
    """
    from .moe import _movement
    e,c=ids.shape;c=config.prepared.physical_capacity(c)
    d,h=x.shape[1],down.shape[2]
    gathered=torch.empty((e,c,d),device=x.device,dtype=x.dtype)
    _movement(x,ids,weights,inverse,gathered.flatten(0,1),config,'gather')
    slot_dy=torch.empty_like(gathered)
    _movement(dy,ids,weights,inverse,slot_dy.flatten(0,1),config,'slot_dy')
    qx,qgu,qd,qdt,qdy,qdyt,qgut,qxt=prepare(((gathered,'row',False),(gu,'row',True),
        (down,'row',False),(down,'transpose',False),(slot_dy,'row',False),
        (slot_dy,'transpose',False),(gu,'transpose',False),(gathered,'transpose',False)),quant,transpose_mode,config.prepared.quant_grid)
    del gathered,slot_dy
    dh=torch.empty((e,c,h),device=x.device,dtype=x.dtype)
    packed_gemm(qdy,qdt,dh,schedules['down_dx'],transport=schedules.get('down_dx_transport','tma'))
    del qdy,qdt
    def pair(rows,cols):
        return (torch.empty((e,rows,cols),device=x.device,dtype=torch.float8_e4m3fn),
                torch.empty((e,rows,cols//32),device=x.device,dtype=torch.float8_e8m0fnu))
    qh,qht,qp,qpt=pair(c,h),pair(h,c),pair(c,2*h),pair(2*h,c)
    fn=load_kernel_symbol('mxfp8_moe_prepared','compile_expert_joint_recompute')(
        e,MXFP8Problem(c,2*h,qx[0].shape[2]),schedules['recompute'],quant)
    # qP is viewed as [2*C,H] only for the derivative emitter. Its physical
    # rows remain [gate,up] for each token, i.e. the GEMM's [C,2H] operand.
    aux=(dh.flatten(0,1),qh[0].flatten(0,1),qh[1].flatten(0,1),
         qht[0].flatten(0,1),qht[1].flatten(0,1),
         qp[0].view(e*2*c,h),qp[1].view(e*2*c,h//32),
         qpt[0].flatten(0,1),qpt[1].flatten(0,1))
    fn(qx[0].flatten(0,1),qgu[0].flatten(0,1),qx[1].flatten(0,1),qgu[1].flatten(0,1),
       dh.flatten(0,1),aux)
    del qx,qgu,dh,aux
    outputs=torch.empty((e,c,d),device=x.device,dtype=x.dtype)
    packed_gemm(qh,qd,outputs,schedules['down'],transport=schedules.get('down_transport','tma'))
    del qh,qd
    dweights=torch.empty_like(weights)
    _movement(dy,ids,weights,inverse,outputs.flatten(0,1),config,'route_grad',
              extra=outputs.flatten(0,1),auxiliary=dweights)
    del outputs
    ddown=torch.empty_like(down)
    packed_gemm(qdyt,qht,ddown,schedules['down_dw'],transport=schedules.get('down_dw_transport','tma'))
    dxg=torch.empty((e,c,d),device=x.device,dtype=x.dtype)
    packed_gemm(qp,qgut,dxg,schedules['gate_dx'],transport=schedules.get('gate_dx_transport','tma'))
    dgu=torch.empty_like(gu)
    packed_gemm(qpt,qxt,dgu,schedules['gate_dw'],transport=schedules.get('gate_dw_transport','tma'))
    dx=torch.empty_like(x)
    _movement(dxg.flatten(0,1),ids,weights,inverse,dx,config,'dx')
    return dx,dgu,ddown,dweights
