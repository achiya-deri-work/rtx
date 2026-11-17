"""Compiler-visible MoE storage and allocation-free expert launch sequences.

The public autograd wrapper owns allocations and logical views. The private
mutation op owns only CUDA launches. Scratch arenas have one canonical dtype:
cross-dtype aliases are not safe under Torch 2.13 functionalization.
"""
from dataclasses import dataclass
from functools import lru_cache
import json
import math
from threading import RLock

import torch

from .runtime import BoundedCache


_COMPILED=BoundedCache(128)
_COMPILE_LOCK=RLock()


@dataclass(frozen=True)
class Buffer:
    shape: tuple
    dtype: torch.dtype
    birth: int
    output: bool = False


class Builder:
    def __init__(self, shape, config, phase):
        self.shape,self.config,self.phase=shape,config,phase
        self.buffers=[];self.steps=[];self.outputs=[]

    def tensor(self,shape,dtype=torch.bfloat16,*,output=False):
        ref=len(self.buffers)
        if math.prod(shape)>=2**31:
            raise ValueError('Physical expert buffer exceeds native signed index range')
        self.buffers.append(Buffer(tuple(shape),dtype,len(self.steps),output))
        if output:self.outputs.append(ref)
        return ref

    def step(self,module,symbol,parameters,arguments):
        self.steps.append((module,symbol,parameters,arguments))

    def movement(self,source,out,kind,extra=None,auxiliary=None):
        n,d,e,c,k,h=self.shape
        cs=self.config.prepared.physical_capacity(c) if self.config.family=='grouped_prepared' else c
        placeholder=out if kind in ('gather','slot_dy') else source
        self.step('mxfp8_moe','compile_movement',
                  (n,d,e,c,k,self.config.movement_threads,kind,
                   self.config.gather_mode if kind=='gather' else 'flat',cs),
                  (source,-4,-6,-5,placeholder if extra is None else extra,out,
                   -5 if auxiliary is None else auxiliary))

    def prepare(self,operands):
        # operands: (reference, batch_count, source_rows, source_cols,
        #            orientation, interleave, explicit_batch_count)
        e=self.shape[2];p=self.config.prepared
        specs=[];sources=[];packed=[];scales=[]
        for ref,batches,rows,k,orientation,interleave,explicit in operands:
            if orientation=='transpose':rows,k=k,rows
            width=(k+31)//32*32
            spec=(rows,k,orientation,interleave)
            specs.append((*spec,batches) if explicit else spec)
            sources.append(ref)
            packed.append(self.tensor((batches*rows,width),torch.float8_e4m3fn))
            scales.append(self.tensor((batches*rows,width//32),torch.uint8))
        self.step('mxfp8_moe_buffers','compile_expert_quant',
                  (e,tuple(specs),p.quant,p.transpose_mode,p.quant_grid),
                  (sources[0],tuple(sources),tuple(packed),tuple(scales)))
        return tuple(zip(packed,scales))

    def pair(self,rows,k):
        e=self.shape[2];width=(k+31)//32*32
        return (self.tensor((e*rows,width),torch.float8_e4m3fn),
                self.tensor((e*rows,width//32),torch.uint8))

    def gemm(self,a,b,out,stage,m,n,k,kind='linear',aux=None):
        from .kernels.mxfp8 import MXFP8Problem
        p=self.config.prepared
        transport=p.transports[('gate_up','down','recompute','down_dx','down_dw','gate_dx','gate_dw').index(stage)]
        self.step('mxfp8_moe_buffers','compile_expert_packed_gemm',
                  (self.shape[2],MXFP8Problem(m,n,k),getattr(p,stage),kind,transport),
                  (a[0],b[0],a[1],b[1],out,out if aux is None else aux))

    def forward_operands(self,copy_bf16):
        n,d,e,c,k,h=self.shape;p=self.config.prepared;cs=p.physical_capacity(c)
        once=getattr(p,'input_preparation','gather_bf16')=='quantize_once'
        gathered=self.tensor((e*cs,d)) if copy_bf16 or not once else -1
        if once:
            global_x,qgu,qd=self.prepare(((-1,1,n,d,'row',False,True),
                (-2,e,2*h,d,'row',True,False),(-3,e,d,h,'row',False,False)))
            qx=self.gather_packed(global_x,gathered,copy_bf16)
        else:
            self.movement(-1,gathered,'gather')
            qx,qgu,qd=self.prepare(((gathered,e,cs,d,'row',False,False),
                (-2,e,2*h,d,'row',True,False),(-3,e,d,h,'row',False,False)))
        return gathered,qx,qgu,qd

    def gather_packed(self,global_x,gathered,copy_bf16):
        n,d,e,c,k,h=self.shape;p=self.config.prepared;cs=p.physical_capacity(c)
        qx=self.pair(cs,d)
        self.step('mxfp8_moe_buffers','compile_packed_input_gather',
                  (n,d,e,c,k,cs,self.config.movement_threads,p.quant.scale_layout,copy_bf16),
                  (global_x[0],global_x[1],-1,-4,-6,qx[0],qx[1],gathered if copy_bf16 else -1))
        return qx

    def projections(self,qx,qgu,qd):
        n,d,e,c,k,h=self.shape;p=self.config.prepared;cs=p.physical_capacity(c)
        qh=self.pair(cs,h)
        self.gemm(qx,qgu,qh[0],'gate_up',cs,2*h,(d+31)//32*32,'glu',qh[1])
        down_scales=qh[1]
        if p.quant.scale_layout=='mma128':
            down_scales=self.tensor((e*cs,((h+31)//32)),torch.uint8)
            flat_source=self.alias(qh[1],(e*cs*((h+31)//32),))
            flat_destination=self.alias(down_scales,(e*cs*((h+31)//32),))
            self.step('mxfp8_moe_buffers','compile_native_hidden_scales',(e,cs,(h+31)//32*32),(flat_source,flat_destination))
        out=self.tensor((e*cs,d))
        self.gemm((qh[0],down_scales),qd,out,'down',cs,d,(h+31)//32*32)
        return out,qh

    def build(self):
        if self.config.family!='grouped_prepared':return self.legacy()
        n,d,e,c,k,h=self.shape;p=self.config.prepared;cs=p.physical_capacity(c)
        if self.phase=='forward':
            _,qx,qgu,qd=self.forward_operands(False)
            out,_=self.projections(qx,qgu,qd)
            result=self.tensor((n,d),torch.float32,output=True)
            self.movement(out,result,'combine')
            return self
        if p.backward_preparation=='joint':return self.joint_backward()
        gathered,qx,qgu,qd=self.forward_operands(True)
        out,qh=self.projections(qx,qgu,qd)
        slot_dy=self.tensor((e*cs,d));dw_route=self.tensor((e,c),torch.float32,output=True)
        self.movement(-7,slot_dy,'slot_grad',out,dw_route)
        hidden=self.tensor((e*cs,h))
        self.step('mxfp8_moe_buffers','compile_dequant_hidden',(e*cs,h),(qh[0],qh[1],hidden))
        qdy,qdt,qdyt,qht=self.prepare(((slot_dy,e,cs,d,'row',False,False),
            (-3,e,d,h,'transpose',False,False),(slot_dy,e,cs,d,'transpose',False,False),
            (hidden,e,cs,h,'transpose',False,False)))
        dh=self.tensor((e*cs,h));ddown=self.tensor((e*d,h),output=True)
        self.gemm(qdy,qdt,dh,'down_dx',cs,h,(d+31)//32*32)
        self.gemm(qdyt,qht,ddown,'down_dw',d,h,(cs+31)//32*32)
        dp=self.tensor((e*cs,2*h))
        self.gemm(qx,qgu,dp,'recompute',cs,2*h,(d+31)//32*32,'derivative',dh)
        qp,qpt,qgut,qxt=self.prepare(((dp,e,cs,2*h,'row',False,False),
            (dp,e,cs,2*h,'transpose',False,False),(-2,e,2*h,d,'transpose',False,False),
            (gathered,e,cs,d,'transpose',False,False)))
        dxg=self.tensor((e*cs,d));dgu=self.tensor((e*2*h,d),output=True)
        self.gemm(qp,qgut,dxg,'gate_dx',cs,d,(2*h+31)//32*32)
        self.gemm(qpt,qxt,dgu,'gate_dw',2*h,d,(cs+31)//32*32)
        dx=self.tensor((n,d),output=True);self.movement(dxg,dx,'dx')
        self.outputs=[dx,dgu,ddown,dw_route]
        return self

    def legacy_linear(self,a,b,out,dimensions,config,a_orientation='row',b_orientation='row',scales=None):
        from .kernels.mxfp8 import MXFP8Problem
        self.step('mxfp8_moe_buffers','compile_grouped_linear',
                  (self.shape[2],MXFP8Problem(*dimensions),config,a_orientation,b_orientation,scales is not None),
                  (a,b,out,*((out if scales is None else scales),)*4))

    def legacy_projections(self,gathered):
        from .kernels.mxfp8 import MXFP8Problem
        n,d,e,c,k,h=self.shape
        width=(h+31)//32*32
        hidden=self.tensor((e*c,width),torch.float8_e4m3fn)
        scales=self.tensor((e*c,width//32),torch.uint8)
        self.step('mxfp8_moe','compile_grouped_glu',
                  (e,MXFP8Problem(c,2*h,d),self.config.gate_up),
                  (gathered,-2,hidden,scales,scales,scales,scales))
        out=self.tensor((e*c,d))
        self.legacy_linear(hidden,-3,out,(c,d,h),self.config.down,scales=scales)
        return out,hidden,scales

    def legacy(self):
        """Expose storage for the existing staged/on-chip numerical kernels."""
        from .kernels.mxfp8 import MXFP8Problem
        n,d,e,c,k,h=self.shape;config=self.config
        gathered=self.tensor((e*c,d));self.movement(-1,gathered,'gather')
        if self.phase=='forward':
            if config.family=='grouped_onchip':
                out=self.tensor((e*c,d))
                self.step('mxfp8_moe_onchip','compile_onchip_experts',
                          (e,MXFP8Problem(c,2*h,d),config.gate_up,config.down,d,
                           config.onchip_down_stages,config.onchip_pipeline),
                          (gathered,-2,out,-3,-3,-3,-3))
            else:out,_,_=self.legacy_projections(gathered)
            result=self.tensor((n,d),torch.float32,output=True)
            self.movement(out,result,'combine');return self
        out,qh,sf=self.legacy_projections(gathered)
        slot_dy=self.tensor((e*c,d));dw_route=self.tensor((e,c),torch.float32,output=True)
        self.movement(-7,slot_dy,'slot_grad',out,dw_route)
        hidden=self.tensor((e*c,h))
        self.step('mxfp8_moe_buffers','compile_dequant_hidden',(e*c,h),(qh,sf,hidden))
        dh=self.tensor((e*c,h));self.legacy_linear(slot_dy,-3,dh,(c,h,d),config.down_dx,b_orientation='transpose')
        ddown=self.tensor((e*d,h),output=True)
        self.legacy_linear(slot_dy,hidden,ddown,(d,h,c),config.down_dw,'transpose','transpose')
        dp=self.tensor((e*c,2*h))
        self.step('mxfp8_moe','compile_grouped_glu',
                  (e,MXFP8Problem(c,2*h,d),config.recompute,True),
                  (gathered,-2,dp,dh,dh,dh,dh))
        dxg=self.tensor((e*c,d));self.legacy_linear(dp,-2,dxg,(c,d,2*h),config.gate_dx,b_orientation='transpose')
        dgu=self.tensor((e*2*h,d),output=True)
        self.legacy_linear(dp,gathered,dgu,(2*h,d,c),config.gate_dw,'transpose','transpose')
        dx=self.tensor((n,d),output=True);self.movement(dxg,dx,'dx')
        self.outputs=[dx,dgu,ddown,dw_route];return self

    def joint_backward(self):
        from .kernels.mxfp8 import MXFP8Problem
        n,d,e,c,k,h=self.shape;p=self.config.prepared;cs=p.physical_capacity(c)
        # Keep the selected separate/joint policy independent from input reuse.
        gathered=self.tensor((e*cs,d))
        slot_dy=self.tensor((e*cs,d))
        if getattr(p,'input_preparation','gather_bf16')=='quantize_once':
            global_x,qgu,qd,qdt,qgut=self.prepare(((-1,1,n,d,'row',False,True),
                (-2,e,2*h,d,'row',True,False),(-3,e,d,h,'row',False,False),
                (-3,e,d,h,'transpose',False,False),(-2,e,2*h,d,'transpose',False,False)))
            qx=self.gather_packed(global_x,gathered,True)
            self.movement(-7,slot_dy,'slot_dy')
            qdy,qdyt,qxt=self.prepare(((slot_dy,e,cs,d,'row',False,False),
                (slot_dy,e,cs,d,'transpose',False,False),(gathered,e,cs,d,'transpose',False,False)))
        else:
            self.movement(-1,gathered,'gather');self.movement(-7,slot_dy,'slot_dy')
            qx,qgu,qd,qdt,qdy,qdyt,qgut,qxt=self.prepare(((gathered,e,cs,d,'row',False,False),
                (-2,e,2*h,d,'row',True,False),(-3,e,d,h,'row',False,False),
                (-3,e,d,h,'transpose',False,False),(slot_dy,e,cs,d,'row',False,False),
                (slot_dy,e,cs,d,'transpose',False,False),(-2,e,2*h,d,'transpose',False,False),
                (gathered,e,cs,d,'transpose',False,False)))
        dh=self.tensor((e*cs,h));self.gemm(qdy,qdt,dh,'down_dx',cs,h,(d+31)//32*32)
        qh,qht,qp,qpt=self.pair(cs,h),self.pair(h,cs),self.pair(cs,2*h),self.pair(2*h,cs)
        # Same-dtype shape aliases are constructed by the wrapper, not launchers.
        qp_view=self.alias(qp[0],(e*2*cs,h));sp_view=self.alias(qp[1],(e*2*cs,h//32))
        aux=(dh,qh[0],qh[1],qht[0],qht[1],qp_view,sp_view,qpt[0],qpt[1])
        self.step('mxfp8_moe_buffers','compile_expert_joint_recompute',
            (e,MXFP8Problem(cs,2*h,(d+31)//32*32),p.recompute,p.quant),
            (qx[0],qgu[0],qx[1],qgu[1],dh,aux))
        out=self.tensor((e*cs,d));self.gemm(qh,qd,out,'down',cs,d,h)
        dw_route=self.tensor((e,c),torch.float32,output=True)
        self.movement(-7,out,'route_grad',out,dw_route)
        ddown=self.tensor((e*d,h),output=True);self.gemm(qdyt,qht,ddown,'down_dw',d,h,cs)
        dxg=self.tensor((e*cs,d));self.gemm(qp,qgut,dxg,'gate_dx',cs,d,2*h)
        dgu=self.tensor((e*2*h,d),output=True);self.gemm(qpt,qxt,dgu,'gate_dw',2*h,d,cs)
        dx=self.tensor((n,d),output=True);self.movement(dxg,dx,'dx')
        self.outputs=[dx,dgu,ddown,dw_route];return self

    def alias(self,reference,shape):
        if not hasattr(self,'aliases'):self.aliases={}
        ref=self.tensor(shape,self.buffers[reference].dtype)
        self.aliases[ref]=reference
        return ref


def references(value):
    if isinstance(value,int):yield value
    else:
        for child in value:yield from references(child)


@lru_cache(maxsize=128)
def plan(key):
    from .moe import _parse
    shape,config_key,phase=json.loads(key)
    builder=Builder(tuple(shape),_parse(config_key),phase).build()
    aliases=getattr(builder,'aliases',{})
    last=[b.birth for b in builder.buffers]
    for index,(_,_,_,arguments) in enumerate(builder.steps):
        for ref in references(arguments):
            if ref>=0:
                while ref in aliases:ref=aliases[ref]
                last[ref]=max(last[ref],index)
    dtypes=[];sizes=[];active={};free={};locations=[]
    for i,b in enumerate(builder.buffers):
        if i in aliases:
            locations.append(locations[aliases[i]]);continue
        if b.output:
            locations.append((-1,0));continue
        if b.dtype not in dtypes:
            dtypes.append(b.dtype);sizes.append(0)
        group=dtypes.index(b.dtype);alignment=max(1,16//b.dtype.itemsize)
        remaining=[]
        for end,offset,count in active.get(group,[]):
            if end<b.birth:free.setdefault(group,[]).append((offset,count))
            else:remaining.append((end,offset,count))
        count=math.prod(b.shape);offset=None
        for j,(start,length) in enumerate(free.get(group,[])):
            aligned=(start+alignment-1)//alignment*alignment
            if aligned+count<=start+length:
                offset=aligned;free[group].pop(j)
                if start+length>aligned+count:free[group].append((aligned+count,start+length-aligned-count))
                break
        if offset is None:
            offset=(sizes[group]+alignment-1)//alignment*alignment;sizes[group]=offset+count
        remaining.append((last[i],offset,count));active[group]=remaining
        locations.append((group,offset))
    layout=tuple((b.shape,b.dtype,*loc) for b,loc in zip(builder.buffers,locations))
    return builder,tuple(dtypes),tuple(sizes),layout,tuple(builder.outputs)


@lru_cache(maxsize=128)
@torch.compiler.assume_constant_result
def layout(shape,config_key,phase):
    key=json.dumps((shape,config_key,phase),sort_keys=True)
    _,dtypes,sizes,views,outputs=plan(key)
    return key,dtypes,sizes,views,outputs


def allocate(x,dtypes,sizes,views):
    arenas=[torch.empty(size,device=x.device,dtype=dtype) for size,dtype in zip(sizes,dtypes)]
    return [torch.empty(shape,device=x.device,dtype=dtype) if group<0 else
            arenas[group].narrow(0,offset,math.prod(shape)).view(shape)
            for shape,dtype,group,offset in views]


def compiled_plan(key,device):
    identity=key,device
    selected=_COMPILED.get(identity)
    if selected is not None:return selected
    with _COMPILE_LOCK:
        selected=_COMPILED.get(identity)
        if selected is None:
            selected=_compile_plan(key,device)
            _COMPILED[identity]=selected
        return selected


def _compile_plan(key,device):
    from .runtime import load_kernel_symbol
    builder,*_=plan(key)
    functions=[];lines=['def launch(inputs, buffers):']
    def expression(value):
        if isinstance(value,int):return f'inputs[{-value-1}]' if value<0 else f'buffers[{value}]'
        return '('+','.join(expression(v) for v in value)+',)'
    with torch.cuda.device(device):
        for module,symbol,parameters,arguments in builder.steps:
            functions.append(load_kernel_symbol(module,symbol)(*parameters))
            lines.append(f'    kernels[{len(functions)-1}]('+','.join(expression(v) for v in arguments)+')')
    namespace={'kernels':tuple(functions)}
    exec(compile('\n'.join(lines),'<rtx-moe-launch-plan>','exec'),namespace)
    launch=namespace['launch']
    launch.__rtx_resources__=tuple(functions)
    return launch


@torch.library.custom_op('rtx::moe_execute',mutates_args=('buffers',),device_types='cuda')
def execute(inputs:list[torch.Tensor],buffers:list[torch.Tensor],key:str)->None:
    compiled_plan(key,inputs[0].device.index)(inputs,buffers)


@execute.register_fake
def execute_fake(inputs,buffers,key):
    pass


class PreparedExperts(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,gu,down,ids,weights,inverse,config_key):
        shape=(x.shape[0],x.shape[1],gu.shape[0],ids.shape[1],inverse.shape[1],down.shape[2])
        key,dtypes,sizes,views,outputs=layout(shape,config_key,'forward')
        buffers=allocate(x,dtypes,sizes,views)
        execute([x,gu.flatten(0,1),down.flatten(0,1),ids,weights,inverse],buffers,key)
        ctx.save_for_backward(x,gu,down,ids,weights,inverse)
        ctx.shape,ctx.config_key=shape,config_key
        return buffers[outputs[0]]

    @staticmethod
    def backward(ctx,dy):
        x,gu,down,ids,weights,inverse=ctx.saved_tensors
        key,dtypes,sizes,views,outputs=layout(ctx.shape,ctx.config_key,'backward')
        buffers=allocate(x,dtypes,sizes,views)
        execute([x,gu.flatten(0,1),down.flatten(0,1),ids,weights,inverse,dy.contiguous()],buffers,key)
        dx,dgu,ddown,dweights=[buffers[i] for i in outputs]
        return dx,dgu.view(gu.shape),ddown.view(down.shape),None,dweights,None,None


def prepared_experts(x,gu,down,ids,weights,inverse,config_key):
    return PreparedExperts.apply(x,gu,down,ids,weights,inverse,config_key)


def run_phase(x,gu,down,ids,weights,inverse,config,dy=None):
    """The same execution/memory objective used by offline autotuning."""
    from .moe import _config_key
    shape=(x.shape[0],x.shape[1],gu.shape[0],ids.shape[1],inverse.shape[1],down.shape[2])
    key,dtypes,sizes,views,outputs=layout(shape,_config_key(config),'forward' if dy is None else 'backward')
    buffers=allocate(x,dtypes,sizes,views)
    inputs=[x,gu.flatten(0,1),down.flatten(0,1),ids,weights,inverse]
    if dy is not None:inputs.append(dy.contiguous())
    with torch.no_grad():execute(inputs,buffers,key)
    if dy is None:return buffers[outputs[0]]
    dx,dgu,ddown,dweights=[buffers[i] for i in outputs]
    return dx,dgu.view(gu.shape),ddown.view(down.shape),dweights
