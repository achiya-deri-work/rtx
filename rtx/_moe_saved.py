"""Experimental saved-state routed experts. No change to inference or default training."""
from dataclasses import replace

import torch

from ._moe_execution import Builder, allocate, execute, layout


class SavedBuilder(Builder):
    def build(self):
        n,d,e,c,k,h = self.shape
        p = self.config.prepared
        if (self.config.family != 'grouped_prepared' or p.forward_engine != 'packed'
                or p.forward_row_layout != 'original' or p.active_prefix
                or p.quant.scale_layout != 'row_major' or h % 32 or d % 32
                or p.physical_capacity(c) % 32 or p.gate_dw_engine == 'indexed'):
            raise ValueError("Saved routed experts require original, aligned row-major prepared operands")
        cs = p.physical_capacity(c)
        if self.phase == 'saved_forward':
            _,qx,qgu,qd = self.forward_operands(False)
            out,qh = self.projections(qx,qgu,qd)
            y = self.tensor((n,d),torch.float32,output=True)
            self.movement(out,y,'combine')
            # Keep only the actual typed tensors, never their scratch arenas.
            self.outputs = [y,out,*qh,*qx,*qgu]
            for ref in self.outputs:
                self.buffers[ref] = replace(self.buffers[ref],output=True)
            return self
        vendor = self.phase == 'saved_backward_blas'
        if not vendor and self.phase != 'saved_backward_mx':
            raise ValueError("Unknown saved-state phase")
        # Inputs 0..6 match ordinary backward; 7..13 are forward-owned state.
        out, qh, qx, qgu = -8,(-9,-10),(-11,-12),(-13,-14)
        gathered = self.tensor((e*cs,d)); self.movement(-1,gathered,'gather')
        slot_dy = self.tensor((e*cs,d))
        dr = self.tensor((e,c),torch.float32,output=True)
        self.movement(-7,slot_dy,'slot_grad',out,dr)
        hidden = self.tensor((e*cs,h))
        self.step('mxfp8_moe_buffers','compile_dequant_hidden',(e*cs,h),(*qh,hidden))
        operands = ((slot_dy,e,cs,d,'row',False,False),(-3,e,d,h,'transpose',False,False),
                    (-2,e,2*h,d,'transpose',False,False))
        if not vendor:
            operands += ((slot_dy,e,cs,d,'transpose',False,False),(hidden,e,cs,h,'transpose',False,False),
                         (gathered,e,cs,d,'transpose',False,False))
        qdy,qdt,qgut,*transposed = self.prepare(operands)
        dh = self.tensor((e*cs,h)); self.gemm(qdy,qdt,dh,'down_dx',cs,h,d)
        dp = self.tensor((e*cs,2*h)); self.gemm(qx,qgu,dp,'recompute',cs,2*h,d,'derivative',dh)
        operand = (dp,e,cs,2*h,'row',False,False)
        qp,*qpt = self.prepare((operand,) if vendor else (operand,(dp,e,cs,2*h,'transpose',False,False)))
        dxg = self.tensor((e*cs,d)); self.gemm(qp,qgut,dxg,'gate_dx',cs,d,2*h)
        dx = self.tensor((n,d),output=True); self.movement(dxg,dx,'dx')
        dgu = self.tensor((e*2*h,d),output=True); dd = self.tensor((e*d,h),output=True)
        if vendor:
            self.step('mxfp8_moe_buffers','compile_batched_bf16_dw',(e,2*h,d,cs),(dp,gathered,dgu))
            self.step('mxfp8_moe_buffers','compile_batched_bf16_dw',(e,d,h,cs),(slot_dy,hidden,dd))
        else:
            qdyt,qht,qxt = transposed
            self.gemm(qdyt,qht,dd,'down_dw',d,h,cs)
            self.gemm(qpt[0],qxt,dgu,'gate_dw',2*h,d,cs)
        self.outputs = [dx,dgu,dd,dr]
        return self


class SavedExperts(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,gu,down,ids,weights,inverse,config_key,bf16_dw):
        shape=(x.shape[0],x.shape[1],gu.shape[0],ids.shape[1],inverse.shape[1],down.shape[2])
        key,dtypes,sizes,views,outputs=layout(shape,config_key,'saved_forward')
        buffers=allocate(x,dtypes,sizes,views)
        execute([x,gu.flatten(0,1),down.flatten(0,1),ids,weights,inverse],buffers,key)
        ctx.save_for_backward(x,gu,down,ids,weights,inverse,*(buffers[i] for i in outputs[1:]))
        ctx.shape,ctx.config_key,ctx.vendor=shape,config_key,bf16_dw
        return buffers[outputs[0]]

    @staticmethod
    def backward(ctx,dy):
        x,gu,down,ids,weights,inverse,*saved=ctx.saved_tensors
        phase='saved_backward_blas' if ctx.vendor else 'saved_backward_mx'
        key,dtypes,sizes,views,outputs=layout(ctx.shape,ctx.config_key,phase)
        buffers=allocate(x,dtypes,sizes,views)
        execute([x,gu.flatten(0,1),down.flatten(0,1),ids,weights,inverse,dy.contiguous(),*saved],buffers,key)
        dx,dgu,dd,dr=(buffers[i] for i in outputs)
        return dx,dgu.view(gu.shape),dd.view(down.shape),None,dr,None,None,None


def saved_train_call(operands,dy,config):
    """Same saved operand handoff as bound training, including for graph timing."""
    from .moe import _config_key
    x,gu,down,ids,weights,inverse=operands
    shape=(x.shape[0],x.shape[1],gu.shape[0],ids.shape[1],inverse.shape[1],down.shape[2])
    inputs=[x,gu.flatten(0,1),down.flatten(0,1),ids,weights,inverse]
    cfg=_config_key(config)
    key,types,sizes,views,outputs=layout(shape,cfg,'saved_forward')
    buffers=allocate(x,types,sizes,views)
    execute(inputs,buffers,key)
    y,*saved=(buffers[i] for i in outputs)
    del buffers
    phase='saved_backward_blas' if config.training_state=='saved_bf16_dw' else 'saved_backward_mx'
    key,types,sizes,views,outputs=layout(shape,cfg,phase)
    buffers=allocate(x,types,sizes,views)
    execute([*inputs,dy.contiguous(),*saved],buffers,key)
    dx,dgu,dd,dr=(buffers[i] for i in outputs)
    return y,dx,dgu.view_as(gu),dd.view_as(down),dr
