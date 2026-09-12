"""Version-pinned direct lowerings for the recomputed MLP's three custom ops."""
import math
import torch


class _Launcher:
    def __init__(self,key):
        self.kind,self.activation,self.config_key,self.hidden=key

    def __call__(self,*inputs,out):
        from .mlp import _parse,_run_forward
        if self.kind=="forward":
            _run_forward(*inputs,self.activation,_parse(self.config_key),out)
        elif self.kind=="dequant":
            from .kernels.mxfp8_mlp import compile_dequant_hidden
            data,scales=inputs
            with torch.cuda.device(data.device):
                compile_dequant_hidden(data.shape[0],self.hidden)(data,scales.view(torch.float8_e8m0fnu),out)
        else:
            from .mlp import _run_derivative
            x,weight,dh=inputs
            policy=_parse(self.config_key)
            _run_derivative(x,weight,dh,self.activation,policy,out)


class _Registry(dict):
    def __missing__(self,key):
        value=_Launcher(key)
        self[key]=value
        return value


LAUNCHERS=_Registry()


def register():
    from torch._inductor import ir
    from torch._inductor.lowering import register_lowering,view
    torch._rtx_mlp_launchers=LAUNCHERS

    def node(inputs,rows,cols,key):
        inputs=[ir.ExternKernel.require_contiguous(ir.ExternKernel.realize_input(t)) for t in inputs]
        return ir.TensorBox.create(ir.ExternKernelOut(
            ir.FixedLayout(inputs[0].get_device(),torch.bfloat16,[rows,cols],[cols,1]),inputs,
            python_kernel_name=f"torch._rtx_mlp_launchers[{key!r}]"))

    @register_lowering(torch.ops.rtx.mxfp8_mlp.default,type_promotion_kind=None)
    def forward(x,gu,down,activation,key):
        leading=x.get_size()[:-1]
        m,k=math.prod(leading),x.get_size()[-1]
        result=node([view(x,[m,k]),gu,down],m,down.get_size()[0],("forward",activation,key,0))
        return view(result,[*leading,down.get_size()[0]])

    @register_lowering(torch.ops.rtx.mxfp8_mlp_dequant.default,type_promotion_kind=None)
    def dequant(data,scales,hidden):
        return node([data,scales],data.get_size()[0],hidden,("dequant","","",hidden))

    @register_lowering(torch.ops.rtx.mxfp8_mlp_derivative.default,type_promotion_kind=None)
    def derivative(x,weight,dh,activation,key):
        from .mlp import _parse
        if _parse(key).backward_recompute=="joint":
            m=x.get_size()[0]
            return node([x,weight,dh],((2*m+7)//8)*8+m,weight.get_size()[0]//2,("derivative",activation,key,0))
        return node([x,weight,dh],x.get_size()[0],weight.get_size()[0],("derivative",activation,key,0))
