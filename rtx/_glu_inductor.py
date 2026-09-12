"""Version-pinned Inductor integration; no custom-op dispatcher in hot calls.

Outputs are allocated by Inductor, including both MXFP8 data and scales.
Quantized operand workspaces are cached per
shape/device/stream; every call still requantizes the current X and W.
"""
import json
import math
from threading import RLock

import torch
from torch._inductor import ir

from .runtime import BoundedCache, validate_runtime_environment
from .configs.glu import glu_config_from_dict,prequant_glu_config_from_dict
from .kernels.mxfp8 import MXFP8Problem


class _GLUMultiOut(ir.ExternKernelAlloc):
    """External launch with independently allocated, scheduler-owned outputs."""

    out_arg_names = ("out", "scales")

    def codegen(self, wrapper):
        self.codegen_comment(wrapper)
        wrapper.generate_extern_kernel_multi_out(self)


class _Launcher:
    def __init__(self,key):
        self.activation,self.output_format,config_key=key
        values=json.loads(config_key)
        self.prequant=values.get("backend")=="prequant"
        self.config=(prequant_glu_config_from_dict if self.prequant else glu_config_from_dict)(values)
        self.runners=BoundedCache(8)
        self.lock=RLock()

    def __call__(self,x,weight,*,out,scales=None):
        m,k=x.shape
        h=weight.shape[0]//2
        if self.output_format=="bf16":
            scales=out
        key=(x.device.index,int(torch._C._cuda_getCurrentRawStream(x.device.index)),m,k,h)
        runner=self.runners.get(key)
        if runner is None:
            with self.lock:
                runner=self.runners.get(key)
                if runner is None:
                    from .glu import _validate
                    _validate(x,weight,self.activation,self.output_format)
                    validate_runtime_environment()
                    if torch.cuda.get_device_capability(x.device)[0]!=12:
                        raise RuntimeError("MXFP8 GLU requires SM12x")
                    with torch.cuda.device(x.device):
                        if self.prequant:
                            from .kernels.mxfp8_glu_prequant import build_prequant_glu_runner
                            runner=build_prequant_glu_runner(x,weight,out,scales,self.config,self.activation,self.output_format,bind_inputs=False)
                        else:
                            from .kernels.mxfp8_glu import compile_mxfp8_glu
                            compiled=compile_mxfp8_glu(MXFP8Problem(m,2*h,k),self.config,self.activation,self.output_format)
                            def runner(x,w,out,scales):
                                return compiled(x,w,out,scales,scales,scales,scales)
                    self.runners[key]=runner
        runner(x,weight,out,scales)
        if self.output_format=="mxfp8":
            return out,scales


class _Registry(dict):
    def __missing__(self,key):
        result=_Launcher(key)
        self[key]=result
        return result


LAUNCHERS=_Registry()


def register():
    from torch._inductor import ir
    from torch._inductor.lowering import register_lowering,empty_strided,view
    torch._rtx_glu_launchers=LAUNCHERS

    @register_lowering(torch.ops.rtx.mxfp8_glu.default,type_promotion_kind=None)
    def lower(x,weight,activation,output_format,config_key):
        m=math.prod(x.get_size()[:-1])
        k=x.get_size()[-1]
        h=weight.get_size()[0]//2
        x=view(x,[m,k])
        inputs=[ir.ExternKernel.require_contiguous(ir.ExternKernel.realize_input(t)) for t in (x,weight)]
        key=(activation,output_format,config_key)
        name=f"torch._rtx_glu_launchers[{key!r}]"
        if output_format=="bf16":
            output=ir.ExternKernelOut(ir.FixedLayout(x.get_device(),torch.bfloat16,[m,h],[h,1]),inputs,
                                      python_kernel_name=name)
            scales=empty_strided([m,0],[0,1],dtype=torch.uint8,device=x.get_device())
            return ir.TensorBox.create(output),scales
        width=(h+31)//32*32
        node=_GLUMultiOut(ir.MultiOutputLayout(device=x.get_device()),inputs,python_kernel_name=name)
        layouts=[ir.FixedLayout(x.get_device(),torch.float8_e4m3fn,[m,width],[width,1]),
                 ir.FixedLayout(x.get_device(),torch.uint8,[m,width//32],[width//32,1])]
        outputs=[ir.AllocatingMultiOutput(layout,node,[(tuple,i)]) for i,layout in enumerate(layouts)]
        node.outputs=outputs
        node.out_variant_output_nodes=outputs
        return tuple(ir.TensorBox.create(output) for output in outputs)
