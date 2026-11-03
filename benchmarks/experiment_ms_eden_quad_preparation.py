"""Exact four-operand producer fusion; retained arithmetic and finalizers."""
from contextlib import contextmanager
from functools import lru_cache
import inspect
from benchmarks import experiment_ms_eden_dependency as study


@lru_cache(maxsize=32)
def compile_plans(specs):
    from rtx.kernels.nvfp4_posthoc import compile_hadamard_preparation
    from rtx.kernels.ms_eden_quad_preparation import PreparationRole,compile_quad_preparation
    rename=dict(tile_order='producer_order',load_mode='producer_load',signs_mode='producer_signs',
                stages='producer_stages',async_copy='producer_async_copy')
    roles=[]
    for shape,config,schedule,transpose in specs:
        kwargs={}
        for name in inspect.signature(compile_hadamard_preparation).parameters:
            source=rename.get(name,name)
            if hasattr(schedule,source):kwargs[name]=getattr(schedule,source)
        roles.append(PreparationRole(*shape,config,4,transpose,**kwargs))
    return compile_quad_preparation(tuple(roles)),tuple(k.partial_count for k in roles)


def launch(dx,dw,live_dx,live_dw,device):
    import torch
    import rtx.experimental_ms_eden as eden
    plans=(dx.prepare_a,dx.prepare_b,dw.prepare_a,dw.prepare_b)
    if not all(p.original_bf16 for p in plans):raise ValueError('original BF16 required')
    specs=tuple((p.source_shape,p.config,p.schedule,p.transpose) for p in plans)
    runner,counts=compile_plans(specs)
    if counts!=tuple(p.partials.numel() for p in plans):raise ValueError('partial shape mismatch')
    args=tuple(v for p in plans for v in (p.source,p.signs,p.packed,p.temporary,p.partials))
    caller=torch.cuda.current_stream(device)
    entry=eden._operand_streams(device,caller)
    if entry is None:dx();dw();return
    sides,lock=entry;dx_b,dw_a,dw_b=sides
    with lock:
        try:
            for value in live_dx:value.record_stream(dx_b)
            for side in (dw_a,dw_b):
                for value in live_dw:value.record_stream(side)
            runner(*args)
            # Producer completion only: all waits precede caller finalization.
            for side in sides:side.wait_stream(caller)
            with torch.cuda.stream(dx_b):dx.prepare_b.finalize()
            with torch.cuda.stream(dw_a):dw.prepare_a.finalize()
            with torch.cuda.stream(dw_b):dw.prepare_b.finalize()
            dx.prepare_a.finalize()
            caller.wait_stream(dx_b);dx.finish()
            with torch.cuda.stream(dw_a):
                dw_a.wait_stream(dw_b);dw.finish()
        except BaseException:
            for side in sides:
                try:caller.wait_stream(side)
                except BaseException:pass
            raise
        try:caller.wait_stream(dw_a)
        except BaseException:
            for side in sides:
                try:caller.wait_stream(side)
                except BaseException:pass
            raise


@contextmanager
def patched(mode):
    import rtx.experimental_ms_eden as eden
    original=eden._launch_operand_paired
    if mode=='quad':eden._launch_operand_paired=launch
    elif mode!='control':raise ValueError('unknown mode')
    try:yield
    finally:eden._launch_operand_paired=original


if __name__=='__main__':
    study.patched=patched
    study.main(('control','quad'))
