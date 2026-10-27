"""Research paired finalizer launch fusion; production defaults unchanged."""
from contextlib import contextmanager
from benchmarks import experiment_ms_eden_dependency as study


def launcher(mode,inline_max=False,inline_scope='warp',pair_values=False,bit_brackets=False):
    if mode not in ('dx','dw','both'):raise ValueError('invalid fusion mode')
    def launch(dx,dw,live_dx,live_dw,device):
        import torch
        import rtx.experimental_ms_eden as eden
        from rtx.kernels.ms_eden_dual_finalizer import compile_dual_finalizer
        fused={}
        for name,leg in (('dx',dx),('dw',dw)):
            if mode not in (name,'both'):continue
            a,b=leg.prepare_a,leg.prepare_b
            if a.finalizer_args is None or b.finalizer_args is None:
                raise ValueError('explicit uniforms required')
            if any(p.schedule.amax_reduction!='staged' for p in (a,b)):
                raise ValueError('paired finalizer requires staged maxima')
            ra,ga,ca=a.finalizer_spec;rb,gb,cb=b.finalizer_spec
            runner=compile_dual_finalizer(ra,ga,rb,gb,ca,cb,a.partials.numel(),b.partials.numel(),inline_max=inline_max,inline_scope=inline_scope,pair_values=pair_values,bit_brackets=bit_brackets)
            fused[name]=(runner,(*a.finalizer_args,*b.finalizer_args))
        caller=torch.cuda.current_stream(device)
        entry=eden._operand_streams(device,caller)
        if entry is None:dx();dw();return
        sides,lock=entry;dx_b,dw_a,dw_b=sides
        with lock:
            try:
                for side in sides:side.wait_stream(caller)
                for value in live_dx:value.record_stream(dx_b)
                for side in (dw_a,dw_b):
                    for value in live_dw:value.record_stream(side)
                with torch.cuda.stream(dx_b):
                    if 'dx' in fused:dx.prepare_b.produce()
                    else:dx.prepare_b()
                with torch.cuda.stream(dw_a):
                    if 'dw' in fused:dw.prepare_a.produce()
                    else:dw.prepare_a()
                with torch.cuda.stream(dw_b):
                    if 'dw' in fused:dw.prepare_b.produce()
                    else:dw.prepare_b()
                if 'dx' in fused:dx.prepare_a.produce()
                else:dx.prepare_a()
                caller.wait_stream(dx_b)
                if 'dx' in fused:
                    runner,args=fused['dx'];runner(*args)
                dx.finish()
                with torch.cuda.stream(dw_a):
                    dw_a.wait_stream(dw_b)
                    if 'dw' in fused:
                        runner,args=fused['dw'];runner(*args)
                    dw.finish()
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
    return launch


@contextmanager
def patched(mode,inline_max=False,inline_scope='warp',pair_values=False,bit_brackets=False):
    import rtx.experimental_ms_eden as eden
    original=eden._launch_operand_paired
    # Reuse the existing guarded study CLI without introducing another parser.
    if mode!='control':eden._launch_operand_paired=launcher(mode,inline_max=inline_max,inline_scope=inline_scope,pair_values=pair_values,bit_brackets=bit_brackets)
    try:yield
    finally:eden._launch_operand_paired=original


if __name__=='__main__':
    study.patched=patched
    study.main(('control','dx','dw','both'))
