"""Complete-backward bounded exact reciprocal experiment; never auto-promotes."""
from contextlib import contextmanager
import inspect
from benchmarks import experiment_ms_eden_dependency as study


@contextmanager
def patched(mode):
    from rtx.kernels import nvfp4_posthoc as module
    from rtx.kernels.ms_eden_direct_refine import compile_direct_refine
    original=module.compile_hadamard_preparation
    signature=inspect.signature(original)
    def compile(*args,**kwargs):
        bound=signature.bind(*args,**kwargs);bound.apply_defaults()
        values=dict(bound.arguments)
        transpose=values['transpose']
        vpl=values['producer_transpose_values_per_lane' if transpose else 'producer_values_per_lane']
        selected=mode=='both' or (mode=='row' and not transpose) or (mode=='transpose' and transpose)
        if selected and vpl in (8,16):return compile_direct_refine(**values)
        return original(*args,**kwargs)
    module.compile_hadamard_preparation=compile
    try:yield
    finally:module.compile_hadamard_preparation=original


if __name__=='__main__':
    import hashlib
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    metadata={'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ('rtx/kernels/ms_eden_direct_refine.py','rtx/kernels/ms_eden_grouped_refine.py',
                     'rtx/kernels/nvfp4_posthoc.py','benchmarks/experiment_ms_eden_direct_refine.py')}}
    study.patched=patched
    study.main(('control','row','transpose','both'),experiment_metadata=metadata)
