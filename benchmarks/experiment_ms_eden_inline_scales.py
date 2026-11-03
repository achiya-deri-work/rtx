"""Complete backward with stochastic scale finalization inside GEMMs."""
from contextlib import contextmanager
from dataclasses import replace
from benchmarks import experiment_ms_eden_dependency as study


@contextmanager
def patched(mode):
    import rtx.experimental_ms_eden as eden
    from rtx.configs.nvfp4 import NVFP4Problem
    from rtx.fp4 import _packed_fp4_view
    from rtx.kernels.nvfp4_posthoc import compile_partial_maximum
    from rtx.kernels.ms_eden_inline_scales import compile_inline_scale_gemm
    original=eden._launch_operand_paired

    def launch(dx,dw,live_dx,live_dw,device):
        for leg in (dx,dw):
            a,b=leg.prepare_a,leg.prepare_b
            if leg.output is None or any(p.finalizer_args is None or not p.original_bf16
                or p.schedule.amax_reduction!='staged' or p.schedule.gemm_split_reduction!=1 for p in (a,b)):
                raise ValueError('inline experiment requires original BF16, explicit RNG, staged maxima and unsplit GEMM')
            m,ga,_=a.finalizer_spec;n,gb,_=b.finalizer_spec
            if ga!=gb:raise ValueError('operand reduction mismatch')
            k=ga*16
            config=replace(a.schedule.gemm,scale_layout='row_major',scale_role=mode,scale_load_vec=1)
            runner=compile_inline_scale_gemm(NVFP4Problem(m,n,k),config,a.schedule.probability_math)
            metadata=(a.finalizer_args[2],b.finalizer_args[2],a.finalizer_args[3],b.finalizer_args[3])
            args=(_packed_fp4_view(a.packed.view(m,k//2)),_packed_fp4_view(b.packed.view(n,k//2)),
                  a.temporary.view(m,ga),b.temporary.view(n,gb),leg.output,metadata)
            # All tensor views/compilation occur before the kernel-only chain.
            for p in (a,b):
                reducer=compile_partial_maximum(p.partials.numel())
                partials,maximum=p.partials,p.finalizer_args[2]
                p.finalize=lambda reducer=reducer,partials=partials,maximum=maximum:reducer(partials,maximum)
            leg.finish=lambda runner=runner,args=args:runner(*args)
        return original(dx,dw,live_dx,live_dw,device)

    if mode!='control':eden._launch_operand_paired=launch
    try:yield
    finally:eden._launch_operand_paired=original


if __name__=='__main__':
    import hashlib
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    metadata={'effective_gemm_overrides':{'scale_layout':'row_major','scale_load_vec':1,
        'scale_role':'selected by --frontier; control retains the original schedule'},
        'source_sha256':{name:hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ('rtx/kernels/ms_eden_inline_scales.py','rtx/kernels/mxfp8_gemm.py',
                     'rtx/experimental_ms_eden.py','benchmarks/experiment_ms_eden_inline_scales.py')}}
    study.patched=patched
    study.main(('control','producer','consumers'),experiment_metadata=metadata)
