"""Research full-backward H16 row preparation plus retained TMA transpose."""
from contextlib import contextmanager
import functools
import inspect


@contextmanager
def patched_preparation(*, grouped_output=False, recompute_coefficients=False):
    from rtx.kernels import nvfp4_posthoc
    from rtx.kernels.ms_eden_h16_preparation import compile_h16_preparation
    original=nvfp4_posthoc.compile_hadamard_preparation
    signature=inspect.signature(original)
    records=[]
    @functools.wraps(original)
    def compile(*args,**kwargs):
        bound=signature.bind(*args,**kwargs);bound.apply_defaults();v=bound.arguments
        eligible=(not v['transpose'] and v['mantissa_bits']==4
            and v['tile_order']=='row_major' and v['correction_decode']=='native'
            and v['normalization_math']=='divide' and v['producer_scale_math']=='divide'
            and v['partial_max_scope']=='warp')
        records.append(dict(rows=v['rows'],cols=v['cols'],transpose=v['transpose'],
                            selected=('h16_grouped_row' if grouped_output else 'h16_row') if eligible else 'retained'))
        if eligible:
            options=dict(grouped_output=grouped_output)
            if recompute_coefficients:options['recompute_coefficients']=True
            return compile_h16_preparation(v['rows'],v['cols'],v['config'],**options)
        return original(*args,**kwargs)
    nvfp4_posthoc.compile_hadamard_preparation=compile
    try:yield records
    finally:nvfp4_posthoc.compile_hadamard_preparation=original


if __name__=='__main__':
    import argparse
    import sys
    import hashlib
    from pathlib import Path
    from benchmarks.experiment_ms_eden_bank_layout import patched
    from benchmarks import experiment_ms_eden_dependency as study
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--h16-grouped-output',action='store_true')
    parser.add_argument('--h16-recompute-coefficients',action='store_true')
    args,remaining=parser.parse_known_args()
    if args.h16_recompute_coefficients and not args.h16_grouped_output:
        parser.error('--h16-recompute-coefficients requires --h16-grouped-output')
    sys.argv[1:]=remaining
    root=Path(__file__).resolve().parents[1]
    paths=('rtx/kernels/ms_eden_h16_preparation.py',
           'benchmarks/experiment_ms_eden_h16_preparation.py',
           'rtx/kernels/ms_eden_tma_preparation.py','benchmarks/experiment_ms_eden_bank_layout.py')
    with patched('tma_xor5_1'),patched_preparation(grouped_output=args.h16_grouped_output,
            recompute_coefficients=args.h16_recompute_coefficients) as records:
        study.main(('control',),experiment_metadata={
            'preparation_variant':'h16_row_tma_xor5_transpose','dispatch':records,
            'h16_grouped_output':args.h16_grouped_output,
            'h16_recompute_coefficients':args.h16_recompute_coefficients,
            'source_sha256':{p:hashlib.sha256((root/p).read_bytes()).hexdigest() for p in paths}})
