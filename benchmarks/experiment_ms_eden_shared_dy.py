"""Complete backward experiment for true shared dY input-tile reuse."""
from contextlib import contextmanager
from benchmarks import experiment_ms_eden_dependency as study


def compile_shared(shape,row_config,row_schedule,trans_config,trans_schedule,*,row_vector_bits=16,swizzle=0,warp_roles=False):
    from rtx.kernels.ms_eden_shared_dy import compile_shared_dy
    if (row_config.threads,row_config.blocks)!=(trans_config.threads,trans_config.blocks):
        raise ValueError('shared dY requires matching source partial geometry')
    if row_schedule.producer_values_per_lane!=trans_schedule.producer_transpose_values_per_lane:
        raise ValueError('shared dY requires matching values-per-lane')
    for schedule in (row_schedule,trans_schedule):
        for name,value in dict(partial_max_scope='warp',normalization_math='divide',
            producer_scale_math='divide',correction_decode='native',producer_signs='register',
            producer_grouped_store_bits=32).items():
            if getattr(schedule,name)!=value:raise ValueError(f'unsupported shared dY {name}')
    runner,count=compile_shared_dy(*shape,row_config.threads,row_config.blocks,
        row_schedule.producer_values_per_lane,row_vector_bits,swizzle,warp_roles=warp_roles)
    return runner,(count,count)


@contextmanager
def patched(mode):
    import rtx.experimental_ms_eden as eden
    from benchmarks import experiment_ms_eden_dual_dy as dual
    original_launch=eden._launch_operand_paired
    original_compile=dual.compile_for_schedules
    if mode=='shared':
        dual.compile_for_schedules=compile_shared
        eden._launch_operand_paired=dual.launch_dual_dy
    elif mode!='control':raise ValueError('unknown mode')
    try:yield
    finally:
        eden._launch_operand_paired=original_launch
        dual.compile_for_schedules=original_compile


if __name__=='__main__':
    import argparse
    from functools import partial
    import hashlib
    from pathlib import Path
    import sys
    parser=argparse.ArgumentParser(add_help=False)
    parser.add_argument('--row-vector-bits',type=int,choices=(16,128),default=16)
    parser.add_argument('--swizzle',type=int,choices=(0,1),default=0)
    parser.add_argument('--warp-roles',action='store_true')
    extra,remaining=parser.parse_known_args()
    sys.argv=[sys.argv[0],*remaining]
    compile_shared=partial(compile_shared,row_vector_bits=extra.row_vector_bits,swizzle=extra.swizzle,warp_roles=extra.warp_roles)
    study.patched=patched
    root=Path(__file__).resolve().parents[1]
    metadata=dict(vars(extra))
    metadata['source_sha256']={name:hashlib.sha256((root/name).read_bytes()).hexdigest()
        for name in ('rtx/kernels/ms_eden_shared_dy.py',
                     'benchmarks/experiment_ms_eden_shared_dy.py',
                     'benchmarks/experiment_ms_eden_dual_dy.py',
                     'benchmarks/experiment_ms_eden_dependency.py')}
    study.main(('control','shared'),experiment_metadata=metadata)
