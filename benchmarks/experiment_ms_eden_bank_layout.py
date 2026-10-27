"""Full-backward bank-layout experiment; no automatic winner promotion."""
from contextlib import contextmanager
import inspect
import functools
from benchmarks import experiment_ms_eden_dependency as study


@contextmanager
def patched(mode, *, records=None, strict=False, tma_tile_rows=32):
    from rtx.kernels import nvfp4_posthoc as module
    from rtx.kernels.ms_eden_bank_layout import compile_bank_layout
    from rtx.kernels.ms_eden_packed_signs import compile_packed_rademacher
    from rtx.kernels.ms_eden_group32 import compile_group32
    from rtx.kernels.ms_eden_tma_preparation import compile_tma_preparation
    original = module.compile_hadamard_preparation
    signature = inspect.signature(original)

    @functools.wraps(original)
    def compile(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = dict(bound.arguments)
        eligible = (values['transpose'] and values['async_copy']
                    and values['config'].threads == 128
                    and values['load_mode'] == 'coalesced_x8'
                    and values['producer_transpose_values_per_lane'] == 16
                    and values['producer_copy_bits'] == 32)
        tma = mode in ('tma1', 'tma2', 'tma3', 'tma_xor4_1', 'tma_xor4_2', 'tma_xor4_3', 'tma_xor5_1', 'tma_xor5_2', 'tma_xor5_3', 'tma_columns_1', 'tma_columns_2')
        if strict and tma and values['transpose'] and not eligible:
            raise ValueError('requested TMA transpose preparation would silently fall back')
        if records is not None:
            records.append(dict(rows=values['rows'], cols=values['cols'],
                transpose=values['transpose'], selected=mode if tma and eligible else 'native',
                stages=int(mode[-1]) if tma and eligible else values['stages'],
                tile_rows=tma_tile_rows if tma and eligible else None))
        if tma and eligible:
            values['producer_unroll_rows'] = False
            values['stages'] = int(mode[-1])
            return compile_tma_preparation(**values, tma_tile_rows=tma_tile_rows, consumer_order='columns' if 'columns' in mode else 'xor5' if 'xor5' in mode else 'xor4' if 'xor4' in mode else 'retained')
        if mode in ('group32', 'group32_reload', 'group32_stores', 'group32_reload_stores') and eligible:
            values['producer_unroll_rows'] = False
            if mode.startswith('group32_reload'):
                values['signs_mode'] = 'reload'
            return compile_group32(**values, address_hoist=True, combined_stores=mode.endswith('_stores'))
        if mode in ('xor', 'hoist', 'compact', 'compact_fma', 'compact2', 'compact3', 'compact_hint7', 'compact_hint8', 'packed', 'packed_hint7', 'packed_hint8') and eligible:
            if mode in ('compact', 'compact_fma', 'compact2', 'compact3', 'compact_hint7', 'compact_hint8', 'packed', 'packed_hint7', 'packed_hint8'):
                values['producer_unroll_rows'] = False
            if mode in ('compact2', 'compact3'):
                values['stages'] = int(mode[-1])
            if mode in ('compact_hint7', 'compact_hint8'):
                values['stages'] = 1
                return compile_bank_layout(**values, address_hoist=True, min_blocks_hint=int(mode[-1]))
            if mode == 'compact_fma':
                return compile_bank_layout(**values, address_hoist=True, butterfly_fma=True)
            if mode.startswith('packed') and values['signs_mode'] == 'register':
                hint = {'packed': 0, 'packed_hint7': 7, 'packed_hint8': 8}[mode]
                return compile_packed_rademacher(**values, address_hoist=True, min_blocks_hint=hint)
            return compile_bank_layout(**values, address_hoist=mode != 'xor')
        return original(*args, **kwargs)

    module.compile_hadamard_preparation = compile
    try:
        yield
    finally:
        module.compile_hadamard_preparation = original


if __name__ == '__main__':
    import faulthandler
    import hashlib
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    metadata = {'source_sha256': {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in ('rtx/kernels/ms_eden_bank_layout.py', 'rtx/kernels/ms_eden_tma_preparation.py', 'rtx/kernels/ms_eden_group32.py', 'rtx/kernels/ms_eden_packed_signs.py', 'rtx/kernels/nvfp4_posthoc.py',
                     'benchmarks/experiment_ms_eden_bank_layout.py')}}
    study.patched = patched
    faulthandler.enable()
    faulthandler.dump_traceback_later(60, repeat=True)
    try:
        study.main(('control', 'xor', 'hoist', 'compact', 'tma1', 'tma2', 'tma3', 'tma_xor4_1', 'tma_xor4_2', 'tma_xor5_1', 'tma_xor5_2', 'tma_columns_1', 'tma_columns_2', 'group32', 'group32_reload', 'group32_stores', 'group32_reload_stores', 'compact_fma', 'compact2', 'compact3', 'compact_hint7', 'compact_hint8', 'packed', 'packed_hint7', 'packed_hint8'), experiment_metadata=metadata)
    finally:
        faulthandler.cancel_dump_traceback_later()
