"""Research interleaved dual-dY producer, retaining original BF16/FP32 math."""
from contextlib import contextmanager
from functools import lru_cache
import gc
import hashlib
import json
from pathlib import Path
import tempfile

from benchmarks import compare_ms_eden_pipeline as comparison


@lru_cache(maxsize=64)
def compile_for_schedules(shape,row_config,row_schedule,trans_config,trans_schedule):
    from rtx.kernels.nvfp4_posthoc import HadamardPreparationKernel
    from rtx.kernels.ms_eden_dual_dy_preparation import compile_dual_dy_preparation
    def make(config,schedule,transpose):
        return HadamardPreparationKernel(*shape,config,4,transpose,
            tile_order=schedule.producer_order,load_mode=schedule.producer_load,
            signs_mode=schedule.producer_signs,stages=schedule.producer_stages,
            async_copy=schedule.producer_async_copy,correction_decode=schedule.correction_decode,
            partial_max_scope=schedule.partial_max_scope,normalization_math=schedule.normalization_math,
            producer_copy_bits=schedule.producer_copy_bits,producer_unroll_rows=schedule.producer_unroll_rows,
            producer_row_tile_batch=schedule.producer_row_tile_batch,producer_row_load_bits=schedule.producer_row_load_bits,
            producer_values_per_lane=schedule.producer_values_per_lane,
            producer_transpose_values_per_lane=schedule.producer_transpose_values_per_lane,
            producer_scale_math=schedule.producer_scale_math,producer_grouped_store_bits=schedule.producer_grouped_store_bits)
    return compile_dual_dy_preparation(make(row_config,row_schedule,False),make(trans_config,trans_schedule,True))


def launch_dual_dy(dx,dw,live_dx,live_dw,device):
    import torch
    import rtx.experimental_ms_eden as eden
    row,trans=dx.prepare_a,dw.prepare_a
    if not (row.original_bf16 and trans.original_bf16 and not row.transpose and trans.transpose
            and row.source_shape==trans.source_shape and row.source.data_ptr()==trans.source.data_ptr()):
        raise ValueError("dual producer requires the same original BF16 dY")
    runner,counts=compile_for_schedules(row.source_shape,row.config,row.schedule,trans.config,trans.schedule)
    if counts!=(row.partials.numel(),trans.partials.numel()):raise ValueError("dual partial count mismatch")
    caller=torch.cuda.current_stream(device)
    entry=eden._operand_streams(device,caller)
    if entry is None:raise RuntimeError("dual research launcher requires a warmed bounded stream entry")
    sides,lock=entry;dx_b,dw_a,dw_b=sides
    with lock:
        try:
            for side in sides:side.wait_stream(caller)
            for value in live_dx:value.record_stream(dx_b)
            for side in (dw_a,dw_b):
                for value in live_dw:value.record_stream(side)
            runner(row.source,row.signs,row.packed,row.temporary,row.partials,
                   trans.signs,trans.packed,trans.temporary,trans.partials)
            # Record precisely the fused-producer completion, before row
            # finalization or unrelated caller joins extend that frontier.
            dw_a.wait_stream(caller)
            with torch.cuda.stream(dx_b):dx.prepare_b()
            with torch.cuda.stream(dw_b):dw.prepare_b()
            row.finalize()
            with torch.cuda.stream(dw_a):trans.finalize()
            caller.wait_stream(dx_b)
            dx.finish()
            with torch.cuda.stream(dw_a):
                dw_a.wait_stream(dw_b)
                dw.finish()
        except BaseException:
            for side in sides:
                try:caller.wait_stream(side)
                except BaseException:pass
            raise
        try:
            caller.wait_stream(dw_a)
        except BaseException:
            for side in (dx_b,dw_b):
                try:caller.wait_stream(side)
                except BaseException:pass
            raise


@contextmanager
def patched_dual_dy():
    import rtx.experimental_ms_eden as eden
    original=eden._launch_operand_paired
    eden._launch_operand_paired=launch_dual_dy
    try:yield
    finally:eden._launch_operand_paired=original


def main(argv=None):
    from benchmarks.experiment_ms_eden_operand_order import exact_smoke
    parser=comparison.parser();args=parser.parse_args(argv)
    if args.candidate_pipeline!="operand_paired" or args.baseline_policy!="stable_mxfp8":
        parser.error("requires operand_paired candidate and stable_mxfp8 baseline")
    if args.validate_exact:parser.error("full-shape exact gate is mandatory before timing")
    if args.output.exists():raise FileExistsError(args.output)
    shape=comparison.shape_value(args.shape)
    common,dx,dw=(json.loads(value) for value in (args.candidate_json,args.candidate_dx_json,args.candidate_dw_json))
    schedules=comparison.schedule_pair(common,dx,dw)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with patched_dual_dy():
        exact=exact_smoke(shape,schedules,args.seed);gc.collect()
        with tempfile.TemporaryDirectory(prefix="dual-dy-",dir=args.output.parent) as staging:
            result=comparison.compare(shape,Path(staging)/"comparison.json",common,
                dx_overrides=dx,dw_overrides=dw,pipeline="operand_paired",rounds=args.rounds,
                replays=args.replays,min_batch_ms=args.min_batch_ms,warmup_ms=args.warmup_ms,
                seed=args.seed,baseline_policy="stable_mxfp8",mxfp8_config_json=args.mxfp8_config_json,
                scopes=("backward","total") if args.scope=="both" else (args.scope,))
    result["dual_dy_experiment"]=dict(full_shape_exact=exact,
        producer_order="fused DY first, then independent W/X",both_dy_finalizers_wait_whole_fused_kernel=True,
        defaults_unchanged=True)
    root=Path(__file__).resolve().parents[1]
    for name in ("benchmarks/experiment_ms_eden_dual_dy.py","rtx/kernels/ms_eden_dual_dy_preparation.py"):
        result["source_sha256"][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    with args.output.open("x") as handle:json.dump(result,handle,indent=2)
    print(json.dumps(result["timings"],indent=2))


if __name__=="__main__":main()
