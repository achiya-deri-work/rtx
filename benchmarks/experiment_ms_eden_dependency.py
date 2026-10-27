"""Research-only dW producer admission after a selected dX frontier."""
import argparse
from contextlib import contextmanager, ExitStack
import json
from pathlib import Path


def launcher(frontier):
    if frontier not in ('producer','finalizer','finish'):
        raise ValueError('unknown dX frontier')
    def launch(dx,dw,live_dx,live_dw,device):
        import torch
        import rtx.experimental_ms_eden as eden
        caller=torch.cuda.current_stream(device)
        entry=eden._operand_streams(device,caller)
        if entry is None:
            dx();dw();return
        sides,lock=entry
        dx_b,dw_a,dw_b=sides
        with lock:
            try:
                for side in sides:side.wait_stream(caller)
                for value in live_dx:value.record_stream(dx_b)
                for side in (dw_a,dw_b):
                    for value in live_dw:value.record_stream(side)
                with torch.cuda.stream(dx_b):dx.prepare_b()
                with torch.cuda.stream(dw_b):dw.prepare_b()
                dx.prepare_a.produce()
                if frontier=='producer':dw_a.wait_stream(caller)
                dx.prepare_a.finalize()
                if frontier=='finalizer':dw_a.wait_stream(caller)
                # For producer/finalizer, enqueue before DX finish to allow
                # ordinary eager overlap too; graph dependency is identical.
                if frontier!='finish':
                    with torch.cuda.stream(dw_a):dw.prepare_a()
                caller.wait_stream(dx_b)
                dx.finish()
                if frontier=='finish':
                    dw_a.wait_stream(caller)
                    with torch.cuda.stream(dw_a):dw.prepare_a()
                with torch.cuda.stream(dw_a):
                    dw_a.wait_stream(dw_b)
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
def patched(frontier):
    import rtx.experimental_ms_eden as eden
    original=eden._launch_operand_paired
    if frontier!='control':eden._launch_operand_paired=launcher(frontier)
    try:yield
    finally:eden._launch_operand_paired=original


def main(frontiers=('control','producer','finalizer','finish'),experiment_metadata=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--frontier',choices=frontiers,required=True)
    parser.add_argument('--shape',default='decoder1536')
    parser.add_argument('--grouped-store-bits', type=int, choices=(8,16,32,64), default=32)
    parser.add_argument('--rng-mode', choices=('explicit', 'philox'), default='explicit')
    parser.add_argument('--philox-packet-size', type=int, choices=(1, 4), default=4)
    from benchmarks.experiment_ms_eden_operand_order import order_value
    parser.add_argument('--operand-order', type=order_value,
                        help='research enqueue-order interaction, no additional waits')
    parser.add_argument('--warmup-sync-each', action='store_true',
                        help='Diagnostic only: synchronize between graph warmups; timed windows unchanged')
    parser.add_argument('--finish-admission', choices=('independent','dx_a','dx_ready'),
                        default='independent', help='Research dW GEMM admission dependency; producers remain concurrent')
    parser.add_argument('--scale-finalizer', choices=('retained','paired','dual_dx','dual_both'),
                        default='retained', help='Research finalizer interaction with preparation variants')
    parser.add_argument('--probability-math',choices=('divide','power2'),
                        help='Explicit exact finalizer rounding-probability arithmetic override')
    parser.add_argument('--dual-inline-max',action='store_true',
                        help='Research dual finalizer: redundant warp maxima instead of a separate reduction launch')
    parser.add_argument('--dual-inline-scope',choices=('warp','cta'),default='warp')
    parser.add_argument('--dual-pair-values',action='store_true')
    parser.add_argument('--dual-bit-brackets',action='store_true')
    args=parser.parse_args()
    if args.dual_pair_values and not args.scale_finalizer.startswith('dual_'):
        parser.error('--dual-pair-values requires a dual scale finalizer')
    if args.dual_bit_brackets and (args.dual_pair_values or not args.scale_finalizer.startswith('dual_')):
        parser.error('--dual-bit-brackets requires a dual finalizer without paired conversion')
    if args.dual_inline_scope!='warp' and not args.dual_inline_max:
        parser.error('--dual-inline-scope cta requires --dual-inline-max')
    if args.dual_inline_max and not args.scale_finalizer.startswith('dual_'):
        parser.error('--dual-inline-max requires a dual scale finalizer')
    if args.operand_order and args.frontier in ('producer', 'finalizer', 'finish'):
        parser.error('operand order cannot override a dependency-frontier experiment')
    if args.finish_admission != 'independent' and args.frontier in ('producer','finalizer','finish'):
        parser.error('finish admission cannot override a producer dependency experiment')
    if args.scale_finalizer.startswith('dual_') and (args.operand_order or args.finish_admission != 'independent'
            or args.frontier in ('producer','finalizer','finish')):
        parser.error('dual finalizer cannot override another launch-topology experiment')
    if args.scale_finalizer != 'retained' and args.rng_mode != 'explicit':
        parser.error('research scale finalizers require explicit uniforms')
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    import torch._inductor.config as compiler_config
    from benchmarks.compare_ms_eden_pipeline import compare,shape_value
    from benchmarks.experiment_ms_eden_operand_order import exact_smoke
    record=json.loads(args.selection.read_text())
    selection=record.get('candidate',record.get('config'))
    if not isinstance(selection,dict) or set(selection) != {'dx','dw','backward_pipeline'}:
        raise ValueError('selection requires an explicit candidate or trial config')
    for phase in ('dx','dw'):selection[phase]['producer_grouped_store_bits']=args.grouped_store_bits
    selection['dx']['producer_values_per_lane']=16
    for phase in ('dx', 'dw'):
        selection[phase]['rng_mode'] = args.rng_mode
        if args.probability_math is not None:
            selection[phase]['probability_math'] = args.probability_math
        if args.rng_mode == 'philox':
            selection[phase]['philox_packet_size'] = args.philox_packet_size
    schedules={phase:json.dumps(selection[phase],sort_keys=True) for phase in ('dx','dw')}
    shape=shape_value(args.shape)
    with ExitStack() as contexts:
        contexts.enter_context(patched(args.frontier))
        if args.scale_finalizer == 'paired':
            from unittest.mock import patch
            from rtx.kernels import nvfp4_posthoc
            from rtx.kernels.ms_eden_paired_finalizer import compile_paired_scales
            contexts.enter_context(patch.object(nvfp4_posthoc,'compile_posthoc_scales',compile_paired_scales))
        elif args.scale_finalizer.startswith('dual_'):
            from benchmarks.experiment_ms_eden_dual_finalizer import patched as dual_patch
            contexts.enter_context(dual_patch(args.scale_finalizer.removeprefix('dual_'),inline_max=args.dual_inline_max,inline_scope=args.dual_inline_scope,pair_values=args.dual_pair_values,bit_brackets=args.dual_bit_brackets))
        if args.operand_order or args.finish_admission != 'independent':
            from benchmarks.experiment_ms_eden_operand_order import patched_order
            contexts.enter_context(patched_order(args.operand_order or ('dx_b','dw_a','dw_b','dx_a'),args.finish_admission))
        exact=exact_smoke(shape,schedules,941)
        with compiler_config.patch(combo_kernels=True,combo_kernels_pointwise_only=True):
            result=compare(shape,args.output.with_suffix('.raw.json'),{},
                dx_overrides=selection['dx'],dw_overrides=selection['dw'],pipeline='operand_paired',
                rounds=24,replays=16,min_batch_ms=2.,warmup_ms=15000.,seed=941,
                baseline_policy='stable_mxfp8',scopes=('backward',),
                warmup_sync_each=args.warmup_sync_each)
    result['dependency_experiment']=dict(frontier=args.frontier,exact=exact,
        operand_order=args.operand_order,
        finish_admission=args.finish_admission,
        scale_finalizer=args.scale_finalizer,
        dual_inline_max=args.dual_inline_max,
        dual_inline_scope=args.dual_inline_scope,
        dual_pair_values=args.dual_pair_values,
        dual_bit_brackets=args.dual_bit_brackets,
        probability_math=args.probability_math,
        compiler_options=dict(combo_kernels=True,combo_kernels_pointwise_only=True))
    import hashlib
    for source in (Path(__file__), Path(__file__).with_name('experiment_ms_eden_operand_order.py')):
        result['source_sha256']['benchmarks/'+source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
    if args.scale_finalizer != 'retained':
        root=Path(__file__).resolve().parents[1]
        for name in ('rtx/kernels/ms_eden_paired_finalizer.py','rtx/kernels/ms_eden_dual_finalizer.py',
                     'benchmarks/experiment_ms_eden_dual_finalizer.py'):
            result['source_sha256'][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    if experiment_metadata is not None:result['experiment_metadata']=experiment_metadata
    with args.output.open('x') as handle:json.dump(result,handle,indent=2)


if __name__=='__main__':main()
