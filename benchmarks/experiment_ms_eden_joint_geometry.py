"""Bounded pairwise-balanced four-producer grid study with the existing tuner."""
from dataclasses import asdict,replace
import argparse
import json
from pathlib import Path
import subprocess
import sys


def balanced_indices():
    # OA(16,4,4,2) over GF(4), polynomial x^2+x+1. Every two coordinates
    # contain each ordered level pair exactly once, unlike modulo-four sums.
    times_two=(0,2,3,1)
    return tuple((a,b,a^b,a^times_two[b]) for a in range(4) for b in range(4))


def portfolio(initial):
    from rtx.autotune.posthoc import PosthocOperandGeometry
    blocks=(64,128,256,512)
    results=[initial]
    for indices in balanced_indices():
        changes={}
        for phase,offset in (('dx',0),('dw',2)):
            schedule=getattr(initial,phase)
            operands={}
            for index,operand in enumerate(('a','b')):
                existing=getattr(schedule,operand+'_geometry') or PosthocOperandGeometry()
                operands[operand+'_geometry']=replace(existing,producer_blocks=blocks[indices[offset+index]])
            changes[phase]=replace(schedule,**operands)
        results.append(replace(initial,**changes))
    return results


def compact_gemm_portfolio(initial):
    """Full two-leg warp-geometry interaction, then deeper buffering challengers."""
    results = [initial]
    for dx_atoms in (2, 4, 8):
        for dw_atoms in (2, 4, 8):
            results.append(replace(initial, **{
                phase: replace(getattr(initial, phase), gemm=replace(
                    getattr(initial, phase).gemm, atom_layout_m=atoms))
                for phase, atoms in (('dx', dx_atoms), ('dw', dw_atoms))}))
    for phases in (('dx',), ('dw',), ('dx', 'dw')):
        results.append(replace(initial, **{
            phase: replace(getattr(initial, phase), gemm=replace(
                getattr(initial, phase).gemm, stages=3)) for phase in phases}))
    return list(dict.fromkeys(results))


def tma_finalizer_portfolio(initial):
    """Four independent scale grids, then equal-thread fused-leg challengers."""
    from rtx.autotune.posthoc import PosthocOperandGeometry
    results=[initial]
    levels=(26,52,104,208)
    for indices in balanced_indices():
        phases={}
        for phase,offset in (('dx',0),('dw',2)):
            base=getattr(initial,phase)
            changes={operand+'_geometry':replace(
                getattr(base,operand+'_geometry') or PosthocOperandGeometry(),
                finalizer_blocks=levels[indices[offset+i]]) for i,operand in enumerate(('a','b'))}
            phases[phase]=replace(base,**changes)
        results.append(replace(initial,**phases))
    for threads in (64,256,512):
        phases={}
        for phase in ('dx','dw'):
            base=getattr(initial,phase)
            changes={operand+'_geometry':replace(
                getattr(base,operand+'_geometry') or PosthocOperandGeometry(),
                finalizer_threads=threads) for operand in ('a','b')}
            phases[phase]=replace(base,**changes)
        results.append(replace(initial,**phases))
    return list(dict.fromkeys(results))


def tma_loop_portfolio(initial):
    """One-leg and joint instruction-scheduling challengers.

    Native mma128 scale staging fixes all CTA tile dimensions at128. Explore
    the legal loop and register-load choices without relaxing that contract.
    Preparation, reduction precision and numerical policy remain unchanged.
    """
    changes=tuple(dict(k_tile_unroll=n) for n in (2,3,4,6,8,12)) + (
        dict(mma_schedule='preload'),dict(a_ldmatrix_matrices=2),
        dict(b_ldmatrix_matrices=2))
    results=[initial]
    for change in changes:
        for phases in (('dx',),('dw',),('dx','dw')):
            results.append(replace(initial,**{phase:replace(getattr(initial,phase),
                gemm=replace(getattr(initial,phase).gemm,**change)) for phase in phases}))
    return list(dict.fromkeys(results))


def tma_persistence_portfolio(initial):
    """Limit resident GEMM grid waves independently on both gradient legs."""
    return list(dict.fromkeys([initial]+[
        replace(initial,**{phase:replace(getattr(initial,phase),
            gemm=replace(getattr(initial,phase).gemm,persistent_waves=waves))
            for phase,waves in (('dx',dx),('dw',dw))})
        for dx in (0,1,2) for dw in (0,1,2)]))


def worker(args):
    from contextlib import ExitStack
    import torch._inductor.config as compiler
    from benchmarks.compare_ms_eden_pipeline import compare,shape_value
    from benchmarks.experiment_ms_eden_operand_order import exact_smoke
    from rtx.autotune.ms_eden_pipeline import MSEdenPipelineSchedule
    config=MSEdenPipelineSchedule(**json.loads(args.config_json))
    shape=shape_value(args.shape)
    schedules={p:json.dumps(asdict(getattr(config,p)),sort_keys=True) for p in ('dx','dw')}
    from benchmarks.experiment_ms_eden_bank_layout import patched
    dispatch_records=[]
    with ExitStack() as contexts:
        if args.space in ('compact_gemms','compact64_grids'):
            contexts.enter_context(patched('compact'))
        elif args.space in ('tma_finalizers','tma_cta_finalizers','tma_producer_grids','tma_gemms','tma_loops','h16_grouped_grids','tma_persistence','tma_paired_finalizers'):
            from benchmarks.experiment_ms_eden_dual_finalizer import patched as dual_patch
            contexts.enter_context(patched('tma_xor5_1'))
            inline=args.space=='tma_cta_finalizers'
            contexts.enter_context(dual_patch('both',inline_max=inline,
                                             inline_scope='cta' if inline else 'warp',
                                             pair_values=args.space=='tma_paired_finalizers'))
            if args.space=='h16_grouped_grids':
                from benchmarks.experiment_ms_eden_h16_preparation import patched_preparation
                dispatch_records=contexts.enter_context(patched_preparation(grouped_output=True))
        exact=exact_smoke(shape,schedules,941)
        with compiler.patch(combo_kernels=True,combo_kernels_pointwise_only=True):
            result=compare(shape,args.output.with_suffix('.raw.json'),{},
                dx_overrides=asdict(config.dx),dw_overrides=asdict(config.dw),
                pipeline=config.backward_pipeline,baseline_policy='stable_mxfp8',scopes=('backward',),
                rounds=8,replays=16,min_batch_ms=2.,warmup_ms=5000.,seed=941)
    result['joint_geometry_experiment']={'same_draw':exact,'compiler_options':{
        'combo_kernels':True,'combo_kernels_pointwise_only':True}, 'space': args.space,
        'preparation_dispatch':dispatch_records}
    with args.output.open('x') as handle:json.dump(result,handle,indent=2)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--shape',default='decoder1536')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--selection',type=Path)
    p.add_argument('--trials',type=int,default=17)
    p.add_argument('--seconds',type=float,default=1800.)
    p.add_argument('--space', choices=('producer_grids', 'compact_gemms', 'compact64_grids','tma_finalizers','tma_cta_finalizers','tma_producer_grids','tma_gemms','tma_loops','h16_grouped_grids','tma_persistence','tma_paired_finalizers'), default='producer_grids')
    p.add_argument('--trial-timeout', type=float, default=180.)
    p.add_argument('--config-json',help=argparse.SUPPRESS)
    args=p.parse_args(argv)
    if args.config_json:return worker(args)
    from benchmarks.compare_ms_eden_pipeline import shape_value
    from benchmarks.tune_ms_eden_pipeline import runtime_environment
    from rtx.autotune.ms_eden_pipeline import MSEdenPipelineSchedule,make_ms_eden_pipeline_adapter,pipeline_outcome
    from rtx.autotune.first_hit import FirstHitPolicy,tune_first_hit
    from rtx.autotune.store import JsonlTuningStore
    from rtx.autotune.outcomes import TrialOutcome
    if args.selection is None:p.error('--selection is required')
    selection=json.loads(args.selection.read_text())['candidate']
    for phase in ('dx','dw'):
        selection[phase]['producer_grouped_store_bits']=64 if args.space in ('compact64_grids','tma_finalizers','tma_cta_finalizers','tma_producer_grids','tma_gemms','tma_loops','h16_grouped_grids','tma_persistence','tma_paired_finalizers') else 32
    selection['dx']['producer_values_per_lane']=16
    initial=MSEdenPipelineSchedule(**{**selection,'backward_pipeline':'operand_paired'})
    shape=shape_value(args.shape)
    device,environment=runtime_environment()
    import hashlib
    environment['source_sha256']['benchmarks/experiment_ms_eden_joint_geometry.py']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    for name in ('rtx/kernels/ms_eden_bank_layout.py', 'benchmarks/experiment_ms_eden_bank_layout.py',
                 'rtx/kernels/ms_eden_tma_preparation.py','rtx/kernels/ms_eden_dual_finalizer.py',
                 'benchmarks/experiment_ms_eden_dual_finalizer.py',
                 'rtx/kernels/ms_eden_h16_preparation.py','benchmarks/experiment_ms_eden_h16_preparation.py'):
        environment['source_sha256'][name] = hashlib.sha256((Path(__file__).resolve().parents[1]/name).read_bytes()).hexdigest()
    root=args.output.resolve();root.mkdir(parents=True,exist_ok=True)
    def evaluate(config):
        identity=adapter.config_id(config)
        import uuid
        trial=root/(identity+'-'+uuid.uuid4().hex[:8]);trial.mkdir()
        command=[sys.executable,'-m','benchmarks.experiment_ms_eden_joint_geometry',
            '--shape',args.shape,'--output',str(trial/'result.json'),
            '--space', args.space,
            '--config-json',json.dumps(asdict(config),sort_keys=True)]
        with (trial/'manifest.json').open('x') as handle:json.dump({'command':command,'config':asdict(config)},handle,indent=2)
        print('START',identity,flush=True)
        with (trial/'run.log').open('x') as log:
            try:completed=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=args.trial_timeout)
            except subprocess.TimeoutExpired:
                return TrialOutcome('runtime_error',error='isolated worker timeout',metadata={'artifact':str(trial)})
        if completed.returncode:
            outcome=TrialOutcome('runtime_error',error=(trial/'run.log').read_text()[-4000:])
        else:
            result=json.loads((trial/'result.json').read_text())
            outcome=pipeline_outcome(result,expected_config=config,expected_shape=shape,require_order_quality=True)
        outcome.metadata['artifact']=str(trial)
        print('DONE',identity,outcome.status,outcome.median_ms,flush=True)
        return outcome
    adapter=make_ms_eden_pipeline_adapter(shape,evaluate,device=device,environment=environment,
        initial=initial,measurement_contract=args.space+'_combo_stable_mxfp8_abba8_warm5s_order_gated_v3')
    # evaluate already performs the strict raw-result -> TrialOutcome conversion.
    # The adapter factory otherwise wraps its evaluator with that same conversion.
    adapter.evaluator=evaluate
    candidates=(tma_finalizer_portfolio(initial) if args.space in ('tma_finalizers','tma_cta_finalizers','tma_paired_finalizers') else
                tma_persistence_portfolio(initial) if args.space == 'tma_persistence' else
                tma_loop_portfolio(initial) if args.space == 'tma_loops' else
                compact_gemm_portfolio(initial) if args.space in ('compact_gemms','tma_gemms') else portfolio(initial))
    candidates=[c for c in candidates if adapter.rejection(c) is None]
    with (root/'portfolio.json').open('x') as handle:json.dump([asdict(c) for c in candidates],handle,indent=2)
    # No local mutations outside the controlled four-grid portfolio.
    adapter.axes={}
    result=tune_first_hit(adapter,JsonlTuningStore(root/'residuals'),candidates,
        policy=FirstHitPolicy(max_trials=args.trials,seconds=args.seconds,pool_size=len(candidates),confirmation_repeats=0))
    with (root/'selection.json').open('x') as handle:json.dump({'candidate':asdict(result.config),
        'median_ms':result.median_ms,'automatic_promotion':False,'requires_independent_confirmation':True},handle,indent=2)
    print('SELECTION',result.median_ms,flush=True)


if __name__=='__main__':main()
