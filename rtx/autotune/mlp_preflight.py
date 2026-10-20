"""Disposable compiled first-hit MLP preflight with durable-cache verification.

Run before starting the training process, using its exact deferred MLP spec.
This is a synthetic same-shape capacity check, not full-model qualification.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time

from .replacement_benchmark import source_manifest


def record_selected_plan(report, token):
    """Record the forward-owned plan, not a cache search or inferred filename."""
    from .._execution_plan_token import read_plan_token
    from ..mlp import MXFP8MLPConfig
    config = MXFP8MLPConfig.from_dict(json.loads(read_plan_token(token))).to_dict()
    selection = dict(config=config, sha256=hashlib.sha256(
        json.dumps(config, sort_keys=True, allow_nan=False).encode()).hexdigest())
    if 'selected_plan' in report and report['selected_plan'] != selection:
        raise RuntimeError('selected MLP plan changed within preflight worker')
    report['selected_plan'] = selection


def require_selected_plan(report, expected=None):
    """Validate serialized selection evidence and optional cross-worker identity."""
    try:
        plan = report['selected_plan']
        if set(plan) != {'config', 'sha256'} or not isinstance(plan['config'], dict):
            raise ValueError('invalid selection structure')
        digest = hashlib.sha256(json.dumps(plan['config'], sort_keys=True,
                                          allow_nan=False).encode()).hexdigest()
        if digest != plan['sha256'] or (expected is not None and plan != expected):
            raise ValueError('selection identity differs')
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f'MLP preflight selected-plan gate failed: {exc}') from exc
    return plan


def require_numerical_gate(report):
    """Recompute acceptance from raw metrics, never trust a reported pass bit."""
    names = ('output', 'dx', 'dgate', 'ddown') if report.get('phase') == 'train' else ('output',)
    try:
        if report['status'] != 'ok' or report['phase'] not in ('train', 'forward'):
            raise ValueError('invalid status/phase')
        if report['oracle'] != 'independent_mxfp8_recipe_v1':
            raise ValueError('unknown oracle')
        if [case['seed'] for case in report['cases']] != [932, 933]:
            raise ValueError('missing seed coverage')
        for case in report['cases']:
            if set(case['metrics']) != set(names):
                raise ValueError('missing output/gradient coverage')
            for metrics in case['metrics'].values():
                if metrics['finite'] is not True:
                    raise ValueError('nonfinite result/reference')
                for key in ('max_abs', 'rms_error', 'reference_rms', 'normalized_rms'):
                    value = metrics[key]
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise ValueError('invalid numerical metric')
                error = metrics['rms_error'] / max(metrics['reference_rms'], 1e-12)
                if not math.isclose(error, metrics['normalized_rms'], rel_tol=1e-6, abs_tol=1e-12):
                    raise ValueError('inconsistent normalized error')
                if (metrics['max_abs'] > 1e-7 if metrics['reference_rms'] <= 1e-12 else error > .025):
                    raise ValueError('numerical tolerance exceeded')
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f'MLP preflight numerical gate failed: {exc}') from exc


def numerical_worker(args):
    """Separate strict-cache worker: oracle storage is NOT capacity evidence."""
    import torch
    from .._mlp_autotune import selected_mlp
    from .mlp import reference
    original = args.spec.read_text()
    spec = json.loads(original)
    if spec.get('mode') not in ('balanced', 'cache') or not spec.get('cache_dir'):
        raise ValueError('numerical preflight requires an explicit prepared cache')
    spec.update(mode='cache', cache_miss='error')
    spec_key = json.dumps(spec, sort_keys=True)
    training = args.phase == 'train'
    m,k,h,o = map(int, args.shape.split(','))
    report = dict(status='running', phase=args.phase, activation=args.activation,
        shape=[m,k,h,o], source_hashes_before=source_manifest(), cases=[],
        oracle='independent_mxfp8_recipe_v1', verify_cache=True,
        spec_sha256=hashlib.sha256(original.encode()).hexdigest(),
        torch=str(torch.__version__), cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), measurement_mode='selected_direct',
        relative_tolerance=.025, rms_floor=1e-12, near_zero_absolute_tolerance=1e-7,
        capacity_evidence=False)
    try:
        fn = torch.compile(lambda *v: selected_mlp(*v,args.activation,spec_key,training),
                           fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
        for seed in (932, 933):
            torch.manual_seed(seed)
            values = tuple((torch.randn(*s,device='cuda',dtype=torch.bfloat16)*scale).requires_grad_(training)
                for s,scale in (((m,k),.1),((2*h,k),.02),((o,h),.02)))
            dy = torch.randn(m,o,device='cuda',dtype=torch.bfloat16)*.1
            with torch.set_grad_enabled(training):
                y, plan_token = fn(*values)
            record_selected_plan(report, plan_token)
            del plan_token
            actual = (y,*torch.autograd.grad(y,values,dy)) if training else (y,)
            with torch.no_grad():
                expected = reference(*values,dy,args.activation)
                metrics = {}
                for name,a,r in zip(('output','dx','dgate','ddown'),actual,expected):
                    a,r = a.float(),r.float()
                    diff = a-r
                    rms = float(diff.square().mean().sqrt())
                    ref_rms = float(r.square().mean().sqrt())
                    metrics[name] = dict(finite=bool(a.isfinite().all() & r.isfinite().all()),
                        max_abs=float(diff.abs().max()), rms_error=rms, reference_rms=ref_rms,
                        normalized_rms=rms/max(ref_rms,1e-12))
                    for key,value in metrics[name].items():
                        if isinstance(value,float) and not math.isfinite(value):
                            metrics[name][key]=None
            report['cases'].append(dict(seed=seed,metrics=metrics))
            del actual,expected,values,y,dy
        report['status']='ok'
        require_numerical_gate(report)
        report['numerical_gate']={'passed':True}
    except Exception as exc:
        report.update(status='error',error=repr(exc),numerical_gate={'passed':False})
        raise
    finally:
        report['source_hashes_after']=source_manifest()
        with (args.output/'report.json').open('x') as stream:
            json.dump(report,stream,indent=2,allow_nan=False)


def require_preflight_sources(report, expected):
    """Reject missing or changed source evidence, including between workers."""
    if (not expected or report.get('source_hashes_before') != expected
            or report.get('source_hashes_after') != expected
            or source_manifest() != expected):
        raise RuntimeError('MLP preflight source identity changed or is missing')


def live_memory_gate(samples):
    """Fixed-workload gate: one cold pair, then three identical live samples."""
    if len(samples) != 4 or any(type(v) is not int or v < 0 for v in samples):
        raise ValueError('expected four nonnegative live allocation samples')
    warm = samples[1:]
    return dict(passed=len(set(warm)) == 1, warmup_pairs=1,
                measured_pairs=3, spread_bytes=max(warm)-min(warm),
                metric='synchronized_live_allocated_bytes', tolerance_bytes=0)


def capacity_gate(status, allocator_events, oom_events):
    """Completion is weaker than an OOM-free capacity qualification.

    A tuner may recover by rejecting a finalist. Keep that useful completion,
    but never present it as evidence that cold tuning fits without failures.
    Allocation retries alone are not terminal allocation failures.
    """
    oom_count = max(allocator_events.get('num_ooms', 0), len(oom_events))
    return dict(completed=status == 'ok', oom_free=oom_count == 0,
                passed=status == 'ok' and oom_count == 0,
                oom_count=oom_count,
                allocation_retries=allocator_events.get('num_alloc_retries', 0))


def require_preflight_gates(report):
    """Fail closed on missing, inconsistent, or failed worker evidence."""
    try:
        events = report['allocator_event_deltas']
        for key in ('num_ooms', 'num_alloc_retries'):
            if type(events[key]) is not int or events[key] < 0:
                raise ValueError(f'invalid allocator counter {key}')
        if not isinstance(report['oom_events'], list):
            raise ValueError('invalid OOM event list')
        capacity = capacity_gate(report['status'], events, report['oom_events'])
        live = live_memory_gate(report['live_allocated'])
        if report['capacity_gate'] != capacity or report['live_memory_gate'] != live:
            raise ValueError('inconsistent recorded gates')
        if not capacity['passed'] or not live['passed']:
            raise ValueError(f'capacity={capacity}, live_memory={live}')
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f'MLP preflight deployment gate failed: {exc}') from exc


def worker(args):
    sources_before = source_manifest()
    import torch
    from .memory_diagnostics import install_oom_recorder
    from ..runtime import runtime_cache_stats
    from .._mlp_autotune import selected_mlp
    import rtx.autotune.mlp as tuning
    root = Path(args.output)
    events = install_oom_recorder(root)
    spec = Path(args.spec).read_text()
    payload = json.loads(spec)
    if payload.get('mode') not in ('balanced', 'cache'):
        raise ValueError('preflight requires the exact cache/balanced deferred spec')
    if not payload.get('cache_dir'):
        raise ValueError('use an explicit shared cache_dir in the deferred spec')
    if args.verify_cache:
        from . import winners
        load = winners.load_runtime_winner
        def require_winner(*a, **kw):
            result = load(*a, **kw)
            if result is None:
                raise RuntimeError('verification found no eligible durable winner')
            return result
        winners.load_runtime_winner = require_winner
        def forbid(*a, **kw):
            raise RuntimeError('verification attempted retuning instead of durable reuse')
        tuning.tune_mxfp8_mlp = forbid
    torch.manual_seed(932)
    training = args.phase == 'train'
    m, k, h, o = map(int, args.shape.split(','))
    report = dict(shape=[m,k,h,o], spec=json.loads(spec), phase=args.phase,
                  source_hashes_before=sources_before,
                  activation=args.activation, measurement_mode='selected_direct',
                  spec_sha256=hashlib.sha256(spec.encode()).hexdigest(),
                  torch=str(torch.__version__), cuda=torch.version.cuda,
                  verify_cache=args.verify_cache, reserve_mib=args.reserve_mib,
                  status='running', environment_qualified=False)
    report['device_memory_before_reserve']=list(torch.cuda.mem_get_info())
    report['gpu']=torch.cuda.get_device_name()
    allocator_before=torch.cuda.memory_stats()
    started = time.monotonic()
    try:
        reserve = torch.empty(args.reserve_mib * 1024 * 1024, device='cuda', dtype=torch.uint8)
        values = tuple((torch.randn(*shape, device='cuda', dtype=torch.bfloat16)*scale).requires_grad_(training)
                       for shape, scale in (((m,k),.1), ((2*h,k),.02), ((o,h),.02)))
        dy = torch.randn(m,o,device='cuda',dtype=torch.bfloat16)*.1 if training else None
        fn = torch.compile(lambda *v: selected_mlp(*v,args.activation,spec,training),
                           fullgraph=True,dynamic=False,options={'triton.cudagraphs':False})
        torch.cuda.reset_peak_memory_stats()
        report['input_allocated'] = torch.cuda.memory_allocated()
        live = []
        report['live_allocated'] = live
        for step in range(4):
            if step == 0:
                torch.cuda.memory._dump_snapshot(str(root/'before-first-forward.pickle'))
            with torch.set_grad_enabled(training):
                y, plan_token = fn(*values)
            record_selected_plan(report, plan_token)
            del plan_token
            if step == 0 and training:
                torch.cuda.memory._dump_snapshot(str(root/'before-first-backward.pickle'))
            grads = torch.autograd.grad(y, values, dy) if training else ()
            if any(not bool(t.isfinite().all()) for t in (y,*grads)):
                raise RuntimeError('nonfinite preflight output/gradient')
            del y, grads
            torch.cuda.synchronize()
            if step == 0:
                report['first_iteration_peak_allocated'] = torch.cuda.max_memory_allocated()
                report['first_iteration_peak_reserved'] = torch.cuda.max_memory_reserved()
                if training:
                    report['first_pair_peak_allocated'] = report['first_iteration_peak_allocated']
                    report['first_pair_peak_reserved'] = report['first_iteration_peak_reserved']
            live.append(torch.cuda.memory_allocated())
        report.update(live_allocated=live, live_memory_gate=live_memory_gate(live),
                      peak_allocated=torch.cuda.max_memory_allocated(),
                      peak_reserved=torch.cuda.max_memory_reserved(),
                      reserved=torch.cuda.memory_reserved(),
                      synthetic_reserve_bytes=reserve.numel(), caches=runtime_cache_stats())
        if not report['live_memory_gate']['passed']:
            raise RuntimeError(f'post-warmup live allocation is not stable: {live}')
        report['status']='ok'
    except Exception as exc:
        report.update(status='error', error=repr(exc))
        if 'nvmlInit_v2_' in str(exc):
            report['oom_observer_limitation']='NVML initialization failed inside allocator error handling; observer may not have run'
        raise
    finally:
        report['source_hashes_after'] = source_manifest()
        allocator_after=torch.cuda.memory_stats()
        report['allocator_event_deltas']={key:allocator_after.get(key,0)-allocator_before.get(key,0)
            for key in ('num_alloc_retries','num_ooms','num_sync_all_streams')}
        report.update(elapsed_s=time.monotonic()-started, oom_events=events)
        report['capacity_gate'] = capacity_gate(
            report['status'], report['allocator_event_deltas'], events)
        with (root/'report.json').open('x') as stream:
            json.dump(report,stream,indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape',required=True,help='M,K,H,O')
    parser.add_argument('--spec',required=True,type=Path,help='Exact deferred spec JSON from the target MLP')
    parser.add_argument('--output',required=True,type=Path)
    parser.add_argument('--reserve-mib',type=int,default=0)
    parser.add_argument('--phase',choices=('train','forward'),default='train')
    parser.add_argument('--activation',choices=('silu','gelu','relu'),default='silu')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--verify-cache',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--check-numerics',action='store_true',
                        help='Run a separate strict-cache independent-reference worker after capacity checks')
    parser.add_argument('--numerical-worker',action='store_true',help=argparse.SUPPRESS)
    args=parser.parse_args()
    if args.reserve_mib < 0 or len(args.shape.split(',')) != 4 or any(int(v)<=0 for v in args.shape.split(',')):
        parser.error('positive M,K,H,O and nonnegative reserve required')
    if args.numerical_worker:
        numerical_worker(args)
        return
    if args.worker:
        worker(args)
        return
    args.output.mkdir(parents=True,exist_ok=False)
    expected_sources = source_manifest()
    selected_plan = None
    for phase in (('cold','reuse','numerics') if args.check_numerics else ('cold','reuse')):
        out=args.output/phase
        out.mkdir()
        command=[sys.executable,'-m','rtx.autotune.mlp_preflight','--worker',
                 '--shape',args.shape,'--spec',str(args.spec.resolve()),
                 '--output',str(out.resolve()),'--reserve-mib',str(args.reserve_mib),
                 '--phase',args.phase,'--activation',args.activation]
        if phase=='reuse':
            command.append('--verify-cache')
        if phase=='numerics':
            command.append('--numerical-worker')
        with (out/'worker.log').open('x') as log:
            result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT)
        print(f'{phase}: exit={result.returncode} report={out / "report.json"}',flush=True)
        if result.returncode:
            raise SystemExit(result.returncode)
        # Successful worker completion may hide a rejected OOM finalist.
        # Preserve all evidence, but do not proceed to reuse/launch approval.
        report = json.loads((out/'report.json').read_text())
        if phase=='numerics':
            require_numerical_gate(report)
        else:
            require_preflight_gates(report)
        require_preflight_sources(report, expected_sources)
        selected_plan = require_selected_plan(report, selected_plan)


if __name__=='__main__':
    main()
