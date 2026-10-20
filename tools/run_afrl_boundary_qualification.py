"""Authorized, isolated owner-application experiment; no production edits.

Reuses the owner's checked qualification harness through an injected setup
module. All experiment-owned writable paths reside under the supplied RTX
output directory. The checkpoint intervention is process-local and opt-in.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import types

RTX = Path(__file__).resolve().parents[1]


def profiling_harness_tree(tree):
    """Insert one profiling callback after the owner's synchronized step count."""
    matches=[]
    class InsertProfile(ast.NodeTransformer):
        def visit_AugAssign(self,node):
            if ast.unparse(node)=="state['update'] += 1":
                matches.append(node)
                callback=ast.Expr(value=ast.Call(func=ast.Name(id='profile_update',ctx=ast.Load()),
                    args=[ast.Subscript(value=ast.Name(id='state',ctx=ast.Load()),
                        slice=ast.Constant(value='update'),ctx=ast.Load())],keywords=[]))
                return [node,ast.copy_location(callback,node)]
            return node
    tree=InsertProfile().visit(tree)
    if len(matches)!=1:
        raise RuntimeError('owner update boundary changed; refusing ambiguous profile capture')
    return ast.fix_missing_locations(tree)


class WarmUpdateProfiler:
    """Diagnostic capture only; no shapes, stacks, tensor references or memory history."""
    def __init__(self,torch,output):
        self.torch=torch
        self.output=output
        self.profiler=None
        self.active=False
        self.completed=False

    def update(self,number):
        if number==2:
            if self.profiler is not None or self.completed:
                raise RuntimeError('profiling boundary was entered twice')
            self.profiler=self.torch.profiler.profile(
                activities=[self.torch.profiler.ProfilerActivity.CPU,
                            self.torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False,profile_memory=False,with_stack=False)
            self.profiler.start()
            self.active=True
        elif number==3:
            if not self.active:
                raise RuntimeError('profiling update 3 without update 2 warmup boundary')
            self.active=False
            self.profiler.stop()
            self.profiler.export_chrome_trace(str(self.output/'steady-update.trace.json.gz'))
            # Avoid materializing a multi-GB Python event forest. The raw trace
            # is analyzed separately using launch correlations and streaming IO.
            self.profiler=None
            (self.output/'steady-update.profile.json').write_text(json.dumps(dict(
                schema_version=1,update=3,diagnostic_only=True,record_shapes=False,
                profile_memory=False,with_stack=False,
                scope='after synchronized optimizer update 2 through synchronized update 3',
                limitations=['runtime RTX labels do not cover the whole tokenizer',
                    'nested aggregate durations must not be summed',
                    'profile timings and capacity are not ordinary qualification measurements'],
                raw_trace='steady-update.trace.json.gz',aggregate_events_omitted=True),indent=2))
            self.completed=True

    def wrap(self,fn,phase):
        def wrapped(x,gu,*args,**kwargs):
            if not self.active:
                return fn(x,gu,*args,**kwargs)
            role='tokenizer' if tuple(gu.shape)==(1536,256) else 'backbone'
            name=f'rtx_mlp/{role}/{phase}/m{x.numel()//x.shape[-1]}'
            with self.torch.profiler.record_function(name):
                return fn(x,gu,*args,**kwargs)
        return wrapped

    def close(self):
        if self.active:
            self.active=False
            self.profiler.stop()
            self.profiler.export_chrome_trace(str(self.output/'incomplete-update.trace.json.gz'))


def fixed_harness_tree(source):
    """Replace exactly the selected-op coverage assertion, not numerical gates."""
    tree=ast.parse(source)
    matches=[]
    class ReplaceCoverage(ast.NodeTransformer):
        def visit_Assert(self,node):
            if ast.unparse(node.test)=="len(SEEN) == (6 if mode == 'candidate' else 4)":
                matches.append(node)
                return ast.copy_location(ast.Expr(value=ast.Call(func=ast.Name(id='verify_fixed_execution',ctx=ast.Load()),
                    args=[ast.Name(id='SEEN',ctx=ast.Load())],keywords=[])),node)
            return node
    tree=ReplaceCoverage().visit(tree)
    if len(matches)!=1:
        raise RuntimeError('owner coverage assertion changed; refusing to weaken unknown harness')
    return ast.fix_missing_locations(tree)


def qualified_backbone_index(owner, bundle):
    """Accept fresh qualified plans, never reinterpret historical qualification."""
    from rtx import MLPCandidate, MXFP8MLPConfig
    from rtx.autotune.replacement_benchmark import source_manifest
    if bundle.get('schema') != 1 or bundle.get('scope') != 'backbone_only_bf16_tokenizer':
        raise ValueError('invalid backbone-only qualification bundle')
    required=('manifest.json','request-index.json','application.py','application-gates.json')
    if set(bundle.get('owner_files',{})) != set(required):
        raise ValueError('incomplete owner harness provenance')
    for name in required:
        if hashlib.sha256((owner/name).read_bytes()).hexdigest()!=bundle['owner_files'][name]:
            raise RuntimeError(f'owner harness changed: {name}')
    if bundle.get('rtx_sources') != source_manifest():
        raise RuntimeError('backbone plan bundle RTX sources changed')
    expected={r['name']:r for r in json.loads((owner/'request-index.json').read_text())
              if r['name'].startswith('backbone-')}
    rows=bundle.get('requests',[])
    if len(rows)!=4 or len(expected)!=4 or {r['name'] for r in rows}!=set(expected):
        raise ValueError('exactly the four owner backbone contexts are required')
    for row in rows:
        original=expected[row['name']]
        if any(row.get(key)!=original.get(key) for key in ('shape','phase','input_shape')):
            raise ValueError('backbone shape/phase/input layout changed')
        request=Path(row['request']).resolve()
        if not request.is_relative_to(RTX/'autotune_reports'):
            raise ValueError('fresh backbone requests must reside in RTX/autotune_reports')
        candidate=MLPCandidate.load(request/'candidate.json')
        checked=candidate.inspect(request)
        if (checked['status']!='preflight_passed' or not checked['current_sources_match']
                or checked['requires_requalification']):
            raise RuntimeError('backbone request is not currently qualified')
        if (candidate.shape!=tuple(row['shape']) or candidate.phase!=row['phase']
                or candidate.activation!='silu'):
            raise ValueError('backbone request contract differs from index')
        selected=MXFP8MLPConfig.from_dict(checked['selected_plan']['config']).to_dict()
        if selected!=MXFP8MLPConfig.from_dict(row['config']).to_dict():
            raise ValueError('backbone index must contain the actually qualified selected plan')
    return rows


def write_backbone_plan_bundle(owner, requests, output):
    """Export an opt-in experiment bundle from already-completed RTX requests."""
    from rtx import MLPCandidate
    from rtx.autotune.replacement_benchmark import source_manifest
    owner=Path(owner).resolve(); output=Path(output).resolve()
    if not output.is_relative_to(RTX/'autotune_reports'):
        raise ValueError('bundle must reside in RTX/autotune_reports')
    rows=[dict(row) for row in json.loads((owner/'request-index.json').read_text())
          if row['name'].startswith('backbone-')]
    if set(requests)!={row['name'] for row in rows}:
        raise ValueError('provide exactly the owner backbone request names')
    for row in rows:
        request=Path(requests[row['name']]).resolve()
        candidate=MLPCandidate.load(request/'candidate.json')
        row.update(request=str(request),config=candidate.inspect(request)['selected_plan']['config'])
    bundle=dict(schema=1,scope='backbone_only_bf16_tokenizer',rtx_sources=source_manifest(),
        owner_files={name:hashlib.sha256((owner/name).read_bytes()).hexdigest()
                     for name in ('manifest.json','request-index.json','application.py','application-gates.json')},
        requests=rows,application_qualified=False)
    qualified_backbone_index(owner,bundle)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x') as stream:
        json.dump(bundle,stream,indent=2)
    return bundle


def verify(owner, backbone_plans=None):
    manifest=json.loads((owner/'manifest.json').read_text())
    afrl=owner.parents[2]
    if backbone_plans is not None:
        bundle=json.loads(Path(backbone_plans).read_text())
        qualified_backbone_index(owner,bundle)
        manifest={**manifest,'current':{f'rtx/{name}':digest for name,digest in bundle['rtx_sources'].items()},
                  'backbone_plan_bundle_sha256':hashlib.sha256(Path(backbone_plans).read_bytes()).hexdigest()}
    for group,root in (('afrl',afrl),('current',RTX)):
        for name,digest in manifest[group].items():
            if hashlib.sha256((root/name).read_bytes()).hexdigest()!=digest:
                raise RuntimeError(f'{group} source differs from qualification: {name}')
    return manifest,afrl


def prepare_backbone_study(owner, reports, output, *, resume=False):
    """Requalify pinned control/candidate plans in fresh request-local caches."""
    from rtx import MLPCandidate, AutotunePolicy, MXFP8MLPConfig
    owner=Path(owner).resolve(); reports=Path(reports).resolve(); output=Path(output).resolve()
    if not output.is_relative_to(RTX/'autotune_reports'):
        raise ValueError('study must reside in RTX/autotune_reports')
    manifest=json.loads((owner/'manifest.json').read_text())
    for name,digest in manifest['afrl'].items():
        if hashlib.sha256((owner.parents[2]/name).read_bytes()).hexdigest()!=digest:
            raise RuntimeError(f'afrl source differs from qualification: {name}')
    rows=[r for r in json.loads((owner/'request-index.json').read_text())
          if r['name'].startswith('backbone-')]
    if len(rows)!=4 or len({r['name'] for r in rows})!=4:
        raise ValueError('expected exactly four distinct backbone contexts')
    proposals={}
    for row in rows:
        proposals['control',row['name']]=MLPCandidate.load(Path(row['request'])/'candidate.json')
        proposals['candidate',row['name']]=MLPCandidate.from_qualification(
            reports/f"m{row['shape'][0]}_policy_matched_{row['phase']}213.json",field='challenger')
    output.mkdir(parents=True,exist_ok=resume)
    # One trial requalifies the exact selected plan, rather than changing the
    # experiment through additional search. Separate eight-trial capacity
    # experiments test search; this bundle tests the proposed application plans.
    policy=AutotunePolicy(max_trials=1,max_seconds=120,warmup=1,samples=3,
                         calls_per_sample=1,confirmation_repeats=0,finalist_rounds=0)
    for arm in ('control','candidate'):
        requests={}
        for row in rows:
            candidate=proposals[arm,row['name']]
            request=output/arm/row['name']
            if request.exists():
                if not resume or MLPCandidate.load(request/'candidate.json')!=candidate:
                    raise ValueError('existing request differs from proposed study; use a new output directory')
                print(f'RECHECK {arm} {row["name"]} request={request}',flush=True)
            else:
                prepared=candidate.prepare(request,policy=policy)
                print(f'QUALIFY {arm} {row["name"]} request={request}',flush=True)
                subprocess.run(prepared['command'],check=True)
            checked=candidate.inspect(request)
            if MXFP8MLPConfig.from_dict(checked['selected_plan']['config']).to_dict()!=candidate.config.to_dict():
                raise RuntimeError('qualification changed the proposed application plan')
            requests[row['name']]=request
        bundle=output/f'{arm}-plans.json'
        if bundle.exists():
            if not resume:
                raise FileExistsError(bundle)
            existing=qualified_backbone_index(owner,json.loads(bundle.read_text()))
            if {r['name']:Path(r['request']).resolve() for r in existing}!=requests:
                raise ValueError('existing bundle request paths differ from study')
        else:
            write_backbone_plan_bundle(owner,requests,bundle)
        verify(owner,bundle)
        print(f'READY {arm} bundle={bundle}',flush=True)


def worker(args):
    manifest,afrl=verify(args.owner,args.backbone_plans)
    if args.backbone_plans and os.environ.get('QUALIFICATION_MODE')!='baseline':
        raise ValueError('backbone-only experiment must keep the tokenizer BF16')
    os.chdir(afrl)
    sys.path.insert(0,str(RTX))
    sys.path.insert(1,str(afrl))
    import rtx
    import torch
    from torch.utils.checkpoint import checkpoint
    index=(json.loads(args.backbone_plans.read_text())['requests'] if args.backbone_plans
           else json.loads((args.owner/'request-index.json').read_text()))
    setup=types.ModuleType('setup_experiment')
    setup.ROOT=args.output
    setup.SOURCE='current'
    setup.CAP=8
    setup.SEEN=set()
    setup.KEYS={(*r['shape'],r['phase']=='train'):json.dumps(rtx.MXFP8MLPConfig.from_dict(r['config']).to_dict(),sort_keys=True) for r in index}
    rtx.configure_workspace(rtx.WorkspacePolicy(paired_bytes=256*1024**2,partial_bytes=128*1024**2,paired_entries=8,partial_entries=8))
    setup.workspace=lambda previous=None:rtx.workspace_report(previous=previous)
    capture=None
    if args.profile_update:
        import rtx._mlp_autotune as runtime
        capture=WarmUpdateProfiler(torch,args.output/'application')
        runtime._selected_forward=capture.wrap(runtime._selected_forward,'forward')
        runtime._selected_backward=capture.wrap(runtime._selected_backward,'backward')
    fixed_seen={}
    fixed_finite=[]
    if args.variant=='fixed':
        import rtx.mlp as mlp_impl
        image_rows=[r for r in index if r['name'].startswith('image-')]
        policies={r['phase']:rtx.MXFP8MLPConfig.from_dict(r['config']) for r in image_rows}
        if policies['train'].to_dict()!=policies['forward'].to_dict():
            raise RuntimeError('this fixed-boundary observer requires identical owner image phase policies')
        expected=policies['train'].to_dict()
        original_forward=mlp_impl._run_forward
        original_backward=mlp_impl._prepare_recomputed_all_bwd
        def observe_forward(x,gu,down,activation,config,out):
            result=original_forward(x,gu,down,activation,config,out)
            if tuple(gu.shape)==(1536,256):
                assert x.numel()==20480*256 and tuple(down.shape)==(256,768)
                assert config.to_dict()==expected and activation=='silu'
                if 'forward' not in fixed_seen:
                    fixed_finite.append(out.detach().isfinite().all())
                    fixed_seen['forward']=config.to_dict()
            return result
        def observe_backward(x,weight,grad_h,dy,down,activation,config_key,gate_key,down_key):
            if tuple(weight.shape)==(1536,256):
                assert tuple(x.shape)==(20480,256) and tuple(down.shape)==(256,768)
                assert json.loads(config_key)==expected and activation=='silu'
                fixed_seen['backward']=json.loads(config_key)
            return original_backward(x,weight,grad_h,dy,down,activation,config_key,gate_key,down_key)
        mlp_impl._run_forward=observe_forward
        mlp_impl._prepare_recomputed_all_bwd=observe_backward
        def verify_fixed_execution(seen):
            required={key for key in setup.KEYS if key[0]!=20480}
            assert seen==required,('backbone execution coverage',seen,required)
            assert set(fixed_seen)=={'forward','backward'},fixed_seen
            assert fixed_finite and all(bool(v) for v in fixed_finite)
            (args.output/'application/fixed-execution.json').write_text(json.dumps(dict(
                backbone_contexts=[list(k) for k in sorted(seen)],image=fixed_seen,
                phase_policy_identical=True,application_qualified=False),indent=2))
            fixed_finite.clear()
        setup.verify_fixed_execution=verify_fixed_execution
    def bind(layer,is_image):
        selected=[r for r in index if r['name'].startswith('image-' if is_image else 'backbone-')]
        result=rtx.bind_qualified_mlp_plans_(layer,[(r['request'],tuple(r['input_shape'])) for r in selected])
        if is_image and args.variant=='checkpoint':
            original=layer.forward
            def wrapped(*inputs,**kwargs):
                return checkpoint(original,*inputs,use_reentrant=False,**kwargs)
            layer.forward=wrapped
            result['experimental_checkpoint']='tokenizer MLP only, non-reentrant, process-local'
        if is_image and args.variant=='fixed':
            def fixed_forward(x):
                if tuple(x.shape)!=(32,640,256):
                    raise ValueError('fixed owner experiment only supports the qualified image shape')
                return rtx.mxfp8_mlp(x,layer.gate_up_weight,layer.down_weight,
                    activation=layer.activation,config=policies['train'])
            layer.forward=fixed_forward
            result['experimental_boundary']='fixed functional; identical qualified train/no-grad policy'
        return result
    setup.bind=bind
    sys.modules['setup_experiment']=setup
    sys.argv=[str(args.owner/'application.py'),'--num-epochs','5','--rope-geometry','v3',
        '--muon-implementation','helion','--muon-autotune-cache',str(args.output/'muon-cache'),
        '--limit-steps',str(args.steps),'--skip-eval']
    try:
        if args.variant=='fixed' or args.profile_update:
            program=args.owner/'application.py'
            tree=fixed_harness_tree(program.read_text()) if args.variant=='fixed' else ast.parse(program.read_text())
            if args.profile_update:
                tree=profiling_harness_tree(tree)
            exec(compile(tree,str(program),'exec'),dict(__name__='__main__',__file__=str(program),
                verify_fixed_execution=getattr(setup,'verify_fixed_execution',None),
                profile_update=capture.update if capture else None))
        else:
            runpy.run_path(str(args.owner/'application.py'),run_name='__main__')
    finally:
        if capture:
            capture.close()
        if verify(args.owner,args.backbone_plans)[0]!=manifest:
            raise RuntimeError('qualification manifest or bundle changed during application')
        (args.output/'source-postcheck.json').write_text(json.dumps({'unchanged':True}))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--owner',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',choices=('control','checkpoint','fixed'),required=True)
    p.add_argument('--steps',type=int,default=3)
    p.add_argument('--backbone-plans',type=Path,
        help='fresh qualified backbone-only bundle; keep tokenizer BF16, control variant only')
    p.add_argument('--prepare-backbone-study',type=Path,metavar='REPORT_DIR',
        help='prepare fresh control/candidate backbone bundles from policy-matched reports; no application launch')
    p.add_argument('--resume-backbone-study',action='store_true',
        help='reuse only fully verified identical requests from an interrupted preparation')
    p.add_argument('--history',action='store_true')
    p.add_argument('--profile-update',action='store_true',
        help='diagnostic CPU/CUDA capture of warmed update 3; no shape/memory/stack retention')
    p.add_argument('--cache-source',type=Path,
        help='copy caches from a completed RTX experiment into this fresh output; never modify source caches')
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    args=p.parse_args()
    if args.prepare_backbone_study:
        if args.worker or args.backbone_plans or args.variant!='control' or args.profile_update or args.history or args.cache_source:
            raise ValueError('study preparation is separate from application execution')
        return prepare_backbone_study(args.owner,args.prepare_backbone_study,args.output,
                                      resume=args.resume_backbone_study)
    if args.resume_backbone_study:
        raise ValueError('resume flag requires study preparation')
    if args.backbone_plans:
        if args.variant!='control':
            raise ValueError('backbone-only plans require control variant')
        args.backbone_plans=args.backbone_plans.resolve()
        if not args.backbone_plans.is_relative_to(RTX/'autotune_reports'):
            raise ValueError('backbone bundle must reside in RTX/autotune_reports')
    if args.profile_update and (args.history or args.variant!='control'):
        raise ValueError('profile capture currently requires control without allocation history')
    if args.variant=='fixed' and args.history:
        raise ValueError('selected-token ownership capture is not yet adapted to fixed-boundary history')
    args.owner=args.owner.resolve();args.output=args.output.resolve()
    if not args.output.is_relative_to(RTX/'autotune_reports'):
        raise ValueError('output must be inside RTX/autotune_reports')
    if args.steps<3:
        raise ValueError('at least two warmup updates and one steady update required')
    if args.worker:
        return worker(args)
    manifest,afrl=verify(args.owner,args.backbone_plans)
    if args.cache_source:
        args.cache_source=args.cache_source.resolve()
        if not args.cache_source.is_relative_to(RTX/'autotune_reports') or not (args.cache_source/'completion.json').is_file():
            raise ValueError('cache source must be a completed RTX experiment')
    args.output.mkdir(parents=True,exist_ok=False)
    (args.output/'application').mkdir()
    (args.output/'manifest.json').write_text(json.dumps(manifest,indent=2))
    shutil.copyfile(args.owner/'application-gates.json',args.output/'application-gates.json')
    shutil.copytree(afrl/'.cache/muon-runtime',args.output/'muon-cache')
    if args.cache_source:
        for name in ('inductor-cache','triton-cache','cuda-cache','xdg-cache','cute-cache'):
            source=args.cache_source/name
            if source.is_dir():
                shutil.copytree(source,args.output/name)
    env=dict(os.environ,QUALIFICATION_MODE='baseline' if args.backbone_plans else 'candidate',QUALIFICATION_NAME='application',
        RTX_QUAL_HISTORY=str(int(args.history)),OMP_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',
        PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',PYTHONPATH=f'{RTX}:{afrl}',
        WANDB_MODE='offline',WANDB_DIR=str(args.output),
        AFRL_SAVES_DIR=str(args.output/'application/checkpoints'),
        AFRL_LOGS_DIR=str(args.output/'application/logs'))
    for variable,name in [('TORCHINDUCTOR_CACHE_DIR','inductor-cache'),('TRITON_CACHE_DIR','triton-cache'),
        ('CUDA_CACHE_PATH','cuda-cache'),('XDG_CACHE_HOME','xdg-cache'),('CUTE_DSL_CACHE_DIR','cute-cache')]:
        env[variable]=str(args.output/name)
    cmd=[str(afrl/'.venv/bin/python'),'-m','torch.distributed.run','--standalone','--nproc_per_node=1',
        str(Path(__file__).resolve()),'--worker','--owner',str(args.owner),'--output',str(args.output),
        '--variant',args.variant,'--steps',str(args.steps)]
    if args.history:
        cmd.append('--history')
    if args.backbone_plans:
        cmd.extend(['--backbone-plans',str(args.backbone_plans)])
    if args.profile_update:
        cmd.append('--profile-update')
    (args.output/'launch.json').write_text(json.dumps(dict(command=cmd,environment={k:v for k,v in env.items()
        if k.startswith(('QUALIFICATION','RTX_QUAL','AFRL_','WANDB_','TORCHINDUCTOR','TRITON_','CUDA_CACHE','CUTE_DSL','XDG_','PYTHON'))},
        variant=args.variant,profile_update=args.profile_update,
        backbone_plans=str(args.backbone_plans) if args.backbone_plans else None,
        tokenizer_backend='bf16' if args.backbone_plans else 'mxfp8',
        cache_source=str(args.cache_source) if args.cache_source else None,
        application_qualified=False,tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2))
    with (args.output/'stdout.log').open('x') as log:
        process=subprocess.Popen(cmd,cwd=afrl,env=env,stdout=log,stderr=subprocess.STDOUT)
        (args.output/'process.json').write_text(json.dumps({'pid':process.pid}))
        print(f'START pid={process.pid} log={args.output}/stdout.log',flush=True)
        code=process.wait()
    if verify(args.owner,args.backbone_plans)[0]!=manifest:
        raise RuntimeError('qualification manifest or bundle changed during application')
    rows=[json.loads(line) for line in (args.output/'application/updates.jsonl').read_text().splitlines()] if (args.output/'application/updates.jsonl').exists() else []
    passed=code==0 and len(rows)==args.steps and all(all(r[k] for k in ('losses_finite','outputs_finite','gradients_finite','identities_unchanged'))
        and r['ooms']==0 and r['allocation_retries']==0 for r in rows)
    if args.profile_update:
        passed=passed and (args.output/'application/steady-update.profile.json').is_file()
    (args.output/'completion.json').write_text(json.dumps({'exit_code':code,'updates':len(rows),'gates_passed':passed,
        'diagnostic_profile':args.profile_update,'application_qualified':False},indent=2))
    print(f'DONE exit={code} updates={len(rows)} gates_passed={passed}',flush=True)
    if not passed:
        raise SystemExit(code or 1)


if __name__=='__main__':
    main()
