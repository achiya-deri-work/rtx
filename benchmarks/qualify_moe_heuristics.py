"""Bounded post-study profiling/panels; invoke under the study's GPU flock."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import sys

from benchmarks.profile_moe_active_budget import write
from rtx.autotune.supervisor import supervise_command,DEVICE_FAULT_EXIT_CODE


def prospective_prior(args,source):
    from types import SimpleNamespace
    import hashlib
    from benchmarks.study_moe_heuristics import study_contexts,result_validation
    from benchmarks.profile_moe_active_budget import identity,worker_command
    if args.prior is None or not args.prior.exists():raise ValueError('A frozen research prior is required')
    if args.shard not in (0,1):raise ValueError('Use shard 0 or 1')
    # Flat routing is the largest observed low-budget weakness. Qualify it
    # first on shard zero; both arms and both held-out contexts still run.
    chosen=(30,24) if args.shard==0 else (26,27)
    rows=[study_contexts()[index] for index in chosen]
    if args.flat_transfer:
        # Additional unseen flat geometry; not part of discovery training.
        row=dict(study_contexts()[26]);row['layout']='flat'
        row['id']=row['id'].removesuffix('_packed')+'_flat'
        row['axis']='prospective_flat_transfer';rows=[row]
    settings=SimpleNamespace(expert_trials=8,shared_trials=8,router_trials=8,
                             tune_seconds=150.,rounds=3,samples=7)
    stamp=identity(source);prior_hash=hashlib.sha256(args.prior.read_bytes()).hexdigest()
    manifest=dict(schema=1,identity=stamp,contexts=rows,settings=vars(settings),
        prior=str(args.prior.resolve()),prior_sha256=prior_hash,
        purpose='Prospective eight actual trials: default versus discovery-only shared-library prior',
        qualification='No automatic deployment; paired BF16 control uses each selected router')
    if (args.output/'manifest.json').exists():
        if json.loads((args.output/'manifest.json').read_text())!=manifest:
            raise ValueError('Prospective resume contract differs')
    else:write(args.output/'manifest.json',manifest)
    for index,row in enumerate(rows):
        # Rotate arm order between workloads rather than always warming one first.
        for mode in (('control','prior') if index%2==0 else ('prior','control')):
            folder=args.output/row['id']/mode;folder.mkdir(parents=True,exist_ok=True)
            target=folder/'result.json'
            if target.exists() and json.loads(target.read_text()).get('status')=='complete':
                result_validation(json.loads(target.read_text()),row,stamp,vars(settings));continue
            if identity(source)!=stamp or hashlib.sha256(args.prior.read_bytes()).hexdigest()!=prior_hash:
                raise ValueError('Source/prior changed after freezing')
            command=worker_command(source,target,row,settings)
            if mode=='prior':command+=['--prior-artifact',str(args.prior.resolve()),'--allow-research-prior']
            write(folder/'command.json',command)
            print('START',row['id'],mode,flush=True)
            with (folder/'run.log').open('a') as log:
                code=supervise_command(command,cwd=source,environment={**os.environ,
                    'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TORCHINDUCTOR_COMPILE_THREADS':'4'},
                    output=log,stall_timeout_s=240,wall_timeout_s=1200)
            write(folder/'exit.json',dict(exit_code=code))
            print('DONE',row['id'],mode,code,flush=True)
            if code==DEVICE_FAULT_EXIT_CODE:raise RuntimeError('CUDA fault; no follow-on GPU work')
            if not code:result_validation(json.loads(target.read_text()),row,stamp,vars(settings))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--action',choices=('profile','prior'),default='profile')
    p.add_argument('--study',required=True,type=Path)
    p.add_argument('--result',type=Path)
    p.add_argument('--extra-result',type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--panel-trials',type=int,default=96)
    p.add_argument('--no-scale-pair',action='store_true')
    p.add_argument('--prior',type=Path);p.add_argument('--shard',type=int,default=0)
    p.add_argument('--flat-transfer',action='store_true',help='Additional unseen D512/E48 global-pool first-hit comparison')
    p.add_argument('--dependency',type=Path,help='Earlier same-GPU qualification progress; stop on a recorded CUDA fault')
    args=p.parse_args();source=Path.cwd();args.output.mkdir(parents=True,exist_ok=True)
    progress=args.study/'progress.jsonl'
    if progress.exists() and any(json.loads(line)['exit_code']==DEVICE_FAULT_EXIT_CODE
                                for line in progress.read_text().splitlines() if line.strip()):
        raise RuntimeError('Study had a CUDA fault; no automatic follow-on GPU work')
    if args.dependency and args.dependency.exists():
        if any(r.get('exit_code')==DEVICE_FAULT_EXIT_CODE for r in json.loads(args.dependency.read_text())):
            raise RuntimeError('Earlier GPU qualification fault; no automatic follow-on GPU work')
    if args.action=='prior':return prospective_prior(args,source)
    if args.result is None:p.error('--result is required for profiling')
    events=[]
    def run(label,command,expected=None,wall=600):
        path=args.output/(label+'.log')
        if expected is not None and expected.exists():
            saved=json.loads(expected.read_text())
            if saved.get('status')=='complete' or saved.get('evaluated_this_launch',0)>0:
                print('REUSE',label,flush=True);return True
        write(args.output/(label+'.command.json'),command)
        print('START',label,flush=True)
        with path.open('a') as log:
            code=supervise_command(command,cwd=source,environment={**os.environ,
                'OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4','TORCHINDUCTOR_COMPILE_THREADS':'4'},
                output=log,stall_timeout_s=180,wall_timeout_s=wall)
        events.append(dict(label=label,exit_code=code,log=str(path.resolve())))
        write(args.output/'progress.json',events)
        print('DONE',label,code,flush=True)
        if code==DEVICE_FAULT_EXIT_CODE:raise RuntimeError('CUDA fault; stopped without reset')
        return code==0
    cases=[('anchor',args.result)]
    if args.extra_result is not None:cases.append(('contrast',args.extra_result))

    def flat_probe():
        # Preserve the 32-elements/lane column scan and test its actual hard
        # outputs, rather than assuming reduction-order equivalence.
        for path in sorted(args.study.glob('*/result.json')):
            saved=json.loads(path.read_text());c=saved['case']
            if saved.get('layout')!='flat' or c['num_experts']>64:continue
            if sum(saved['lengths'])*c['top_k']%c['num_experts']:continue
            target=args.output/'flat-preserving-forward.json'
            command=[sys.executable,'-u','-m','benchmarks.qualify_flat_router_forward',
                '--fixture',str(path.resolve()),'--control-forward','128,1,4','--forward','64,1,2']
            run('flat-preserving-forward',[*command,'--output',str(target)],target)
            if (c['d_model'],c['top_k'],c['expert_hidden_dim'])==(384,4,320):
                target=args.output/'flat-preserving-heldout.json'
                run('flat-preserving-heldout',[*command,'--output',str(target),'--heldout'],target)
            break

    for label,result in cases:
        if not result.exists():
            print('MISSING source result',result,flush=True);continue
        folder=args.output/label;folder.mkdir(exist_ok=True)
        fixture=folder/'expert.pt';times=folder/'times.json'
        command=[sys.executable,'-u','-m','benchmarks.profile_moe_routing_cost',
                 '--result',str(result.resolve()),'--output',str(times),'--fixture',str(fixture)]
        if not run(label+'-times',command,times):continue
        if label=='anchor':flat_probe()
        for phase in ('expert_step','router_step'):
            output=folder/(phase+'.json')
            run(label+'-trace-'+phase,[*command[:4],'--result',str(result.resolve()),
                '--output',str(output),'--mode','trace','--phase',phase],output)
        panel=folder/'schedule-panel'
        run(label+'-panel',[sys.executable,'-u','-m','benchmarks.benchmark_moe_geometry_frontier',
            '--result',str(result.resolve()),'--fixture',str(fixture),'--output',str(panel),
            '--mode','schedule','--stages','down','down_dx','gate_dx',
            '--trials',str(args.panel_trials),'--max-seconds','360'],panel/'result.json',wall=900)
        if label!='anchor':continue
        nsys=shutil.which('nsys')
        if nsys:
            prefix=folder/'whole-timeline'
            run('whole-timeline',[nsys,'profile','--trace=cuda,nvtx,osrt','--sample=none','--cuda-graph-trace=node',
                '--cpuctxsw=none','--capture-range=cudaProfilerApi','--capture-range-end=stop',
                '--force-overwrite=true','--output',str(prefix),sys.executable,'-u','-m',
                'benchmarks.profile_moe_routing_cost','--result',str(result.resolve()),
                '--output',str(folder/'whole-profile.json'),'--mode','nsys',
                '--phase','whole_step','--graph-replay'],folder/'whole-profile.json')
            if prefix.with_suffix('.nsys-rep').exists():
                run('export-timeline',[nsys,'export','--type','sqlite','--force-overwrite=true',
                    '--output',str(prefix.with_suffix('.sqlite')),str(prefix.with_suffix('.nsys-rep'))])
        ncu=shutil.which('ncu')
        if ncu:
            prefix=folder/'movement-counters'
            # Router preparation happened outside injection. Only the saved
            # real expert route executes in this disposable NCU worker.
            ok=run('movement-counters',[ncu,'--profile-from-start','off','--replay-mode','kernel',
                '--graph-profiling','node','--clock-control','none','--cache-control','none',
                '--metrics','gpu__time_duration.sum,dram__bytes_op_read.sum,dram__bytes_op_write.sum,lts__t_bytes.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum,l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum',
                '--kernel-name-base','demangled','-k','regex:CompactGather|CompactCombine|PackedInputGather|PackedGather|RoutedDYQuant|TokenReduce|IndexedExpertTranspose|ExpertWarpTransposeOperand',
                '--launch-count','8','--force-overwrite','-o',str(prefix),sys.executable,'-u','-m',
                'benchmarks.profile_moe_frozen_experts','--source-result',str(result.resolve()),
                '--fixture',str(fixture),'--output',str(folder/'counter-worker.json'),
                '--precision','mxfp8','--capture','counters'],wall=600)
            if ok and prefix.with_suffix('.ncu-rep').exists():
                run('export-counters',[ncu,'--import',str(prefix.with_suffix('.ncu-rep')),'--csv','--page','raw'])
    if not args.no_scale_pair:
        # Same board, weights, input, total tokens and active compute budget;
        # only independent sequence lengths differ. K6/E128 has half-filled
        # 32-row groups at L1024, versus full groups at L2048. Routing semantics
        # differ intentionally: this is not an interchangeable dispatch A/B.
        from types import SimpleNamespace
        from benchmarks.profile_moe_active_budget import worker_command
        saved=json.loads(args.result.read_text())
        c={k:saved['case'][k] for k in ('d_model','top_k','expert_hidden_dim','shared_hidden_dim','num_experts')}
        c.update(top_k=6,num_experts=128)
        c['expert_hidden_dim']=192 if c['d_model']==384 else 256
        c['shared_hidden_dim']=320 if c['d_model']==384 else 384
        settings=SimpleNamespace(expert_trials=32,shared_trials=32,router_trials=32,
                                 tune_seconds=300.,rounds=3,samples=7)
        pair=[]
        for b,length in ((32,1024),(16,2048)):
            folder=args.output/f'b{b}_l{length}';folder.mkdir(exist_ok=True)
            target=folder/'result.json'
            row=dict(case=c,lengths=[length]*b,layout='packed',seed=17391,stable_experts=True)
            if run(f'scale-b{b}',worker_command(source,target,row,settings),target,wall=1200):
                pair.append(json.loads(target.read_text()))
        if len(pair)==2:
            first,second=pair
            for key in ('input','upstream','parameters'):
                if first['identities'][key]!=second['identities'][key]:
                    raise AssertionError(('Scale-pair initialization mismatch',key))
            write(args.output/'scale-pair.json',dict(status='complete',case=c,total_tokens=32768,
                matched_inputs_upstream_parameters=True,
                contexts=[dict(lengths=r['lengths'],expert_plan=r['expert_plan'],
                               timings=r['timings'],identities=r['identities']) for r in pair],
                caveat='Different independent balance pools and hard routes; compare same-board costs, not semantic equivalence'))


if __name__=='__main__':main()
