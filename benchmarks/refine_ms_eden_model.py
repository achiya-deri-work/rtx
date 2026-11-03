"""Refine qualified per-shape kernels using the real compiled decoder objective."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--batch',type=int,default=72)
    p.add_argument('--trials',type=int,default=8)
    p.add_argument('--seconds',type=float,default=900.)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    base=json.loads((args.study/f'batch-{args.batch}/fp4-designs.json').read_text())
    forward=json.loads((args.study/f'batch-{args.batch}/forward.json').read_text())
    commands=[]
    for path in args.study.glob('commands-gpu*.jsonl'):
        for line in path.read_text().splitlines():
            command=json.loads(line)['command']
            if ('benchmarks.train_precision20m' in command and command[command.index('--mode')+1]=='nvfp4_full'
                and f'/batch-{args.batch}/pilot-' in command[command.index('--output')+1]):commands.append(command)
    assert len(commands)==1
    original=commands[0]
    from rtx.autotune import DeviceFingerprint,JsonlTuningStore,TuningBudget
    from rtx.autotune.ms_eden_model import make_ms_eden_model_adapter,MSEdenModelSchedule
    from rtx.autotune.orchestrator import AutotuneOrchestrator,ConfirmationPolicy
    from rtx.autotune.bandit import UCB1Scheduler
    from rtx.autotune.strategies import CoordinateLocalSearch
    from rtx.autotune.first_hit import PortfolioSearch
    from rtx.autotune.core import Proposal,stable_id
    from rtx.autotune.outcomes import TrialOutcome
    from rtx.autotune.measurement import runtime_timing_metadata
    import statistics
    provenance={}

    def evaluate(designs,*,steps=32,destination=None):
        root=destination or args.output/f'trial-{stable_id(designs)}-{uuid.uuid4().hex[:8]}'
        root.mkdir()
        design_path=root/'designs.json'
        design_path.write_text(json.dumps(designs,indent=2))
        command=list(original)
        command[0]=sys.executable
        changes={'--output':root/'training.jsonl','--ms-eden-designs':design_path,
            '--steps':steps,'--warmup-steps':2,'--log-interval':4 if steps==32 else 8,
            '--eval-every':steps,'--checkpoint-interval':steps}
        for key,value in changes.items():command[command.index(key)+1]=str(value)
        before=time.monotonic()
        with (root/'run.log').open('x') as log:
            process=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600)
        (root/'command.json').write_text(json.dumps(dict(command=command,returncode=process.returncode,
            wall_s=time.monotonic()-before),indent=2))
        if process.returncode:
            return TrialOutcome('runtime_error',error=f'model worker failed: {root}')
        rows=[json.loads(s) for s in (root/'training.jsonl').read_text().splitlines()]
        environment=rows[0];terminal=rows[-1]
        if terminal.get('event')!='result' or terminal['step']!=steps or terminal['interrupted'] or not terminal['finite_parameters']:
            return TrialOutcome('correctness_error',error='incomplete/nonfinite model run')
        if terminal['peak_reserved_bytes']>.95*terminal['device_total_bytes']:
            return TrialOutcome('runtime_error',error='model exceeds declared 5% memory headroom')
        for key in ('initial_weights_sha256','corpus_sha256','validation_offsets_sha256','source_sha256','model_config'):
            provenance.setdefault(key,environment[key])
            if provenance[key]!=environment[key]:
                return TrialOutcome('correctness_error',error=f'model provenance mismatch: {key}')
        if environment.get('ms_eden_design_map')!=designs:
            return TrialOutcome('correctness_error',error='design map did not reach training')
        for key,design in designs.items():
            shape=tuple(map(int,key.split(',')))
            actual=environment['projection_configs'][str(shape)]
            if actual['forward']!=forward[key] or any(actual[phase]!=design['schedule'][phase] for phase in ('dx','dw')) or actual['backward_pipeline']!=design['schedule']['backward_pipeline']:
                return TrialOutcome('correctness_error',error=f'projection dispatch mismatch: {key}')
        blocks=[r for r in rows if r['event']=='train'][1:]
        timings=[r['block_training_wall_s']/r['block_steps']*1000 for r in blocks]
        return TrialOutcome('ok',median_ms=statistics.median(timings),timings_ms=timings,
            metadata=dict(validated=True,scope='complete_compiled_model_training_step',
                candidate_designs=designs,artifact=str(root),peak_allocated_bytes=terminal['peak_allocated_bytes'],
                **runtime_timing_metadata(timings)))

    root=Path(__file__).resolve().parents[1]
    environment={'source_sha256':{str(path.relative_to(root)):hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in ('rtx','benchmarks') for path in sorted((root/folder).rglob('*.py'))}}
    adapter=make_ms_eden_model_adapter(base,evaluate,batch=args.batch,sequence=512,
        model_context=dict(model='decoder205852672_situ',steps=32,warmup_steps=2,
            log_interval=4,precision='exact_jit_nvfp4_original_bf16_ms_eden'),
        device=asdict(DeviceFingerprint.current()),environment=environment)
    initial=adapter.initial_config
    uniform=MSEdenModelSchedule(('paired_dw_first',)*len(initial.pipelines))
    seeds=[initial,uniform]
    for index in range(len(initial.pipelines)):
        values=list(initial.pipelines);values[index]='paired_dw_first'
        seeds.append(MSEdenModelSchedule(tuple(values)))
    for index in range(len(initial.pipelines)):
        values=list(uniform.pipelines);values[index]=initial.pipelines[index]
        seeds.append(MSEdenModelSchedule(tuple(values)))
    search=AutotuneOrchestrator(adapter,JsonlTuningStore(args.output/'residuals'),
        [PortfolioSearch(seeds),CoordinateLocalSearch(shuffle_coordinates=True)],UCB1Scheduler(),
        TuningBudget(args.trials,args.seconds),transfer_history=False,resume=False,
        confirmation=ConfirmationPolicy(0,.02,True),progress=print,
        coverage_proposals=[Proposal(c,'model_schedule_coverage') for c in seeds])
    result=search.tune()
    assignments=adapter.serialize(result.config)['pipelines']
    selected=json.loads(json.dumps(base))
    for key,value in assignments.items():selected[key]['schedule']['backward_pipeline']=value
    confirmed=evaluate(selected,steps=64,destination=args.output/'confirmation')
    (args.output/'confirmation.json').write_text(json.dumps(confirmed.as_dict(),indent=2))
    if not confirmed.timing_eligible:raise RuntimeError('model winner failed independent timing/correctness confirmation')
    (args.output/'designs.json').write_text(json.dumps(selected,indent=2))
    (args.output/'selection.json').write_text(json.dumps(dict(config=adapter.serialize(result.config),
        search_median_ms=result.median_ms,confirmed_median_ms=confirmed.median_ms,
        context_id=result.context_id,automatic_promotion=False,provenance=provenance),indent=2))
    print('CONFIRMED',confirmed.median_ms,flush=True)


if __name__=='__main__':main()
