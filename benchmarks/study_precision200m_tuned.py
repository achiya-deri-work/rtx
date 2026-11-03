"""Batch-aware fresh tuning, full-model requalification, then matched training.

No global runtime cache is changed. All tuning is in disposable processes;
training consumes explicit maps with autotuning disabled in its timed loop.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

PROJECTIONS=((3072,1024),(1024,1024),(5632,1024),(1024,2816))
MODES=('nvfp4_mxfp8','nvfp4_full')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tokens',type=Path,required=True)
    p.add_argument('--probe-root',type=Path,required=True,
                   help='parent containing batch200m_MODE_20261028/summary.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--trials',type=int,default=32)
    p.add_argument('--seconds',type=float,default=300.)
    p.add_argument('--model-trials',type=int,default=8)
    p.add_argument('--model-seconds',type=float,default=900.)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()

    def run(gpu,module,arguments,log,timeout=1800):
        command=[sys.executable,'-u','-m',module,*map(str,arguments)]
        print(f'START gpu={gpu} {module} {arguments}',flush=True)
        before=time.monotonic()
        with Path(log).open('x') as stream:
            result=subprocess.run(command,stdout=stream,stderr=subprocess.STDOUT,
                env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu)},timeout=timeout)
        with (args.output/f'commands-gpu{gpu}.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(command=command,gpu=gpu,returncode=result.returncode,
                wall_s=time.monotonic()-before,log=str(log)))+'\n')
        if result.returncode:raise RuntimeError(f'failed command; inspect {log}')

    summaries=[args.probe_root/f'batch200m_{mode}_20261028/summary.json' for mode in MODES]
    deadline=time.monotonic()+3600
    while not all(path.exists() for path in summaries):
        if time.monotonic()>deadline:raise RuntimeError('capacity probes did not complete; no tuning launched')
        time.sleep(15)
    capacity=[json.loads(path.read_text()) for path in summaries]
    maximum=min(record['maximum_safe_tested_batch'] for record in capacity)
    batches=sorted({maximum,max(8,(maximum*3//4)//8*8)},reverse=True)
    (args.output/'plan.json').write_text(json.dumps(dict(capacity=capacity,batches=batches,
        projections=PROJECTIONS,trials=args.trials,seconds_per_context=args.seconds,
        training_clock_excludes_all_preparation=True),indent=2))

    def tune_backward4(batch,root):
        from rtx.autotune.ms_eden_design import MSEdenDesign
        from rtx.autotune.ms_eden_pipeline import MSEdenPipelineSchedule
        initial=root/'fp4-seed.json'
        initial.write_text(json.dumps({'candidate':asdict(MSEdenDesign(
            MSEdenPipelineSchedule(backward_pipeline='paired')))}))
        for n,k in PROJECTIONS:
            key=f'{batch*512},{n},{k}'
            run(1,'benchmarks.tune_ms_eden_design',[
                '--shape',key,'--selection',initial,'--output',root/f'fp4-{n}-{k}',
                '--trials',args.trials,'--seconds',args.seconds,'--rounds',8,
                '--allow-row-major','--warmup-ms',5000],root/f'fp4-{n}-{k}.log')

    def tune_forward_backward8(batch,root):
        for phase in ('forward','mxfp8_backward'):
            mapping={}
            for n,k in PROJECTIONS:
                directory=root/f'{phase}-{n}-{k}'
                run(0,'benchmarks.tune_precision200m_component',[
                    '--phase',phase,'--batch',batch,'--n',n,'--k',k,
                    '--trials',args.trials,'--seconds',args.seconds,'--output',directory],
                    directory.with_suffix('.log'))
                mapping[f'{batch*512},{n},{k}']=json.loads((directory/'selection.json').read_text())['config']
            (root/f'{phase}.json').write_text(json.dumps(mapping,indent=2))

    def select_fp4(batch,root):
        from benchmarks.select_ms_eden_training_designs import choose_common
        from rtx.autotune.ms_eden_design import MSEdenDesign,make_ms_eden_design_adapter
        records=[]
        keys=[f'{batch*512},{n},{k}' for n,k in PROJECTIONS]
        for n,k in PROJECTIONS:
            key=f'{batch*512},{n},{k}'
            for path in (root/f'fp4-{n}-{k}').glob('*/result.json'):
                raw=json.loads(path.read_text())
                design=MSEdenDesign(**raw['design'])
                adapter=make_ms_eden_design_adapter((batch*512,n,k),lambda _:raw,
                    initial=design,allow_row_major=True)
                outcome=adapter.evaluate(design)
                if outcome.timing_eligible and outcome.metadata.get('sampling',{}).get('collection',{}).get('stationary') is True:
                    records.append(dict(shape=key,design=asdict(design),median_ms=outcome.median_ms,path=str(path)))
        selected=choose_common(records,required=keys)
        (root/'fp4-selection.json').write_text(json.dumps(selected,indent=2))
        mapping={key:row['design'] for key,row in selected['selected'].items()}
        (root/'fp4-designs.json').write_text(json.dumps(mapping,indent=2))
        for key,design in mapping.items():
            selection=root/f'confirm-input-{key}.json'
            selection.write_text(json.dumps({'candidate':design}))
            run(1,'benchmarks.tune_ms_eden_design',[
                '--shape',key,'--selection',selection,'--output',root/f'confirm-{key}',
                '--confirm-only','--rounds',24,'--warmup-ms',15000,
                '--seconds',240],root/f'confirm-{key}.log',timeout=600)

    selected_model_designs=None
    def model_run(batch,root,phase,gpu):
        mode=MODES[gpu]
        steps=64 if phase=='pilot' else 100663296//(batch*512)
        output=root/f'{phase}-{mode}.jsonl'
        options=['--tokens',args.tokens,'--output',output,'--mode',mode,
            '--layers',16,'--width',1024,'--intermediate',2816,'--heads',16,
            '--batch',batch,'--sequence',512,'--corpus-tokens',100000000,
            '--mlp-activation','situ','--cudagraphs','--combo-kernels',
            '--steps',steps,'--warmup-steps',2 if phase=='pilot' else max(1,8192//batch),
            '--log-interval',8 if phase=='pilot' else 64,
            '--eval-every',steps if phase=='pilot' else max(1,16384//batch),
            '--eval-batches',4 if phase=='pilot' else 32,
            '--checkpoint-interval',steps,
            '--forward-configs',root/'forward.json']
        options += ['--mxfp8-backward-configs',root/'mxfp8_backward.json'] if gpu==0 else [
            '--ms-eden-designs',selected_model_designs or root/'fp4-designs.json']
        run(gpu,'benchmarks.train_precision20m',options,output.with_suffix('.log'),timeout=7200)
        rows=[json.loads(line) for line in output.read_text().splitlines()]
        terminal=rows[-1]
        if terminal.get('event')!='result' or terminal['interrupted'] or not terminal['finite_parameters']:
            raise RuntimeError('model run failed completion/numerical gates')
        if terminal['peak_reserved_bytes']>.95*terminal['device_total_bytes']:
            raise RuntimeError('tuned model exceeds declared memory headroom')
        return terminal

    pilots=[]
    for batch in batches:
        root=args.output/f'batch-{batch}'
        root.mkdir()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures=[pool.submit(tune_forward_backward8,batch,root),pool.submit(tune_backward4,batch,root)]
            for future in futures:future.result()
        select_fp4(batch,root)
        with ThreadPoolExecutor(max_workers=2) as pool:
            terminals=list(pool.map(lambda gpu:model_run(batch,root,'pilot',gpu),(0,1)))
        run(0,'benchmarks.analyze_precision20m',['--baseline-mode','nvfp4_mxfp8',
            root/'pilot-nvfp4_mxfp8.jsonl',root/'pilot-nvfp4_full.jsonl',
            '--output',root/'pilot-comparison.json'],root/'pilot-analysis.log')
        pilots.append(dict(batch=batch,results=terminals))
    # User requested largest sustainable common batch; preserve throughput of
    # the smaller tuned candidate as an explicit alternative, not a silent swap.
    (args.output/'pilots.json').write_text(json.dumps(pilots,indent=2))
    selected=max(record['batch'] for record in pilots)
    refinement=args.output/'model-refinement'
    run(1,'benchmarks.refine_ms_eden_model',['--study',args.output,'--output',refinement,
        '--batch',selected,'--trials',args.model_trials,'--seconds',args.model_seconds],
        args.output/'model-refinement.log',timeout=int(args.model_seconds)+900)
    selected_model_designs=refinement/'designs.json'
    (args.output/'preparation.json').write_text(json.dumps(dict(selected_batch=selected,
        search_and_qualification_wall_s=time.monotonic()-started,
        excluded_from_training_time=True),indent=2))
    root=args.output/f'batch-{selected}'
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda gpu:model_run(selected,root,'train',gpu),(0,1)))
    run(0,'benchmarks.analyze_precision20m',['--baseline-mode','nvfp4_mxfp8',
        root/'train-nvfp4_mxfp8.jsonl',root/'train-nvfp4_full.jsonl',
        '--output',root/'train-comparison.json'],root/'train-analysis.log')
    print('FINISHED fresh tuned matched training',flush=True)


if __name__=='__main__':main()
