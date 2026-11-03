"""Hot versus cache-disrupted isolated backward, using the exact tuned schedules."""
import argparse
from contextlib import ExitStack
from dataclasses import asdict
import json
from pathlib import Path
import random
import statistics
import time


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--configs',type=Path,required=True)
    p.add_argument('--shape',default='36864,5632,1024')
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    import torch
    from benchmarks.compare_ms_eden_pipeline import options
    from benchmarks.compare_mixed_nvfp4_backward import run_worker
    from benchmarks.ms_eden_design_dispatch import dispatch
    from rtx.autotune.ms_eden_design import MSEdenDesign
    shape=tuple(map(int,args.shape.split(',')))
    design=MSEdenDesign(**json.loads((args.configs/'fp4-designs.json').read_text())[args.shape])
    mx=json.loads((args.configs/'mxfp8_backward.json').read_text())[args.shape]
    schedules={phase:json.dumps(asdict(getattr(design.schedule,phase))) for phase in ('dx','dw')}
    bundles={}
    with ExitStack() as stack:
        stack.enter_context(torch._inductor.config.patch(combo_kernels=True,combo_kernels_pointwise_only=True))
        stack.enter_context(dispatch(design))
        for name,policy in (('mxfp8','stable_mxfp8'),('nvfp4','fp4_fp4')):
            _,bundles[name]=run_worker(options(shape,schedules,policy=policy,
                pipeline=design.schedule.backward_pipeline,
                mxfp8_config_json=json.dumps(mx)),capture_only=True,
                shared_inputs=None if not bundles else bundles['mxfp8']['inputs'])
        graphs={name:b['pins'][1] for name,b in bundles.items()}
        flush=torch.empty(512*1024*1024,device='cuda',dtype=torch.uint8)
        until=time.monotonic()+5
        while time.monotonic()<until:
            for g in graphs.values():g.replay()
            torch.cuda.synchronize()
        samples=[]
        rng=random.Random(941)
        for i in range(40):
            order=[(mode,name) for mode in ('hot','disrupted') for name in graphs]
            rng.shuffle(order)
            for mode,name in order:
                if mode=='hot':graphs[name].replay()
                else:flush.fill_(i)
                start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                start.record();graphs[name].replay();end.record();end.synchronize()
                samples.append(dict(round=i,mode=mode,policy=name,ms=start.elapsed_time(end)))
        result=dict(shape=shape,device=str(torch.cuda.get_device_properties(0)),samples=samples,
            summary={f'{mode}_{name}':statistics.median(s['ms'] for s in samples
                if s['mode']==mode and s['policy']==name) for mode in ('hot','disrupted') for name in graphs},
            scope='single replay with preceding identical replay or 512MiB device fill excluded from timing; diagnostic not training')
        with args.output.open('x') as f:json.dump(result,f,indent=2)
        print(result['summary'],flush=True)


if __name__=='__main__':main()
