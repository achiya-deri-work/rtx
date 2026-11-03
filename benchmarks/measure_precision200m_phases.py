"""Diagnostic CUDA-event phase attribution around the unchanged training harness."""
import argparse
import json
from pathlib import Path
import statistics


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',required=True)
    p.add_argument('--subparts',action='store_true')
    p.add_argument('--span-only',action='store_true')
    args=p.parse_args()
    import torch
    from benchmarks import train_precision20m as study
    commands=[]
    for path in args.study.glob('commands-gpu*.jsonl'):
        for line in path.read_text().splitlines():
            c=json.loads(line)['command']
            if ('benchmarks.train_precision20m' in c and
                c[c.index('--mode')+1]==args.mode and
                '/batch-72/pilot-' in c[c.index('--output')+1]):commands.append(c)
    assert len(commands)==1
    c=commands[0]
    cli=c[c.index('benchmarks.train_precision20m')+1:]
    cli[cli.index('--output')+1]=str(args.output)
    parsed=study.parser().parse_args(cli)
    events={name:[] for name in ('forward_train','backward','clip','optimizer')}
    captured=[]
    if args.subparts:
        # External graph event nodes are diagnostic instrumentation. Keep their
        # handles alive, then read last-replay durations after synchronization.
        def capture_span(label,fn):
            def call(*a,**kw):
                if not torch.cuda.is_current_stream_capturing():return fn(*a,**kw)
                start=torch.cuda.Event(enable_timing=True,external=True)
                end=torch.cuda.Event(enable_timing=True,external=True)
                start.record()
                result=fn(*a,**kw)
                end.record()
                captured.append((label,start,end))
                return result
            return call
        import rtx.fp8_bwd as mx
        import rtx.experimental_ms_eden as fp4
        mx._BwdRunner.__call__=capture_span('linear_backward_span',mx._BwdRunner.__call__)
        fp4._launch_paired=capture_span('linear_backward_span',fp4._launch_paired)
        original_build=fp4._build_gradient
        def build(*a,**kw):
            plan,out,live=original_build(*a,**kw)
            phase=a[5]
            for name in ('prepare_a','prepare_b'):
                preparation=getattr(plan,name)
                preparation.produce=capture_span(phase+'_'+name+'_produce',preparation.produce)
                preparation.finalize=capture_span(phase+'_'+name+'_finalize',preparation.finalize)
            plan.finish=capture_span(phase+'_gemm_finish',plan.finish)
            return plan,out,live
        if not args.span_only:
            fp4._build_gradient=build
    def measured(name,fn):
        def call(*a,**kw):
            if name=='forward_train' and not torch.is_grad_enabled():return fn(*a,**kw)
            start,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
            start.record()
            result=fn(*a,**kw)
            end.record()
            events[name].append((start,end))
            return result
        return call
    original_compile=torch.compile
    def compile_fn(fn,*a,**kw):
        result=original_compile(fn,*a,**kw)
        name={'objective':'forward_train','gradient_norm_and_clip':'clip'}.get(fn.__name__)
        return measured(name,result) if name else result
    original_backward=torch.Tensor.backward
    original_step=torch.optim.AdamW.step
    torch.compile=compile_fn
    torch.Tensor.backward=measured('backward',original_backward)
    torch.optim.AdamW.step=measured('optimizer',original_step)
    try:study.run(parsed)
    finally:
        torch.compile=original_compile
        torch.Tensor.backward=original_backward
        torch.optim.AdamW.step=original_step
    torch.cuda.synchronize()
    # Discard compilation, capture, and first optimizer-state allocations.
    samples={name:[a.elapsed_time(b) for a,b in pairs][-48:] for name,pairs in events.items()}
    result=dict(scope='diagnostic per-phase CUDA events; not replacement for unobstructed timing',
        samples_ms=samples,summary={name:dict(median_ms=statistics.median(v),mean_ms=statistics.mean(v),
            minimum_ms=min(v),samples=len(v)) for name,v in samples.items()})
    if args.subparts:
        parts={}
        for label,a,b in captured:parts.setdefault(label,[]).append(a.elapsed_time(b))
        result['captured_last_replay_ms']=parts
        result['captured_totals']={label:dict(count=len(v),sum_ms=sum(v)) for label,v in parts.items()}
    args.output.with_suffix('.phases.json').write_text(json.dumps(result,indent=2))


if __name__=='__main__':main()
