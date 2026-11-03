"""Controlled full-model CUDA-graph scheduling diagnostics, never runtime policy."""
import argparse
from collections import Counter
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--study',type=Path,required=True)
    p.add_argument('--batch',type=int,default=72)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--variant',choices=('normal','serial','before','after','both','timed',
        'reverse_submission','dw_prefill','dx_prefill','pre_noop',
        'dw_caller','dw_caller_main_first','integrated'),default='normal')
    p.add_argument('--dump',action='store_true')
    p.add_argument('--profile-step',type=int,default=-1)
    args=p.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    import torch
    from benchmarks import train_precision20m as study
    import rtx.experimental_ms_eden as fp4
    commands=[]
    for path in args.study.glob('commands-gpu*.jsonl'):
        for line in path.read_text().splitlines():
            c=json.loads(line)['command']
            if ('benchmarks.train_precision20m' in c and
                c[c.index('--mode')+1]=='nvfp4_full' and
                f'/batch-{args.batch}/pilot-' in c[c.index('--output')+1]):commands.append(c)
    assert len(commands)==1
    c=commands[0]
    cli=c[c.index('benchmarks.train_precision20m')+1:]
    cli[cli.index('--output')+1]=str(args.output/'training.jsonl')
    if args.profile_step>=0:
        cli+=['--profile-step',str(args.profile_step)]
        def annotate(label,fn):
            def call(*a,**kw):
                torch.cuda.nvtx.range_push(label)
                try:return fn(*a,**kw)
                finally:torch.cuda.nvtx.range_pop()
            return call
        original_compile=torch.compile
        def compile_fn(fn,*a,**kw):
            return annotate(fn.__name__,original_compile(fn,*a,**kw))
        torch.compile=compile_fn
        torch.Tensor.backward=annotate('model_backward',torch.Tensor.backward)
        torch.optim.AdamW.step=annotate('adamw',torch.optim.AdamW.step)
    if args.variant in ('serial','integrated'):
        path=Path(cli[cli.index('--ms-eden-designs')+1])
        mapping=json.loads(path.read_text())
        for value in mapping.values():
            value['schedule']['backward_pipeline']='serial' if args.variant=='serial' else 'paired_dw_first'
        new=args.output/'selected-designs.json'
        new.write_text(json.dumps(mapping,indent=2))
        cli[cli.index('--ms-eden-designs')+1]=str(new)
    records=Counter()
    streams=Counter()
    original_stream=fp4._paired_stream
    def stream(device,caller):
        result=original_stream(device,caller)
        capture=torch.cuda.is_current_stream_capturing()
        records[f'lookup_capture={capture}_fallback={result is None}']+=1
        streams[str(caller.cuda_stream)]+=1
        return result
    fp4._paired_stream=stream
    retained=[]
    original_pair=fp4._launch_paired
    if args.variant in ('dw_caller','dw_caller_main_first'):
        original_build=fp4._build_gradient
        class Plan:
            def __init__(self,plan,live):self.plan,self.live=plan,live
            def __getattr__(self,name):return getattr(self.plan,name)
            def __call__(self):return self.plan()
        def build(*a,**kw):
            plan,out,live=original_build(*a,**kw)
            return Plan(plan,live),out,live
        fp4._build_gradient=build
    def paired(*a,**kw):
        if args.variant in ('dw_caller','dw_caller_main_first'):
            dx,dw,live,device=a
            caller=torch.cuda.current_stream(device)
            entry=fp4._paired_stream(device,caller)
            if entry is None:return original_pair(*a,**kw)
            side,lock=entry
            with lock:
                side.wait_stream(caller)
                try:
                    for value in dx.live:value.record_stream(side)
                    if args.variant=='dw_caller_main_first':dw()
                    with torch.cuda.stream(side):dx()
                    if args.variant=='dw_caller':dw()
                finally:caller.wait_stream(side)
            return
        if args.variant=='dw_prefill':
            dx,dw,live,device=a
            dw.prepare_a()
            return original_pair(dx,lambda: (dw.prepare_b(),dw.finish()),live,device)
        if args.variant=='dx_prefill':
            dx,dw,live,device=a
            dx.prepare_a()
            return original_pair(lambda: (dx.prepare_b(),dx.finish()),dw,live,device)
        if args.variant=='reverse_submission':
            dx,dw,live,device=a
            caller=torch.cuda.current_stream(device)
            entry=fp4._paired_stream(device,caller)
            if entry is None:return original_pair(*a,**kw)
            side,lock=entry
            with lock:
                side.wait_stream(caller)
                try:
                    for value in live:value.record_stream(side)
                    dx()
                    with torch.cuda.stream(side):dw()
                finally:caller.wait_stream(side)
            return
        if not torch.cuda.is_current_stream_capturing():return original_pair(*a,**kw)
        records['captured_pairs']+=1
        if args.variant=='pre_noop':torch.cuda._sleep(1)
        if args.variant in ('before','both','timed'):
            event=torch.cuda.Event(enable_timing=args.variant=='timed',external=True)
            event.record();retained.append(event)
        result=original_pair(*a,**kw)
        if args.variant in ('after','both','timed'):
            event=torch.cuda.Event(enable_timing=args.variant=='timed',external=True)
            event.record();retained.append(event)
        return result
    fp4._launch_paired=paired
    graph_records=[]
    if args.dump:
        from cuda.bindings import runtime
        original_graph=torch.cuda.CUDAGraph
        class DiagnosticGraph(original_graph):
            def __new__(cls,keep_graph=False):
                return super().__new__(cls,True)
            def __init__(self,keep_graph=False):
                super().__init__(True)
            def capture_end(self):
                result=super().capture_end()
                self.instantiate()
                destination=args.output/f'graph-{len(graph_records):03d}.dot'
                status,=runtime.cudaGraphDebugDotPrint(self.raw_cuda_graph(),str(destination).encode(),1)
                if status!=0:raise RuntimeError(f'cudaGraphDebugDotPrint failed: {status}')
                graph_records.append(str(destination))
                try:
                    data=self.get_graph_data()
                    destination.with_suffix('.json').write_text(json.dumps(data,indent=2))
                except RuntimeError as exc:
                    destination.with_suffix('.error.txt').write_text(str(exc))
                return result
        torch.cuda.CUDAGraph=DiagnosticGraph
    try:study.run(study.parser().parse_args(cli))
    finally:
        fp4._paired_stream=original_stream
        fp4._launch_paired=original_pair
    torch.cuda.synchronize()
    payload=dict(variant=args.variant,records=dict(records),caller_streams=dict(streams),
        paired_cache_size=len(fp4._PAIRED_STREAMS),external_events=len(retained),graphs=graph_records)
    (args.output/'diagnostic.json').write_text(json.dumps(payload,indent=2))
    print(json.dumps(payload),flush=True)


if __name__=='__main__':main()
