"""Attribute actual GPU kernels using launch correlations, not nested profiler totals.

RTX ranges cover runtime MLP calls, not the whole image tokenizer. Kernel-time
shares are not wall-time speedups; unlabelled work is deliberately left unknown.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path


def trace_events(stream,chunk_size=1024*1024):
    """Stream the top-level traceEvents array without retaining a multi-GB JSON."""
    decoder=json.JSONDecoder()
    buffer=''
    pos=0
    eof=False
    def more():
        nonlocal buffer,pos,eof
        buffer=buffer[pos:]+stream.read(chunk_size)
        pos=0
        if not buffer:
            eof=True
            return False
        return True
    def whitespace():
        nonlocal pos
        while True:
            while pos<len(buffer) and buffer[pos].isspace():
                pos+=1
            if pos<len(buffer):
                return
            if not more():
                return
    def token(expected):
        nonlocal pos
        whitespace()
        if pos>=len(buffer) or buffer[pos]!=expected:
            raise ValueError(f'expected JSON token {expected!r}')
        pos+=1
    def value():
        nonlocal pos,buffer,eof
        whitespace()
        while True:
            try:
                result,end=decoder.raw_decode(buffer,pos)
                # A scalar number may end at a read boundary, e.g. 1 | 23.
                if end==len(buffer) and not eof:
                    extra=stream.read(chunk_size)
                    if extra:
                        buffer=buffer[pos:]+extra;pos=0
                        continue
                    eof=True
                pos=end
                return result
            except json.JSONDecodeError:
                extra=stream.read(chunk_size)
                if not extra:
                    raise
                buffer=buffer[pos:]+extra;pos=0
    token('{')
    found=False
    whitespace()
    if pos<len(buffer) and buffer[pos]=='}':
        raise ValueError('missing traceEvents')
    while True:
        key=value()
        if not isinstance(key,str):
            raise ValueError('JSON object key must be a string')
        token(':')
        if key=='traceEvents':
            if found:
                raise ValueError('duplicate traceEvents')
            found=True
            token('[')
            whitespace()
            if pos<len(buffer) and buffer[pos]!=']':
                while True:
                    event=value()
                    if not isinstance(event,dict):
                        raise ValueError('trace event must be an object')
                    yield event
                    whitespace()
                    if pos<len(buffer) and buffer[pos]==']':
                        break
                    token(',')
            token(']')
        else:
            value()
        whitespace()
        if pos<len(buffer) and buffer[pos]=='}':
            token('}')
            break
        token(',')
    whitespace()
    if pos<len(buffer):
        raise ValueError('trailing JSON data')
    if not found:
        raise ValueError('missing traceEvents')


def union_us(intervals):
    total=0.0
    end=None
    for start,stop in sorted(intervals):
        if stop<start:
            raise ValueError('negative event duration')
        if end is None or start>end:
            total+=stop-start
        else:
            total+=max(0.0,stop-end)
        end=stop if end is None else max(end,stop)
    return total


def audit_ownership_counts(report,records):
    expected=defaultdict(int)
    for row in records:
        shape=row['x']['shape']
        m=1
        for dimension in shape[:-1]:
            m*=dimension
        role='tokenizer' if tuple(row['gu']['shape'])==(1536,256) else 'backbone'
        if row['phase'] not in ('forward','backward'):
            raise ValueError('unknown ownership phase')
        expected[f"rtx_mlp/{role}/{row['phase']}/m{m}"]+=1
    observed={name:row['calls'] for name,row in report['runtime_ranges'].items()}
    return dict(passed=dict(expected)==observed,expected=dict(expected),observed=observed,
                scope='exact runtime call counts against an independently recorded owner update')


def analyze(events):
    timelines=defaultdict(list)
    launches=defaultdict(list)
    kernels=[]
    ranges=defaultdict(lambda:dict(calls=0,host_range_us=0.0,kernels=0,kernel_us=0.0))
    for e in events:
        if e.get('ph')!='X':
            continue
        if e.get('dur',0)<0:
            raise ValueError('negative event duration')
        cat=e.get('cat')
        if cat not in ('user_annotation','cuda_runtime','cuda_driver','kernel'):
            continue
        # Drop verbose launch/kernel metadata before retaining timeline events.
        e={k:e[k] for k in ('cat','name','pid','tid','ts','dur','args') if k in e}
        e['args']={k:v for k,v in e.get('args',{}).items() if k in ('correlation','device')}
        thread=(e.get('pid'),e.get('tid'))
        if cat=='user_annotation' and e.get('name','').startswith('rtx_mlp/'):
            # End before start at a shared boundary; API events sort last.
            if e['dur']>0:
                timelines[thread].append((e['ts'],1,e))
                timelines[thread].append((e['ts']+e['dur'],0,e))
            ranges[e['name']]['calls']+=1
            ranges[e['name']]['host_range_us']+=e['dur']
        elif cat in ('cuda_runtime','cuda_driver'):
            if 'correlation' in e.get('args',{}):
                timelines[thread].append((e['ts'],2,e))
        elif cat=='kernel':
            kernels.append(e)
    for timeline in timelines.values():
        active={}
        for _,kind,e in sorted(timeline,key=lambda item:(item[0],item[1])):
            if kind==0:
                active.pop(id(e),None)
            elif kind==1:
                active[id(e)]=e
            else:
                # Pick the narrowest containing range; never add nested scopes.
                containing=[r for r in active.values() if e['ts']+e['dur']<=r['ts']+r['dur']]
                label=min(containing,key=lambda r:r['dur'])['name'] if containing else None
                launches[e['args']['correlation']].append(label)
    categories=defaultdict(lambda:dict(kernels=0,kernel_us=0.0))
    top=defaultdict(lambda:dict(kernels=0,kernel_us=0.0))
    intervals=defaultdict(list)
    for kernel in kernels:
        candidates=launches.get(kernel.get('args',{}).get('correlation'),[])
        if len(candidates)==1:
            label=candidates[0] or 'unlabelled'
        else:
            label='ambiguous_correlation' if candidates else 'missing_correlation'
        duration=kernel['dur']
        categories[label]['kernels']+=1
        categories[label]['kernel_us']+=duration
        if label in ranges:
            ranges[label]['kernels']+=1
            ranges[label]['kernel_us']+=duration
        top[kernel['name']]['kernels']+=1
        top[kernel['name']]['kernel_us']+=duration
        device=kernel.get('args',{}).get('device',kernel.get('pid'))
        intervals[str(device)].append((kernel['ts'],kernel['ts']+duration))
    summed=sum(k['dur'] for k in kernels)
    for row in categories.values():
        row['summed_kernel_time_fraction']=row['kernel_us']/summed if summed else None
    return dict(schema_version=1,diagnostic_only=True,kernel_count=len(kernels),
        summed_kernel_us=summed,busy_kernel_union_us_by_device={d:union_us(v) for d,v in intervals.items()},
        categories=dict(categories),runtime_ranges=dict(ranges),
        top_kernels=[dict(name=k,**v) for k,v in sorted(top.items(),key=lambda kv:kv[1]['kernel_us'],reverse=True)[:40]],
        limitations=['RTX labels cover MLP only, not whole tokenizer',
            'fractions use summed kernel duration, not update wall time',
            'overlapping streams count multiply in sums, once per device in union',
            'GPU annotation spans and CPU aggregate device times are not added',
            'missing or ambiguous correlations remain unattributed'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('trace',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--ownership',type=Path,
        help='optional owner saved-storage-ownership.json for exact call-count coverage audit')
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    opener=gzip.open if args.trace.suffix=='.gz' else open
    with opener(args.trace,'rt') as stream:
        report=analyze(trace_events(stream))
    with args.trace.open('rb') as stream:
        digest=hashlib.file_digest(stream,'sha256').hexdigest()
    report['source']=dict(path=str(args.trace.resolve()),sha256=digest)
    if args.ownership:
        raw=args.ownership.read_bytes()
        report['ownership_count_audit']=audit_ownership_counts(report,json.loads(raw))
        report['ownership_count_audit']['source']=dict(path=str(args.ownership.resolve()),
            sha256=hashlib.sha256(raw).hexdigest())
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(report,stream,indent=2)
    print(json.dumps({k:report[k] for k in ('kernel_count','summed_kernel_us','categories')},indent=2))
    if args.ownership and not report['ownership_count_audit']['passed']:
        raise SystemExit('runtime call counts differ; inspect retained audit before interpreting attribution')


if __name__=='__main__':
    main()
