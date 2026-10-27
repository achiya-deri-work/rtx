"""Research-only four-operand enqueue-order experiment; run each order freshly.

Stream ownership, dependencies, lifetime recording and kernels are unchanged.
Only the host order of the four independent preparation calls is permuted.
No production API enum or environment variable is changed.
"""
import argparse
from contextlib import contextmanager
import gc
import hashlib
import inspect
import json
from pathlib import Path
import tempfile

from benchmarks import compare_ms_eden_pipeline as comparison


OPERANDS=("dx_a","dx_b","dw_a","dw_b")
# Eight balanced priorities; the last entry is the production enqueue order.
FIRST_SCREEN=(
    ("dx_a","dw_a","dx_b","dw_b"),
    ("dx_a","dx_b","dw_a","dw_b"),
    ("dx_a","dw_b","dw_a","dx_b"),
    ("dw_a","dx_a","dw_b","dx_b"),
    ("dw_a","dw_b","dx_a","dx_b"),
    ("dw_a","dx_b","dx_a","dw_b"),
    ("dx_b","dx_a","dw_a","dw_b"),
    ("dx_b","dw_a","dw_b","dx_a"),
)


def order_value(value):
    parts=tuple(value.split(",")) if isinstance(value,str) else tuple(value)
    if len(parts)!=4 or set(parts)!=set(OPERANDS):
        raise argparse.ArgumentTypeError("order requires dx_a,dx_b,dw_a,dw_b exactly once each")
    return parts


def ordered_launcher(order, finish_admission='independent'):
    """Keep the production topology, replacing only its four enqueue lines."""
    order=order_value(order)
    if finish_admission not in ('independent','dx_a','dx_ready'):
        raise ValueError('unknown finish admission')
    import torch
    import rtx.experimental_ms_eden as eden
    def launch(dx,dw,live_dx,live_dw,device):
        caller=torch.cuda.current_stream(device)
        entry=eden._operand_streams(device,caller)
        if entry is None:
            dx();dw()
            return
        sides,lock=entry
        dx_b,dw_a,dw_b=sides
        preparations={"dx_a":(caller,dx.prepare_a),"dx_b":(dx_b,dx.prepare_b),
                      "dw_a":(dw_a,dw.prepare_a),"dw_b":(dw_b,dw.prepare_b)}
        with lock:
            try:
                # All startup waits precede ANY preparation, including DX-A.
                for side in sides:side.wait_stream(caller)
                for value in live_dx:value.record_stream(dx_b)
                for side in (dw_a,dw_b):
                    for value in live_dw:value.record_stream(side)
                for name in order:
                    stream,prepare=preparations[name]
                    if name=="dx_a":prepare()
                    else:
                        with torch.cuda.stream(stream):prepare()
                if finish_admission == 'dx_a':
                    dw_a.wait_stream(caller)
                caller.wait_stream(dx_b)
                if finish_admission == 'dx_ready':
                    dw_a.wait_stream(caller)
                dx.finish()
                with torch.cuda.stream(dw_a):
                    dw_a.wait_stream(dw_b)
                    dw.finish()
            except BaseException:
                for side in sides:
                    try:caller.wait_stream(side)
                    except BaseException:pass
                raise
            try:
                caller.wait_stream(dw_a)
            except BaseException:
                for side in (dx_b,dw_b):
                    try:caller.wait_stream(side)
                    except BaseException:pass
                raise
    return launch


@contextmanager
def patched_order(order, finish_admission='independent'):
    """Fresh-process research patch; always restore even when a gate fails."""
    import rtx.experimental_ms_eden as eden
    previous=eden._launch_operand_paired
    eden._launch_operand_paired=ordered_launcher(order,finish_admission)
    try:yield
    finally:eden._launch_operand_paired=previous


def exact_smoke(shape,schedules,seed):
    """Full-shape identical-draw serial versus patched operand pipeline gate."""
    import torch
    from rtx.experimental_ms_eden import ms_eden_paired_gradient
    torch.manual_seed(seed)
    m,n,k=shape
    dy=torch.randn(m,n,device="cuda",dtype=torch.bfloat16)
    x=torch.randn(m,k,device="cuda",dtype=torch.bfloat16)
    w=torch.randn(n,k,device="cuda",dtype=torch.bfloat16)
    signs=[torch.ones(128,device="cuda") for _ in range(2)]
    uniforms=[torch.rand(count//16,device="cuda") for count in (dy.numel(),w.numel(),dy.numel(),x.numel())]
    seeded = {phase: json.loads(schedules[phase]).get('rng_mode', 'explicit') == 'philox'
              for phase in ('dx', 'dw')}
    for phase, offset in (('dx', 0), ('dw', 2)):
        if seeded[phase]:
            key = torch.randint(0, 2**32, (2,), device='cuda', dtype=torch.int64)
            uniforms[offset] = uniforms[offset + 1] = key
    seed_flags = dict(dx_seeded=seeded['dx'], dw_seeded=seeded['dw'])
    args=(dy,x,w,signs[0],uniforms[0],uniforms[1],signs[1],uniforms[2],uniforms[3])
    def run(pipeline):
        return ms_eden_paired_gradient(*args,dx_schedule_json=schedules["dx"],
            dw_schedule_json=schedules["dw"],pipeline=pipeline,**seed_flags)
    compiled=torch.compile(lambda *values:ms_eden_paired_gradient(*values,
        dx_schedule_json=schedules["dx"],dw_schedule_json=schedules["dw"],pipeline="operand_paired",**seed_flags),
        fullgraph=True,options={"triton.cudagraphs":False})
    # torch.cuda.graph without an explicit stream selects a different stream
    # from ordinary eager warmup. Warm the actual capture-stream cache entry.
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):compiled(*args)
    stream.synchronize()
    graph=torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph,stream=stream):actual=compiled(*args)
    checks=[]
    for mode in ("gaussian","outlier","zero"):
        for value in (dy,x,w):
            value.normal_(0,.25)
            if mode=="outlier":value[0,0]=4096
            if mode=="zero":value.zero_()
        for value in signs:value.random_(0,2).mul_(2).sub_(1)
        for value in uniforms:
            if value.dtype == torch.int64:
                value.random_(0, 2**32)
            else:
                value.uniform_()
        expected=run("serial")
        graph.replay();torch.cuda.synchronize()
        for phase,reference,candidate in zip(("dx","dw"),expected,actual):
            if not torch.isfinite(candidate).all() or not torch.equal(reference.view(torch.uint8),candidate.view(torch.uint8)):
                raise RuntimeError(f"same-draw operand-order exact gate failed: {mode}/{phase}")
        checks.append(dict(distribution=mode,dx_bitwise=True,dw_bitwise=True))
    return dict(shape=list(shape),checks=checks,fullgraph=True,changed_input_graph_replay=True,
        seeded=seeded,
        oracle="same schedule, signs and explicit uniforms or Philox seeds; serial complete dX+dW")


def parser():
    p=comparison.parser()
    p.description=__doc__
    p.set_defaults(candidate_pipeline="operand_paired",baseline_policy="stable_mxfp8")
    p.add_argument("--order",type=order_value,required=True,help="four comma-separated preparation names, e.g. dx_a,dw_a,dx_b,dw_b")
    return p


def main(argv=None):
    p=parser();args=p.parse_args(argv)
    if args.candidate_pipeline!="operand_paired":p.error("operand-order experiment requires --candidate-pipeline operand_paired")
    if args.validate_exact:p.error("same-draw full-shape exact smoke is mandatory before timing; --validate-exact is not a timing mode")
    if args.output.exists():raise FileExistsError(args.output)
    shape=comparison.shape_value(args.shape)
    common,dx,dw=(json.loads(value) for value in (args.candidate_json,args.candidate_dx_json,args.candidate_dw_json))
    schedules=comparison.schedule_pair(common,dx,dw)
    with patched_order(args.order):
        smoke=exact_smoke(shape,schedules,args.seed)
        # Smoke graph/outputs must not remain live in the timing comparison.
        gc.collect()
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="operand-order-",dir=args.output.parent) as staging:
            result=comparison.compare(shape,Path(staging)/"comparison.json",common,dx_overrides=dx,dw_overrides=dw,
                pipeline="operand_paired",rounds=args.rounds,replays=args.replays,min_batch_ms=args.min_batch_ms,
                warmup_ms=args.warmup_ms,seed=args.seed,baseline_policy=args.baseline_policy,
                mxfp8_config_json=args.mxfp8_config_json,
                scopes=("backward","total") if args.scope=="both" else (args.scope,))
    source=Path(__file__).resolve()
    result["operand_order_experiment"]=dict(preparation_order=list(args.order),same_draw_exact=smoke,
        source_path=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        launcher_source_sha256=hashlib.sha256(inspect.getsource(ordered_launcher).encode()).hexdigest(),
        production_default_unchanged=True,stream_topology="caller=dx_a, side1=dx_b, side2=dw_a, side3=dw_b",
        first_screen=[list(order) for order in FIRST_SCREEN])
    result["source_sha256"]["benchmarks/experiment_ms_eden_operand_order.py"]=result["operand_order_experiment"]["source_sha256"]
    with args.output.open("x") as handle:json.dump(result,handle,indent=2)
    print(json.dumps(result["timings"],indent=2))
    return result


if __name__=="__main__":main()
