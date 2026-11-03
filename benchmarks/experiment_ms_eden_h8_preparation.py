"""Research-only fused H8 complete-backward comparison; defaults unchanged.

The transform changes FP32 association. Preparation qualification must precede
this measurement; this harness also retains the fullgraph FP32 gradient gates
of compare_ms_eden_pipeline. It is not exact equivalence to scalar MS-EDEN.
"""
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
import tempfile

from benchmarks import compare_ms_eden_pipeline as comparison


@contextmanager
def patched_preparation(scope):
    if scope not in ("row", "vpl4"):
        raise ValueError("scope must be row or vpl4")
    from rtx.kernels import nvfp4_posthoc as kernel
    from rtx.kernels.ms_eden_h8_preparation import compile_h8_preparation
    original=kernel.compile_hadamard_preparation
    signature=inspect.signature(original)
    dispatch=[]
    def compile_candidate(*args,**kwargs):
        bound=signature.bind(*args,**kwargs);bound.apply_defaults()
        values=dict(bound.arguments)
        transpose=values["transpose"]
        vpl=values["producer_transpose_values_per_lane" if transpose else "producer_values_per_lane"]
        selected=vpl==4 and (scope=="vpl4" or not transpose)
        dispatch.append(dict(rows=values["rows"],cols=values["cols"],transpose=transpose,
                             values_per_lane=vpl,h8=selected))
        return compile_h8_preparation(**values) if selected else original(*args,**kwargs)
    kernel.compile_hadamard_preparation=compile_candidate
    try:
        yield dispatch
    finally:
        kernel.compile_hadamard_preparation=original


def main(argv=None):
    p=comparison.parser()
    p.description=__doc__
    p.set_defaults(candidate_pipeline="operand_paired",baseline_policy="stable_mxfp8")
    p.add_argument("--transform-scope",choices=("row","vpl4"),default="row")
    args=p.parse_args(argv)
    if args.baseline_policy!="stable_mxfp8":p.error("requires the untouched stable MXFP8 baseline")
    if args.validate_exact:p.error("H8 reassociates FP32; use dedicated preparation qualification")
    if args.output.exists():raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    common,dx,dw=(json.loads(value) for value in (args.candidate_json,args.candidate_dx_json,args.candidate_dw_json))
    with patched_preparation(args.transform_scope) as dispatch:
        with tempfile.TemporaryDirectory(prefix="h8-",dir=args.output.parent) as staging:
            result=comparison.compare(comparison.shape_value(args.shape),Path(staging)/"comparison.json",
                common,dx_overrides=dx,dw_overrides=dw,pipeline=args.candidate_pipeline,
                rounds=args.rounds,replays=args.replays,min_batch_ms=args.min_batch_ms,
                warmup_ms=args.warmup_ms,seed=args.seed,baseline_policy=args.baseline_policy,
                mxfp8_config_json=args.mxfp8_config_json,
                scopes=("backward","total") if args.scope=="both" else (args.scope,))
    if not any(item["h8"] for item in dispatch):raise RuntimeError("no H8 producer compiled")
    result["h8_experiment"]=dict(transform_scope=args.transform_scope,dispatch=dispatch,
        production_default_unchanged=True,precision="FP32 correction/statistics; guarded exact BF16 input conversion; H128 reassociated")
    root=Path(__file__).resolve().parents[1]
    for name in ("rtx/kernels/ms_eden_h8_preparation.py","benchmarks/experiment_ms_eden_h8_preparation.py"):
        result["source_sha256"][name]=hashlib.sha256((root/name).read_bytes()).hexdigest()
    with args.output.open("x") as handle:json.dump(result,handle,indent=2)
    print(json.dumps(result["timings"],indent=2))
    return result


if __name__=="__main__":main()
