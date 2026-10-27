"""Isolated, persistent first-hit tuning of fixed-policy FP4 backward prep."""
import argparse
from dataclasses import asdict,replace
import json
from pathlib import Path
import random
import subprocess
import sys
import uuid

from rtx.autotune.posthoc import make_posthoc_preparation_adapter,preparation_grid_candidates,preparation_resource_seeds
from rtx.autotune.first_hit import tune_first_hit,FirstHitPolicy
from rtx.autotune.legacy import DeviceFingerprint
from rtx.autotune.outcomes import TrialOutcome
from rtx.autotune.store import JsonlTuningStore


def evaluation_command(args,config,directory):
    """Keep explicit-input and public-RNG trial contracts visibly separate."""
    if args.public_rng:
        return [sys.executable,"-m","benchmarks.compare_ms_eden_amortization",
                "--shape",args.shape,"--evaluate-phase",args.phase,
                "--schedule-json",json.dumps(asdict(config)),"--output",str(directory/"result.json")]
    return [sys.executable,"-m","benchmarks.study_ms_eden_backward",
            "--shape",args.shape,"--phase",args.phase,"--layout","mma128",
            "--source-policy",args.source_policy,
            "--schedule-json",str(directory/"schedule.json"),"--output",str(directory/"result.json"),
            *(["--packed-reader"] if args.packed_reader else [])]


def focused_candidates(adapter, initial, focus):
    """Round-robin nontrivial coordinates with the other phase frozen."""
    if focus not in ("preparation", "gemm"):
        raise ValueError("focus must be preparation or gemm")
    axes=[(name,values) for name,values in adapter.axes.items()
          if name.startswith(("gemm:", "gemm_")) == (focus=="gemm")]
    if focus=="preparation":
        # New knobs must not push the existing reduction strategy entirely
        # outside a small-budget first round of structural probes.
        priorities=("producer_load","producer_blocks","producer_threads","amax_reduction",
                    "producer_signs","rng_mode")
        axes.sort(key=lambda item:priorities.index(item[0]) if item[0] in priorities else len(priorities))
    portfolios=[]
    for name,values in axes:
        proposals=[adapter.update_fn(initial,name,value) for value in values]
        portfolios.append([c for c in proposals if adapter.config_id(c)!=adapter.config_id(initial)])
    return [initial]+[portfolio[index] for index in range(max(map(len,portfolios),default=0))
                      for portfolio in portfolios if index<len(portfolio)]


def public_rng_seeds(initial):
    """Symmetric small-budget coverage, regardless of the supplied incumbent."""
    return [initial]+[
        replace(initial,rng_mode=rng,producer_signs=signs)
        for signs in ("reload","register") for rng in ("explicit","philox")
    ]+[replace(initial,rng_mode="philox",philox_packet_size=1)]


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--shape",default="2048,768,1536")
    p.add_argument("--phase",choices=("dx","dw"),required=True)
    p.add_argument("--trials",type=int,default=16)
    p.add_argument("--output",required=True)
    p.add_argument("--refine",action="store_true")
    p.add_argument("--initial-selection",help="Validated original-BF16 selection file/directory to refine")
    p.add_argument("--initial-schedule-json",help="Explicit serialized starting schedule; still measured and validated")
    p.add_argument("--focus",choices=("all","preparation","gemm"),default="all")
    p.add_argument("--public-rng",action="store_true",
                   help="measure compiled caller RNG plus gradient; enables RNG implementation choices")
    p.add_argument("--source-policy",choices=("original_bf16","forward_current_roundtrip_bf16"),default="original_bf16")
    p.add_argument("--packed-reader",action="store_true")
    p.add_argument("--grid-only",action="store_true",help="Controlled producer/finalizer grid experiment; fixed GEMM and threads")
    p.add_argument("--producer-order",choices=("row_major","reduction_major"),help="Restrict traversal; grid-only defaults to row_major")
    p.add_argument("--amax-reduction",choices=("inline","staged"),help="Restrict maximum reduction; grid-only defaults to inline")
    p.add_argument("--producer-load",choices=("direct","coalesced","coalesced_wide","coalesced_x4","coalesced_x8"),help="Restrict load layout; grid-only defaults to direct")
    args=p.parse_args()
    if args.initial_selection and args.initial_schedule_json:
        p.error("choose an initial selection or an explicit schedule, not both")
    if args.public_rng and (args.source_policy!="original_bf16" or args.packed_reader):
        p.error("public RNG tuning currently requires original BF16 sources")
    root=Path(args.output).resolve(); root.mkdir(parents=True,exist_ok=True)
    def evaluate(config):
        directory=root/str(uuid.uuid4()); directory.mkdir()
        (directory/"schedule.json").write_text(json.dumps(asdict(config)))
        with (directory/"run.log").open("w") as log:
            try:
                proc=subprocess.run(evaluation_command(args,config,directory),
                    stdout=log,stderr=subprocess.STDOUT,timeout=180)
            except subprocess.TimeoutExpired:
                return TrialOutcome("runtime_error",error="isolated trial timeout",metadata={"artifact":str(directory)})
        if proc.returncode:
            return TrialOutcome("runtime_error",error=(directory/"run.log").read_text()[-4000:])
        result=json.loads((directory/"result.json").read_text())
        if args.public_rng:
            return TrialOutcome("ok",median_ms=result["timings"]["median_ms"],
                timings_ms=result["timings"]["samples_ms"],metadata={
                    "artifact":str(directory/"result.json"),"metrics":result["metrics"],
                    **result["quality"]})
        return TrialOutcome("ok",median_ms=result["timings"]["full"]["median_ms"],
            timings_ms=result["timings"]["full"]["samples_ms"],metadata={
                "artifact":str(directory/"result.json"),"metrics":result["metrics"],
                **result["quality"]["full"]})
    import torch
    import importlib.metadata
    device=DeviceFingerprint.current("cuda:0").as_dict()
    props=torch.cuda.get_device_properties(0)
    from benchmarks.compare_ms_eden_amortization import PUBLIC_PHASE_MEASUREMENT_CONTRACT
    device["l2_cache_size"]=int(getattr(props,"L2_cache_size",getattr(props,"l2_cache_size",0)) or 0)
    adapter=make_posthoc_preparation_adapter(tuple(map(int,args.shape.split(","))),args.phase,"mma128",evaluate,
        source_policy=args.source_policy,packed_reader=args.packed_reader,
        randomness="public" if args.public_rng else "explicit",
        measurement_contract=PUBLIC_PHASE_MEASUREMENT_CONTRACT if args.public_rng else None,
        device=device,environment={"torch":torch.__version__,
            "cuda":torch.version.cuda,"cutlass_dsl":importlib.metadata.version("nvidia-cutlass-dsl")})
    if args.initial_selection:
        if args.source_policy!="original_bf16" or args.packed_reader:
            p.error("initial selection currently requires original BF16 sources")
        from benchmarks.compare_nvfp4_backward_policies import load_schedule
        adapter.initial_config=load_schedule(args.initial_selection,tuple(map(int,args.shape.split(","))),args.phase)
    if args.initial_schedule_json:
        from rtx.autotune.posthoc import PosthocPreparationSchedule
        adapter.initial_config=PosthocPreparationSchedule(**json.loads(args.initial_schedule_json))
    initial=adapter.initial_config
    orders=(args.producer_order,) if args.producer_order else (("row_major",) if args.grid_only else adapter.axes["producer_order"])
    reductions=(args.amax_reduction,) if args.amax_reduction else (("inline",) if args.grid_only else adapter.axes["amax_reduction"])
    loads=(args.producer_load,) if args.producer_load else (("direct",) if args.grid_only else adapter.axes["producer_load"])
    portfolios=[]
    for mode in adapter.axes["reader_mode"]:
        for order in orders:
            for reduction in reductions:
                for load in loads:
                    portfolios.append(preparation_grid_candidates(tuple(map(int,args.shape.split(","))),args.phase,
                        torch.cuda.get_device_properties(0).multi_processor_count,
                        replace(initial,reader_mode=mode,producer_order=order,amax_reduction=reduction,producer_load=load)))
    candidates=[portfolio[index] for index in range(max(map(len,portfolios)))
                for portfolio in portfolios if index<len(portfolio)]
    grid_candidates=list(candidates)
    for mode in adapter.axes["reader_mode"]:
        for stages in (2,3):
            candidates.append(replace(initial,reader_mode=mode,
                gemm=replace(initial.gemm,atom_layout_m=2,atom_layout_n=2,stages=stages)))
    if args.packed_reader:
        candidates.extend(replace(initial,reader_mode="staged",decoder_threads=threads,decoder_blocks=blocks)
                          for threads,blocks in ((256,512),(256,1024),(512,512),(128,512)))
    axes=list(adapter.axes.items())
    for index in range(max(map(lambda v:len(v[1]),axes))):
        for name,values in axes:
            for mode in adapter.axes["reader_mode"]:
                if index<len(values):
                    candidates.append(adapter.update_fn(replace(initial,reader_mode=mode),name,values[index]))
    # Never materialize the Cartesian product: GEMM axes multiply this space
    # into millions of configurations. Random combinations retain reachability
    # with bounded host memory/work, independently of the trial budget.
    rng=random.Random(808)
    for _ in range(512):
        candidate=initial
        for name,values in axes:
            candidate=adapter.update_fn(candidate,name,rng.choice(values))
        candidates.append(candidate)
    if args.grid_only:
        candidates=grid_candidates
    else:
        seeds=[preparation_resource_seeds(tuple(map(int,args.shape.split(","))),args.phase,
                   props.multi_processor_count,replace(initial,reader_mode=mode))
               for mode in adapter.axes["reader_mode"]]
        candidates=[portfolio[index] for index in range(max(map(len,seeds)))
                    for portfolio in seeds if index<len(portfolio)]+candidates
    if args.producer_order:
        candidates=[replace(c,producer_order=args.producer_order) for c in candidates]
    if args.amax_reduction:
        candidates=[replace(c,amax_reduction=args.amax_reduction) for c in candidates]
    if args.producer_load:
        candidates=[replace(c,producer_load=args.producer_load) for c in candidates]
    if args.focus != "all":
        # Round-robin nontrivial axis probes around the measured winner. Avoid
        # spending the entire small budget on the earlier grid portfolios.
        candidates=focused_candidates(adapter,initial,args.focus)
    if args.public_rng and args.focus!="gemm" and not args.grid_only:
        # Spend the first few trials on the actual competing amortization
        # implementations, not only hundreds of grid variants of one policy.
        candidates=public_rng_seeds(initial)+candidates
    result=tune_first_hit(adapter,JsonlTuningStore(root/"residuals"),candidates,
        policy=FirstHitPolicy(max_trials=args.trials,seconds=900,pool_size=128,confirmation_repeats=1),refine=args.refine,
        candidate_group=lambda c:(f"{c.reader_mode}:{c.rng_mode}:{c.producer_signs}"
                                  if args.public_rng else c.reader_mode))
    with (root/f"selection-{uuid.uuid4()}.json").open("x") as handle:
        json.dump(dict(config=adapter.serialize(result.config),median_ms=result.median_ms,
                       context=adapter.context.as_dict(),arguments=vars(args)),handle,indent=2)
    print("WINNER",result.median_ms,adapter.serialize(result.config))


if __name__=="__main__":
    main()
