"""Isolated matched-packed GEMM tuning via the shared first-hit autotuner.

Run with the optional Quartet/FlashInfer environment used by
compare_nvfp4_packed_gemm. Quantization and scale conversion are not timed GEMM.
"""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import random
import subprocess
import sys
import uuid

from rtx.autotune.adapters import make_nvfp4_fully_prequant_adapter
from rtx.autotune.first_hit import FirstHitPolicy, tune_first_hit
from rtx.autotune.legacy import DeviceFingerprint
from rtx.autotune.outcomes import TrialOutcome
from rtx.autotune.preparation import expand_preparation_candidates
from rtx.autotune.store import JsonlTuningStore
from rtx.configs.nvfp4 import NVFP4Problem, NVFP4FullyPrequantConfig, NVFP4GemmConfig
from rtx.nvfp4_inference_autotune import NVFP4_FULLY_PREQUANT_SEARCH_SPACE


def geometry_seeds(adapter):
    """Coupled legal seeds: K=64 requires matching SMEM swizzles."""
    seeds,seen=[adapter.initial_config],{adapter.config_id(adapter.initial_config)}
    for value in adapter.axes["gemm_geometry"]:
        base=adapter.update_fn(adapter.initial_config,"gemm_geometry",value)
        for swizzle in ("none","32b","64b","128b"):
            for stages in (1,2,3):
                config=replace(base,gemm=replace(base.gemm,a_swizzle=swizzle,b_swizzle=swizzle,stages=stages))
                cid=adapter.config_id(config)
                if cid not in seen and adapter.rejection(config) is None:
                    seeds.append(config)
                    seen.add(cid)
    return seeds


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shape",default="2048,768,1536")
    parser.add_argument("--trials",type=int,default=24)
    parser.add_argument("--output",required=True)
    parser.add_argument("--seed",type=int,default=20261023)
    parser.add_argument("--refine",action="store_true")
    parser.add_argument("--native-scales",action="store_true")
    args=parser.parse_args()
    root=Path(args.output).resolve()
    root.mkdir(parents=True,exist_ok=True)
    with (root/f"invocation-{uuid.uuid4()}.json").open("x") as handle:
        json.dump(vars(args),handle,indent=2)
    problem=NVFP4Problem(*map(int,args.shape.split(",")))
    def evaluate(config):
        trial=root/str(uuid.uuid4())
        trial.mkdir()
        config_path,result_path=trial/"config.json",trial/"result.json"
        config_path.write_text(json.dumps(adapter.serialize(config)))
        with (trial/"run.log").open("w") as log:
            try:
                proc=subprocess.run([sys.executable,"-m","benchmarks.compare_nvfp4_packed_gemm",
                    "--shape",args.shape,"--config-json",str(config_path),"--output",str(result_path)],
                    stdout=log,stderr=subprocess.STDOUT,timeout=300)
            except subprocess.TimeoutExpired:
                return TrialOutcome("runtime_error",error="packed diagnostic exceeded 300s",
                    metadata={"trial_directory":str(trial)})
        if proc.returncode:
            return TrialOutcome("runtime_error",error=(trial/"run.log").read_text()[-4000:])
        data=json.loads(result_path.read_text())
        timing=data["timings"]["rtx_gemm"]
        return TrialOutcome("ok",median_ms=timing["median_ms"],timings_ms=timing["samples_ms"],
            max_abs_error=data["metrics"]["rtx"]["max_abs"],metadata={
                "artifact":str(result_path),**data["timing_quality"]["rtx_gemm"]})
    axes={key:value for key,value in NVFP4_FULLY_PREQUANT_SEARCH_SPACE.items() if key!="global_l2_fetch"}
    initial_gemm=NVFP4GemmConfig()
    if args.native_scales:
        initial_gemm=replace(initial_gemm,scale_layout="mma128",scale_role="tma")
    adapter=make_nvfp4_fully_prequant_adapter(problem,evaluate,
        initial=NVFP4FullyPrequantConfig(gemm=initial_gemm),axes=axes,
        device=DeviceFingerprint.current("cuda:0"),tags={"oracle":"decoded_packed_fp32",
            "quantization":"Quartet_FourOverSix_fixed_seed", "scale_layout":initial_gemm.scale_layout,
            "measurement":"graph8_replay10_samples15", "timing_quality_revision":1})
    # Diagnostic layout conversion is not the TorchAO fully-packed frontend
    # contract. Never make these records directly promotable to that family.
    adapter.context=replace(adapter.context,family="nvfp4_packed_diagnostic")
    candidates=expand_preparation_candidates(adapter,[adapter.initial_config],per_seed=128,
        max_examined=4096,accept=lambda c:c.gemm.scale_layout==initial_gemm.scale_layout)
    refinements=candidates[1:]
    random.Random(args.seed).shuffle(refinements)
    result=tune_first_hit(adapter,JsonlTuningStore(root/"residuals"),
        [*geometry_seeds(adapter),*refinements],refine=args.refine,
        candidate_group=lambda c:(c.gemm.tile_m,c.gemm.tile_n,c.gemm.tile_k,c.gemm.atom_layout_m,c.gemm.atom_layout_n),
        policy=FirstHitPolicy(max_trials=args.trials,seconds=900,confirmation_repeats=1,pool_size=256))
    document={"config":adapter.serialize(result.config),"median_ms":result.median_ms,
        "context":adapter.context.as_dict(),"trials":result.evaluated_trials}
    with (root/f"selection-{uuid.uuid4()}.json").open("x") as handle:
        json.dump(document,handle,indent=2)
    print(json.dumps(document),flush=True)


if __name__=="__main__":
    main()
