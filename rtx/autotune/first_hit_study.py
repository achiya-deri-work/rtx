"""Matched MLP anchor collection and grouped offline first-hit evaluation.

CLI: python -m rtx.autotune.first_hit_study {run,worker,fit,export}
Workers reuse production numerical checks and the existing residual journal.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
import zipfile

import numpy as np

from .core import stable_id
from .first_hit import contract, model_features


def source_id():
    root=Path(__file__).resolve().parents[1]
    return stable_id({str(p.relative_to(root)):stable_id(p.read_text()) for p in sorted(root.rglob("*.py"))})


def anchor_documents(paths):
    for path in map(Path,paths):
        if path.is_dir():
            for file in path.rglob("anchor.json"):
                yield json.loads(file.read_text())
        elif path.suffix==".zip":
            with zipfile.ZipFile(path) as archive:
                for name in archive.namelist():
                    if name.endswith("/anchor.json"):
                        yield json.loads(archive.read(name))


def audit_anchors(paths):
    grouped=defaultdict(set)
    sources=set()
    for document in anchor_documents(paths):
        grouped[(document["manifest_id"],document["task"]["id"])].add(document["portfolio_id"])
        sources.add(document.get("source_id","unknown"))
    mismatches=[str(k) for k,v in grouped.items() if len(v)>1]
    return {"tasks":len(grouped),"source_ids":sorted(sources),"portfolio_mismatches":mismatches}


def atomic_json(path, document):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f".{os.getpid()}.tmp")
    with temporary.open("w") as stream:
        json.dump(document,stream,indent=2); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary,path)


def study_tasks(manifest):
    if manifest.get("schema_version")!=1:
        raise ValueError("unsupported anchor manifest schema")
    milestones=manifest["milestones"]
    if not milestones or milestones!=sorted(set(milestones)) or min(milestones)<1:
        raise ValueError("milestones must be strictly increasing positive integers")
    if any(len(s)!=4 or any(not isinstance(d,int) or d<1 for d in s) for s in manifest["shapes"]):
        raise ValueError("shapes must contain four positive integer dimensions")
    if not manifest["shapes"] or not manifest["phases"] or not manifest["precisions"]:
        raise ValueError("anchor study must not be empty")
    if set(manifest["phases"])-{"forward","train"} or set(manifest["precisions"])-{"mxfp8","bf16_gate"}:
        raise ValueError("unsupported study policy")
    for key in ("slice_seconds","worker_timeout_seconds"):
        if not math.isfinite(manifest[key]) or manifest[key]<=0:raise ValueError("worker budgets must be positive")
    tasks=[]
    # Phase/precision alternate within each shape, ensuring breadth on stop.
    for shape in manifest["shapes"]:
        for phase in manifest["phases"]:
            for precision in manifest["precisions"]:
                task={"shape":shape,"phase":phase,"precision":precision,
                      "activation":manifest["activation"],"seed":manifest["seed"]}
                tasks.append({**task,"id":stable_id(task)})
    return tasks


def anchor_candidates(adapter, precision, phase, count, seed):
    """Deterministic raw portfolio BEFORE device legality filtering.

    Same source revision/manifest yields the same candidate payloads on each
    device. Rejections remain observations, not device-dependent pool shifts.
    """
    from .mlp import mlp_structural_seeds
    bases=[adapter.initial_config,*mlp_structural_seeds(adapter,precision,phase)]
    result=[]; seen=set()
    def add(candidate):
        key=adapter.config_id(candidate)
        if key not in seen:
            seen.add(key); result.append(candidate)
    for base in bases:
        add(base)
    rng=random.Random(seed)
    axes=list(adapter.axes)
    rng.shuffle(axes)
    # A fixed cap keeps malformed spaces from hanging collection preparation.
    for attempt in range(count*32):
        if len(result)>=count:
            break
        axis=axes[attempt%len(axes)]
        base=bases[(attempt//len(axes))%len(bases)]
        try:
            add(adapter.update_fn(base,axis,rng.choice(adapter.axes[axis])))
        except (TypeError,ValueError,KeyError):
            pass
    return result[:count]


def worker(manifest, output, index, milestone, device):
    from .compile_pipeline import compile_pipeline
    lease_path=Path(output)/"gpu_execution.lock"
    with compile_pipeline(lease_path,device):
        return _worker(manifest,output,index,milestone,device)


def _worker(manifest, output, index, milestone, device):
    import torch
    from .mlp import make_mlp_adapter
    from .store import ResidualTuningStore
    from .orchestrator import AutotuneOrchestrator, ConfirmationPolicy, SequentialScheduler
    from .core import TuningBudget
    from .first_hit import PortfolioSearch
    task=study_tasks(manifest)[index]
    torch.manual_seed(task["seed"])
    m,k,h,o=task["shape"]
    x=torch.randn(m,k,device=device,dtype=torch.bfloat16)*.2
    gu=torch.randn(2*h,k,device=device,dtype=torch.bfloat16)*.02
    down=torch.randn(o,h,device=device,dtype=torch.bfloat16)*.02
    adapter=make_mlp_adapter(x,gu,down,activation=task["activation"],
        phase=task["phase"],precision=task["precision"])
    candidates=anchor_candidates(adapter,task["precision"],task["phase"],
                                 max(manifest["milestones"]),task["seed"])
    root=Path(output)/task["id"]/adapter.context.identifier
    store=ResidualTuningStore(root)
    payload=[dict(adapter.serialize(c)) for c in candidates]
    metadata={"task":task,"context":adapter.context.as_dict(),
              "contract":contract(adapter.context),"portfolio_id":stable_id(payload),
              "candidate_ids":[adapter.config_id(c) for c in candidates],
              "manifest_id":stable_id(manifest),"source_id":source_id()}
    from .compile_pipeline import adopt_identity
    adopt_identity(root/"anchor.json",metadata,os.getenv("RTX_AUTOTUNE_ADOPT_SOURCE")=="1")
    strategy=PortfolioSearch(candidates)
    print(f"ANCHOR {index} {task['shape']} {task['precision']} {task['phase']} "
          f"portfolio={metadata['portfolio_id']} trials={milestone}",flush=True)
    AutotuneOrchestrator(adapter,store,[strategy],SequentialScheduler(((strategy.name,None),)),
        TuningBudget(milestone,manifest["slice_seconds"]),transfer_history=False,
        max_trials_includes_resumed=True,confirmation=ConfirmationPolicy(1,.02,True),
        progress=lambda s:print(s,flush=True)).tune()


def run(manifest_path, output, hours, device):
    import fcntl
    manifest=json.loads(Path(manifest_path).read_text())
    root=Path(output); root.mkdir(parents=True,exist_ok=True)
    with (root/"collection.lock").open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity={"manifest":manifest,"manifest_id":stable_id(manifest),"source_id":source_id()}
        from .compile_pipeline import adopt_identity,run_workers
        adopt_identity(root/"study.json",identity,os.getenv("RTX_AUTOTUNE_ADOPT_SOURCE")=="1")
        deadline=time.monotonic()+hours*3600
        workers=max(1,int(os.getenv("RTX_AUTOTUNE_CPU_WORKERS","4")))
        print(f"PIPELINE cpu_workers={workers} GPU measurement leases=1",flush=True)
        for milestone in manifest["milestones"]:
            pending=[]
            for index,task in enumerate(study_tasks(manifest)):
                # Do not repeatedly initialize CUDA for completed anchors when
                # a combined campaign revisits this collector in short slices.
                journals=list((root/task["id"]).glob("*/observations.jsonl"))
                if journals:
                    completed=0
                    for line in journals[0].read_text().splitlines():
                        try:
                            json.loads(line);completed+=1
                        except json.JSONDecodeError:
                            continue
                    if completed>=milestone:
                        continue
                pending.append((index,task))
            if not run_workers(pending,manifest,manifest_path,root,milestone,device,deadline,workers):
                print("DEADLINE residuals are resumable",flush=True);return
        print("COMPLETE planned anchors exhausted; no unbounded search started",flush=True)
        atomic_json(root/"collection_complete.json",identity)


def export(paths, output):
    import csv
    from .pretrained import load_offline_observations
    rows,report=load_offline_observations(paths)
    report["anchors"]=audit_anchors(paths)
    if report["anchors"]["portfolio_mismatches"]:
        raise ValueError("mismatched cross-device portfolios; do not merge as a matched study")
    root=Path(output); root.mkdir(parents=True,exist_ok=True)
    records=[{"context_id":r.context_id,"family":r.family,"revision":r.kernel_revision,
              "config_id":r.config_id,"status":r.outcome.status,"latency_ms":r.outcome.median_ms,
              "elapsed_s":r.elapsed_s,"config_json":json.dumps(r.serialized_config,sort_keys=True),
              "features_json":json.dumps(r.features,sort_keys=True),
              "outcome_json":json.dumps(r.outcome.as_dict(),sort_keys=True)} for r in rows]
    if records:
        with (root/"observations.csv").open("w",newline="") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
        import pyarrow as pa
        import pyarrow.parquet as pq
        pq.write_table(pa.Table.from_pylist(records),root/"observations.parquet")
    atomic_json(root/"ingestion.json",report)


def grouped_key(row, split):
    features=row.features
    if split=="shape":
        return tuple(features.get("context.workload."+k,0) for k in ("m","n","k","h","o"))
    # Device name survives only in raw context metadata; numeric hardware
    # signature deliberately excludes UUIDs, clocks and software versions.
    return tuple(sorted((k,v) for k,v in features.items() if k.startswith("context.device.")
        and any(part in k for part in ("multiprocessor_count","total_memory","major","minor"))))


def fit(paths, output, split="shape", backend="builtin"):
    """Grouped offline replay. Never automatically qualifies a deployable model."""
    from .pretrained import load_offline_observations,ContextRankingModel,_observation_documents
    from .cost_model import GradientBoostedCostModel
    observations,report=load_offline_observations(paths)
    report["anchors"]=audit_anchors(paths)
    if report["anchors"]["portfolio_mismatches"]:
        raise ValueError("mismatched cross-device portfolios")
    benchmark_keys={}
    for _,record in _observation_documents(paths):
        benchmark_keys[record["context_id"]]=tuple(sorted((k,v) for k,v in record.get("features",{}).items()
            if k.startswith(("context.tags.","context.regime="))))
    contracts={}
    for document in anchor_documents(paths):
        contracts[stable_id(document["context"])]=document["contract"]
    rows=[replace(r,features=model_features(r.features)) for r in observations if r.successful]
    families=defaultdict(list)
    # Separate semantic modes and exact benchmark tags, not only revisions.
    for row in rows:
        semantic=tuple(sorted((k,v) for k,v in row.features.items()
            if k.startswith("context.workload.") and "=" in k))
        families[(row.family,row.kernel_revision,semantic,benchmark_keys.get(row.context_id,()))].append(row)
    root=Path(output);root.mkdir(parents=True,exist_ok=True)
    metrics=[]
    def new_model():
        if backend=="xgboost":
            from .first_hit import XGBRankerModel
            return XGBRankerModel()
        return ContextRankingModel(GradientBoostedCostModel(n_estimators=16,ensembles=2,max_depth=3))
    for family,data in families.items():
        groups={grouped_key(r,split) for r in data}
        if len(groups)<2:
            metrics.append({"family":str(family),"status":"insufficient_groups","groups":len(groups)})
            continue
        for heldout in sorted(groups):
            train=[r for r in data if grouped_key(r,split)!=heldout]
            test=[r for r in data if grouped_key(r,split)==heldout]
            model=new_model()
            model.fit(train)
            contexts=defaultdict(list)
            for row in test: contexts[row.context_id].append(row)
            for context,candidates in contexts.items():
                # Replicates are aggregated before ranking; no repeated config
                # can occupy multiple shortlist slots.
                unique={}
                for row in candidates: unique.setdefault(row.config_id,[]).append(row)
                candidates=[replace(v[0],outcome=replace(v[0].outcome,
                    median_ms=float(np.median([r.score for r in v])))) for v in unique.values()]
                if len(candidates)<8: continue
                predicted,_=model.predict([r.features for r in candidates])
                ranked=sorted(range(len(candidates)),key=lambda i:float(predicted[i]))
                heuristic=sorted(range(len(candidates)),key=lambda i:(candidates[i].sequence,candidates[i].config_id))
                oracle=min(r.score for r in candidates)
                training_winners={}
                for row in train:
                    old=training_winners.get(row.context_id)
                    if old is None or row.score<old.score:training_winners[row.context_id]=row
                def distance(row):
                    return sum(abs(math.log2(max(1,row.features.get("context.workload."+axis,1))/
                        max(1,candidates[0].features.get("context.workload."+axis,1))))
                        for axis in ("m","n","k","h","o"))
                retrieved_ids=[r.config_id for r in sorted(training_winners.values(),key=distance)]
                retrieved=sorted(range(len(candidates)),key=lambda i:(
                    retrieved_ids.index(candidates[i].config_id) if candidates[i].config_id in retrieved_ids else len(retrieved_ids),
                    candidates[i].sequence,candidates[i].config_id))
                for budget in (1,2,4,8):
                    random_regrets=[]
                    for seed in range(100):
                        selected=random.Random(seed).sample(candidates,budget)
                        random_regrets.append(min(r.score for r in selected)/oracle-1)
                    metrics.append({"family":str(family),"context":context,"split":split,
                        "heldout":str(heldout),"budget":budget,"candidates":len(candidates),
                        "ranker_regret":min(candidates[i].score for i in ranked[:budget])/oracle-1,
                        "ordered_portfolio_regret":min(candidates[i].score for i in heuristic[:budget])/oracle-1,
                        "retrieval_regret":min(candidates[i].score for i in retrieved[:budget])/oracle-1,
                        "random_regret_mean":float(np.mean(random_regrets)),
                        "status":"offline_replay_successful_candidates_only"})
        final_model=new_model();final_model.fit(data)
        known_contracts=[contracts[r.context_id] for r in data if r.context_id in contracts]
        if known_contracts and all(c==known_contracts[0] for c in known_contracts):
            artifact={"schema_version":1,"deployment_enabled":False,"backend":backend,
                      "contract":known_contracts[0],"model":final_model.state_dict(),
                      "training_contexts":sorted({r.context_id for r in data}),
                      "training_observations_id":stable_id(sorted(r.observation_id for r in data))}
            atomic_json(root/(stable_id(str(family))+"-ranker.json"),artifact)
    atomic_json(root/"evaluation.json",{"schema_version":1,"deployment_enabled":False,"backend":backend,
        "ingestion":report,"split":split,"metrics":metrics,
        "limitations":["Offline replay, not prospective validation",
                       "Successful observed candidates only; no unseen-candidate or failure-cost claim"]})
    print(f"EVALUATION {root/'evaluation.json'} deployment_enabled=false",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="command",required=True)
    for name in ("run","worker"):
        sub=commands.add_parser(name);sub.add_argument("manifest");sub.add_argument("--output",required=True)
        sub.add_argument("--device",default="cuda:0")
        if name=="run": sub.add_argument("--hours",type=float,default=4)
        else:
            sub.add_argument("--index",type=int,required=True);sub.add_argument("--milestone",type=int,required=True)
    for name in ("fit","export"):
        sub=commands.add_parser(name);sub.add_argument("paths",nargs="+");sub.add_argument("--output",required=True)
        if name=="fit":
            sub.add_argument("--split",choices=("shape","device"),default="shape")
            sub.add_argument("--backend",choices=("builtin","xgboost"),default="builtin")
    args=parser.parse_args()
    if args.command=="run":
        if not math.isfinite(args.hours) or args.hours<=0:parser.error("hours must be finite and positive")
        def stop(signum,frame):
            raise KeyboardInterrupt("collection supervisor stopped")
        signal.signal(signal.SIGTERM,stop)
        run(args.manifest,args.output,args.hours,args.device)
    elif args.command=="worker":
        worker(json.loads(Path(args.manifest).read_text()),args.output,args.index,args.milestone,args.device)
    elif args.command=="export":export(args.paths,args.output)
    else:fit(args.paths,args.output,args.split,args.backend)


if __name__=="__main__":main()
