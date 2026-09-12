"""Balanced SiLU/GELU shape study; isolated workers and per-trial residual files."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def source_fingerprint():
    digest=hashlib.sha256()
    for path in sorted((*Path("rtx").rglob("*.py"),*Path("benchmarks").glob("*.py"))):
        digest.update(str(path).encode());digest.update(path.read_bytes())
    packages={}
    for name in ("torch","torchao","nvidia-cutlass-dsl","apache-tvm-ffi"):
        try:
            packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name]=None
    try:
        probe=subprocess.run(["nvidia-smi","--query-gpu=name,uuid,driver_version",
                              "--format=csv,noheader"],capture_output=True,text=True,timeout=10)
        hardware=probe.stdout.strip() if probe.returncode==0 else "unavailable"
    except (OSError,subprocess.TimeoutExpired):
        hardware="unavailable"
    environment={"packages":packages,"hardware":hardware,"python":sys.version,
                 "cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES")}
    digest.update(json.dumps(environment,sort_keys=True).encode())
    return digest.hexdigest(),environment


SHAPES = (
    "128,256,512,256", "512,768,1536,768", "2048,512,2048,512",
    "1024,1024,4096,1024", "8192,256,768,512", "129,128,256,128",
)


def tasks():
    contexts=[(shape,act) for shape in SHAPES for act in ("silu","gelu")]
    # Cover all contexts before searching any single one deeply.
    for policy in ("bf16_gate_fwd","fused","bf16"):
        for shape,act in contexts:
            yield "coverage",shape,act,policy,[]
    for shape,act in contexts:
        for policy in ("bf16_gate_fwd","fused"):
            yield "numerics",shape,act,policy,["--distribution","uniform_half","--seed","31"]
    for options in (
        ["--tile-m","128","--stages","2"],
        ["--down-tile-m","128","--down-stages","2"],
        ["--tile-m","128","--stages","2","--down-tile-m","64","--down-stages","1"],
        ["--persistent-waves","1"],
    ):
        for shape,act in contexts:
            if shape.startswith("129,"):
                continue
            yield "search",shape,act,"bf16_gate_fwd",options


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output-dir",required=True)
    p.add_argument("--seconds",type=float,default=2700)
    p.add_argument("--worker-seconds",type=float,default=240)
    a=p.parse_args()
    if min(a.seconds,a.worker_seconds)<=0:
        p.error("time budgets must be positive")
    root=Path(a.output_dir); root.mkdir(parents=True,exist_ok=True)
    start=time.monotonic()
    revision,environment=source_fingerprint()
    (root/f"environment-{revision[:16]}.json").write_text(json.dumps(environment,indent=2))
    for index,(wave,shape,act,policy,extra) in enumerate(tasks()):
        if time.monotonic()-start>=a.seconds:
            break
        args=["--shape",shape,"--activation",act,"--policy",policy,"--backend","materialized",
              "--materialized-layout","row_major","--tile-m","64","--stages","1",*extra]
        if shape.startswith("129,"):
            args += ["--preset","default","--backward-mode","default"]
        key=hashlib.sha256(json.dumps([revision,args]).encode()).hexdigest()[:16]
        directory=root/key; directory.mkdir(exist_ok=True)
        outcome=directory/"outcome.json"
        if outcome.exists():
            try:
                previous=json.loads(outcome.read_text())
                if previous.get("status") in ("ok","recovered","error","timeout"):
                    print("RESUME",index,wave,shape,act,policy,key,flush=True)
                    continue
            except json.JSONDecodeError:
                pass
            outcome.rename(directory/f"incomplete-outcome-{time.time_ns()}.json")
        result=directory/"result.json"
        # A worker interrupted after its result write is recoverable.
        if result.exists():
            try:
                report=json.loads(result.read_text())
                if "train" in report:
                    outcome.write_text(json.dumps({"status":"recovered","args":args,"source":revision}))
                    continue
            except json.JSONDecodeError:
                pass
            result.rename(directory/f"incomplete-{time.time_ns()}.json")
        command=[sys.executable,"-m","benchmarks.benchmark_mlp_quality",*args,"--output",str(result)]
        print("START",index,wave,shape,act,policy,key,flush=True)
        before=time.monotonic()
        with (directory/"worker.log").open("w") as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            try:
                code=proc.wait(timeout=min(a.worker_seconds,max(1,a.seconds-(before-start))))
                status="ok" if code==0 else "error"
            except subprocess.TimeoutExpired:
                # Only this worker's newly created process group is targeted.
                os.killpg(proc.pid,signal.SIGTERM)
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid,signal.SIGKILL); proc.wait()
                status="timeout"
        record={"status":status,"args":args,"source":revision,"wave":wave,
                "wall_seconds":time.monotonic()-before,"command":command}
        if status=="ok":
            report=json.loads(result.read_text())
            record["train_us"]=report["train"]["median_us"]
        outcome.write_text(json.dumps(record,indent=2))
        print("DONE",index,status,record.get("train_us"),flush=True)


if __name__=="__main__":
    main()
