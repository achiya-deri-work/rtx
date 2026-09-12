"""Alternate MLP anchors and linear diversity collection under one deadline."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from .first_hit_study import atomic_json, source_id


def command(kind, root, seconds, device):
    base=[sys.executable,"-u","-m"]
    if kind=="mlp":
        return base+["rtx.autotune.first_hit_study","run",
            "autotune_manifests/blackwell_first_hit_anchors_v1.json",
            "--output",str(root/"mlp"),"--hours",str(seconds/3600),"--device",device]
    return base+["rtx.autotune.linear_pipeline","--output",str(root/"linears"),
                 "--device",device,"--seconds",str(seconds)]


def run(output, hours=6, slice_minutes=15, device="cuda:0",adopt_source=False):
    if not all(math.isfinite(v) and v>0 for v in (hours,slice_minutes)):
        raise ValueError("duration and slice must be finite and positive")
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
    # One combined collector per visible-device mapping, even across different
    # output directories. This does not claim to lock unrelated applications.
    from .core import stable_id
    lock_id=stable_id([device,os.getenv("CUDA_VISIBLE_DEVICES","")])
    lock_path=Path("autotune_logs")/f"combined-{lock_id}.lock"
    lock_path.parent.mkdir(parents=True,exist_ok=True)
    with lock_path.open("w") as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        identity={"source_id":source_id(),"device":device,"schema_version":1}
        identity_path=root/"identity.json"
        from .compile_pipeline import adopt_identity
        adopt_identity(identity_path,identity,adopt_source)
        if adopt_source:os.environ["RTX_AUTOTUNE_ADOPT_SOURCE"]="1"
        state_path=root/"cursor.json"
        state=json.loads(state_path.read_text()) if state_path.exists() else {"next":"mlp","visit":0}
        deadline=time.monotonic()+hours*3600
        failures=0
        while deadline-time.monotonic()>10:
            kind=state["next"]
            if kind=="mlp" and (root/"mlp/collection_complete.json").exists():
                kind="linears"
            seconds=min(slice_minutes*60,deadline-time.monotonic()-10)
            state["visit"]+=1
            log=root/f"visit-{state['visit']:04d}-{kind}.log"
            print(f"START {kind} slice={seconds:.1f}s log={log}",flush=True)
            with log.open("a") as stream:
                process=subprocess.Popen(command(kind,root,seconds,device),
                    stdout=stream,stderr=subprocess.STDOUT,start_new_session=True)
                try:
                    status=process.wait(timeout=min(seconds+5,deadline-time.monotonic()))
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait()
                    status=124
                except BaseException:
                    os.killpg(process.pid,signal.SIGTERM)
                    try:process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid,signal.SIGKILL);process.wait()
                    raise
            state["next"]="linears" if kind=="mlp" else "mlp"
            atomic_json(state_path,state)
            with (root/"visits.jsonl").open("a") as stream:
                stream.write(json.dumps({**state,"kind":kind,"status":status,"allocated_seconds":seconds})+"\n")
                stream.flush();os.fsync(stream.fileno())
            print(f"DONE {kind} status={status}",flush=True)
            failures=failures+1 if status not in (0,75,124) else 0
            if failures>=3:
                raise RuntimeError("three collection slices failed; inspect visit logs")
            # A completed linear plan should not cause rapid restart loops.
            if kind=="linears" and status==0 and "DEADLINE" not in log.read_text():
                print("LINEARS exited normally; inspect log for completion status",flush=True)
                if (root/"mlp/collection_complete.json").exists():break
        print(f"DONE output={root}; journals are resumable; export after collection",flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--hours",type=float,default=6)
    parser.add_argument("--slice-minutes",type=float,default=15)
    parser.add_argument("--device",default="cuda:0")
    parser.add_argument("--adopt-source",action="store_true",help="Explicitly adopt execution-only source changes while preserving all workload/portfolio identities")
    args=parser.parse_args()
    def stop(signum,frame):raise KeyboardInterrupt("combined collection stopped")
    signal.signal(signal.SIGTERM,stop)
    run(args.output,args.hours,args.slice_minutes,args.device,args.adopt_source)


if __name__=="__main__":main()
