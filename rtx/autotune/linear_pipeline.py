"""Parallel linear campaign shards with serialized GPU work and CuTe compile overlap."""
import argparse
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


def run(output,seconds,device,worker_index=None,workers=None):
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("seconds must be finite and positive")
    if worker_index is not None and (workers is None or not 0 <= worker_index < workers):
        raise ValueError("worker index must be within shard count")
    from .compile_pipeline import compile_pipeline
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=True)
    if worker_index is not None:
        from .dataset import main as dataset_main
        with compile_pipeline(root/"gpu_execution.lock",device):
            dataset_main(["run","autotune_manifests/blackwell_diversity_atlas_v1.json",
                "--output-dir",str(root),"--device",device,"--format","none",
                "--wall-time",f"{seconds}s","--context-slice","45s",
                "--trial-milestones","1,2,4,8,16,32,64,128","--initial-promote","1",
                "--context-orchestration","breadth_first","--max-milestone-lead","1",
                "--strategy-orchestration","bandit","--reuse-deterministic-failures",
                "--adopt-existing-context-identity-if-present",
                "--shard-index",str(worker_index),"--shard-count",str(workers)])
        return
    import torch
    requested=int(os.getenv("RTX_AUTOTUNE_CPU_WORKERS","4"))
    workers=max(1,min(requested,int(torch.cuda.get_device_properties(device).total_memory/8_000_000_000)))
    print(f"LINEAR PIPELINE workers={workers} GPU execution leases=1",flush=True)
    active=[];deadline=time.monotonic()+seconds
    def stop(process):
        if process.poll() is not None:return
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
    try:
        for index in range(workers):
            log=root/f"compiler-shard-{index:03d}.log"
            stream=log.open("a")
            env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","TORCHINDUCTOR_COMPILE_THREADS":"1"}
            process=subprocess.Popen([sys.executable,"-u","-m","rtx.autotune.linear_pipeline",
                "--output",str(root),"--seconds",str(seconds),"--device",device,
                "--worker-index",str(index),"--workers",str(workers)],
                stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,env=env)
            active.append((process,stream))
            print(f"SHARD {index} pid={process.pid} log={log}",flush=True)
        while any(p.poll() is None for p,_ in active) and time.monotonic()<deadline:
            time.sleep(.1)
        if time.monotonic()>=deadline:print("DEADLINE linear pipeline slice exhausted",flush=True)
        elif any(p.returncode for p,_ in active):raise RuntimeError("linear shard failed; inspect compiler-shard logs")
    finally:
        # Notify every child before waiting, so the outer supervisor's grace
        # period cannot leave later shards running after this process exits.
        for process,_ in active:
            if process.poll() is None:
                try:os.killpg(process.pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for process,stream in active:stop(process);stream.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True);parser.add_argument("--seconds",type=float,required=True)
    parser.add_argument("--device",default="cuda:0");parser.add_argument("--worker-index",type=int)
    parser.add_argument("--workers",type=int)
    args=parser.parse_args()
    def stop(signum,frame):raise KeyboardInterrupt("linear pipeline stopped")
    signal.signal(signal.SIGTERM,stop)
    run(args.output,args.seconds,args.device,args.worker_index,args.workers)


if __name__=="__main__":main()
