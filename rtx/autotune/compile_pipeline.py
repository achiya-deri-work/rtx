"""Process-isolated CuTe compilation with an exclusive GPU execution lease.

Only cute.compile may run outside the lease. Tensor creation, Torch tracing,
validation, capture and timing stay serialized. This is a worker-only hook,
never installed in application/runtime processes.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time
import signal
import subprocess
import sys


class GPULease:
    def __init__(self,path):
        self.path=Path(path)
        self.stream=None

    def event(self,kind,**fields):
        record={"pid":os.getpid(),"time":time.monotonic(),"event":kind,**fields}
        with self.path.with_suffix(".jsonl").open("a") as stream:
            stream.write(json.dumps(record)+"\n")

    def acquire(self):
        started=time.monotonic()
        fcntl.flock(self.stream,fcntl.LOCK_EX)
        self.event("gpu_acquired",wait_s=time.monotonic()-started)

    def release(self):
        self.event("gpu_released")
        fcntl.flock(self.stream,fcntl.LOCK_UN)

    def __enter__(self):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        self.stream=self.path.open("a")
        self.acquire()
        return self

    def __exit__(self,*args):
        self.release();self.stream.close()


@contextmanager
def compile_pipeline(path,device):
    import torch
    import cutlass.cute as cute
    with GPULease(path) as lease:
        original=cute.compile
        def compile_unlocked(*args,**kwargs):
            # Drain this process before another worker can measure. Pure CuTe
            # compilation uses fake tensors/streams, not workload launches.
            torch.cuda.synchronize(device)
            lease.release()
            started=time.monotonic()
            lease.event("compile_started")
            try:
                return original(*args,**kwargs)
            finally:
                lease.event("compile_finished",elapsed_s=time.monotonic()-started)
                lease.acquire()
        cute.compile=compile_unlocked
        try:
            yield lease
        finally:
            cute.compile=original
            # A poisoned context may raise here; closing the lease still
            # releases the OS lock through the outer context manager.
            torch.cuda.synchronize(device)


def adopt_identity(path,expected,allow=False):
    """Explicit source-only migration; workload/portfolio identities stay strict."""
    from .first_hit_study import atomic_json
    path=Path(path)
    if path.exists():
        previous=json.loads(path.read_text())
        fields=lambda d:{k:v for k,v in d.items() if k not in ("source_id","source_history")}
        if fields(previous)!=fields(expected):
            raise ValueError(f"workload/portfolio identity changed: {path}")
        changed=previous.get("source_id")!=expected.get("source_id")
        if changed and not allow:
            raise ValueError(f"source changed: explicitly adopt execution-only changes or use a new output: {path}")
        history=list(previous.get("source_history",[]))
        if changed:history.append(previous.get("source_id"))
        if history:expected={**expected,"source_history":history}
    atomic_json(path,expected)


def run_workers(pending,manifest,manifest_path,root,milestone,device,deadline,workers):
    """Bounded process pool: coarse VRAM admission and one GPU execution token."""
    pending=list(pending);active=[];failures=0
    import torch
    budget=int(torch.cuda.get_device_properties(device).total_memory*.65) if torch.cuda.is_available() else 4*1024**3
    def reserve(task):
        m,k,h,o=task["shape"]
        return 768*1024**2+8*m*(k+6*h+3*o)+32*(2*h*k+o*h)
    compiler_timeout=float(os.getenv("RTX_AUTOTUNE_COMPILE_TIMEOUT","60"))
    def stop(process):
        if process.poll() is not None:return
        os.killpg(process.pid,signal.SIGTERM)
        try:process.wait(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid,signal.SIGKILL);process.wait()
    try:
        while pending or active:
            if time.monotonic()>=deadline:return False
            # Conservative byte-accounted admission, not four full allocator
            # pools regardless of the GPU's capacity. Oversized jobs run alone.
            while pending and len(active)<workers:
                index,task=pending[0]
                memory=reserve(task)
                if active and memory+sum(item["memory"] for item in active)>budget:break
                pending.pop(0)
                log=root/f"worker-{index:03d}-{milestone:03d}.log"
                command=[sys.executable,"-m","rtx.autotune.first_hit_study","worker",str(Path(manifest_path).resolve()),
                         "--output",str(root.resolve()),"--index",str(index),"--milestone",str(milestone),"--device",device]
                env={**os.environ,"OMP_NUM_THREADS":"1","MKL_NUM_THREADS":"1","TORCHINDUCTOR_COMPILE_THREADS":"1"}
                stream=log.open("a")
                process=subprocess.Popen(command,stdout=stream,stderr=subprocess.STDOUT,
                                         start_new_session=True,env=env)
                active.append({"process":process,"stream":stream,"started":time.monotonic(),
                               "task":task,"index":index,"memory":memory})
                print(f"START task={index} milestone={milestone} pid={process.pid} active={len(active)} log={log}",flush=True)
            events={}
            lease_log=root/"gpu_execution.jsonl"
            if lease_log.exists():
                # A few events per kernel compile; seek only a bounded tail.
                with lease_log.open("rb") as stream:
                    stream.seek(max(0,lease_log.stat().st_size-131072))
                    for line in stream:
                        try:
                            event=json.loads(line);events[event["pid"]]=event
                        except (json.JSONDecodeError,UnicodeDecodeError):pass
            for item in list(active):
                process=item["process"];status=process.poll()
                event=events.get(process.pid,{})
                compile_stall=(event.get("event")=="compile_started" and
                               time.monotonic()-event["time"]>compiler_timeout)
                if status is None and (compile_stall or time.monotonic()-item["started"]>
                                       manifest["worker_timeout_seconds"]*workers):
                    print(f"WATCHDOG pid={process.pid} compile_stall={compile_stall}",flush=True)
                    stop(process);status=124
                if status is None:continue
                item["stream"].close();active.remove(item)
                with (root/"supervisor.jsonl").open("a") as stream:
                    stream.write(json.dumps({"task":item["task"]["id"],"milestone":milestone,
                        "status":status,"elapsed_s":time.monotonic()-item["started"],"cpu_workers":workers})+"\n")
                    stream.flush();os.fsync(stream.fileno())
                print(f"DONE task={item['index']} milestone={milestone} status={status}",flush=True)
                failures=failures+1 if status else 0
                if failures>=3:raise RuntimeError("three consecutive workers failed; inspect worker logs")
            if active:time.sleep(.1)
        return True
    finally:
        for item in active:
            if item["process"].poll() is None:
                try:os.killpg(item["process"].pid,signal.SIGTERM)
                except ProcessLookupError:pass
        for item in active:
            stop(item["process"]);item["stream"].close()
