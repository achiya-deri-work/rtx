"""Run an existing application against RTX lifecycle fixes without editing it.

The application must expose benchmarks.memory_audit.MemoryAudit and a
graphs_warm boundary with no concurrent host submissions (FlashDETR harness).
This wrapper is diagnostic only; it does not alter model math or tuning policy.
"""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
from pathlib import Path
import runpy
import sys


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--application-root',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--split-workspace-mib',type=int,default=16)
    p.add_argument('--explicit-resources', action='store_true',
                   help='historical diagnostic control; normal execution owns resources automatically')
    p.add_argument('--allow-application-torch',action='store_true',
                   help='diagnostic-only use of the application Torch version; not native release qualification')
    args,command=p.parse_known_args()
    if command and command[0]=='--':command=command[1:]
    source=Path(__file__).resolve().parents[1]
    sys.path[:0]=[str(source),str(args.application_root),str(args.application_root/'fp8dinov3')]
    import benchmarks
    benchmarks.__path__=[str(args.application_root/'benchmarks'),*list(benchmarks.__path__)]
    import torch
    import rtx
    if args.allow_application_torch:
        import rtx.runtime as runtime
        runtime._SUPPORTED_TORCH=runtime._SUPPORTED_TORCH|{str(torch.__version__).split('+')[0]}
    from benchmarks.memory_audit import MemoryAudit
    capture=MemoryAudit.capture
    owner=(rtx.ExecutionResources(split_workspace_bytes=args.split_workspace_mib<<20)
           if args.explicit_resources else None)
    reports={}
    reports['source_snapshot']=str(source)
    reports['source_sha256']={name:hashlib.sha256((source/name).read_bytes()).hexdigest() for name in
        ('rtx/runtime.py','rtx/_workspace.py','rtx/_resources.py','rtx/execution.py','rtx/_managed.py',
         'rtx/fp8_bwd.py','rtx/fp8.py','rtx/_graph_replay.py')}
    def audited(audit,label,**kwargs):
        if label=='graphs_warm':
            reports['before']=rtx.runtime_memory_report()
            free,total=torch.cuda.mem_get_info();reports['device_used_before']=total-free
            if owner is not None:
                reports['retirement']=owner.retire(streams=[torch.cuda.default_stream()],device='cuda:0')
                gc.collect();torch.cuda.empty_cache();torch.cuda.synchronize()
            free,total=torch.cuda.mem_get_info();reports['device_used_after']=total-free
            reports['after']=rtx.runtime_memory_report()
            reports['explicit_resources']=args.explicit_resources
            if owner is not None:reports['arena']=owner.stats()
            reports['torch']=str(torch.__version__)
            reports['application_runtime_override']=args.allow_application_torch
            args.output.write_text(json.dumps(reports,indent=2))
        return capture(audit,label,**kwargs)
    MemoryAudit.capture=audited
    sys.argv=[str(args.application_root/'benchmarks/full_dummy_training.py'),*command]
    with owner.activate() if owner is not None else nullcontext():
        runpy.run_path(sys.argv[0],run_name='__main__')


if __name__=='__main__':main()
