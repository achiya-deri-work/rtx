"""Optional CUDA/CUB benchmark baseline; never imported by production RTX."""
import ctypes
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from threading import RLock

_libraries={}
_lock=RLock()


def _toolkit():
    candidates=[os.getenv('CUDA_TOOLKIT_PATH'),os.getenv('CUDA_HOME'),
                str(Path(sys.prefix)/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages/nvidia/cu13')]
    nvcc=shutil.which('nvcc')
    if nvcc:candidates.append(str(Path(nvcc).resolve().parent.parent))
    for value in candidates:
        if value and (Path(value)/'bin/nvcc').is_file():return Path(value)
    raise RuntimeError('Native routing requires a CUDA 13 toolkit with NVCC/CCCL; '
                       'set CUDA_TOOLKIT_PATH to a CUDA 13.2 NVCC/CCCL installation. '
                       "backend='torch' remains the explicit reference path.")


def library():
    import torch
    capability=torch.cuda.get_device_capability()
    with _lock:
        if capability in _libraries:return _libraries[capability]
        if capability[0]!=12:raise RuntimeError('Native routing targets RTX SM12x only')
        root=_toolkit()
        # Compile with the selected toolkit, but link against Torch's runtime
        # when it is present. Loading a second libcudart from a system toolkit
        # can split kernel registration from graph construction in one process.
        environment_runtime=Path(sys.prefix)/'lib'/f'python{sys.version_info.major}.{sys.version_info.minor}'/'site-packages/nvidia/cu13/lib'
        libdir=next((p for p in (environment_runtime,root/'lib',root/'lib64')
                     if (p/'libcudart.so.13').exists()),None)
        if libdir is None:raise RuntimeError('CUDA toolkit must supply libcudart.so.13')
        source=Path(__file__).with_name('routing_cub.cu')
        version=subprocess.check_output([str(root/'bin/nvcc'),'--version'],text=True)
        runtime=libdir/'libcudart.so.13'
        fingerprint=hashlib.sha256(source.read_bytes()+str((root,version,capability,torch.version.cuda,
            runtime.resolve(),runtime.stat().st_mtime_ns)).encode()).hexdigest()[:24]
        cache=Path(os.getenv('RTX_ROUTING_BUILD_CACHE',str(Path(os.getenv('XDG_CACHE_HOME',Path.home()/'.cache'))/'rtx/routing')))/fingerprint
        cache.mkdir(parents=True,exist_ok=True,mode=0o700)
        output=cache/'routing.so'
        import fcntl
        with (cache/'build.lock').open('a') as lockfile:
            fcntl.flock(lockfile,fcntl.LOCK_EX)
            if not output.exists():
                fd,name=tempfile.mkstemp(prefix='routing-',suffix='.so',dir=cache);os.close(fd)
                temporary=Path(name)
                command=[str(root/'bin/nvcc'),'-std=c++17','-O3','-shared','--cudart=none','--threads=4',
                    '-Xcompiler=-fPIC',f'-arch=sm_{capability[0]}{capability[1]}',
                    '-I'+str(root/'include/cccl'),str(source),'-L'+str(libdir),
                    '-Xlinker=-l:libcudart.so.13','-Xlinker=-rpath','-Xlinker='+str(libdir),'-o',str(temporary)]
                result=subprocess.run(command,text=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
                (cache/'build.log').write_text(result.stdout)
                if result.returncode:
                    temporary.unlink(missing_ok=True)
                    raise RuntimeError(f'Routing native build failed: {cache / "build.log"}\n{result.stdout}')
                temporary.replace(output)
        lib=ctypes.CDLL(str(output))
        lib.bimoe_repair_graph_create.argtypes=[ctypes.c_void_p]*4+[ctypes.c_int]*3+[ctypes.POINTER(ctypes.c_void_p)]
        lib.bimoe_repair_graph_create.restype=ctypes.c_int
        lib.bimoe_repair_graph_launch.argtypes=[ctypes.c_void_p,ctypes.c_void_p]
        lib.bimoe_repair_graph_launch.restype=ctypes.c_int
        lib.bimoe_repair_graph_destroy.argtypes=[ctypes.c_void_p]
        lib.bimoe_repair_graph_destroy.restype=None
        lib.bimoe_quota_workspace.argtypes=[ctypes.c_int,ctypes.c_int,ctypes.POINTER(ctypes.c_size_t)]
        lib.bimoe_quota_workspace.restype=ctypes.c_int
        lib.bimoe_quota_apply.argtypes=[ctypes.c_void_p]*13+[ctypes.c_size_t]+[ctypes.c_void_p]*5+[
            ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_void_p]
        lib.bimoe_quota_apply.restype=ctypes.c_int
        lib.bimoe_quota_error.argtypes=[ctypes.c_int];lib.bimoe_quota_error.restype=ctypes.c_char_p
        lib.rtx_routing_copy.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_size_t,ctypes.c_void_p]
        lib.rtx_routing_copy.restype=ctypes.c_int
        lib.rtx_routing_quantile.argtypes=[ctypes.c_void_p,ctypes.c_void_p]+[ctypes.c_int]*5+[ctypes.c_void_p]
        lib.rtx_routing_quantile.restype=ctypes.c_int
        _libraries[capability]=lib
        return lib


def checked(lib,result):
    if result:raise RuntimeError(f'Native routing CUDA error: {lib.bimoe_quota_error(result).decode()} ({result})')
