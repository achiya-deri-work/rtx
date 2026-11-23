"""Launch BF16 dY.T @ X into a caller-owned contiguous gradient buffer."""

import ctypes as C
from functools import lru_cache

import torch


_OP_N, _OP_T = 0, 1
_CUDA_R_16BF = 14
_COMPUTE_32F = 68
_GEMM_DEFAULT = -1
_DISALLOW_REDUCED_PRECISION_REDUCTION = 16
_POINTER_MODE_HOST = 0
_ONE, _ZERO = C.c_float(1), C.c_float(0)


@lru_cache(maxsize=1)
def _bindings():
    from cuda.pathfinder import load_nvidia_dynamic_lib

    loader = C.WinDLL if hasattr(C, "WinDLL") else C.CDLL
    lib = loader(load_nvidia_dynamic_lib("cublas").abs_path)
    ptr, num = C.c_void_p, C.c_int
    lib.cublasGemmEx.argtypes = (
        ptr, num, num, num, num, num, ptr, ptr, num, num,
        ptr, num, num, ptr, ptr, num, num, num, num,
    )
    lib.cublasGemmEx.restype = num
    for mode, suffix in (("MathMode", ""), ("PointerMode", "_v2")):
        get = getattr(lib, f"cublasGet{mode}{suffix}")
        get.argtypes, get.restype = (ptr, C.POINTER(num)), num
        set_mode = getattr(lib, f"cublasSet{mode}{suffix}")
        set_mode.argtypes, set_mode.restype = (ptr, num), num
    return lib


def _check(status, action):
    if status:
        raise RuntimeError(f"cuBLAS dW {action} failed (status={status})")


class CublasBF16DW:
    """Uses PyTorch's current device/stream BLAS handle; creates no tensors."""

    __slots__ = ("lib", "rows", "cols", "tokens")

    def __init__(self, problem):
        self.rows, self.cols, self.tokens = problem.m, problem.n, problem.k
        self.lib = _bindings()

    def __call__(self, dy, x, out):
        lib = self.lib
        with torch.cuda.device(dy.device):
            handle = torch.cuda.current_blas_handle()
            math_mode, pointer_mode = C.c_int(), C.c_int()
            _check(lib.cublasGetMathMode(handle, C.byref(math_mode)), "get math mode")
            _check(lib.cublasGetPointerMode_v2(handle, C.byref(pointer_mode)), "get pointer mode")
            strict = math_mode.value | _DISALLOW_REDUCED_PRECISION_REDUCTION
            change_math = strict != math_mode.value
            change_pointer = pointer_mode.value != _POINTER_MODE_HOST
            if change_math:
                _check(lib.cublasSetMathMode(handle, strict), "enable FP32 reduction")
            try:
                if change_pointer:
                    _check(lib.cublasSetPointerMode_v2(handle, _POINTER_MODE_HOST), "set host scalars")
                try:
                    # Row-major dW = dY.T @ X corresponds to a column-major
                    # dW.T = X.T @ dY. No transpose view or copy is required.
                    _check(lib.cublasGemmEx(
                        handle, _OP_N, _OP_T, self.cols, self.rows, self.tokens,
                        C.byref(_ONE), x.data_ptr(), _CUDA_R_16BF, self.cols,
                        dy.data_ptr(), _CUDA_R_16BF, self.rows, C.byref(_ZERO),
                        out.data_ptr(), _CUDA_R_16BF, self.cols, _COMPUTE_32F, _GEMM_DEFAULT,
                    ), "GemmEx")
                finally:
                    if change_pointer:
                        _check(lib.cublasSetPointerMode_v2(handle, pointer_mode.value), "restore pointer mode")
            finally:
                if change_math:
                    _check(lib.cublasSetMathMode(handle, math_mode.value), "restore math mode")
