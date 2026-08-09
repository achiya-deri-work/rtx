"""FP32 workspace epilogues for split-reduction MXFP8 backward GEMMs."""

from __future__ import annotations

from functools import lru_cache
import math
import os

os.environ.setdefault("CUTE_DSL_ARCH", "sm_120a")
os.environ.setdefault("QUACK_ARCH", "sm_120a")

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass import BFloat16, Float32


class MXFP8WorkspaceReduceKernel:
    """Reduce ``[split, M, N]`` FP32 partials once, then cast to BF16."""

    def __init__(
        self,
        m: int,
        n: int,
        split: int,
        *,
        algorithm: str,
        threads: int,
        vector: int,
        persistent_waves: int,
    ) -> None:
        if min(m, n) <= 0 or split not in (1, 2, 4, 8, 16, 32):
            raise ValueError("invalid split-reduction shape")
        if algorithm not in ("serial", "tree", "persistent_tree"):
            raise ValueError(f"invalid workspace reduction {algorithm!r}")
        if threads not in (64, 128, 256, 512, 1024):
            raise ValueError("reduction threads must be 64, 128, 256, 512, or 1024")
        if vector not in (1, 2, 4, 8):
            raise ValueError("reduction vector must be 1, 2, 4, or 8")
        if persistent_waves not in (1, 2, 3, 4, 6, 8):
            raise ValueError("reduction persistent waves must be 1, 2, 3, 4, 6, or 8")
        self.m = m
        self.n = n
        self.split = split
        self.algorithm = algorithm
        self.threads = threads
        self.vector = vector
        self.persistent_waves = persistent_waves

    @cute.jit
    def __call__(
        self,
        workspace: cute.Tensor,
        out: cute.Tensor,
        stream: cuda.CUstream,
    ):
        elements = self.m * self.n
        natural_ctas = cute.ceil_div(elements, self.threads * self.vector)
        grid_ctas = natural_ctas
        if cutlass.const_expr(self.algorithm == "persistent_tree"):
            sm_count = utils.HardwareInfo().get_device_multiprocessor_count()
            grid_ctas = cutlass.min(
                natural_ctas, sm_count * self.persistent_waves
            )
        self.kernel(workspace, out).launch(
            grid=(grid_ctas, 1, 1),
            block=(self.threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def kernel(self, workspace: cute.Tensor, out: cute.Tensor):
        tidx, _, _ = cute.arch.thread_idx()
        block, _, _ = cute.arch.block_idx()
        grid, _, _ = cute.arch.grid_dim()
        elements = self.m * self.n
        out_flat = cute.make_tensor(
            out.iterator, cute.make_layout((elements,), stride=(1,))
        )
        first = (block * self.threads + tidx) * self.vector
        step = grid * self.threads * self.vector
        for base in cutlass.range(first, elements, step, unroll=1):
            for lane in cutlass.range_constexpr(self.vector):
                index = base + lane
                if index < elements:
                    values = [Float32(0.0)] * self.split
                    for part in cutlass.range_constexpr(self.split):
                        values[part] = workspace[part * elements + index]
                    if cutlass.const_expr(self.algorithm == "serial"):
                        total = Float32(0.0)
                        for part in cutlass.range_constexpr(self.split):
                            total += values[part]
                    else:
                        width = self.split
                        for _level in cutlass.range_constexpr(
                            int(math.log2(self.split))
                        ):
                            for pair in cutlass.range_constexpr(width // 2):
                                values[pair] = values[2 * pair] + values[2 * pair + 1]
                            width //= 2
                        total = values[0]
                    out_flat[index] = BFloat16(total)


@lru_cache(maxsize=None)
def compile_mxfp8_workspace_reduce(
    m: int,
    n: int,
    split: int,
    *,
    algorithm: str,
    threads: int,
    vector: int,
    persistent_waves: int,
):
    kernel = MXFP8WorkspaceReduceKernel(
        m,
        n,
        split,
        algorithm=algorithm,
        threads=threads,
        vector=vector,
        persistent_waves=persistent_waves,
    )
    workspace = cute.runtime.make_fake_tensor(
        Float32,
        (split * m * n,),
        stride=(1,),
        assumed_align=16,
    )
    out = cute.runtime.make_fake_tensor(
        BFloat16,
        (m, n),
        stride=(n, 1),
        assumed_align=16,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        kernel,
        workspace,
        out,
        stream,
        options="--enable-tvm-ffi --opt-level 3 --ptxas-options '-O3 -v'",
    )


__all__ = ["MXFP8WorkspaceReduceKernel", "compile_mxfp8_workspace_reduce"]
