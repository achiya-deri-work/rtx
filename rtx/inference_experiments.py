"""Calibrated MXFP8 benchmarks for persistent inference operands."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Literal, Mapping

import torch

from .autotune.hardware import compiled_resource_metadata
from .configs import MXFP8FullyPrequantConfig, MXFP8WeightPrequantConfig
from .formats import MXFP8Tensor, make_mxfp8_tensor
from .formats.mxfp8 import mxfp8_qdata_2d, mxfp8_scales_for_kernel
from .fp8 import (
    _allocate_scales,
    _ensure_l2_fetch_granularity,
    _set_l2_fetch_granularity,
    compile_mxfp8_gemm,
    compile_mxfp8_quant,
)
from .prequant_experiments import (
    BenchmarkProtocol,
    CacheRegime,
    CandidateCompileError,
    CandidateCorrectnessError,
    PrequantBenchmarkHarness,
    ShapeSpec,
)


InferenceOperandState = Literal["weight_prequantized", "fully_prequantized"]


def _pair_key(x: torch.Tensor, weight: torch.Tensor) -> tuple[int, int]:
    return id(x), id(weight)


def _materialize(
    source: torch.Tensor,
    config,
) -> MXFP8Tensor:
    rows, k = map(int, source.shape)
    storage_k = (k + 31) // 32 * 32
    data = torch.empty(
        (rows, storage_k), dtype=torch.float8_e4m3fn, device=source.device
    )
    scales = _allocate_scales(rows, k, config.scale_layout, source.device)
    compile_mxfp8_quant(rows, k, config)(source, data, scales)
    return make_mxfp8_tensor(
        data, scales, tuple(source.shape), config.scale_layout
    )


@dataclass(slots=True)
class _InferenceRunner:
    state: InferenceOperandState
    gemm: object
    packed_pairs: Mapping[
        tuple[int, int], tuple[MXFP8Tensor | None, MXFP8Tensor]
    ]
    quant_x: object | None = None
    qx: torch.Tensor | None = None
    sx: torch.Tensor | None = None
    l2_fetch_granularity: int | None = None

    def __call__(
        self, x: torch.Tensor, weight: torch.Tensor, out: torch.Tensor
    ) -> None:
        _ensure_l2_fetch_granularity(self.l2_fetch_granularity)
        packed_x, packed_weight = self.packed_pairs[_pair_key(x, weight)]
        if self.state == "weight_prequantized":
            assert self.quant_x is not None and self.qx is not None and self.sx is not None
            self.quant_x(x, self.qx, self.sx)
            qx, sx = self.qx, self.sx
        else:
            assert packed_x is not None
            qx = mxfp8_qdata_2d(packed_x)
            sx = mxfp8_scales_for_kernel(packed_x)
        self.gemm(
            qx,
            mxfp8_qdata_2d(packed_weight),
            sx,
            mxfp8_scales_for_kernel(packed_weight),
            out,
        )


@dataclass(slots=True)
class _PreparedInferenceCandidate:
    config: MXFP8WeightPrequantConfig | MXFP8FullyPrequantConfig
    runner: _InferenceRunner
    out: torch.Tensor
    compile_ms: float
    max_abs_error: float
    compiled_resources: Mapping[str, object]


class MXFP8InferenceBenchmarkHarness(PrequantBenchmarkHarness):
    """Benchmark only per-invocation work for persistent operand states."""

    def __init__(
        self,
        state: InferenceOperandState,
        shape: ShapeSpec,
        regime: CacheRegime,
        protocol: BenchmarkProtocol,
        *,
        device: torch.device | str = "cuda",
        seed: int = 0,
    ) -> None:
        if state not in ("weight_prequantized", "fully_prequantized"):
            raise ValueError(f"unsupported inference operand state {state!r}")
        self.operand_state = state
        super().__init__(shape, regime, protocol, device=device, seed=seed)

    def _make_input_ring(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        if self.regime == "hot":
            return [(self.x, self.weight)]
        bf16_bytes = (
            self.x.numel() * self.x.element_size()
            + self.weight.numel() * self.weight.element_size()
        )
        x_packed = self.x.numel() + self.shape.m * (self.shape.k // 32)
        w_packed = self.weight.numel() + self.shape.n * (self.shape.k // 32)
        runtime_bytes = (
            self.x.numel() * self.x.element_size() + w_packed
            if self.operand_state == "weight_prequantized"
            else x_packed + w_packed
        )
        future_packed = w_packed + (
            x_packed if self.operand_state == "fully_prequantized" else 0
        )
        allocation_per_pair = bf16_bytes + future_packed
        free_bytes, _total_bytes = torch.cuda.mem_get_info(self.device)
        budget = min(
            self.protocol.max_rotation_bytes,
            int(free_bytes * 0.30),
        )
        if budget < 2 * allocation_per_pair:
            raise RuntimeError(
                "rotating inference benchmark needs two BF16+packed operand "
                f"sets ({2 * allocation_per_pair} bytes), budget is {budget}"
            )
        l2_target = max(
            runtime_bytes * 2,
            int(self._l2_bytes() * self.protocol.rotation_l2_multiple),
        )
        count = max(
            2,
            min(
                self.protocol.max_rotation_buffers,
                max(1, budget // allocation_per_pair),
                math.ceil(l2_target / runtime_bytes),
            ),
        )
        ring = [(self.x, self.weight)]
        for _ in range(1, count):
            ring.append((self.x.clone(), self.weight.clone()))
        torch.cuda.synchronize(self.device)
        return ring

    def prepare(
        self,
        config: MXFP8WeightPrequantConfig | MXFP8FullyPrequantConfig,
    ) -> _PreparedInferenceCandidate:
        rejection = config.rejection(self.problem)
        if rejection is not None:
            raise CandidateCompileError(rejection)
        previous_l2: int | None = None
        if config.l2_fetch_granularity is not None:
            previous_l2 = _set_l2_fetch_granularity(
                config.l2_fetch_granularity
            )
        started = time.monotonic()
        try:
            try:
                storage_problem = type(self.problem)(
                    self.problem.m, self.problem.n, self.problem.storage_k
                )
                gemm = compile_mxfp8_gemm(storage_problem, config.gemm)
                packed_pairs: dict[
                    tuple[int, int], tuple[MXFP8Tensor | None, MXFP8Tensor]
                ] = {}
                if isinstance(config, MXFP8WeightPrequantConfig):
                    quant_x = compile_mxfp8_quant(
                        self.problem.m, self.problem.k, config.quant_x
                    )
                    qx = torch.empty(
                        (self.problem.m, self.problem.storage_k),
                        dtype=torch.float8_e4m3fn,
                        device=self.device,
                    )
                    sx = _allocate_scales(
                        self.problem.m,
                        self.problem.k,
                        config.quant_x.scale_layout,
                        self.device,
                    )
                    for x, weight in self._inputs:
                        packed_pairs[_pair_key(x, weight)] = (
                            None,
                            _materialize(weight, config.weight_packing_quant()),
                        )
                    runner = _InferenceRunner(
                        self.operand_state,
                        gemm,
                        packed_pairs,
                        quant_x,
                        qx,
                        sx,
                        config.l2_fetch_granularity,
                    )
                else:
                    for x, weight in self._inputs:
                        packed_pairs[_pair_key(x, weight)] = (
                            _materialize(x, config.activation_packing_quant()),
                            _materialize(weight, config.weight_packing_quant()),
                        )
                    runner = _InferenceRunner(
                        self.operand_state,
                        gemm,
                        packed_pairs,
                        l2_fetch_granularity=config.l2_fetch_granularity,
                    )
                torch.cuda.synchronize(self.device)
            except Exception as exc:
                raise CandidateCompileError(
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            compile_ms = (time.monotonic() - started) * 1000
            out = torch.empty(
                (self.shape.m, self.shape.n),
                device=self.device,
                dtype=torch.bfloat16,
            )
            runner(self.x, self.weight, out)
            torch.cuda.synchronize(self.device)
            max_abs_error = float(
                (out.float() - self._expected.float()).abs().max()
            )
            if not torch.allclose(
                out,
                self._expected,
                rtol=self.protocol.correctness_rtol,
                atol=self.protocol.correctness_atol,
                equal_nan=True,
            ):
                raise CandidateCorrectnessError(
                    f"candidate differs from reference (max abs {max_abs_error})"
                )
            return _PreparedInferenceCandidate(
                config,
                runner,
                out,
                compile_ms,
                max_abs_error,
                compiled_resource_metadata(runner),
            )
        finally:
            if previous_l2 is not None:
                _set_l2_fetch_granularity(previous_l2)

    def _measure_components(
        self,
        prepared: _PreparedInferenceCandidate,
        calls: int,
        samples: int,
    ) -> dict[str, object]:
        previous_l2: int | None = None
        if prepared.config.l2_fetch_granularity is not None:
            previous_l2 = _set_l2_fetch_granularity(
                prepared.config.l2_fetch_granularity
            )
        try:
            return self._measure_components_with_l2(prepared, calls, samples)
        finally:
            if previous_l2 is not None:
                _set_l2_fetch_granularity(previous_l2)

    def _measure_components_with_l2(
        self,
        prepared: _PreparedInferenceCandidate,
        calls: int,
        samples: int,
    ) -> dict[str, object]:
        # AOT packing is intentionally absent. The authoritative E2E timing is
        # produced by the inherited calibrated measure/race implementation.
        runner = prepared.runner
        component_calls = max(1, min(calls, 1024))
        component_samples = max(3, min(samples, 5))
        results: dict[str, list[float]] = {}
        if runner.quant_x is not None:
            assert runner.qx is not None and runner.sx is not None
            results["x_quant"] = [
                self._time_callable(
                    lambda index: runner.quant_x(
                        self._inputs[index % len(self._inputs)][0],
                        runner.qx,
                        runner.sx,
                    ),
                    component_calls,
                )
                for _ in range(component_samples)
            ]
        first_x, first_w = self._inputs[0]
        packed_x, packed_w = runner.packed_pairs[_pair_key(first_x, first_w)]
        qx = runner.qx if packed_x is None else mxfp8_qdata_2d(packed_x)
        sx = (
            runner.sx
            if packed_x is None
            else mxfp8_scales_for_kernel(packed_x)
        )
        assert qx is not None and sx is not None
        results["gemm_hot_packed"] = [
            self._time_callable(
                lambda _index: runner.gemm(
                    qx,
                    mxfp8_qdata_2d(packed_w),
                    sx,
                    mxfp8_scales_for_kernel(packed_w),
                    prepared.out,
                ),
                component_calls,
            )
            for _ in range(component_samples)
        ]
        from .prequant_experiments import robust_summary

        return {
            name: {
                "timings_ms": timings,
                "summary_ms": robust_summary(
                    timings,
                    seed=len(name) ^ calls,
                    bootstrap_resamples=self.protocol.bootstrap_resamples,
                ).as_dict(),
            }
            for name, timings in results.items()
        }


class WeightPrequantBenchmarkHarness(MXFP8InferenceBenchmarkHarness):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("weight_prequantized", *args, **kwargs)


class FullyPrequantBenchmarkHarness(MXFP8InferenceBenchmarkHarness):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("fully_prequantized", *args, **kwargs)


__all__ = [
    "FullyPrequantBenchmarkHarness",
    "MXFP8InferenceBenchmarkHarness",
    "WeightPrequantBenchmarkHarness",
]
