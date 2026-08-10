"""Calibrated NVFP4 benchmarks for persistent inference operand states."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Literal, Mapping

import torch

from .autotune.hardware import compiled_resource_metadata
from .configs.nvfp4 import (
    NVFP4DynamicConfig,
    NVFP4FullyPrequantConfig,
    NVFP4Problem,
    NVFP4WeightPrequantConfig,
)
from .formats import NVFP4Tensor
from .formats.nvfp4 import nvfp4_tensor_scale
from .fp4 import (
    NVFP4ForwardConfig,
    _make_block_dynamic_runner,
    _current_tensor_scale,
    _packed_fp4_view,
    compile_nvfp4_gemm,
    compile_nvfp4_quant,
    quantize_nvfp4,
)
from .fp8 import _ensure_l2_fetch_granularity, _set_l2_fetch_granularity
from .prequant_experiments import (
    BenchmarkProtocol,
    CacheRegime,
    CandidateCompileError,
    CandidateCorrectnessError,
    PrequantBenchmarkHarness,
    ShapeSpec,
)


InferenceOperandState = Literal["weight_prequantized", "fully_prequantized"]


@dataclass(slots=True)
class _PreparedNVFP4Dynamic:
    config: NVFP4DynamicConfig
    runner: object
    out: torch.Tensor
    compile_ms: float
    max_abs_error: float
    compiled_resources: Mapping[str, object]


class NVFP4DynamicBenchmarkHarness(PrequantBenchmarkHarness):
    """Measure block-scale quantize-both plus native NVFP4 GEMM schedules."""

    def __init__(
        self,
        shape: ShapeSpec,
        regime: CacheRegime,
        protocol: BenchmarkProtocol,
        *,
        device: torch.device | str = "cuda",
        seed: int = 0,
    ) -> None:
        self.shape = shape
        self.problem = NVFP4Problem(shape.m, shape.n, shape.k)
        self.regime = regime
        self.protocol = protocol
        self.device = torch.device(device)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        self.x = torch.randn(
            shape.m,
            shape.k,
            device=self.device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        self.weight = torch.randn(
            shape.n,
            shape.k,
            device=self.device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        self._inputs = self._make_input_ring()
        reference = self._build_runner(NVFP4DynamicConfig())
        expected = torch.empty(
            (shape.m, shape.n), device=self.device, dtype=torch.bfloat16
        )
        reference(self.x, self.weight, expected)
        torch.cuda.synchronize(self.device)
        self._expected = expected.clone()

    def _build_runner(
        self, config: NVFP4DynamicConfig
    ) -> object:
        return _make_block_dynamic_runner(
            self.problem,
            NVFP4ForwardConfig.from_materialized(config),
            self.device,
        )

    def prepare(self, config: NVFP4DynamicConfig) -> _PreparedNVFP4Dynamic:
        rejection = config.rejection(self.problem)
        if rejection is not None:
            raise CandidateCompileError(rejection)
        previous_l2: int | None = None
        if config.l2_fetch_granularity is not None:
            previous_l2 = _set_l2_fetch_granularity(config.l2_fetch_granularity)
        started = time.monotonic()
        try:
            try:
                runner = self._build_runner(config)
                torch.cuda.synchronize(self.device)
            except Exception as exc:
                raise CandidateCompileError(
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            compile_ms = (time.monotonic() - started) * 1000
            out = torch.empty_like(self._expected)
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
            return _PreparedNVFP4Dynamic(
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

    def _measure_components(self, prepared, calls: int, samples: int):
        return {}


def _pair_key(x: torch.Tensor, weight: torch.Tensor) -> tuple[int, int]:
    return id(x), id(weight)


@dataclass(slots=True)
class _NVFP4InferenceRunner:
    state: InferenceOperandState
    gemm: object
    packed_pairs: Mapping[
        tuple[int, int], tuple[NVFP4Tensor | None, NVFP4Tensor]
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
        weight_scale = nvfp4_tensor_scale(packed_weight).reshape(1)
        if self.state == "weight_prequantized":
            assert self.quant_x is not None
            assert self.qx is not None and self.sx is not None
            x_scale = _current_tensor_scale(x)
            self.quant_x(x, self.qx, self.sx, x_scale)
            qx = _packed_fp4_view(self.qx)
            sx = self.sx
        else:
            assert packed_x is not None
            x_scale = nvfp4_tensor_scale(packed_x).reshape(1)
            qx = packed_x.qdata
            sx = packed_x.scale
        output_scale = x_scale * weight_scale
        self.gemm(
            qx,
            packed_weight.qdata,
            sx,
            packed_weight.scale,
            out,
            output_scale,
        )
        out._nvfp4_scale_lifetime = (x_scale, weight_scale, output_scale)


@dataclass(slots=True)
class _PreparedNVFP4Inference:
    config: NVFP4WeightPrequantConfig | NVFP4FullyPrequantConfig
    runner: _NVFP4InferenceRunner
    out: torch.Tensor
    compile_ms: float
    max_abs_error: float
    compiled_resources: Mapping[str, object]


class NVFP4InferenceBenchmarkHarness(PrequantBenchmarkHarness):
    """Measure the exact per-call path for one persistent operand state."""

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
            raise ValueError(f"unsupported NVFP4 inference state {state!r}")
        self.operand_state = state
        self.shape = shape
        self.problem = NVFP4Problem(shape.m, shape.n, shape.k)
        self.regime = regime
        self.protocol = protocol
        self.device = torch.device(device)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(seed)
        self.x = torch.randn(
            shape.m,
            shape.k,
            device=self.device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        self.weight = torch.randn(
            shape.n,
            shape.k,
            device=self.device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        self._inputs = self._make_input_ring()
        # AOT operands are a property of the inference context, not of a
        # candidate schedule. Pack each rotating input exactly once so search
        # time measures compiler/schedule work instead of repeating free AOT
        # preparation hundreds of times.
        self._packed_pairs = self._pack_inputs()
        initial = (
            NVFP4WeightPrequantConfig()
            if state == "weight_prequantized"
            else NVFP4FullyPrequantConfig()
        )
        runner = self._build_runner(initial)
        expected = torch.empty(
            (shape.m, shape.n), device=self.device, dtype=torch.bfloat16
        )
        runner(self.x, self.weight, expected)
        torch.cuda.synchronize(self.device)
        self._expected = expected.clone()

    def _pack_inputs(
        self,
    ) -> dict[tuple[int, int], tuple[NVFP4Tensor | None, NVFP4Tensor]]:
        packed_pairs = {}
        for x, weight in self._inputs:
            packed_pairs[_pair_key(x, weight)] = (
                quantize_nvfp4(x)
                if self.operand_state == "fully_prequantized"
                else None,
                quantize_nvfp4(weight),
            )
        return packed_pairs

    def _build_runner(
        self,
        config: NVFP4WeightPrequantConfig | NVFP4FullyPrequantConfig,
    ) -> _NVFP4InferenceRunner:
        gemm = compile_nvfp4_gemm(self.problem, config.gemm)
        if isinstance(config, NVFP4WeightPrequantConfig):
            qx = torch.empty(
                (self.problem.m, self.problem.k // 2),
                dtype=torch.uint8,
                device=self.device,
            )
            sx = torch.empty(
                (self.problem.m, self.problem.k // 16),
                dtype=torch.float8_e4m3fn,
                device=self.device,
            )
            return _NVFP4InferenceRunner(
                self.operand_state,
                gemm,
                self._packed_pairs,
                compile_nvfp4_quant(
                    self.problem.m, self.problem.k, config.quant_x
                ),
                qx,
                sx,
                config.l2_fetch_granularity,
            )
        return _NVFP4InferenceRunner(
            self.operand_state,
            gemm,
            self._packed_pairs,
            l2_fetch_granularity=config.l2_fetch_granularity,
        )

    def prepare(
        self,
        config: NVFP4WeightPrequantConfig | NVFP4FullyPrequantConfig,
    ) -> _PreparedNVFP4Inference:
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
                runner = self._build_runner(config)
                torch.cuda.synchronize(self.device)
            except Exception as exc:
                raise CandidateCompileError(
                    f"{type(exc).__name__}: {exc}"
                ) from exc
            compile_ms = (time.monotonic() - started) * 1000
            out = torch.empty_like(self._expected)
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
            return _PreparedNVFP4Inference(
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

    def _measure_components(self, prepared, calls: int, samples: int):
        # End-to-end timing includes current X amax, quantization, and GEMM.
        # Isolated components would change allocation/lifetime behavior and are
        # therefore intentionally omitted for this stateful inference path.
        return {}


class NVFP4WeightPrequantBenchmarkHarness(NVFP4InferenceBenchmarkHarness):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("weight_prequantized", *args, **kwargs)


class NVFP4FullyPrequantBenchmarkHarness(NVFP4InferenceBenchmarkHarness):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__("fully_prequantized", *args, **kwargs)


__all__ = [
    "NVFP4DynamicBenchmarkHarness",
    "NVFP4FullyPrequantBenchmarkHarness",
    "NVFP4InferenceBenchmarkHarness",
    "NVFP4WeightPrequantBenchmarkHarness",
]
