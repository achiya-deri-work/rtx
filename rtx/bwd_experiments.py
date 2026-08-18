"""Calibrated end-to-end measurement for MXFP8 linear backward."""

from __future__ import annotations

from dataclasses import dataclass
import math
import time
from typing import Callable, Mapping

import torch

from .autotune.hardware import compiled_resource_metadata
from .fp8_bwd import _build_bwd_runner
from .kernels.mxfp8_bwd import MXFP8BwdConfig
from .prequant_experiments import (
    BenchmarkProtocol,
    CacheRegime,
    ShapeSpec,
    _nvidia_smi_snapshot,
    collect_stable_timing_samples,
    robust_summary,
    stabilize_timing_batches,
)


class BwdCandidateCompileError(RuntimeError):
    pass


class BwdCandidateCorrectnessError(RuntimeError):
    pass


@dataclass(slots=True)
class _PreparedBwdCandidate:
    config: MXFP8BwdConfig
    runner: object
    grad_x: torch.Tensor
    grad_weight: torch.Tensor
    compile_ms: float
    max_abs_error: float
    max_relative_l2_error: float
    compiled_resources: Mapping[str, object]


class BwdBenchmarkHarness:
    """Own one backward workload and produce calibrated, paired GPU timings."""

    def __init__(
        self,
        shape: ShapeSpec,
        protocol: BenchmarkProtocol,
        *,
        regime: CacheRegime = "hot",
        device: torch.device | str = "cuda",
        seed: int = 0,
    ) -> None:
        self.shape = shape
        self.problem = shape.problem
        self.protocol = protocol
        self.regime = regime
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
        self.grad_output = torch.randn(
            shape.m,
            shape.n,
            device=self.device,
            dtype=torch.bfloat16,
            generator=generator,
        )
        self._inputs = self._make_input_ring()
        self._reference = (
            (self.grad_output.float() @ self.weight.float()).to(torch.bfloat16),
            (self.grad_output.float().T @ self.x.float()).to(torch.bfloat16),
        )

    def _l2_bytes(self) -> int:
        props = torch.cuda.get_device_properties(self.device)
        return int(
            getattr(props, "L2_cache_size", getattr(props, "l2_cache_size", 0)) or 0
        )

    def _make_input_ring(
        self,
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        base = (self.grad_output, self.x, self.weight)
        if self.regime == "hot":
            return [base]
        input_bytes = sum(tensor.numel() * tensor.element_size() for tensor in base)
        free_bytes, _total_bytes = torch.cuda.mem_get_info(self.device)
        budget = min(self.protocol.max_rotation_bytes, int(free_bytes * 0.35))
        l2_target = max(
            input_bytes * 2,
            int(self._l2_bytes() * self.protocol.rotation_l2_multiple),
        )
        requested = math.ceil(l2_target / input_bytes)
        count = max(
            2,
            min(
                self.protocol.max_rotation_buffers,
                max(1, budget // input_bytes),
                requested,
            ),
        )
        ring = [base]
        for _ in range(1, count):
            ring.append(
                (self.grad_output.clone(), self.x.clone(), self.weight.clone())
            )
        torch.cuda.synchronize(self.device)
        return ring

    def prepare(self, config: MXFP8BwdConfig) -> _PreparedBwdCandidate:
        reason = config.implementation_rejection(self.problem)
        if reason is not None:
            raise BwdCandidateCompileError(reason)
        started = time.monotonic()
        try:
            runner = _build_bwd_runner(self.problem, config, self.device)
        except Exception as exc:
            raise BwdCandidateCompileError(f"{type(exc).__name__}: {exc}") from exc
        compile_ms = (time.monotonic() - started) * 1000
        grad_x = torch.empty_like(self.x)
        grad_weight = torch.empty_like(self.weight)
        runner(
            self.grad_output,
            self.x,
            self.weight,
            grad_x,
            grad_weight,
        )
        torch.cuda.synchronize(self.device)
        reference_x, reference_weight = self._reference
        max_abs_error = max(
            float((grad_x.float() - reference_x.float()).abs().max()),
            float((grad_weight.float() - reference_weight.float()).abs().max()),
        )
        relative_x = float(
            (grad_x.float() - reference_x.float()).norm()
            / reference_x.float().norm().clamp_min(1e-12)
        )
        relative_weight = float(
            (grad_weight.float() - reference_weight.float()).norm()
            / reference_weight.float().norm().clamp_min(1e-12)
        )
        max_relative_l2_error = max(relative_x, relative_weight)
        if not math.isfinite(max_relative_l2_error) or (
            max_relative_l2_error > self.protocol.correctness_rtol
        ):
            raise BwdCandidateCorrectnessError(
                "candidate differs from FP32 reference "
                f"(relative L2 {max_relative_l2_error}, max abs {max_abs_error})"
            )
        return _PreparedBwdCandidate(
            config,
            runner,
            grad_x,
            grad_weight,
            compile_ms,
            max_abs_error,
            max_relative_l2_error,
            compiled_resource_metadata(runner),
        )

    def _time_batch(
        self, prepared: _PreparedBwdCandidate, calls: int, offset: int = 0
    ) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for call in range(calls):
            grad_output, x, weight = self._inputs[(offset + call) % len(self._inputs)]
            prepared.runner(
                grad_output,
                x,
                weight,
                prepared.grad_x,
                prepared.grad_weight,
            )
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end)) / calls

    def calibrate_calls(
        self, prepared: _PreparedBwdCandidate
    ) -> tuple[int, float]:
        pilot_calls = self.protocol.min_calls_per_sample
        pilot_ms = self._time_batch(prepared, pilot_calls)
        calls = math.ceil(self.protocol.target_batch_ms / max(pilot_ms, 1e-6))
        calls = min(
            self.protocol.max_calls_per_sample,
            max(self.protocol.min_calls_per_sample, calls),
        )
        return calls, pilot_ms

    @staticmethod
    def _time_callable(function: Callable[[], None], calls: int) -> float:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(calls):
            function()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end)) / calls

    def _measure_components(
        self, prepared: _PreparedBwdCandidate, calls: int, samples: int
    ) -> dict[str, object]:
        runner = prepared.runner
        component_calls = max(1, min(calls, 1024))
        component_samples = max(3, min(samples, 5))
        operations: dict[str, Callable[[], None]] = {}
        is_quad = prepared.config.quant_schedule in ("quad", "shared_g_quad")
        if is_quad:
            operations.update(
                {
                    f"{prepared.config.quant_schedule}_quant": lambda: runner.quantize_quad(
                        self.grad_output, self.x, self.weight
                    ),
                    "dx_gemm_hot_materialized": lambda: runner.dx.matmul(
                        prepared.grad_x
                    ),
                    "dw_gemm_hot_materialized": lambda: runner.dw.matmul(
                        prepared.grad_weight
                    ),
                }
            )
        elif prepared.config.dx.backend == "fused":
            operations["dx_fused_quant_mma"] = lambda: runner.dx(
                self.grad_output, self.weight, prepared.grad_x
            )
        else:
            operations.update(
                {
                    "dx_quant": lambda: runner.dx.quantize(
                        self.grad_output, self.weight
                    ),
                    "dx_gemm_hot_materialized": lambda: runner.dx.matmul(
                        prepared.grad_x
                    ),
                }
            )
        if is_quad:
            pass
        elif prepared.config.dw.backend == "fused":
            operations["dw_fused_quant_mma"] = lambda: runner.dw(
                self.grad_output, self.x, prepared.grad_weight
            )
        else:
            operations.update(
                {
                    "dw_quant": lambda: runner.dw.quantize(
                        self.grad_output, self.x
                    ),
                    "dw_gemm_hot_materialized": lambda: runner.dw.matmul(
                        prepared.grad_weight
                    ),
                }
            )
        result: dict[str, object] = {}
        for name, operation in operations.items():
            timings = [
                self._time_callable(operation, component_calls)
                for _ in range(component_samples)
            ]
            result[name] = {
                "timings_ms": timings,
                "summary_ms": robust_summary(
                    timings,
                    seed=len(name) ^ calls,
                    bootstrap_resamples=self.protocol.bootstrap_resamples,
                ).as_dict(),
            }
        return result

    def measure(
        self,
        config: MXFP8BwdConfig,
        *,
        samples: int,
        seed: int,
        components: bool = False,
    ) -> dict[str, object]:
        telemetry_before = (
            _nvidia_smi_snapshot(self.device.index or torch.cuda.current_device())
            if self.protocol.telemetry
            else {"available": False, "disabled": True}
        )
        started = time.monotonic()
        try:
            prepared = self.prepare(config)
        except Exception as exc:
            if isinstance(exc, BwdCandidateCompileError):
                status = "compile_error"
            elif isinstance(exc, BwdCandidateCorrectnessError):
                status = "correctness_error"
            else:
                status = "runtime_error"
            return {
                "status": status,
                "error": f"{type(exc).__name__}: {exc}"[:4000],
                "elapsed_s": time.monotonic() - started,
                "telemetry_before": telemetry_before,
            }
        for _ in range(self.protocol.warmup_calls):
            prepared.runner(
                self.grad_output,
                self.x,
                self.weight,
                prepared.grad_x,
                prepared.grad_weight,
            )
        torch.cuda.synchronize(self.device)
        calls, pilot_ms = self.calibrate_calls(prepared)
        timings, collection = collect_stable_timing_samples(
            lambda sample: self._time_batch(prepared, calls, sample * calls),
            self.protocol,
            requested_samples=samples,
            stabilize=lambda attempt: stabilize_timing_batches(
                lambda batch: self._time_batch(
                    prepared,
                    calls,
                    -((attempt + 1) * self.protocol.stabilization_max_batches + batch)
                    * calls,
                )
                * calls,
                self.protocol,
            ),
            telemetry=(
                lambda: _nvidia_smi_snapshot(
                    self.device.index or torch.cuda.current_device()
                )
            )
            if self.protocol.telemetry
            else None,
        )
        telemetry_after = (
            _nvidia_smi_snapshot(self.device.index or torch.cuda.current_device())
            if self.protocol.telemetry
            else {"available": False, "disabled": True}
        )
        return {
            "status": "ok",
            "compile_ms": prepared.compile_ms,
            "max_abs_error": prepared.max_abs_error,
            "max_relative_l2_error": prepared.max_relative_l2_error,
            "compiled_resources": prepared.compiled_resources,
            "calls_per_sample": calls,
            "pilot_ms_per_call": pilot_ms,
            "rotation_buffers": len(self._inputs),
            "timings_ms": timings,
            "sampling": {
                **self.protocol.sampling_metadata(
                    timings, requested_samples=samples
                ),
                "collection": collection,
            },
            "summary_ms": robust_summary(
                timings,
                seed=seed,
                bootstrap_resamples=self.protocol.bootstrap_resamples,
            ).as_dict(),
            "components": (
                self._measure_components(prepared, calls, samples)
                if components
                else None
            ),
            "elapsed_s": time.monotonic() - started,
            "telemetry_before": telemetry_before,
            "telemetry_after": telemetry_after,
        }

    def race(
        self,
        incumbent: MXFP8BwdConfig,
        challenger: MXFP8BwdConfig,
        *,
        seed: int,
    ) -> dict[str, object]:
        try:
            a = self.prepare(incumbent)
            b = self.prepare(challenger)
        except Exception as exc:
            return {
                "status": "prepare_error",
                "error": f"{type(exc).__name__}: {exc}"[:4000],
            }
        for _ in range(self.protocol.warmup_calls):
            a.runner(
                self.grad_output,
                self.x,
                self.weight,
                a.grad_x,
                a.grad_weight,
            )
            b.runner(
                self.grad_output,
                self.x,
                self.weight,
                b.grad_x,
                b.grad_weight,
            )
        calls_a, _ = self.calibrate_calls(a)
        calls_b, _ = self.calibrate_calls(b)
        stabilization = stabilize_timing_batches(
            lambda batch: (
                self._time_batch(a, calls_a, -(batch + 1) * calls_a) * calls_a
                + self._time_batch(b, calls_b, -(batch + 1) * calls_b) * calls_b
            ),
            self.protocol,
        )
        a_times: list[float] = []
        b_times: list[float] = []
        for round_index in range(self.protocol.race_rounds):
            if round_index % 2:
                b_time = self._time_batch(b, calls_b, round_index * calls_b)
                a_time = self._time_batch(a, calls_a, round_index * calls_a)
            else:
                a_time = self._time_batch(a, calls_a, round_index * calls_a)
                b_time = self._time_batch(b, calls_b, round_index * calls_b)
            a_times.append(a_time)
            b_times.append(b_time)
            if self.protocol.race_complete(a_times, b_times):
                break
        speedups = [
            (a_time - b_time) / a_time
            for a_time, b_time in zip(a_times, b_times)
        ]
        summary = robust_summary(
            speedups,
            seed=seed,
            bootstrap_resamples=self.protocol.bootstrap_resamples,
        )
        threshold = self.protocol.practical_threshold
        if summary.ci_low > threshold:
            decision = "challenger"
        elif summary.ci_high < -threshold:
            decision = "incumbent"
        else:
            decision = "tie"
        return {
            "status": "ok",
            "decision": decision,
            "practical_threshold": threshold,
            "incumbent_timings_ms": a_times,
            "challenger_timings_ms": b_times,
            "paired_speedup": summary.as_dict(),
            "incumbent_calls_per_sample": calls_a,
            "challenger_calls_per_sample": calls_b,
            "rotation_buffers": len(self._inputs),
            "sampling": self.protocol.race_sampling_metadata(a_times, b_times),
            "stabilization": stabilization,
        }


__all__ = [
    "BwdBenchmarkHarness",
    "BwdCandidateCompileError",
    "BwdCandidateCorrectnessError",
]
