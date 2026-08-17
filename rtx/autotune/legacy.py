"""Persistent coordinate-descent autotuning for RTX low-precision kernels.

The tuner is intentionally independent of the kernel implementation: it records
every rejected, failed, incorrect, or timed candidate and writes after every
trial.  A killed process therefore resumes from the last completed coordinate
instead of losing a long tuning run.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import statistics
import tempfile
import time
from typing import Callable, Iterable, Iterator, Mapping, TextIO
import uuid

import torch

from ..kernels.mxfp8 import (
    DEFAULT_MXFP8_FWD_CONFIG,
    FWD_COORDINATE_ORDER,
    FWD_SEARCH_SPACE,
    MXFP8_FWD_KERNEL_REVISION,
    MXFP8FwdConfig,
    MXFP8Problem,
    fwd_config_from_dict,
    fwd_config_id,
    fwd_config_to_dict,
    fwd_search_space_digest,
    normalize_fwd_config,
)
from ..runtime import load_kernel_symbol
from .outcomes import TrialOutcome, TrialStatus


def compile_mxfp8_fwd(*args, **kwargs):
    return load_kernel_symbol("mxfp8_fwd", "compile_mxfp8_fwd")(*args, **kwargs)

try:
    import fcntl
except ImportError:  # pragma: no cover - this project targets Linux/CUDA.
    fcntl = None


SCHEMA_VERSION = 1
KERNEL_NAME = "mxfp8_fwd"
def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _safe_package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def default_cache_dir() -> Path:
    override = os.getenv("RTX_AUTOTUNE_CACHE_DIR")
    if override:
        return Path(override).expanduser()
    xdg = os.getenv("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "rtx" / "autotune"


@dataclass(frozen=True, slots=True)
class DeviceFingerprint:
    device_index: int
    name: str
    capability: tuple[int, int]
    total_memory: int
    multiprocessor_count: int
    uuid: str | None
    torch_version: str
    torch_cuda_version: str | None
    cutlass_dsl_version: str | None
    cuda_python_version: str | None
    python_version: str
    platform: str

    @classmethod
    def current(cls, device: torch.device | str | int | None = None) -> "DeviceFingerprint":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA must be visible to identify an autotuning target")
        if isinstance(device, int):
            resolved = torch.device("cuda", device)
        else:
            resolved = torch.device("cuda" if device is None else device)
        index = torch.cuda.current_device() if resolved.index is None else resolved.index
        props = torch.cuda.get_device_properties(index)
        device_uuid = getattr(props, "uuid", None)
        return cls(
            device_index=index,
            name=props.name,
            capability=(props.major, props.minor),
            total_memory=int(props.total_memory),
            multiprocessor_count=int(props.multi_processor_count),
            uuid=None if device_uuid is None else str(device_uuid),
            torch_version=str(torch.__version__),
            torch_cuda_version=torch.version.cuda,
            cutlass_dsl_version=_safe_package_version("nvidia-cutlass-dsl"),
            cuda_python_version=_safe_package_version("cuda-python"),
            python_version=platform.python_version(),
            platform=platform.platform(),
        )

    def as_dict(self) -> dict[str, object]:
        value = asdict(self)
        value["capability"] = list(self.capability)
        return value

    @property
    def identifier(self) -> str:
        return hashlib.sha256(_canonical_json(self.as_dict()).encode()).hexdigest()[:20]


@dataclass(frozen=True, slots=True)
class CoordinateDescentPolicy:
    time_budget_s: float = 1800.0
    max_trials: int | None = None
    max_passes: int = 4
    restarts: int = 1
    warmup: int = 5
    samples: int = 9
    calls_per_sample: int = 5
    min_improvement: float = 0.002
    correctness_rtol: float = 2e-2
    correctness_atol: float = 2e-1
    correctness: Literal["baseline", "torch", "none"] = "baseline"
    resume: bool = True
    force: bool = False
    randomize_coordinates: bool = False
    seed: int = 0
    coordinate_order: tuple[str, ...] = FWD_COORDINATE_ORDER

    def __post_init__(self) -> None:
        if self.time_budget_s <= 0:
            raise ValueError("time_budget_s must be positive")
        if self.max_trials is not None and self.max_trials <= 0:
            raise ValueError("max_trials must be positive when set")
        if self.max_passes <= 0:
            raise ValueError("max_passes must be positive")
        if self.restarts <= 0:
            raise ValueError("restarts must be positive")
        if self.warmup < 0 or self.samples <= 0 or self.calls_per_sample <= 0:
            raise ValueError("warmup must be nonnegative and timing counts positive")
        if not 0 <= self.min_improvement < 1:
            raise ValueError("min_improvement must be in [0, 1)")


@dataclass(frozen=True, slots=True)
class TuningResult:
    config: MXFP8FwdConfig
    median_ms: float
    database_path: Path
    session_id: str
    evaluated_trials: int
    reused_trials: int
    elapsed_s: float


class JsonTuningDatabase:
    """One versioned, atomically replaced JSON document per problem/device."""

    def __init__(
        self,
        root: Path | str | None,
        fingerprint: DeviceFingerprint,
        problem: MXFP8Problem,
        axes: Mapping[str, Iterable[object]],
    ) -> None:
        self.root = default_cache_dir() if root is None else Path(root).expanduser()
        self.fingerprint = fingerprint
        self.problem = problem
        self.axes = {name: tuple(values) for name, values in axes.items()}
        self.search_digest = fwd_search_space_digest(self.axes)
        self.path = (
            self.root
            / KERNEL_NAME
            / fingerprint.identifier
            / (
                f"m{problem.m}_n{problem.n}_k{problem.k}_"
                f"{self.search_digest[:12]}.json"
            )
        )
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")

    def _new_document(self) -> dict[str, object]:
        now = _utc_now()
        return {
            "schema_version": SCHEMA_VERSION,
            "kernel": KERNEL_NAME,
            "kernel_revision": MXFP8_FWD_KERNEL_REVISION,
            "created_at": now,
            "updated_at": now,
            "fingerprint": self.fingerprint.as_dict(),
            "problem": asdict(self.problem),
            "search_space_digest": self.search_digest,
            "search_space": {name: list(values) for name, values in self.axes.items()},
            "coordinate_order": list(self.axes),
            "best": None,
            "trials": {},
            "sessions": [],
        }

    @contextmanager
    def _locked(self) -> Iterator[None]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_file:
            if fcntl is not None:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _read_unlocked(self) -> dict[str, object]:
        if not self.path.exists():
            return self._new_document()
        with self.path.open("r", encoding="utf-8") as source:
            document = json.load(source)
        expected = (
            document.get("schema_version") == SCHEMA_VERSION
            and document.get("kernel") == KERNEL_NAME
            and document.get("kernel_revision") == MXFP8_FWD_KERNEL_REVISION
            and document.get("search_space_digest") == self.search_digest
            and document.get("problem") == asdict(self.problem)
            and document.get("fingerprint") == self.fingerprint.as_dict()
        )
        if not expected:
            raise RuntimeError(
                f"incompatible autotuning database at {self.path}; move it aside "
                "or use a different cache directory"
            )
        return document

    def _write_unlocked(self, document: dict[str, object]) -> None:
        document["updated_at"] = _utc_now()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.path.parent,
                prefix=self.path.name + ".",
                suffix=".tmp",
                delete=False,
            ) as sink:
                temp_name = sink.name
                json.dump(document, sink, indent=2, sort_keys=True)
                sink.write("\n")
                sink.flush()
                os.fsync(sink.fileno())
            os.replace(temp_name, self.path)
            directory_fd = os.open(self.path.parent, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if temp_name is not None and os.path.exists(temp_name):
                os.unlink(temp_name)

    def read(self) -> dict[str, object]:
        with self._locked():
            return self._read_unlocked()

    def start_session(self, policy: CoordinateDescentPolicy) -> str:
        session_id = uuid.uuid4().hex
        with self._locked():
            document = self._read_unlocked()
            sessions = document["sessions"]
            assert isinstance(sessions, list)
            sessions.append(
                {
                    "id": session_id,
                    "started_at": _utc_now(),
                    "finished_at": None,
                    "status": "running",
                    "policy": asdict(policy),
                }
            )
            self._write_unlocked(document)
        return session_id

    def finish_session(
        self,
        session_id: str,
        *,
        status: Literal["complete", "budget_exhausted", "failed"],
        best_config_id: str | None,
        elapsed_s: float,
    ) -> None:
        with self._locked():
            document = self._read_unlocked()
            sessions = document["sessions"]
            assert isinstance(sessions, list)
            for session in reversed(sessions):
                if session.get("id") == session_id:
                    session.update(
                        finished_at=_utc_now(),
                        status=status,
                        best_config_id=best_config_id,
                        elapsed_s=elapsed_s,
                    )
                    break
            self._write_unlocked(document)

    def get_trial(self, config: MXFP8FwdConfig) -> dict[str, object] | None:
        document = self.read()
        trials = document["trials"]
        assert isinstance(trials, dict)
        trial = trials.get(fwd_config_id(config))
        return None if trial is None else dict(trial)

    def record_trial(
        self,
        config: MXFP8FwdConfig,
        outcome: TrialOutcome,
        *,
        session_id: str,
        pass_index: int,
        coordinate: str,
        coordinate_value: object,
    ) -> None:
        config_id = fwd_config_id(config)
        with self._locked():
            document = self._read_unlocked()
            trials = document["trials"]
            assert isinstance(trials, dict)
            previous = trials.get(config_id)
            attempt = 1 if previous is None else int(previous.get("attempt", 0)) + 1
            history: list[dict[str, object]] = []
            if previous is not None:
                history.extend(previous.get("history", []))
                history.append(
                    {
                        key: value
                        for key, value in previous.items()
                        if key not in {"config", "history"}
                    }
                )
            trial: dict[str, object] = {
                "config_id": config_id,
                "config": fwd_config_to_dict(config),
                "attempt": attempt,
                "recorded_at": _utc_now(),
                "session_id": session_id,
                "pass_index": pass_index,
                "coordinate": coordinate,
                "coordinate_value": coordinate_value,
                "history": history,
                **outcome.as_dict(),
            }
            trials[config_id] = trial
            successful = [
                item
                for item in trials.values()
                if item.get("status") == "ok" and item.get("median_ms") is not None
            ]
            if successful:
                winner = min(successful, key=lambda item: float(item["median_ms"]))
                document["best"] = {
                    "config_id": winner["config_id"],
                    "config": winner["config"],
                    "median_ms": winner["median_ms"],
                    "recorded_at": winner["recorded_at"],
                }
            else:
                document["best"] = None
            self._write_unlocked(document)

    def best(self) -> tuple[MXFP8FwdConfig, float] | None:
        document = self.read()
        best = document.get("best")
        if best is None:
            return None
        return fwd_config_from_dict(best["config"]), float(best["median_ms"])

    def select_best(
        self,
        config: MXFP8FwdConfig,
        median_ms: float,
        *,
        session_id: str,
    ) -> None:
        """Persist the configuration accepted by the search policy.

        ``record_trial`` maintains a provisional raw minimum so interrupted
        sessions remain resumable.  A completed coordinate-descent step calls
        this method because its minimum-improvement threshold can intentionally
        reject tiny, noise-sized timing deltas.
        """

        config_id = fwd_config_id(config)
        with self._locked():
            document = self._read_unlocked()
            trials = document["trials"]
            assert isinstance(trials, dict)
            trial = trials.get(config_id)
            if trial is None or trial.get("status") != "ok":
                raise RuntimeError(
                    f"cannot select unrecorded or unsuccessful config {config_id}"
                )
            document["best"] = {
                "config_id": config_id,
                "config": fwd_config_to_dict(config),
                "median_ms": median_ms,
                "recorded_at": trial["recorded_at"],
                "selected_at": _utc_now(),
                "selected_by_session": session_id,
            }
            self._write_unlocked(document)


Evaluator = Callable[[MXFP8FwdConfig], TrialOutcome]
Validator = Callable[[MXFP8FwdConfig, MXFP8Problem], str | None]
ProgressCallback = Callable[[str], None]


class CoordinateDescentTuner:
    def __init__(
        self,
        problem: MXFP8Problem,
        evaluator: Evaluator,
        database: JsonTuningDatabase,
        policy: CoordinateDescentPolicy,
        *,
        axes: Mapping[str, Iterable[object]] | None = None,
        architecture_validator: Validator | None = None,
        implementation_validator: Validator | None = None,
        progress: ProgressCallback | None = None,
    ) -> None:
        self.problem = problem
        self.evaluator = evaluator
        self.database = database
        self.policy = policy
        self.axes = {
            name: tuple(values)
            for name, values in (FWD_SEARCH_SPACE if axes is None else axes).items()
        }
        self.architecture_validator = architecture_validator or (
            lambda cfg, prob: cfg.architecture_rejection(prob)
        )
        self.implementation_validator = implementation_validator or (
            lambda cfg, prob: cfg.implementation_rejection(prob)
        )
        self.progress = progress
        missing = set(policy.coordinate_order).difference(self.axes)
        if missing:
            raise ValueError(f"coordinate order contains missing axes: {sorted(missing)}")
        self.evaluated_trials = 0
        self.reused_trials = 0
        self._start_time = 0.0
        self._session_id = ""
        self._session_outcomes: dict[str, TrialOutcome] = {}

    def _log(self, message: str) -> None:
        if self.progress is not None:
            elapsed = 0.0 if self._start_time == 0 else time.monotonic() - self._start_time
            self.progress(f"[{elapsed:8.2f}s] {message}")

    def _budget_exhausted(self) -> bool:
        return (
            time.monotonic() - self._start_time >= self.policy.time_budget_s
            or (
                self.policy.max_trials is not None
                and self.evaluated_trials >= self.policy.max_trials
            )
        )

    def _make_restart_seed(
        self,
        initial: MXFP8FwdConfig,
        rng: random.Random,
    ) -> MXFP8FwdConfig:
        """Build a deterministic random legal seed one accepted move at a time.

        Drawing a full Cartesian point produces mostly impossible schedules in
        this coupled search space.  Incremental admission preserves compound
        tile/warp normalization and still moves restarts into different basins.
        """

        candidate = initial
        coordinates = list(self.policy.coordinate_order)
        rng.shuffle(coordinates)
        for coordinate in coordinates:
            values = list(self.axes[coordinate])
            rng.shuffle(values)
            for value in values:
                proposal = normalize_fwd_config(
                    candidate, **{coordinate: value}
                )
                if self.architecture_validator(proposal, self.problem) is not None:
                    continue
                if self.implementation_validator(proposal, self.problem) is not None:
                    continue
                candidate = proposal
                break
        return candidate

    def _evaluate(
        self,
        config: MXFP8FwdConfig,
        *,
        pass_index: int,
        coordinate: str,
        coordinate_value: object,
    ) -> TrialOutcome:
        config_id = fwd_config_id(config)
        memoized = self._session_outcomes.get(config_id)
        if memoized is not None:
            self.reused_trials += 1
            score = (
                ""
                if memoized.median_ms is None
                else f" median={memoized.median_ms:.6f}ms"
            )
            self._log(
                f"MEMO  {config_id} {coordinate}={coordinate_value!r} "
                f"status={memoized.status}{score}"
            )
            return memoized

        if self.policy.resume and not self.policy.force:
            cached = self.database.get_trial(config)
            if cached is not None:
                self.reused_trials += 1
                outcome = TrialOutcome.from_dict(cached)
                self._session_outcomes[config_id] = outcome
                score = (
                    ""
                    if outcome.median_ms is None
                    else f" median={outcome.median_ms:.6f}ms"
                )
                self._log(
                    f"REUSE {config_id} {coordinate}={coordinate_value!r} "
                    f"status={outcome.status}{score}"
                )
                return outcome

        architecture_reason = self.architecture_validator(config, self.problem)
        if architecture_reason is not None:
            outcome = TrialOutcome("architecture_rejected", error=architecture_reason)
        else:
            implementation_reason = self.implementation_validator(config, self.problem)
            if implementation_reason is not None:
                outcome = TrialOutcome(
                    "implementation_rejected", error=implementation_reason
                )
            else:
                self._log(
                    f"RUN   {config_id} {coordinate}={coordinate_value!r} "
                    "compile -> correctness -> benchmark"
                )
                try:
                    outcome = self.evaluator(config)
                except Exception as exc:  # Keep a long run alive after one bad candidate.
                    outcome = TrialOutcome(
                        "runtime_error", error=f"{type(exc).__name__}: {exc}"[:4000]
                    )
        self.evaluated_trials += 1
        self._session_outcomes[config_id] = outcome
        self.database.record_trial(
            config,
            outcome,
            session_id=self._session_id,
            pass_index=pass_index,
            coordinate=coordinate,
            coordinate_value=coordinate_value,
        )
        details = []
        if outcome.compile_ms is not None:
            details.append(f"compile={outcome.compile_ms:.2f}ms")
        if outcome.median_ms is not None:
            details.append(f"median={outcome.median_ms:.6f}ms")
        if outcome.max_abs_error is not None:
            details.append(f"max_error={outcome.max_abs_error:g}")
        if outcome.error is not None:
            details.append(f"reason={outcome.error}")
        suffix = " " + " ".join(details) if details else ""
        self._log(
            f"SAVE  {config_id} {coordinate}={coordinate_value!r} "
            f"status={outcome.status}{suffix}"
        )
        return outcome

    def tune(
        self,
        initial: MXFP8FwdConfig = DEFAULT_MXFP8_FWD_CONFIG,
    ) -> TuningResult:
        self.problem.validate()
        self._start_time = time.monotonic()
        self._session_id = self.database.start_session(self.policy)
        self._log(
            f"START session={self._session_id} problem="
            f"[{self.problem.m},{self.problem.n},{self.problem.k}] "
            f"budget={self.policy.time_budget_s:.1f}s "
            f"passes={self.policy.max_passes} restarts={self.policy.restarts}"
        )
        self._log(f"DB    {self.database.path}")
        session_status: Literal["complete", "budget_exhausted", "failed"] = "failed"
        best_id: str | None = None
        try:
            cached_best = self.database.best() if self.policy.resume else None
            current = cached_best[0] if cached_best is not None else initial
            initial_outcome = self._evaluate(
                current,
                pass_index=-1,
                coordinate="initial",
                coordinate_value=None,
            )
            if not initial_outcome.successful:
                raise RuntimeError(
                    f"initial MXFP8 tuning configuration failed: "
                    f"{initial_outcome.status}: {initial_outcome.error}"
                )
            current_score = float(initial_outcome.median_ms)
            self._log(
                f"SEED  config={fwd_config_id(current)} median={current_score:.6f}ms"
            )

            global_best = current
            global_score = current_score
            rng = random.Random(self.policy.seed)
            budget_stopped = False
            for restart_index in range(self.policy.restarts):
                if restart_index:
                    current = self._make_restart_seed(initial, rng)
                    seed_outcome = self._evaluate(
                        current,
                        pass_index=-(restart_index + 1),
                        coordinate="restart_seed",
                        coordinate_value=restart_index,
                    )
                    if not seed_outcome.successful:
                        self._log(
                            f"SKIP  restart={restart_index + 1} "
                            f"seed={fwd_config_id(current)} status={seed_outcome.status}"
                        )
                        continue
                    current_score = float(seed_outcome.median_ms)
                self._log(
                    f"BASIN {restart_index + 1}/{self.policy.restarts} "
                    f"seed={fwd_config_id(current)} median={current_score:.6f}ms"
                )

                for local_pass in range(self.policy.max_passes):
                    pass_index = restart_index * self.policy.max_passes + local_pass
                    improved_in_pass = False
                    self._log(
                        f"PASS  {local_pass + 1}/{self.policy.max_passes} "
                        f"restart={restart_index + 1}/{self.policy.restarts} "
                        f"current={fwd_config_id(current)} "
                        f"median={current_score:.6f}ms"
                    )
                    coordinates = list(self.policy.coordinate_order)
                    if self.policy.randomize_coordinates:
                        rng.shuffle(coordinates)
                    for coordinate in coordinates:
                        if self._budget_exhausted():
                            session_status = "budget_exhausted"
                            budget_stopped = True
                            self._log("STOP  time budget exhausted")
                            break
                        self._log(
                            f"AXIS  {coordinate} "
                            f"values={list(self.axes[coordinate])!r} "
                            f"current={getattr(current, coordinate)!r}"
                        )
                        axis_best = current
                        axis_best_score = current_score
                        values = list(self.axes[coordinate])
                        current_value = getattr(current, coordinate)
                        if current_value not in values:
                            values.insert(0, current_value)
                        for value in values:
                            if self._budget_exhausted():
                                budget_stopped = True
                                session_status = "budget_exhausted"
                                break
                            candidate = normalize_fwd_config(
                                current, **{coordinate: value}
                            )
                            outcome = self._evaluate(
                                candidate,
                                pass_index=pass_index,
                                coordinate=coordinate,
                                coordinate_value=value,
                            )
                            if not outcome.successful:
                                continue
                            score = float(outcome.median_ms)
                            threshold = axis_best_score * (
                                1.0 - self.policy.min_improvement
                            )
                            if score < threshold:
                                axis_best = candidate
                                axis_best_score = score
                        if axis_best != current:
                            current = axis_best
                            current_score = axis_best_score
                            improved_in_pass = True
                            self._log(
                                f"BEST  axis={coordinate} "
                                f"config={fwd_config_id(current)} "
                                f"median={current_score:.6f}ms"
                            )
                    if budget_stopped:
                        break
                    if not improved_in_pass:
                        self._log(
                            f"DONE  basin={restart_index + 1} converged: "
                            "full pass produced no improvement"
                        )
                        break

                global_threshold = global_score * (
                    1.0 - self.policy.min_improvement
                )
                if current_score < global_threshold:
                    global_best = current
                    global_score = current_score
                    self._log(
                        f"GLOBAL config={fwd_config_id(global_best)} "
                        f"median={global_score:.6f}ms"
                    )
                if budget_stopped:
                    break

            if not budget_stopped:
                session_status = "complete"
            current = global_best
            current_score = global_score

            best_id = fwd_config_id(current)
            self.database.select_best(
                current,
                current_score,
                session_id=self._session_id,
            )
            elapsed = time.monotonic() - self._start_time
            self._log(
                f"FINAL config={best_id} median={current_score:.6f}ms "
                f"evaluated={self.evaluated_trials} reused={self.reused_trials}"
            )
            return TuningResult(
                config=current,
                median_ms=current_score,
                database_path=self.database.path,
                session_id=self._session_id,
                evaluated_trials=self.evaluated_trials,
                reused_trials=self.reused_trials,
                elapsed_s=elapsed,
            )
        finally:
            self.database.finish_session(
                self._session_id,
                status=session_status,
                best_config_id=best_id,
                elapsed_s=time.monotonic() - self._start_time,
            )


def _torch_mxfp8_reference(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    def quantize(value: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = value.shape
        blocks = value.reshape(*shape[:-1], shape[-1] // 32, 32)
        amax = blocks.abs().amax(dim=-1, keepdim=True).float()
        amax_bits = amax.view(torch.int32)
        exponent = ((amax_bits >> 23) & 0xFF) - 127 - 8
        exponent += (amax_bits & 0x7FFFFF) > 0x600000
        scale_code = (exponent.clamp(-127, 128) + 127).to(torch.uint8)
        scale_code = torch.where(
            torch.isnan(amax), torch.full_like(scale_code, 255), scale_code
        )
        scale = (scale_code.int() << 23).view(torch.float32).clamp_min(2**-126)
        quantized = (blocks.float() / scale).clamp(-448, 448)
        return (
            quantized.to(torch.float8_e4m3fn).reshape(shape),
            scale_code.view(torch.float8_e8m0fnu).squeeze(-1),
        )

    qx, sx = quantize(x)
    qw, sw = quantize(weight)
    dx = qx.float() * sx.float().repeat_interleave(32, dim=-1)
    dw = qw.float() * sw.float().repeat_interleave(32, dim=-1)
    return (dx @ dw.T).bfloat16()


class MXFP8ForwardEvaluator:
    """Compile, validate, and time one forward configuration on real inputs."""

    def __init__(
        self,
        x: torch.Tensor,
        weight: torch.Tensor,
        policy: CoordinateDescentPolicy,
    ) -> None:
        if x.ndim != 2 or weight.ndim != 2:
            raise ValueError("autotuner expects flattened 2D activation and weight")
        if x.dtype is not torch.bfloat16 or weight.dtype is not torch.bfloat16:
            raise TypeError("autotuner inputs must be BF16")
        if x.device.type != "cuda" or weight.device != x.device:
            raise ValueError("autotuner inputs must share one CUDA device")
        self.x = x.contiguous()
        self.weight = weight.contiguous()
        self.policy = policy
        self.problem = MXFP8Problem(x.shape[0], weight.shape[0], x.shape[1])
        self._expected: torch.Tensor | None = None

    def _baseline_expected(self) -> torch.Tensor:
        if self._expected is None:
            if self.policy.correctness == "torch":
                self._expected = _torch_mxfp8_reference(self.x, self.weight)
            else:
                launcher = compile_mxfp8_fwd(
                    self.problem, DEFAULT_MXFP8_FWD_CONFIG
                )
                expected = torch.empty(
                    (self.problem.m, self.problem.n),
                    dtype=torch.bfloat16,
                    device=self.x.device,
                )
                launcher(self.x, self.weight, expected)
                torch.cuda.synchronize(self.x.device)
                self._expected = expected.clone()
        return self._expected

    def __call__(self, config: MXFP8FwdConfig) -> TrialOutcome:
        compile_start = time.monotonic()
        try:
            launcher = compile_mxfp8_fwd(self.problem, config)
        except Exception as exc:
            return TrialOutcome(
                "compile_error",
                compile_ms=(time.monotonic() - compile_start) * 1000,
                error=f"{type(exc).__name__}: {exc}"[:4000],
            )
        compile_ms = (time.monotonic() - compile_start) * 1000
        out = torch.empty(
            (self.problem.m, self.problem.n),
            dtype=torch.bfloat16,
            device=self.x.device,
        )
        try:
            launcher(self.x, self.weight, out)
            torch.cuda.synchronize(self.x.device)
        except Exception as exc:
            return TrialOutcome(
                "runtime_error",
                compile_ms=compile_ms,
                error=f"{type(exc).__name__}: {exc}"[:4000],
            )

        max_abs_error: float | None = None
        if self.policy.correctness != "none":
            expected = self._baseline_expected()
            difference = (out.float() - expected.float()).abs()
            max_abs_error = float(difference.max())
            if not torch.allclose(
                out,
                expected,
                rtol=self.policy.correctness_rtol,
                atol=self.policy.correctness_atol,
                equal_nan=True,
            ):
                return TrialOutcome(
                    "correctness_error",
                    compile_ms=compile_ms,
                    max_abs_error=max_abs_error,
                    error="candidate output differs from the configured reference",
                )

        try:
            for _ in range(self.policy.warmup):
                launcher(self.x, self.weight, out)
            torch.cuda.synchronize(self.x.device)
            timings: list[float] = []
            for _ in range(self.policy.samples):
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _call in range(self.policy.calls_per_sample):
                    launcher(self.x, self.weight, out)
                end.record()
                end.synchronize()
                timings.append(
                    float(start.elapsed_time(end)) / self.policy.calls_per_sample
                )
        except Exception as exc:
            return TrialOutcome(
                "runtime_error",
                compile_ms=compile_ms,
                max_abs_error=max_abs_error,
                error=f"benchmark failed: {type(exc).__name__}: {exc}"[:4000],
            )
        return TrialOutcome(
            "ok",
            median_ms=float(statistics.median(timings)),
            timings_ms=timings,
            compile_ms=compile_ms,
            max_abs_error=max_abs_error,
        )


def _make_database(
    problem: MXFP8Problem,
    device: torch.device | str | int | None,
    cache_dir: Path | str | None,
    axes: Mapping[str, Iterable[object]],
) -> JsonTuningDatabase:
    return JsonTuningDatabase(
        cache_dir,
        DeviceFingerprint.current(device),
        problem,
        axes,
    )


def load_cached_mxfp8_fwd_config(
    problem: MXFP8Problem,
    *,
    device: torch.device | str | int | None = None,
    cache_dir: Path | str | None = None,
    axes: Mapping[str, Iterable[object]] | None = None,
) -> MXFP8FwdConfig | None:
    search = FWD_SEARCH_SPACE if axes is None else axes
    database = _make_database(problem, device, cache_dir, search)
    best = database.best()
    if best is None:
        return None
    config, _score = best
    if config.implementation_rejection(problem) is not None:
        return None
    return config


def tune_mxfp8_fwd(
    x: torch.Tensor,
    weight: torch.Tensor,
    *,
    policy: CoordinateDescentPolicy | None = None,
    initial: MXFP8FwdConfig = DEFAULT_MXFP8_FWD_CONFIG,
    cache_dir: Path | str | None = None,
    axes: Mapping[str, Iterable[object]] | None = None,
    progress: ProgressCallback | None = None,
) -> TuningResult:
    selected_policy = policy or CoordinateDescentPolicy()
    x_2d = x.reshape(-1, x.shape[-1])
    evaluator = MXFP8ForwardEvaluator(x_2d, weight, selected_policy)
    search = FWD_SEARCH_SPACE if axes is None else axes
    fingerprint = DeviceFingerprint.current(x.device)
    database = JsonTuningDatabase(
        cache_dir, fingerprint, evaluator.problem, search
    )
    tuner = CoordinateDescentTuner(
        evaluator.problem,
        evaluator,
        database,
        selected_policy,
        axes=search,
        progress=progress,
    )
    return tuner.tune(initial)


def _cli() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m", type=int, required=True)
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seconds", type=float, default=1800.0)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--restarts", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=9)
    parser.add_argument("--calls-per-sample", type=int, default=5)
    parser.add_argument(
        "--correctness", choices=("baseline", "torch", "none"), default="baseline"
    )
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument(
        "--initial-config",
        type=Path,
        help=(
            "seed from a JSON config, a tuner result containing 'config', or "
            "a benchmark artifact containing 'selected.config'"
        ),
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--randomize-coordinates", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--log-file",
        type=Path,
        help="also append timestamped progress messages to this file",
    )
    parser.add_argument("--quiet", action="store_true", help="disable progress output")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        parser.error("CUDA is not available")
    if torch.cuda.get_device_capability()[0] != 12:
        parser.error("MXFP8 tuner requires an SM120/SM121 GPU")
    torch.manual_seed(args.seed)
    x = torch.randn(args.m, args.k, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(args.n, args.k, device="cuda", dtype=torch.bfloat16)
    policy = CoordinateDescentPolicy(
        time_budget_s=args.seconds,
        max_passes=args.passes,
        restarts=args.restarts,
        warmup=args.warmup,
        samples=args.samples,
        calls_per_sample=args.calls_per_sample,
        correctness=args.correctness,
        force=args.force,
        randomize_coordinates=args.randomize_coordinates,
        seed=args.seed,
    )
    initial = DEFAULT_MXFP8_FWD_CONFIG
    if args.initial_config is not None:
        document = json.loads(args.initial_config.read_text(encoding="utf-8"))
        config_values = document.get("config")
        if config_values is None:
            config_values = document.get("selected", {}).get("config")
        if not isinstance(config_values, dict):
            parser.error(
                "--initial-config must contain 'config' or 'selected.config'"
            )
        initial = fwd_config_from_dict(config_values)
    log_stream: TextIO | None = None
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        log_stream = args.log_file.open("a", encoding="utf-8", buffering=1)

    def progress(message: str) -> None:
        line = f"{_utc_now()} {message}"
        if not args.quiet:
            print(line, flush=True)
        if log_stream is not None:
            print(line, file=log_stream, flush=True)

    try:
        result = tune_mxfp8_fwd(
            x,
            weight,
            policy=policy,
            initial=initial,
            cache_dir=args.cache_dir,
            progress=progress,
        )
    finally:
        if log_stream is not None:
            log_stream.close()
    print(
        json.dumps(
            {
                "config": fwd_config_to_dict(result.config),
                "median_ms": result.median_ms,
                "database_path": str(result.database_path),
                "session_id": result.session_id,
                "evaluated_trials": result.evaluated_trials,
                "reused_trials": result.reused_trials,
                "elapsed_s": result.elapsed_s,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    _cli()


__all__ = [
    "CoordinateDescentPolicy",
    "CoordinateDescentTuner",
    "DeviceFingerprint",
    "JsonTuningDatabase",
    "MXFP8ForwardEvaluator",
    "TrialOutcome",
    "TuningResult",
    "default_cache_dir",
    "load_cached_mxfp8_fwd_config",
    "tune_mxfp8_fwd",
]
