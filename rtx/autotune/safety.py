"""Exact-scope deterministic failure reuse for expensive kernel compilers."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from threading import RLock
from typing import Generic, Iterable, Mapping

from .core import ConfigT, FeatureMap, KernelAdapter, KernelContext, stable_id, utc_now
from .outcomes import TrialOutcome

try:
    import fcntl
except ImportError:  # pragma: no cover - deployment targets Linux.
    fcntl = None


DETERMINISTIC_FAILURES = frozenset(
    {"architecture_rejected", "implementation_rejected", "compile_error"}
)


def failure_scope(context: KernelContext, config_id: str) -> dict[str, object]:
    """Build a conservative key; cache regime and experimental tags are absent."""

    device = context.device
    architecture = device.get("architecture", device.get("capability"))
    compiler = device.get("compiler")
    if compiler is None:
        software = device.get("software", {})
        compiler = software if isinstance(software, Mapping) else {}
    return {
        "family": context.family,
        "kernel_revision": context.kernel_revision,
        "workload": dict(context.workload),
        "architecture_id": stable_id(architecture),
        "compiler_id": stable_id(compiler),
        "config_id": config_id,
    }


class JsonlFailureLedger:
    """Append-only exact failure cache safe to share across local processes."""

    def __init__(self, path: Path | str, *, fsync: bool = True) -> None:
        self.path = Path(path).expanduser()
        self.lock_path = self.path.with_suffix(self.path.suffix + ".lock")
        self.fsync = fsync
        self._thread_lock = RLock()
        self._records: dict[str, dict[str, object]] | None = None
        self._loaded_signature: tuple[int, int] | None = None

    def _signature(self) -> tuple[int, int] | None:
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            return None
        return stat.st_mtime_ns, stat.st_size

    def _load(self) -> dict[str, dict[str, object]]:
        signature = self._signature()
        if self._records is not None and signature == self._loaded_signature:
            return self._records
        records: dict[str, dict[str, object]] = {}
        try:
            with self.path.open(encoding="utf-8") as source:
                for line in source:
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if isinstance(value, dict) and value.get("failure_key"):
                        records[str(value["failure_key"])] = value
        except FileNotFoundError:
            pass
        self._records = records
        self._loaded_signature = signature
        return records

    def lookup(
        self, context: KernelContext, config_id: str
    ) -> TrialOutcome | None:
        scope = failure_scope(context, config_id)
        key = stable_id(scope, 32)
        with self._thread_lock:
            record = self._load().get(key)
        if record is None:
            return None
        outcome = TrialOutcome.from_dict(record["outcome"])  # type: ignore[arg-type]
        outcome.metadata = {
            **outcome.metadata,
            "deterministic_failure_cache_hit": True,
            "failure_key": key,
            "original_recorded_at": record.get("recorded_at"),
        }
        return outcome

    def record(
        self,
        context: KernelContext,
        config_id: str,
        outcome: TrialOutcome,
    ) -> bool:
        if outcome.status not in DETERMINISTIC_FAILURES:
            return False
        scope = failure_scope(context, config_id)
        key = stable_id(scope, 32)
        value = {
            "schema_version": 1,
            "failure_key": key,
            "recorded_at": utc_now(),
            "scope": scope,
            "outcome": outcome.as_dict(),
        }
        encoded = (
            json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
        ).encode("utf-8")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._thread_lock:
            records = self._load()
            if key in records:
                return False
            with self.lock_path.open("a+") as lock:
                if fcntl is not None:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
                try:
                    # Another process may have appended since our in-memory load.
                    self._records = None
                    self._loaded_signature = None
                    records = self._load()
                    if key in records:
                        return False
                    with self.path.open("a+b") as sink:
                        sink.seek(0, os.SEEK_END)
                        if sink.tell():
                            sink.seek(-1, os.SEEK_END)
                            if sink.read(1) != b"\n":
                                sink.seek(0, os.SEEK_END)
                                sink.write(b"\n")
                        sink.seek(0, os.SEEK_END)
                        sink.write(encoded)
                        sink.flush()
                        if self.fsync:
                            os.fsync(sink.fileno())
                    records[key] = value
                    self._loaded_signature = self._signature()
                finally:
                    if fcntl is not None:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
        return True


@dataclass(slots=True)
class SafetyAwareAdapter(Generic[ConfigT]):
    """KernelAdapter decorator which reuses exact deterministic failures."""

    adapter: KernelAdapter[ConfigT]
    ledger: JsonlFailureLedger

    @property
    def context(self) -> KernelContext:
        return self.adapter.context

    @property
    def initial_config(self) -> ConfigT:
        return self.adapter.initial_config

    def config_id(self, config: ConfigT) -> str:
        return self.adapter.config_id(config)

    def serialize(self, config: ConfigT) -> Mapping[str, object]:
        return self.adapter.serialize(config)

    def deserialize(self, value: Mapping[str, object]) -> ConfigT:
        return self.adapter.deserialize(value)

    def features(self, config: ConfigT) -> FeatureMap:
        return self.adapter.features(config)

    def rejection(self, config: ConfigT) -> tuple[str, str] | None:
        return self.adapter.rejection(config)

    def evaluate(self, config: ConfigT) -> TrialOutcome:
        return self.evaluate_at_fidelity(config, 1.0)

    def evaluate_at_fidelity(
        self, config: ConfigT, fidelity: float
    ) -> TrialOutcome:
        config_id = self.config_id(config)
        cached = self.ledger.lookup(self.context, config_id)
        if cached is not None:
            return cached
        evaluate_at_fidelity = getattr(self.adapter, "evaluate_at_fidelity", None)
        outcome = (
            self.adapter.evaluate(config)
            if evaluate_at_fidelity is None
            else evaluate_at_fidelity(config, fidelity)
        )
        self.ledger.record(self.context, config_id, outcome)
        return outcome

    def coordinates(self):
        return self.adapter.coordinates()

    def neighbors(self, config: ConfigT) -> Iterable[tuple[str, object, ConfigT]]:
        return self.adapter.neighbors(config)

    def sample(self, rng, count: int, seeds):
        return self.adapter.sample(rng, count, seeds)


__all__ = [
    "DETERMINISTIC_FAILURES",
    "JsonlFailureLedger",
    "SafetyAwareAdapter",
    "failure_scope",
]
