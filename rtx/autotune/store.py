"""Append-only, crash-resilient storage for tuning observations and decisions."""

from __future__ import annotations

from contextlib import contextmanager
import json
import os
from pathlib import Path
from threading import RLock
from typing import Generic, Iterable, Iterator, Mapping, Protocol
import uuid

from .core import ConfigT, KernelContext, Observation, canonical_json, utc_now

try:
    import fcntl
except ImportError:  # pragma: no cover - deployment targets Linux.
    fcntl = None


class TuningStore(Protocol, Generic[ConfigT]):
    @property
    def path(self) -> Path | None: ...

    def start_session(self, context: KernelContext, metadata: Mapping[str, object]) -> str: ...

    def record_observation(self, observation: Observation[ConfigT]) -> None: ...

    def record_event(self, session_id: str, kind: str, payload: Mapping[str, object]) -> None: ...

    def finish_session(self, session_id: str, payload: Mapping[str, object]) -> None: ...

    def records(self, context: KernelContext | None = None) -> Iterable[Mapping[str, object]]: ...


class JsonlTuningStore(Generic[ConfigT]):
    """One JSON object per line; every accepted measurement is immediately durable."""

    def __init__(self, root: Path | str, *, fsync: bool = True) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.observations_path = self.root / "observations.jsonl"
        self.sessions_path = self.root / "sessions.jsonl"
        self.events_path = self.root / "events.jsonl"
        self.lock_path = self.root / ".write.lock"
        self.fsync = fsync
        self._thread_lock = RLock()

    @property
    def path(self) -> Path:
        return self.root

    @contextmanager
    def _lock(self) -> Iterator[None]:
        with self._thread_lock:
            with self.lock_path.open("a+", encoding="utf-8") as handle:
                if fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    if fcntl is not None:
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _append(self, path: Path, value: Mapping[str, object]) -> None:
        encoded = canonical_json(dict(value)) + "\n"
        with self._lock():
            with path.open("a", encoding="utf-8") as handle:
                handle.write(encoded)
                handle.flush()
                if self.fsync:
                    os.fsync(handle.fileno())

    def start_session(
        self,
        context: KernelContext,
        metadata: Mapping[str, object],
    ) -> str:
        session_id = uuid.uuid4().hex
        self._append(
            self.sessions_path,
            {
                "schema_version": 1,
                "event": "start",
                "session_id": session_id,
                "recorded_at": utc_now(),
                "context_id": context.identifier,
                "context": context.as_dict(),
                "metadata": dict(metadata),
            },
        )
        return session_id

    def record_observation(self, observation: Observation[ConfigT]) -> None:
        self._append(self.observations_path, observation.as_dict())

    def record_event(
        self,
        session_id: str,
        kind: str,
        payload: Mapping[str, object],
    ) -> None:
        self._append(
            self.events_path,
            {
                "schema_version": 1,
                "session_id": session_id,
                "kind": kind,
                "recorded_at": utc_now(),
                "payload": dict(payload),
            },
        )

    def finish_session(
        self,
        session_id: str,
        payload: Mapping[str, object],
    ) -> None:
        self._append(
            self.sessions_path,
            {
                "schema_version": 1,
                "event": "finish",
                "session_id": session_id,
                "recorded_at": utc_now(),
                **dict(payload),
            },
        )

    def records(
        self,
        context: KernelContext | None = None,
    ) -> Iterable[Mapping[str, object]]:
        if not self.observations_path.exists():
            return ()

        def iterator() -> Iterator[Mapping[str, object]]:
            with self.observations_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if context is None or value.get("context_id") == context.identifier:
                        yield value

        return iterator()


class InMemoryTuningStore(Generic[ConfigT]):
    """Testing/embedding store with the same event model as the JSONL backend."""

    def __init__(self) -> None:
        self.observations: list[Observation[ConfigT]] = []
        self.events: list[dict[str, object]] = []
        self.sessions: list[dict[str, object]] = []

    @property
    def path(self) -> None:
        return None

    def start_session(
        self,
        context: KernelContext,
        metadata: Mapping[str, object],
    ) -> str:
        session_id = uuid.uuid4().hex
        self.sessions.append(
            {"event": "start", "session_id": session_id, "context": context.as_dict(), **dict(metadata)}
        )
        return session_id

    def record_observation(self, observation: Observation[ConfigT]) -> None:
        self.observations.append(observation)

    def record_event(self, session_id: str, kind: str, payload: Mapping[str, object]) -> None:
        self.events.append({"session_id": session_id, "kind": kind, **dict(payload)})

    def finish_session(self, session_id: str, payload: Mapping[str, object]) -> None:
        self.sessions.append({"event": "finish", "session_id": session_id, **dict(payload)})

    def records(self, context: KernelContext | None = None) -> Iterable[Mapping[str, object]]:
        return tuple(
            item.as_dict()
            for item in self.observations
            if context is None or item.context_id == context.identifier
        )


__all__ = ["InMemoryTuningStore", "JsonlTuningStore", "TuningStore"]
