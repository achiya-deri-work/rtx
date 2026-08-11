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

    def incomplete_candidates(self, context_id: str) -> Iterable[Mapping[str, object]]: ...


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
            with path.open("a+b") as handle:
                # A power loss may leave a partial final JSON object. Isolate
                # it before appending so the first record after resume remains
                # independently decodable.
                handle.seek(0, os.SEEK_END)
                if handle.tell() > 0:
                    handle.seek(-1, os.SEEK_END)
                    if handle.read(1) != b"\n":
                        handle.seek(0, os.SEEK_END)
                        handle.write(b"\n")
                handle.seek(0, os.SEEK_END)
                handle.write(encoded.encode("utf-8"))
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

    def incomplete_candidates(
        self, context_id: str
    ) -> Iterable[Mapping[str, object]]:
        """Return candidates durably issued but never durably completed.

        An observation itself counts as completion.  This closes the tiny
        crash window between saving an observation and appending its explicit
        completion event.
        """

        started: dict[str, Mapping[str, object]] = {}
        completed: set[str] = set()
        if self.events_path.exists():
            with self.events_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    payload = value.get("payload", {})
                    if not isinstance(payload, Mapping):
                        continue
                    attempt_id = str(payload.get("attempt_id", ""))
                    if not attempt_id:
                        continue
                    kind = value.get("kind")
                    if kind in ("candidate_started", "trial_issued"):
                        if payload.get("context_id") == context_id:
                            started[attempt_id] = payload
                    elif kind in ("candidate_completed", "trial_completed"):
                        completed.add(attempt_id)
        if self.observations_path.exists():
            with self.observations_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    try:
                        value = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if value.get("context_id") != context_id:
                        continue
                    metadata = value.get("metadata", {})
                    if isinstance(metadata, Mapping):
                        attempt_id = str(metadata.get("attempt_id", ""))
                        if attempt_id:
                            completed.add(attempt_id)
        return tuple(
            payload
            for attempt_id, payload in started.items()
            if attempt_id not in completed
        )


class ResidualTuningStore(Generic[ConfigT]):
    """Write one task residual while reading transferable sibling residuals.

    Each tuning unit owns independent JSONL files, so a truncated or lost file
    cannot damage an entire family/regime campaign. Reads may span a family
    root to preserve cross-context model training. Malformed tail records are
    skipped independently by ``JsonlTuningStore``.
    """

    def __init__(
        self,
        unit_root: Path | str,
        *,
        transfer_root: Path | str | None = None,
        fsync: bool = True,
    ) -> None:
        self.local = JsonlTuningStore[ConfigT](unit_root, fsync=fsync)
        self.transfer_root = (
            self.local.root
            if transfer_root is None
            else Path(transfer_root).expanduser()
        )

    @property
    def path(self) -> Path:
        return self.local.path

    def start_session(
        self, context: KernelContext, metadata: Mapping[str, object]
    ) -> str:
        return self.local.start_session(context, metadata)

    def record_observation(self, observation: Observation[ConfigT]) -> None:
        self.local.record_observation(observation)

    def record_event(
        self, session_id: str, kind: str, payload: Mapping[str, object]
    ) -> None:
        self.local.record_event(session_id, kind, payload)

    def finish_session(
        self, session_id: str, payload: Mapping[str, object]
    ) -> None:
        self.local.finish_session(session_id, payload)

    def records(
        self, context: KernelContext | None = None
    ) -> Iterable[Mapping[str, object]]:
        paths = sorted(self.transfer_root.rglob("observations.jsonl"))

        def iterator() -> Iterator[Mapping[str, object]]:
            seen: set[tuple[object, object]] = set()
            for path in paths:
                store = JsonlTuningStore[ConfigT](path.parent, fsync=False)
                for value in store.records(context):
                    identity = (
                        value.get("context_id"),
                        value.get("observation_id"),
                    )
                    if identity in seen:
                        continue
                    seen.add(identity)
                    yield value

        return iterator()

    def incomplete_candidates(
        self, context_id: str
    ) -> Iterable[Mapping[str, object]]:
        # Only this residual owns executable attempts. Transfer siblings are
        # training history and must never blacklist a local candidate.
        return self.local.incomplete_candidates(context_id)


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

    def incomplete_candidates(
        self, context_id: str
    ) -> Iterable[Mapping[str, object]]:
        started: dict[str, Mapping[str, object]] = {}
        completed: set[str] = set()
        for event in self.events:
            attempt_id = str(event.get("attempt_id", ""))
            if not attempt_id:
                continue
            if event.get("kind") in ("candidate_started", "trial_issued"):
                if event.get("context_id") == context_id:
                    started[attempt_id] = event
            elif event.get("kind") in ("candidate_completed", "trial_completed"):
                completed.add(attempt_id)
        for observation in self.observations:
            attempt_id = str(observation.metadata.get("attempt_id", ""))
            if observation.context_id == context_id and attempt_id:
                completed.add(attempt_id)
        return tuple(
            payload
            for attempt_id, payload in started.items()
            if attempt_id not in completed
        )


__all__ = [
    "InMemoryTuningStore",
    "JsonlTuningStore",
    "ResidualTuningStore",
    "TuningStore",
]
