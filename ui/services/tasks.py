"""Background task management for transcriptions."""
from __future__ import annotations

import os
import queue
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Dict, Iterable, Optional

default_env = {
    "PYTHONUNBUFFERED": "1",
}


@dataclass(slots=True)
class TaskMessage:
    """Message produced by a running task."""

    stream: str
    content: str


@dataclass(slots=True)
class TranscriptionTask:
    """Wraps a running transcribe.py subprocess."""

    env: Dict[str, str]
    dry_run: bool = False
    process: subprocess.Popen[bytes] | None = None
    _log_queue: "queue.Queue[TaskMessage]" = field(default_factory=queue.Queue, init=False)
    _done_event: Event = field(default_factory=Event, init=False)
    _lock: Lock = field(default_factory=Lock, init=False)

    def start(self) -> None:
        with self._lock:
            if self.process is not None:
                raise RuntimeError("Task already started")
            command = [sys.executable, "transcribe.py"]
            if self.dry_run:
                command.append("--dry-run")
            env = os.environ.copy()
            env.update(default_env)
            env.update(self.env)
            self.process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
                cwd=str(Path.cwd()),
            )
        Thread(target=self._pump_stream, args=(self.process.stdout, "stdout"), daemon=True).start()
        Thread(target=self._pump_stream, args=(self.process.stderr, "stderr"), daemon=True).start()
        Thread(target=self._watcher, daemon=True).start()

    def _pump_stream(self, handle: Optional[object], stream: str) -> None:
        if handle is None:
            return
        for raw_line in iter(handle.readline, b""):
            line = raw_line.decode("utf8", errors="replace").rstrip()
            self._log_queue.put(TaskMessage(stream=stream, content=line))
        handle.close()

    def _watcher(self) -> None:
        if self.process is None:
            return
        self.process.wait()
        self._done_event.set()

    def stop(self) -> None:
        with self._lock:
            if self.process and self.process.poll() is None:
                self.process.terminate()

    def logs(self) -> Iterable[TaskMessage]:
        while True:
            try:
                yield self._log_queue.get_nowait()
            except queue.Empty:
                break

    def wait(self, timeout: float | None = None) -> bool:
        return self._done_event.wait(timeout)

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self.process is not None and self.process.poll() is None


class TaskManager:
    """Manage a single transcription task at a time."""

    def __init__(self) -> None:
        self._task: TranscriptionTask | None = None
        self._lock = Lock()

    def start(self, env: Dict[str, str], *, dry_run: bool = False) -> TranscriptionTask:
        with self._lock:
            if self._task and self._task.is_running:
                raise RuntimeError("Transcription already running")
            self._task = TranscriptionTask(env=env, dry_run=dry_run)
            self._task.start()
            return self._task

    def stop(self) -> None:
        with self._lock:
            if self._task:
                self._task.stop()

    def get_task(self) -> TranscriptionTask | None:
        with self._lock:
            return self._task


__all__ = ["TaskManager", "TranscriptionTask", "TaskMessage"]
