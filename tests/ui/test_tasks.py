from __future__ import annotations

import io
from typing import Any

import pytest

from ui.services.tasks import TaskManager


class DummyProcess:
    def __init__(self) -> None:
        self.stdout = io.BytesIO(b"log1\nlog2\n")
        self.stderr = io.BytesIO(b"err1\n")
        self._returncode: int | None = None

    def wait(self) -> None:
        self._returncode = 0

    def poll(self) -> int | None:
        return self._returncode

    def terminate(self) -> None:
        self._returncode = 0


def test_task_manager_start(monkeypatch: pytest.MonkeyPatch) -> None:
    dummy = DummyProcess()

    def fake_popen(*args: Any, **kwargs: Any) -> DummyProcess:
        return dummy

    monkeypatch.setattr("ui.services.tasks.subprocess.Popen", fake_popen)
    manager = TaskManager()
    task = manager.start({}, dry_run=True)
    task.wait(0.1)
    logs = list(task.logs())
    assert any(message.content == "log1" for message in logs)
    manager.stop()
