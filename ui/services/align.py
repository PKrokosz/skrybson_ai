"""Alignment related helpers."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Thread
from typing import Iterable, List, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@dataclass(slots=True)
class AlignItem:
    path: Path
    selected: bool = True


@dataclass(slots=True)
class AlignmentMessage:
    level: str
    message: str
    candidate: Path | None = None


class AlignmentWorker(Thread):
    """Background worker executing ``align.py`` for selected candidates."""

    def __init__(
        self,
        candidates: Sequence[AlignItem],
        recordings_dir: Path,
        output_dir: Path,
        queue_: "queue.Queue[AlignmentMessage]",
        *,
        diarization_token: str | None = None,
    ) -> None:
        super().__init__(daemon=True)
        self._candidates = list(candidates)
        self._recordings_dir = recordings_dir
        self._output_dir = output_dir
        self._queue = queue_
        self._token = diarization_token
        self._stop_event = Event()
        self._running = Event()
        self._current_process: subprocess.Popen[str] | None = None

    # ------------------------------------------------------------------
    def run(self) -> None:  # pragma: no cover - threaded worker
        self._running.set()
        try:
            for item in self._candidates:
                if self._stop_event.is_set():
                    break
                self._queue.put(
                    AlignmentMessage(level="info", message=f"Start align dla {item.path.name}", candidate=item.path)
                )
                payload = self._load_segments(item.path)
                if payload is None:
                    continue
                audio_path = self._resolve_audio_path(item.path, payload)
                if audio_path is None:
                    continue
                output_path = item.path.with_suffix(".aligned.json")
                if self._stop_event.is_set():
                    break
                self._execute_alignment(audio_path, item.path, output_path)
        finally:
            final_state = "stopped" if self._stop_event.is_set() else "done"
            self._queue.put(AlignmentMessage(level="state", message=final_state))
            self._running.clear()

    # ------------------------------------------------------------------
    def stop(self) -> None:
        self._stop_event.set()
        if self._current_process and self._current_process.poll() is None:
            self._current_process.terminate()

    # ------------------------------------------------------------------
    @property
    def is_running(self) -> bool:
        return self._running.is_set()

    # ------------------------------------------------------------------
    def _load_segments(self, path: Path) -> dict | None:
        try:
            payload = json.loads(path.read_text(encoding="utf8"))
        except json.JSONDecodeError as exc:
            self._queue.put(
                AlignmentMessage(
                    level="error", message=f"Nie mogę odczytać JSON ({exc})", candidate=path
                )
            )
            return None
        if "segments" not in payload:
            self._queue.put(
                AlignmentMessage(level="warning", message="Brak pola 'segments'", candidate=path)
            )
            return None
        return payload

    # ------------------------------------------------------------------
    def _resolve_audio_path(self, json_path: Path, payload: dict) -> Path | None:
        raw_files = payload.get("raw_files")
        audio_candidates: Iterable[str]
        if isinstance(raw_files, list):
            audio_candidates = [str(item) for item in raw_files if isinstance(item, (str, Path))]
        elif isinstance(raw_files, str):
            audio_candidates = [raw_files]
        else:
            audio_candidates = []

        audio_candidates = list(audio_candidates)
        if not audio_candidates:
            self._queue.put(
                AlignmentMessage(level="warning", message="Brak informacji o plikach audio", candidate=json_path)
            )
            return None
        if len(audio_candidates) > 1:
            self._queue.put(
                AlignmentMessage(
                    level="warning",
                    message="Wiele plików audio w JSON – wykorzystuję pierwszy wpis.",
                    candidate=json_path,
                )
            )
        session_relative = self._session_relative_path(json_path)
        audio_path = (self._recordings_dir / session_relative / Path(audio_candidates[0])).resolve()
        if not audio_path.exists():
            self._queue.put(
                AlignmentMessage(
                    level="error",
                    message=f"Brak pliku audio: {audio_candidates[0]}",
                    candidate=json_path,
                )
            )
            return None
        return audio_path

    # ------------------------------------------------------------------
    def _session_relative_path(self, json_path: Path) -> Path:
        try:
            relative = json_path.relative_to(self._output_dir)
        except ValueError:
            return Path()
        parts = list(relative.parts)
        if len(parts) <= 2:
            return Path()
        return Path(*parts[:-2])

    # ------------------------------------------------------------------
    def _execute_alignment(self, audio_path: Path, json_path: Path, output_path: Path) -> None:
        command = [
            sys.executable,
            "align.py",
            str(audio_path),
            str(json_path),
            "--output",
            str(output_path),
        ]
        env = os.environ.copy()
        if self._token:
            env.setdefault("PYANNOTE_AUTH_TOKEN", self._token)
        try:
            self._current_process = subprocess.Popen(
                command,
                cwd=str(PROJECT_ROOT),
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except OSError as exc:
            self._queue.put(
                AlignmentMessage(level="error", message=f"Nie mogę uruchomić align.py: {exc}", candidate=json_path)
            )
            return

        stdout, stderr = self._current_process.communicate()
        return_code = self._current_process.returncode
        self._current_process = None
        if stdout:
            for line in stdout.splitlines():
                self._queue.put(AlignmentMessage(level="info", message=line, candidate=json_path))
        if stderr:
            for line in stderr.splitlines():
                self._queue.put(AlignmentMessage(level="warning", message=line, candidate=json_path))
        if return_code == 0:
            self._queue.put(
                AlignmentMessage(
                    level="success",
                    message=f"Utworzono {output_path.name}",
                    candidate=json_path,
                )
            )
        else:
            self._queue.put(
                AlignmentMessage(
                    level="error",
                    message=f"align.py zakończył się kodem {return_code}",
                    candidate=json_path,
                )
            )


def discover_alignment_candidates(output_dir: Path) -> List[AlignItem]:
    candidates: List[AlignItem] = []
    if not output_dir.exists():
        return candidates
    for path in sorted(output_dir.glob("**/*.json")):
        if path.name.endswith(".aligned.json"):
            continue
        candidates.append(AlignItem(path=path))
    return candidates


__all__ = [
    "AlignItem",
    "AlignmentMessage",
    "AlignmentWorker",
    "discover_alignment_candidates",
]
