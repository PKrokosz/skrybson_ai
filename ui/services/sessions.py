"""Session discovery utilities."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Mapping, Sequence

_STATUS_GLYPHS = {
    "new": "⚪",
    "in-progress": "🟡",
    "transcribed": "🟢",
    "aligned": "📐",
}


@dataclass(slots=True)
class SessionRecording:
    """Representation of a single user recording."""

    user: str
    path: Path
    duration: float | None
    avatar: Path | None = None


@dataclass(slots=True)
class SessionSummary:
    """Summary for a discovered session."""

    session_id: str
    created_at: datetime | None
    channel: str | None
    users: Sequence[str]
    duration: float | None
    size_bytes: int
    status: str
    manifest: Mapping[str, object] | None = None
    recordings: Sequence[SessionRecording] = ()

    @property
    def status_glyph(self) -> str:
        return _STATUS_GLYPHS.get(self.status, "⚪")


def _parse_manifest(manifest_path: Path) -> Mapping[str, object] | None:
    if not manifest_path.exists():
        return None
    with manifest_path.open("r", encoding="utf8") as fp:
        return json.load(fp)


def _folder_size(path: Path) -> int:
    total = 0
    for item in path.glob("**/*"):
        if item.is_file():
            total += item.stat().st_size
    return total


def _derive_status(session_dir: Path) -> str:
    if list(session_dir.glob("*.aligned.json")):
        return "aligned"
    if list(session_dir.glob("*.json")):
        return "transcribed"
    if list(session_dir.glob("*.tmp")):
        return "in-progress"
    return "new"


def discover_sessions(recordings_dir: Path) -> List[SessionSummary]:
    sessions: List[SessionSummary] = []
    if not recordings_dir.exists():
        return sessions
    for session_dir in sorted(p for p in recordings_dir.iterdir() if p.is_dir()):
        manifest = _parse_manifest(session_dir / "manifest.json")
        created_at = None
        channel = None
        users: Sequence[str] = []
        duration = None
        if manifest:
            if date := manifest.get("created_at"):
                try:
                    created_at = datetime.fromisoformat(str(date))
                except ValueError:
                    created_at = None
            channel = manifest.get("channel")  # type: ignore[assignment]
            users = [str(u) for u in manifest.get("users", [])]
            duration = manifest.get("duration")  # type: ignore[assignment]
        recordings = _load_recordings(session_dir, users)
        sessions.append(
            SessionSummary(
                session_id=session_dir.name,
                created_at=created_at,
                channel=channel,
                users=users,
                duration=duration,
                size_bytes=_folder_size(session_dir),
                status=_derive_status(session_dir),
                manifest=manifest,
                recordings=recordings,
            ),
        )
    return sessions


def _load_recordings(session_dir: Path, users: Sequence[str]) -> Sequence[SessionRecording]:
    recordings: List[SessionRecording] = []
    for wav in sorted(session_dir.glob("*.wav")):
        user = wav.stem
        duration = None
        if "_" in user:
            user = user.split("_", 1)[1]
        recordings.append(SessionRecording(user=user, path=wav, duration=duration))
    if not recordings and users:
        recordings.extend(SessionRecording(user=u, path=session_dir / f"{u}.wav", duration=None) for u in users)
    return recordings


__all__ = ["SessionSummary", "SessionRecording", "discover_sessions"]
