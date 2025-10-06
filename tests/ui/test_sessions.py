from __future__ import annotations

import json
from pathlib import Path

from ui.services.sessions import discover_sessions


def test_discover_sessions(tmp_path: Path) -> None:
    session_dir = tmp_path / "session-1"
    session_dir.mkdir()
    manifest = {
        "created_at": "2024-01-01T12:00:00",
        "channel": "general",
        "users": ["alice", "bob"],
        "duration": 123.4,
    }
    (session_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")
    (session_dir / "alice.wav").write_bytes(b"data")

    sessions = discover_sessions(tmp_path)
    assert len(sessions) == 1
    summary = sessions[0]
    assert summary.session_id == "session-1"
    assert summary.channel == "general"
    assert summary.users == ["alice", "bob"]
    assert summary.duration == 123.4
