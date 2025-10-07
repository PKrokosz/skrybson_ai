from __future__ import annotations

import json
from pathlib import Path

from ui.services.sessions import discover_sessions, validate_manifest


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


def test_validate_manifest_success(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    audio_path = session_dir / "alice.wav"
    audio_path.write_bytes(b"data")
    out_dir = tmp_path / "out" / "session" / "transcripts"
    out_dir.mkdir(parents=True)
    (out_dir / "user.json").write_text("{}", encoding="utf8")
    (out_dir / "user.srt").write_text("", encoding="utf8")
    (out_dir / "user.vtt").write_text("", encoding="utf8")
    manifest = {
        "transcripts": {
            "alice": {
                "wav_path": ["alice.wav"],
                "json_path": "../out/session/transcripts/user.json",
                "srt_path": "../out/session/transcripts/user.srt",
                "vtt_path": "../out/session/transcripts/user.vtt",
            }
        }
    }
    (session_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")

    issues = validate_manifest(session_dir)
    assert any(issue.level == "info" for issue in issues)
    assert all(issue.level == "info" for issue in issues)


def test_validate_manifest_missing_files(tmp_path: Path) -> None:
    session_dir = tmp_path / "session"
    session_dir.mkdir()
    manifest = {
        "transcripts": {
            "alice": {
                "wav_path": ["missing.wav"],
                "json_path": "../out/session/transcripts/user.json",
                "srt_path": "",
                "vtt_path": None,
            }
        }
    }
    (session_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf8")

    issues = validate_manifest(session_dir)
    levels = {issue.level for issue in issues}
    assert "error" in levels
    assert "warning" in levels
