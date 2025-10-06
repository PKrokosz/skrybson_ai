"""Utilities for transcribing recorded Discord sessions with metadata."""

import os
import json
import sys
import glob
import pathlib
import re
from datetime import datetime, timezone
from typing import Optional
from faster_whisper import WhisperModel

# --- konfiguracja ---
SESSION_DIR = os.environ.get("SESSION_DIR")
RECORDINGS_DIR = os.environ.get("RECORDINGS_DIR", "/app/recordings")
MODEL_SIZE = os.environ.get("WHISPER_MODEL", "small")

def pick_latest_session(recordings_dir: str) -> str:
    """Return the newest recording session directory in ``recordings_dir``."""

    sessions = sorted(
        [p for p in pathlib.Path(recordings_dir).glob("*") if p.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return str(sessions[0]) if sessions else ""

def norm_text(t: str) -> str:
    """Normalise text for fuzzy duplicate detection."""

    t = t.strip().lower()
    t = re.sub(r"[^\wąćęłńóśżź ]+", "", t)
    t = re.sub(r"\s+", " ", t)
    return t

def parse_iso_to_epoch(value: Optional[str]) -> Optional[float]:
    """Parse ISO8601 string to epoch seconds, returning ``None`` if invalid."""

    if not value:
        return None
    try:
        if value.endswith("Z"):
            value = value[:-1] + "+00:00"
        return datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None

def main():
    """Generate per-user and global transcripts for the selected session."""

    session_dir = SESSION_DIR or pick_latest_session(RECORDINGS_DIR)
    if not session_dir:
        print("Brak sesji do transkrypcji.")
        sys.exit(1)

    raw_dir = os.path.join(session_dir, "raw")
    out_dir = os.path.join(session_dir, "transcripts")
    os.makedirs(out_dir, exist_ok=True)

    files = sorted(glob.glob(os.path.join(raw_dir, "*.wav")))
    files = [f for f in files if os.path.getsize(f) >= 1024]
    if not files:
        print(f"[!] Brak sensownych plików WAV w {raw_dir} (>=1KB).")
        sys.exit(0)

    print(f"[i] Sesja: {session_dir}")
    print(f"[i] Model: {MODEL_SIZE}")

    manifest_path = os.path.join(session_dir, "manifest.json")
    manifest_start_iso = None
    if os.path.exists(manifest_path):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest_start_iso = manifest.get("startISO")
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[!] Nie udało się odczytać manifestu {manifest_path}: {exc}")

    model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="int8")

    buckets = {}
    for f in files:
        name = os.path.basename(f)
        user_prefix = name.rsplit("_seg", 1)[0]
        buckets.setdefault(user_prefix, []).append(f)

    summary_index = []
    conversation_segments = []

    user_payloads = []

    for user_prefix, wavs in buckets.items():
        wavs.sort(key=lambda x: os.path.getmtime(x))
        timeline = []
        print(f"\n=== START USER {user_prefix} ===")

        for wav in wavs:
            try:
                file_t0 = os.path.getmtime(wav)
                print(f"[file] {os.path.basename(wav)} mtime={file_t0}")
                segments, _info = model.transcribe(
                    wav,
                    vad_filter=True,
                    vad_parameters=dict(min_silence_duration_ms=500),
                )
                for seg in segments:
                    pseudo_t = file_t0 + float(seg.start)
                    print(f"  [seg] {wav} {seg.start:.2f}-{seg.end:.2f}s :: {seg.text.strip()}")
                    timeline.append({
                        "pseudo_t": pseudo_t,
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "text": seg.text.strip(),
                        "file": os.path.basename(wav),
                    })
            except Exception as e:
                print(f"[!] Pomijam uszkodzony plik: {wav} ({e})")
                continue

        timeline.sort(key=lambda x: x["pseudo_t"])

        deduped = []
        last_norm = ""
        last_t = -1e9
        for item in timeline:
            nt = norm_text(item["text"])
            if deduped and nt == last_norm and (item["pseudo_t"] - last_t) < 1.5:
                print(f"  [dup] Pomijam duplikat: {item['text']}")
                continue
            deduped.append(item)
            last_norm = nt
            last_t = item["pseudo_t"]

        t0 = deduped[0]["pseudo_t"]
        segments_all = []
        for it in deduped:
            relative_to_user = round(it["pseudo_t"] - t0, 2)
            duration = it["end"] - it["start"]
            segment = {
                "start": relative_to_user,
                "end": round(relative_to_user + duration, 2),
                "text": it["text"],
                "session_epoch": it["pseudo_t"],
            }
            segments_all.append(segment)
            conversation_segments.append({
                "user": user_prefix,
                "text": it["text"],
                "pseudo_t": it["pseudo_t"],
                "duration": duration,
                "file": it["file"],
            })

        user_payloads.append({
            "user": user_prefix,
            "segments": segments_all,
        })
        summary_index.append({"user": user_prefix, "segments": len(segments_all)})

        print(f"=== END USER {user_prefix} ===\n")

    if not conversation_segments:
        print("[!] Brak segmentów do zbudowania osi czasu rozmowy.")

    conversation_segments.sort(key=lambda item: item["pseudo_t"])

    session_t0 = parse_iso_to_epoch(manifest_start_iso)
    if session_t0 is None and conversation_segments:
        session_t0 = conversation_segments[0]["pseudo_t"]
        manifest_start_iso = datetime.fromtimestamp(session_t0, tz=timezone.utc).isoformat().replace("+00:00", "Z")

    timeline_payload = {
        "session_dir": session_dir,
        "session_start_iso": manifest_start_iso,
        "generated_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat().replace("+00:00", "Z"),
        "segments": [],
    }

    if session_t0 is not None:
        for user_data in user_payloads:
            for seg in user_data["segments"]:
                seg["relative_session_start"] = round(seg["session_epoch"] - session_t0, 2)
                duration = seg["end"] - seg["start"]
                seg["relative_session_end"] = round(seg["relative_session_start"] + duration, 2)
        for seg in conversation_segments:
            relative_start = round(seg["pseudo_t"] - session_t0, 2)
            relative_end = round(relative_start + seg["duration"], 2)
            absolute_start = datetime.fromtimestamp(seg["pseudo_t"], tz=timezone.utc).isoformat().replace("+00:00", "Z")
            timeline_payload["segments"].append({
                "user": seg["user"],
                "text": seg["text"],
                "start": relative_start,
                "end": relative_end,
                "absolute_start": absolute_start,
                "file": seg["file"],
            })

    for user_data in user_payloads:
        for seg in user_data["segments"]:
            seg.pop("session_epoch", None)
        user_json_path = os.path.join(out_dir, f"{user_data['user']}.json")
        with open(user_json_path, "w", encoding="utf-8") as f:
            json.dump(user_data, f, ensure_ascii=False, indent=2)
        print(f"[✓] Zapisano: {user_json_path} ({len(user_data['segments'])} segmentów)")

    conversation_path = os.path.join(out_dir, "conversation.json")
    with open(conversation_path, "w", encoding="utf-8") as f:
        json.dump(timeline_payload, f, ensure_ascii=False, indent=2)
    print(f"[✓] Globalna oś czasu: {conversation_path} ({len(timeline_payload['segments'])} wpisów)")

    index_path = os.path.join(out_dir, "index.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "session_dir": session_dir,
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "items": summary_index,
            "conversation_segments": len(timeline_payload["segments"]),
        }, f, ensure_ascii=False, indent=2)
    print(f"[✓] Gotowe. Index: {index_path}")

if __name__ == "__main__":
    main()
