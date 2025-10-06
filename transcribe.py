"""Utilities for transcribing recorded Discord sessions with metadata."""

from __future__ import annotations

import glob
import json
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from faster_whisper import WhisperModel


@dataclass
class TranscribeConfig:
    """Runtime configuration for whisper transcription."""

    recordings_dir: Path
    output_dir: Path
    session_dir: Optional[Path]
    model_size: str
    device: str
    compute_type: str
    beam_size: int
    language: str
    vad_filter: bool
    vad_parameters: Dict[str, int]
    sanitize_lower_noise: bool


def _strtobool_env(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    return value.strip() not in {"0", "false", "False"}


def _parse_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _cuda_available() -> bool:
    try:
        import torch  # type: ignore

        return bool(getattr(torch.cuda, "is_available", lambda: False)())
    except ImportError:
        try:
            import ctranslate2  # type: ignore

            return bool(getattr(ctranslate2, "has_cuda_device", lambda: False)())
        except ImportError:
            return False


def load_config() -> TranscribeConfig:
    """Load configuration from environment variables."""

    recordings_dir = Path(os.environ.get("RECORDINGS_DIR", "./recordings")).expanduser().resolve()
    output_dir = Path(os.environ.get("OUTPUT_DIR", "./out")).expanduser().resolve()
    session_env = os.environ.get("SESSION_DIR")
    session_dir = None
    if session_env:
        raw_session = Path(session_env).expanduser()
        session_dir = raw_session if raw_session.is_absolute() else recordings_dir / raw_session

    requested_device = os.environ.get("WHISPER_DEVICE", "cuda").lower()
    model_size = os.environ.get("WHISPER_MODEL", "large-v3")
    compute_type = os.environ.get("WHISPER_COMPUTE", "int8_float16").lower()

    valid_compute_types = {"int8_float16", "int8", "float16"}
    if compute_type not in valid_compute_types:
        print(
            f"[!] Nieznany WHISPER_COMPUTE={compute_type}, używam domyślnego int8_float16.",
            file=sys.stderr,
        )
        compute_type = "int8_float16"

    beam_size = max(1, _parse_int(os.environ.get("WHISPER_BEAM"), 5))
    language = os.environ.get("WHISPER_LANG", "pl")
    vad_filter = _strtobool_env(os.environ.get("WHISPER_VAD"), True)
    vad_parameters = {"min_silence_duration_ms": 500, "padding_duration_ms": 120}
    sanitize_lower_noise = _strtobool_env(os.environ.get("SANITIZE_LOWER_NOISE"), False)

    if requested_device == "cuda" and not _cuda_available():
        print(
            "[!] CUDA nie jest dostępna, przełączam na CPU z modelem medium i compute_type int8.",
            file=sys.stderr,
        )
        requested_device = "cpu"
        model_size = "medium"
        compute_type = "int8"

    return TranscribeConfig(
        recordings_dir=recordings_dir,
        output_dir=output_dir,
        session_dir=session_dir,
        model_size=model_size,
        device="cuda" if requested_device == "cuda" else "cpu",
        compute_type=compute_type,
        beam_size=beam_size,
        language=language,
        vad_filter=vad_filter,
        vad_parameters=vad_parameters,
        sanitize_lower_noise=sanitize_lower_noise,
    )


def sanitize_text(text: str, *, lower_noise: bool = False) -> str:
    """Normalise whitespace and tame repeated punctuation."""

    text = re.sub(r"\s+", " ", text.strip())
    text = re.sub(r"([?!.,:;])\1+", r"\1", text)
    text = re.sub(r"\s+([?!.,:;])", r"\1", text)
    if lower_noise:
        text = re.sub(r"\b(uhm+|um+|eh+|eee+|yyy+)\b", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s{2,}", " ", text).strip()
    return text


def soft_merge_segments(
    segments: List[Dict[str, object]],
    *,
    user_key: Optional[str] = None,
    max_gap: float = 0.3,
    short_threshold: float = 1.0,
    lower_noise: bool = False,
) -> List[Dict[str, object]]:
    """Merge very short segments when they are close and share the same speaker."""

    if not segments:
        return []

    merged: List[Dict[str, object]] = [dict(segments[0])]
    for seg in segments[1:]:
        current = dict(seg)
        prev = merged[-1]
        gap = float(current["start"]) - float(prev["end"])
        current_duration = float(current["end"]) - float(current["start"])
        prev_duration = float(prev["end"]) - float(prev["start"])
        same_user = True
        if user_key is not None:
            same_user = prev.get(user_key) == current.get(user_key)

        if (
            same_user
            and 0 <= gap <= max_gap
            and (current_duration < short_threshold or prev_duration < short_threshold)
        ):
            prev["end"] = max(float(prev["end"]), float(current["end"]))
            prev_text = str(prev.get("text", ""))
            curr_text = str(current.get("text", ""))
            prev["text"] = sanitize_text(f"{prev_text} {curr_text}", lower_noise=lower_noise)
            if "files" in current:
                prev.setdefault("files", [])
                prev_files = list(prev.get("files", []))
                prev_files.extend(current.get("files", []))
                prev["files"] = prev_files
            continue

        merged.append(current)

    return merged


def _format_timestamp(seconds: float, *, separator: str) -> str:
    seconds = max(0.0, seconds)
    millis = int(round((seconds - int(seconds)) * 1000))
    total_seconds = int(seconds)
    if millis == 1000:
        total_seconds += 1
        millis = 0
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if separator == ",":
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"
    return f"{hours:02}:{minutes:02}:{secs:02}.{millis:03}"


def write_srt(segments: List[Dict[str, float]], path: Path, *, base: float = 0.0) -> None:
    lines = []
    for idx, seg in enumerate(segments, start=1):
        start = float(seg["start"]) - base
        end = float(seg["end"]) - base
        lines.append(str(idx))
        lines.append(
            f"{_format_timestamp(start, separator=',')} --> {_format_timestamp(end, separator=',')}"
        )
        lines.append(str(seg.get("text", "")).strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(segments: List[Dict[str, float]], path: Path, *, base: float = 0.0) -> None:
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = float(seg["start"]) - base
        end = float(seg["end"]) - base
        lines.append(
            f"{_format_timestamp(start, separator='.')} --> {_format_timestamp(end, separator='.')}"
        )
        lines.append(str(seg.get("text", "")).strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

def pick_latest_session(recordings_dir: Path) -> Optional[Path]:
    """Return the newest recording session directory in ``recordings_dir``."""

    sessions = sorted(
        (p for p in recordings_dir.glob("*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return sessions[0] if sessions else None

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

def _resolve_session_path(config: TranscribeConfig) -> Optional[Path]:
    if config.session_dir:
        if config.session_dir.is_dir():
            return config.session_dir.resolve()
        candidate = (config.recordings_dir / config.session_dir.name).resolve()
        return candidate if candidate.is_dir() else None
    return pick_latest_session(config.recordings_dir)


def _relative_session_path(config: TranscribeConfig, session_dir: Path) -> Path:
    try:
        return session_dir.relative_to(config.recordings_dir)
    except ValueError:
        return Path(session_dir.name)


def main():
    """Generate per-user and global transcripts for the selected session."""

    config = load_config()

    if not config.recordings_dir.exists():
        print(f"[!] Brak katalogu nagrań: {config.recordings_dir}")
        sys.exit(1)

    session_dir = _resolve_session_path(config)
    if not session_dir:
        print("Brak sesji do transkrypcji.")
        sys.exit(1)

    raw_dir = session_dir / "raw"
    if not raw_dir.exists():
        print(f"[!] Brak katalogu z plikami RAW: {raw_dir}")
        sys.exit(1)

    output_session_dir = config.output_dir / _relative_session_path(config, session_dir)
    out_dir = output_session_dir / "transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(raw_dir / "*.wav")))
    files = [f for f in files if os.path.getsize(f) >= 1024]
    if not files:
        print(f"[!] Brak sensownych plików WAV w {raw_dir} (>=1KB).")
        sys.exit(0)

    print(f"[i] Sesja: {session_dir}")
    print(
        "[i] Konfiguracja: model={model} device={device} compute_type={compute} "
        "beam={beam} language={lang} vad={vad}".format(
            model=config.model_size,
            device=config.device,
            compute=config.compute_type,
            beam=config.beam_size,
            lang=config.language,
            vad=int(config.vad_filter),
        )
    )

    manifest_path = session_dir / "manifest.json"
    manifest: Dict[str, object] = {}
    manifest_start_iso = None
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest_start_iso = manifest.get("startISO")
        except (json.JSONDecodeError, OSError) as exc:
            print(f"[!] Nie udało się odczytać manifestu {manifest_path}: {exc}")

    model = WhisperModel(config.model_size, device=config.device, compute_type=config.compute_type)

    buckets: Dict[str, list[str]] = {}
    for f in files:
        name = os.path.basename(f)
        user_prefix = name.rsplit("_seg", 1)[0]
        buckets.setdefault(user_prefix, []).append(f)

    summary_index = []
    conversation_segments = []

    user_payloads = []
    manifest_transcripts: Dict[str, Dict[str, object]] = {}

    for user_prefix, wavs in buckets.items():
        wavs.sort(key=lambda x: os.path.getmtime(x))
        timeline = []
        raw_wavs: List[str] = []
        print(f"\n=== START USER {user_prefix} ===")

        id_candidate = user_prefix.rsplit("_", 1)[-1]
        if not id_candidate.isdigit():
            id_candidate = re.sub(r"[^0-9A-Za-z]+", "_", user_prefix).strip("_") or user_prefix
        file_stub = f"user_{id_candidate}"

        for wav in wavs:
            try:
                file_t0 = os.path.getmtime(wav)
                print(f"[file] {os.path.basename(wav)} mtime={file_t0}")
                raw_wavs.append(os.path.relpath(wav, session_dir))
                transcribe_kwargs = dict(
                    beam_size=config.beam_size,
                    language=config.language,
                    vad_filter=config.vad_filter,
                )
                if config.vad_filter:
                    transcribe_kwargs["vad_parameters"] = config.vad_parameters
                segments, _info = model.transcribe(wav, **transcribe_kwargs)
                for seg in segments:
                    pseudo_t = file_t0 + float(seg.start)
                    clean_text = sanitize_text(seg.text, lower_noise=config.sanitize_lower_noise)
                    if not clean_text:
                        continue
                    print(
                        f"  [seg] {wav} {seg.start:.2f}-{seg.end:.2f}s :: {clean_text}"
                    )
                    timeline.append({
                        "pseudo_t": pseudo_t,
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "text": clean_text,
                        "file": os.path.basename(wav),
                    })
            except Exception as e:
                print(f"[!] Pomijam uszkodzony plik: {wav} ({e})")
                continue

        timeline.sort(key=lambda x: x["pseudo_t"])

        if not timeline:
            print(f"[!] Brak segmentów dla użytkownika {user_prefix}")
            continue

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
                "user_id": id_candidate,
                "text": it["text"],
                "start": it["pseudo_t"],
                "end": it["pseudo_t"] + duration,
                "files": [it["file"]],
            })

        raw_wavs = sorted(set(raw_wavs))

        segments_all = soft_merge_segments(segments_all, lower_noise=config.sanitize_lower_noise)
        short_after = sum(1 for seg in segments_all if (seg["end"] - seg["start"]) < 1.0)
        if segments_all:
            short_ratio = short_after / len(segments_all)
            if short_ratio > 0.2:
                print(
                    f"[!] Ostrzeżenie: {short_ratio:.0%} segmentów <1s po scaleniu dla {user_prefix}"
                )

        user_payloads.append({
            "user": user_prefix,
            "user_id": id_candidate,
            "file_stub": file_stub,
            "segments": segments_all,
            "raw_files": raw_wavs,
        })
        summary_index.append({"user": user_prefix, "segments": len(segments_all)})

        print(f"=== END USER {user_prefix} ===\n")

    if not conversation_segments:
        print("[!] Brak segmentów do zbudowania osi czasu rozmowy.")

    conversation_segments.sort(key=lambda item: item["start"])
    conversation_segments = soft_merge_segments(
        conversation_segments,
        user_key="user",
        lower_noise=config.sanitize_lower_noise,
    )
    short_conv = sum(1 for seg in conversation_segments if (seg["end"] - seg["start"]) < 1.0)
    if conversation_segments:
        short_ratio = short_conv / len(conversation_segments)
        if short_ratio > 0.2:
            print(f"[!] Ostrzeżenie: {short_ratio:.0%} globalnych segmentów <1s po scaleniu")

    session_t0 = parse_iso_to_epoch(manifest_start_iso)
    if session_t0 is None and conversation_segments:
        session_t0 = conversation_segments[0]["start"]
        manifest_start_iso = (
            datetime.fromtimestamp(session_t0, tz=timezone.utc).isoformat().replace("+00:00", "Z")
        )

    timeline_payload = {
        "session_dir": str(session_dir),
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
            relative_start = round(seg["start"] - session_t0, 2)
            relative_end = round(seg["end"] - session_t0, 2)
            absolute_start = (
                datetime.fromtimestamp(seg["start"], tz=timezone.utc).isoformat().replace("+00:00", "Z")
            )
            timeline_payload["segments"].append({
                "user": seg["user"],
                "user_id": seg.get("user_id"),
                "text": seg["text"],
                "start": relative_start,
                "end": relative_end,
                "absolute_start": absolute_start,
                "files": seg.get("files", []),
            })

    conversation_srt_segments: List[Dict[str, object]] = []
    for seg in conversation_segments:
        conversation_srt_segments.append({
            "start": seg["start"],
            "end": seg["end"],
            "text": f"{seg['user']}: {seg['text']}",
        })

    for user_data in user_payloads:
        for seg in user_data["segments"]:
            seg.pop("session_epoch", None)
        user_json_path = out_dir / f"{user_data['file_stub']}.json"
        json_payload = {
            "user": user_data["user"],
            "user_id": user_data["user_id"],
            "segments": user_data["segments"],
            "raw_files": user_data["raw_files"],
        }
        with open(user_json_path, "w", encoding="utf-8") as f:
            json.dump(json_payload, f, ensure_ascii=False, indent=2)
        print(f"[✓] Zapisano: {user_json_path} ({len(user_data['segments'])} segmentów)")
        user_srt_path = out_dir / f"{user_data['file_stub']}.srt"
        user_vtt_path = out_dir / f"{user_data['file_stub']}.vtt"
        write_srt(user_data["segments"], user_srt_path)
        write_vtt(user_data["segments"], user_vtt_path)
        manifest_transcripts[user_data["user_id"]] = {
            "wav_path": user_data["raw_files"],
            "json_path": os.path.relpath(user_json_path, session_dir),
            "srt_path": os.path.relpath(user_srt_path, session_dir),
            "vtt_path": os.path.relpath(user_vtt_path, session_dir),
        }

    if conversation_srt_segments:
        base_time = session_t0 if session_t0 is not None else conversation_srt_segments[0]["start"]
        conversation_srt_path = out_dir / "all_in_one.srt"
        write_srt(conversation_srt_segments, conversation_srt_path, base=base_time)
        print(f"[✓] Globalne SRT: {conversation_srt_path}")

    conversation_path = out_dir / "conversation.json"
    with open(conversation_path, "w", encoding="utf-8") as f:
        json.dump(timeline_payload, f, ensure_ascii=False, indent=2)
    print(f"[✓] Globalna oś czasu: {conversation_path} ({len(timeline_payload['segments'])} wpisów)")

    index_path = out_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump({
            "session_dir": str(session_dir),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "items": summary_index,
            "conversation_segments": len(timeline_payload["segments"]),
        }, f, ensure_ascii=False, indent=2)
    print(f"[✓] Gotowe. Index: {index_path}")

    if manifest_transcripts:
        manifest.setdefault("transcripts", {})
        manifest["transcripts"] = manifest_transcripts
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            print(f"[✓] Uaktualniono manifest: {manifest_path}")
        except OSError as exc:
            print(f"[!] Nie udało się zaktualizować manifestu: {exc}")

if __name__ == "__main__":
    main()
