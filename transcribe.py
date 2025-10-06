"""Utilities for transcribing recorded Discord sessions with metadata."""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    SupportsFloat,
    Type,
    cast,
)

if TYPE_CHECKING:  # pragma: no cover - only for type hints
    from faster_whisper import WhisperModel as _WhisperModelType

    from align import WhisperWordAligner
else:  # pragma: no cover - runtime import guard
    _WhisperModelType = object

_RuntimeWhisperModel: Optional[Type["_WhisperModelType"]] = None
try:  # pragma: no cover - optional runtime dependency
    from faster_whisper import WhisperModel as _ImportedWhisperModel

    _RuntimeWhisperModel = cast(Type["_WhisperModelType"], _ImportedWhisperModel)
except ImportError:  # pragma: no cover - handled at runtime
    _RuntimeWhisperModel = None


class NarrativeLogger:
    """Narrator weaving runtime events into a colourful logbook."""

    _COLOR_NARRATION = "\033[94m"
    _COLOR_EVENT = "\033[36m"
    _COLOR_SUCCESS = "\033[92m"
    _COLOR_REFLECTION = "\033[33m"
    _RESET = "\033[0m"

    def __init__(self) -> None:
        self._process_t0 = time.perf_counter()
        self._task_stack: List[tuple[str, float]] = []

    def _timestamp(self) -> str:
        elapsed = time.perf_counter() - self._process_t0
        wall_clock = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
        return f"[{wall_clock} | +{elapsed:7.2f}s]"

    @staticmethod
    def _format_context(context: Optional[Dict[str, object]]) -> str:
        if not context:
            return ""
        fragments = []
        for key, value in context.items():
            fragments.append(f"{key}: {value}")
        return " | ".join(fragments)

    def log_start(self, task_name: str, details: Optional[Dict[str, object]] = None) -> None:
        self._task_stack.append((task_name, time.perf_counter()))
        ctx = self._format_context(details)
        narrative = (
            f"Skryba unosi pióro i rozpoczyna wyprawę '{task_name}'."
            if not ctx
            else f"Skryba unosi pióro i rozpoczyna wyprawę '{task_name}' ({ctx})."
        )
        print(f"{self._timestamp()} {self._COLOR_NARRATION}{narrative}{self._RESET}")

    def log_event(self, event: str, context: Optional[Dict[str, object]] = None) -> None:
        ctx = self._format_context(context)
        storyline = (
            f"Spoglądam na scenę: {event}."
            if not ctx
            else f"Spoglądam na scenę: {event} — {ctx}."
        )
        print(f"{self._timestamp()} {self._COLOR_EVENT}{storyline}{self._RESET}")

    def log_result(
        self,
        result: str,
        stats: Optional[Dict[str, object]] = None,
        *,
        reflection: Optional[str] = None,
    ) -> None:
        task_name = ""
        duration = None
        if self._task_stack:
            task_name, start_t = self._task_stack.pop()
            duration = time.perf_counter() - start_t

        headline = (
            f"Domykam kronikę etapu '{task_name}': {result}."
            if task_name
            else f"Domykam kronikę: {result}."
        )
        print(f"{self._timestamp()} {self._COLOR_SUCCESS}{headline}{self._RESET}")

        block_lines = []
        if duration is not None:
            block_lines.append(f"czas: {duration:0.2f}s")
        if stats:
            for key, value in stats.items():
                block_lines.append(f"{key}: {value}")
        print("  == KONIEC ETAPU ==")
        for line in block_lines:
            print(f"    • {line}")

        if reflection:
            print(
                f"{self._timestamp()} {self._COLOR_REFLECTION}Refleksja systemowa: {reflection}{self._RESET}"
            )

_NOISE_RE = re.compile(
    r"(?:^|\s)(?:[?!.,:;]\s*)*(?:uhm+|um+|eh+|eee+|yyy+)(?:\s*[?!.,:;])*(?=\s|$)",
    flags=re.IGNORECASE,
)


@dataclass
class ModelAttempt:
    """Single attempt to instantiate a Whisper model."""

    device: str
    model_size: str
    compute_type: str
    reason: str


@dataclass
class TranscribeConfig:
    """Runtime configuration for whisper transcription."""

    recordings_dir: Path
    output_dir: Path
    session_dir: Optional[Path]
    requested_device: str
    model_size: str
    device: str
    compute_type: str
    beam_size: int
    language: str
    vad_filter: bool
    vad_parameters: Dict[str, int]
    sanitize_lower_noise: bool
    align_words: bool
    profile: Optional[str]
    mock_transcriber: bool


class WhisperSegment(Protocol):
    start: float
    end: float
    text: str


DEFAULT_POLICIES = {
    "cuda": {"model": "large-v3", "compute": "int8_float16"},
    "cpu": {"model": "medium", "compute": "int8"},
}


PROFILE_PRESETS: Dict[str, Dict[str, object]] = {
    "quality@cuda": {
        "device": "cuda",
        "model": "large-v3",
        "compute": "int8_float16",
        "beam": 5,
        "language": "pl",
    },
    "cpu-fallback": {
        "device": "cpu",
        "model": "medium",
        "compute": "int8",
        "beam": 3,
        "language": "pl",
    },
    "ci-mock": {
        "device": "cpu",
        "model": "tiny",
        "compute": "int8",
        "beam": 1,
        "language": "pl",
        "mock": True,
    },
}


OOM_SIGNATURES: Sequence[str] = (
    "CUDA out of memory",
    "CUDA error: out of memory",
    "failed to allocate GPU memory",
    "CUBLAS_STATUS_ALLOC_FAILED",
    "OOM",
)


def _first_not_none(*values: Optional[str], default: Optional[str] = None) -> Optional[str]:
    for value in values:
        if value is not None:
            stripped = str(value).strip()
            if stripped:
                return stripped
    return default


def _strtobool_env(value: Optional[str], default: bool) -> bool:
    if value is None:
        return default
    coerced = value.strip()
    if not coerced:
        return default
    return coerced.lower() not in {"0", "false", "no"}


def _parse_int(value: Optional[str], default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _cuda_available() -> bool:
    try:
        import torch

        return bool(getattr(torch.cuda, "is_available", lambda: False)())
    except ImportError:
        try:
            import ctranslate2

            return bool(getattr(ctranslate2, "has_cuda_device", lambda: False)())
        except ImportError:
            return False


def _normalise_punctuation(text: str) -> str:
    text = re.sub(r"([?!.,:;])\1+", r"\1", text)
    text = re.sub(r"\s+([?!.,:;])", r"\1", text)
    return text


def _require_whisper_model() -> Type["_WhisperModelType"]:
    """Return the Whisper model class or raise a runtime error."""

    if _RuntimeWhisperModel is None:
        raise RuntimeError(
            "Brak zależności faster-whisper. Zainstaluj pakiet `faster-whisper`, aby korzystać z transkrypcji.",
        )
    return _RuntimeWhisperModel


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Return CLI arguments overriding environment defaults."""

    parser = argparse.ArgumentParser(
        description="Transkrybuj nagrania Discorda przy użyciu faster-whisper.",
    )
    parser.add_argument("--recordings", type=Path, help="Katalog z nagraniami Discorda")
    parser.add_argument("--output", type=Path, help="Katalog wyjściowy na transkrypcje")
    parser.add_argument(
        "--session",
        type=Path,
        help="Konkretna sesja do przepisania (ścieżka względna lub absolutna)",
    )
    parser.add_argument(
        "--profile",
        choices=sorted(PROFILE_PRESETS.keys()),
        help="Nazwa profilu uruchomieniowego (quality@cuda, cpu-fallback, ci-mock)",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], help="Preferowane urządzenie wykonania")
    parser.add_argument("--model", help="Wymuszony rozmiar modelu whisper (np. small, medium, large-v3)")
    parser.add_argument(
        "--compute-type",
        choices=["int8_float16", "int8", "float16"],
        help="Tryb obliczeń dla modelu whisper",
    )
    parser.add_argument("--beam-size", type=int, help="Rozmiar wiązki segmentów")
    parser.add_argument("--language", help="Język docelowy dla modelu (np. pl, en)")
    parser.add_argument(
        "--vad",
        dest="vad_filter",
        action="store_true",
        help="Włącz filtrację ciszy VAD",
    )
    parser.add_argument(
        "--no-vad",
        dest="vad_filter",
        action="store_false",
        help="Wyłącz filtrację ciszy VAD",
    )
    parser.add_argument(
        "--sanitize-lower-noise",
        dest="sanitize_lower_noise",
        action="store_true",
        help="Redukuj drobne wtrącenia (uhm, eee) w wynikach",
    )
    parser.add_argument(
        "--keep-noise",
        dest="sanitize_lower_noise",
        action="store_false",
        help="Pozostaw drobne wtrącenia w tekście",
    )
    parser.add_argument(
        "--align-words",
        dest="align_words",
        action="store_true",
        help="Generuj znaczniki słów (wymaga align/pyannote)",
    )
    parser.add_argument(
        "--no-align-words",
        dest="align_words",
        action="store_false",
        help="Pomiń generowanie znaczników słów",
    )
    parser.set_defaults(vad_filter=None, sanitize_lower_noise=None, align_words=None)
    return parser.parse_args(argv)


def _cli_overrides(args: Optional[argparse.Namespace]) -> bool:
    if args is None:
        return False
    for name in (
        "recordings",
        "output",
        "session",
        "profile",
        "device",
        "model",
        "compute_type",
        "beam_size",
        "language",
        "vad_filter",
        "sanitize_lower_noise",
        "align_words",
    ):
        if getattr(args, name, None) is not None:
            return True
    return False


def load_config(args: Optional[argparse.Namespace] = None) -> TranscribeConfig:
    """Load configuration from environment variables and optional CLI overrides."""

    recordings_override = getattr(args, "recordings", None) if args else None
    if recordings_override is not None:
        recordings_dir = Path(recordings_override).expanduser().resolve()
    else:
        recordings_dir = Path(os.environ.get("RECORDINGS_DIR", "./recordings")).expanduser().resolve()

    output_override = getattr(args, "output", None) if args else None
    if output_override is not None:
        output_dir = Path(output_override).expanduser().resolve()
    else:
        output_dir = Path(os.environ.get("OUTPUT_DIR", "./out")).expanduser().resolve()

    session_override = getattr(args, "session", None) if args else None
    session_env = os.environ.get("SESSION_DIR") if session_override is None else session_override
    session_dir = None
    if session_env:
        raw_session = Path(session_env).expanduser()
        session_dir = raw_session if raw_session.is_absolute() else recordings_dir / raw_session

    profile_override = getattr(args, "profile", None) if args else None
    profile_env = os.environ.get("WHISPER_PROFILE")
    profile_name = _first_not_none(profile_override, profile_env, default=None)
    profile_defaults = PROFILE_PRESETS.get(profile_name or "", {})

    requested_device_override = getattr(args, "device", None) if args else None
    requested_device = (
        _first_not_none(
            requested_device_override,
            os.environ.get("WHISPER_DEVICE"),
            str(profile_defaults.get("device")) if profile_defaults else None,
            "cuda",
        )
        or "cuda"
    ).lower()
    if requested_device not in {"cuda", "cpu"}:
        print(
            f"[!] Nieznany WHISPER_DEVICE={requested_device}, używam domyślnego cuda.",
            file=sys.stderr,
        )
        requested_device = "cuda"

    policy_defaults = DEFAULT_POLICIES[requested_device]

    model_override = getattr(args, "model", None) if args else None
    model_candidate = _first_not_none(
        model_override,
        os.environ.get("WHISPER_MODEL"),
        str(profile_defaults.get("model")) if profile_defaults else None,
    )
    model_size = model_candidate or policy_defaults["model"]

    compute_override = getattr(args, "compute_type", None) if args else None
    compute_type = (
        _first_not_none(
            compute_override,
            os.environ.get("WHISPER_COMPUTE"),
            str(profile_defaults.get("compute")) if profile_defaults else None,
            policy_defaults["compute"],
        )
        or policy_defaults["compute"]
    ).lower()

    valid_compute_types = {"int8_float16", "int8", "float16"}
    if compute_type not in valid_compute_types:
        print(
            f"[!] Nieznany WHISPER_COMPUTE={compute_type}, używam polityki domyślnej {policy_defaults['compute']}.",
            file=sys.stderr,
        )
        compute_type = policy_defaults["compute"]

    device = requested_device
    if requested_device == "cuda" and not _cuda_available():
        print(
            "[policy] CUDA nie jest dostępna, przełączam na CPU z polityką awaryjną.",
            file=sys.stderr,
        )
        device = "cpu"
        cpu_defaults = DEFAULT_POLICIES["cpu"]
        if "WHISPER_MODEL" not in os.environ:
            model_size = cpu_defaults["model"]
        if "WHISPER_COMPUTE" not in os.environ:
            compute_type = cpu_defaults["compute"]

    beam_override = getattr(args, "beam_size", None) if args else None
    if beam_override is not None:
        beam_size = max(1, beam_override)
    else:
        profile_beam = profile_defaults.get("beam") if profile_defaults else None
        beam_size = max(
            1,
            _parse_int(
                _first_not_none(os.environ.get("WHISPER_SEGMENT_BEAM"), str(profile_beam) if profile_beam else None),
                5,
            ),
        )

    language_override = getattr(args, "language", None) if args else None
    language_candidate = _first_not_none(
        language_override,
        os.environ.get("WHISPER_LANG"),
        str(profile_defaults.get("language")) if profile_defaults else None,
    )
    language = language_candidate or "pl"

    vad_override = getattr(args, "vad_filter", None) if args else None
    vad_filter = vad_override if vad_override is not None else _strtobool_env(os.environ.get("WHISPER_VAD"), True)
    vad_parameters = {"min_silence_duration_ms": 500, "padding_duration_ms": 120}
    sanitize_override = getattr(args, "sanitize_lower_noise", None) if args else None
    sanitize_lower_noise = (
        sanitize_override
        if sanitize_override is not None
        else _strtobool_env(os.environ.get("SANITIZE_LOWER_NOISE"), False)
    )
    align_override = getattr(args, "align_words", None) if args else None
    align_words = align_override if align_override is not None else _strtobool_env(os.environ.get("WHISPER_ALIGN"), False)

    mock_default = bool(profile_defaults.get("mock", False)) if profile_defaults else False
    mock_transcriber = _strtobool_env(os.environ.get("WHISPER_MOCK"), mock_default)

    return TranscribeConfig(
        recordings_dir=recordings_dir,
        output_dir=output_dir,
        session_dir=session_dir,
        requested_device=requested_device,
        model_size=model_size,
        device=device,
        compute_type=compute_type,
        beam_size=beam_size,
        language=language,
        vad_filter=vad_filter,
        vad_parameters=vad_parameters,
        sanitize_lower_noise=sanitize_lower_noise,
        align_words=align_words,
        profile=profile_name,
        mock_transcriber=mock_transcriber,
    )


def _is_recoverable_model_error(exc: Exception) -> bool:
    message = str(exc)
    if isinstance(exc, MemoryError):
        return True
    return any(signature.lower() in message.lower() for signature in OOM_SIGNATURES)


def build_model_attempts(config: TranscribeConfig) -> List[ModelAttempt]:
    """Construct a list of model attempts with graceful degradation."""

    attempts: List[ModelAttempt] = []
    seen: set[tuple[str, str, str]] = set()

    def _add(device: str, model_size: str, compute_type: str, reason: str) -> None:
        key = (device, model_size, compute_type)
        if key in seen:
            return
        seen.add(key)
        attempts.append(
            ModelAttempt(
                device=device,
                model_size=model_size,
                compute_type=compute_type,
                reason=reason,
            )
        )

    _add(config.device, config.model_size, config.compute_type, "konfiguracja bazowa")

    if config.device == "cuda":
        _add("cuda", config.model_size, "int8", "cuda: wymuszam int8 po OOM")
        _add("cuda", "medium", "int8_float16", "cuda: zmniejszam model do medium")
        _add("cuda", "medium", "int8", "cuda: medium + int8")

    cpu_defaults = DEFAULT_POLICIES["cpu"]
    _add("cpu", cpu_defaults["model"], cpu_defaults["compute"], "CPU fallback polityki")
    _add("cpu", "small", "int8", "CPU minimalny")

    return attempts


class MockWhisperSegment:
    def __init__(self, text: str, *, start: float = 0.0, end: float = 1.0) -> None:
        self.text = text
        self.start = start
        self.end = end


class MockWhisperModel:
    """Lightweight mock used in CI to avoid downloading heavy models."""

    def __init__(self, language: str = "pl") -> None:
        self.language = language

    def transcribe(self, audio_path: str, **_kwargs: object) -> tuple[List[MockWhisperSegment], Dict[str, object]]:
        base_name = os.path.splitext(os.path.basename(audio_path))[0]
        text = f"[mock:{self.language}] {base_name}"
        segment = MockWhisperSegment(text=text)
        return [segment], {"language": self.language}


def _short_error(exc: Exception) -> str:
    message = str(exc).strip().splitlines()[0] if str(exc).strip() else exc.__class__.__name__
    return message[:160]


def load_whisper_model(
    config: TranscribeConfig,
    narrator: NarrativeLogger,
    whisper_model_cls: Type["_WhisperModelType"],
) -> "_WhisperModelType":
    """Load Whisper model honouring graceful degradation policies."""

    if config.mock_transcriber:
        narrator.log_event(
            "Profil mock - pomijam ładowanie prawdziwego modelu",
            {"język": config.language},
        )
        return MockWhisperModel(language=config.language)

    attempts = build_model_attempts(config)
    baseline_signature = f"{config.model_size}/{config.compute_type}@{config.device}"

    for attempt in attempts:
        narrator.log_event(
            "Próba inicjalizacji wariantu",
            {
                "model": attempt.model_size,
                "device": attempt.device,
                "compute": attempt.compute_type,
                "powód": attempt.reason,
            },
        )
        try:
            model = whisper_model_cls(
                attempt.model_size,
                device=attempt.device,
                compute_type=attempt.compute_type,
            )
        except Exception as exc:  # pragma: no cover - runtime fallback
            if _is_recoverable_model_error(exc):
                narrator.log_event(
                    "Model odrzucił wariant",
                    {
                        "powód": _short_error(exc),
                        "próbowałem": f"{attempt.model_size}/{attempt.compute_type}@{attempt.device}",
                    },
                )
                continue
            raise

        if attempt.reason != "konfiguracja bazowa":
            narrator.log_event(
                "Przełączam się na plan awaryjny",
                {
                    "wcześniej": baseline_signature,
                    "teraz": f"{attempt.model_size}/{attempt.compute_type}@{attempt.device}",
                },
            )

        config.device = attempt.device
        config.model_size = attempt.model_size
        config.compute_type = attempt.compute_type
        return model

    raise RuntimeError("Żaden wariant modelu nie został zainicjalizowany")


def sanitize_text(text: str, *, lower_noise: bool = False) -> str:
    """Normalise whitespace and tame repeated punctuation."""

    text = re.sub(r"\s+", " ", text.strip())
    text = _normalise_punctuation(text)
    if lower_noise:
        text = _NOISE_RE.sub(" ", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = _normalise_punctuation(text).strip()
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

    def _clone_segment(data: Dict[str, object]) -> Dict[str, object]:
        cloned = dict(data)
        if "words" in cloned and isinstance(cloned["words"], list):
            cloned["words"] = [dict(word) for word in cloned["words"]]  # shallow copy
        return cloned

    merged: List[Dict[str, object]] = [_clone_segment(segments[0])]
    for seg in segments[1:]:
        current = _clone_segment(seg)
        prev = merged[-1]
        current_start = cast(float, current["start"])
        current_end = cast(float, current["end"])
        prev_start = cast(float, prev["start"])
        prev_end = cast(float, prev["end"])
        gap = current_start - prev_end
        current_duration = current_end - current_start
        prev_duration = prev_end - prev_start
        same_user = True
        if user_key is not None:
            same_user = prev.get(user_key) == current.get(user_key)

        if (
            same_user
            and 0 <= gap <= max_gap
            and (current_duration < short_threshold or prev_duration < short_threshold)
        ):
            prev["end"] = max(prev_end, current_end)
            prev_text = str(prev.get("text", ""))
            curr_text = str(current.get("text", ""))
            prev["text"] = sanitize_text(f"{prev_text} {curr_text}", lower_noise=lower_noise)
            if "files" in current:
                prev.setdefault("files", [])
                prev_files = list(cast(Iterable[str], prev.get("files", [])))
                prev_files.extend(list(cast(Iterable[str], current.get("files", []))))
                prev["files"] = prev_files
            if "words" in current or "words" in prev:
                prev_words = list(
                    cast(Iterable[Dict[str, object]], prev.get("words", []))
                )
                curr_words = cast(Iterable[Dict[str, object]], current.get("words", []))
                prev_words.extend(list(curr_words))
                prev["words"] = prev_words
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


def write_srt(
    segments: Iterable[Mapping[str, object]], path: Path, *, base: float = 0.0
) -> None:
    lines = []
    for idx, seg in enumerate(segments, start=1):
        start = float(cast(SupportsFloat, seg["start"])) - base
        end = float(cast(SupportsFloat, seg["end"])) - base
        lines.append(str(idx))
        lines.append(
            f"{_format_timestamp(start, separator=',')} --> {_format_timestamp(end, separator=',')}"
        )
        lines.append(str(seg.get("text", "")).strip())
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_vtt(
    segments: Iterable[Mapping[str, object]], path: Path, *, base: float = 0.0
) -> None:
    lines = ["WEBVTT", ""]
    for seg in segments:
        start = float(cast(SupportsFloat, seg["start"])) - base
        end = float(cast(SupportsFloat, seg["end"])) - base
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

    args = parse_args()
    narrator = NarrativeLogger()
    source = "argumenty CLI + zmienne środowiskowe" if _cli_overrides(args) else "zmienne środowiskowe"
    narrator.log_start("Konfiguracja transkrypcji", {"źródło": source})
    config = load_config(args)
    narrator.log_event(
        "Sprawdzam katalogi robocze",
        {
            "nagrania": config.recordings_dir,
            "wyniki": config.output_dir,
        },
    )

    if not config.recordings_dir.exists():
        narrator.log_event("Nie odnajduję katalogu nagrań", {"ścieżka": config.recordings_dir})
        narrator.log_result(
            "Konfiguracja zatrzymała się na brakujących nagraniach",
            {"status": "brak recordings_dir"},
            reflection="Czasem nawet Skryba nie zapisze historii, jeśli nie ma gdzie zajrzeć.",
        )
        sys.exit(1)

    session_dir = _resolve_session_path(config)
    if not session_dir:
        narrator.log_event("W archiwum cisza, brak bieżącej sesji", {"recordings": config.recordings_dir})
        narrator.log_result(
            "Nie znalazłem sesji do przepisania",
            {"status": "brak katalogu sesji"},
            reflection="Cisza w logach bywa równie wymowna jak najgłośniejsza sesja.",
        )
        sys.exit(1)

    raw_dir = session_dir / "raw"
    if not raw_dir.exists():
        narrator.log_event("Sesja pozbawiona katalogu RAW", {"ścieżka": raw_dir})
        narrator.log_result(
            "Brak surowych danych uniemożliwił start",
            {"status": "brak raw"},
            reflection="Bez surowych fal głosowych nie powstanie żadna ballada.",
        )
        sys.exit(1)

    output_session_dir = config.output_dir / _relative_session_path(config, session_dir)
    out_dir = output_session_dir / "transcripts"
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob(str(raw_dir / "*.wav")))
    files = [f for f in files if os.path.getsize(f) >= 1024]
    if not files:
        narrator.log_event("W katalogu RAW panuje cisza", {"ścieżka": raw_dir})
        narrator.log_result(
            "Brak materiału do transkrypcji",
            {"pliki_wav": 0},
            reflection="Czasem najlepszą decyzją jest odnotować ciszę i ruszyć dalej.",
        )
        sys.exit(0)

    narrator.log_event(
        "Dobieram politykę wykonania",
        {
            "urządzenie": f"{config.requested_device}→{config.device}",
            "model": config.model_size,
            "compute": config.compute_type,
            "beam": config.beam_size,
        },
    )
    if config.profile:
        narrator.log_event("Profil wykonania", {"profil": config.profile})
    if config.mock_transcriber:
        narrator.log_event("Tryb transkrypcji", {"wariant": "mock"})
    narrator.log_event(
        "Analizuję filtry i dodatki",
        {
            "VAD": "on" if config.vad_filter else "off",
            "filtr_szumów": "on" if config.sanitize_lower_noise else "off",
            "word_align": "on" if config.align_words else "off",
            "język": config.language,
        },
    )

    narrator.log_result(
        "Konfiguracja gotowa",
        {
            "sesja": session_dir,
            "katalog_wyjściowy": out_dir,
            "pliki_wav": len(files),
        },
        reflection="Dobrze przygotowana mapa sprawia, że dalsza podróż staje się przyjemnością.",
    )

    narrator.log_start(
        "Ładowanie modelu Whisper",
        {
            "model": config.model_size,
            "device": config.device,
            "compute": config.compute_type,
            "profil": config.profile or "brak",
        },
    )
    try:
        whisper_model_cls: Type["_WhisperModelType"]
        if config.mock_transcriber:
            whisper_model_cls = cast(Type["_WhisperModelType"], MockWhisperModel)
        else:
            whisper_model_cls = _require_whisper_model()
    except RuntimeError as exc:
        narrator.log_event("Nie mogę przywołać klasy modelu", {"powód": exc})
        narrator.log_result(
            "Ładowanie modelu nie powiodło się",
            {"status": "brak faster-whisper"},
            reflection="Nawet najcierpliwszy kronikarz potrzebuje narzędzi, by pisać dalej.",
        )
        sys.exit(1)

    try:
        model = load_whisper_model(config, narrator, whisper_model_cls)
    except RuntimeError as exc:  # pragma: no cover - skrajny przypadek
        narrator.log_event(
            "Wyczerpałem wszystkie warianty modelu",
            {"powód": _short_error(exc)},
        )
        narrator.log_result(
            "Nie udało się rozgrzać rdzeni Whispera",
            {"status": "brak działających wariantów"},
            reflection="Czasem trzeba odłożyć pióro, gdy pergamin zaczyna płonąć.",
        )
        sys.exit(1)

    narrator.log_result(
        "Model czeka w pogotowiu",
        {
            "klasa": type(model).__name__,
            "device": config.device,
            "model": config.model_size,
            "compute": config.compute_type,
        },
        reflection="GPU mruczy z zadowoleniem, gotowa na falę fonemów.",
    )

    aligner: Optional["WhisperWordAligner"] = None
    if config.align_words:
        narrator.log_start(
            "Przygotowanie alignera WhisperX",
            {"język": config.language, "device": config.device},
        )
        try:
            from align import AlignerConfig as WhisperAlignConfig
            from align import WhisperWordAligner
        except (RuntimeError, ImportError) as exc:
            narrator.log_event("Aligner odmówił współpracy", {"powód": exc})
            config.align_words = False
            narrator.log_result(
                "Słowny alignment wyłączony",
                {"status": "aligner niedostępny"},
                reflection="Nawet bez słów-perł nasza opowieść może płynąć dalej.",
            )
        else:
            aligner = WhisperWordAligner(
                WhisperAlignConfig(device=config.device, language_code=config.language)
            )
            narrator.log_result(
                "Aligner przygotowany",
                {"klasa": WhisperWordAligner.__name__},
                reflection="Synchronizuję sylaby niczym dyrygent orkiestry czasu.",
            )
    else:
        narrator.log_event("Pomijam alignment słów", {"status": "wyłączony"})

    buckets: Dict[str, list[str]] = {}
    for f in files:
        name = os.path.basename(f)
        user_prefix = name.rsplit("_seg", 1)[0]
        buckets.setdefault(user_prefix, []).append(f)

    narrator.log_start(
        "Inferencja segmentów",
        {"użytkownicy": len(buckets)},
    )

    summary_index = []
    conversation_segments = []
    user_payloads = []
    manifest_transcripts: Dict[str, Dict[str, object]] = {}
    users_processed = 0
    total_segments = 0

    for user_prefix, wavs in buckets.items():
        wavs.sort(key=lambda x: os.path.getmtime(x))
        timeline = []
        raw_wavs: List[str] = []

        narrator.log_event(
            "Rozpoczynam nasłuch użytkownika",
            {"użytkownik": user_prefix, "pliki": len(wavs)},
        )

        id_candidate = user_prefix.rsplit("_", 1)[-1]
        if not id_candidate.isdigit():
            id_candidate = re.sub(r"[^0-9A-Za-z]+", "_", user_prefix).strip("_") or user_prefix
        file_stub = f"user_{id_candidate}"

        for wav in wavs:
            try:
                file_t0 = os.path.getmtime(wav)
                narrator.log_event(
                    "Otwieram falę dźwięku",
                    {"plik": os.path.basename(wav), "mtime": file_t0},
                )
                raw_wavs.append(os.path.relpath(wav, session_dir))
                transcribe_kwargs = dict(
                    beam_size=config.beam_size,
                    language=config.language,
                    vad_filter=config.vad_filter,
                )
                if config.vad_filter:
                    transcribe_kwargs["vad_parameters"] = config.vad_parameters
                raw_segments, _info = model.transcribe(wav, **transcribe_kwargs)
                typed_segments = list(cast(Iterable[WhisperSegment], raw_segments))
                segment_items: List[Dict[str, object]] = []
                for seg in typed_segments:
                    clean_text = sanitize_text(seg.text, lower_noise=config.sanitize_lower_noise)
                    if not clean_text:
                        continue
                    item = {
                        "start": float(seg.start),
                        "end": float(seg.end),
                        "text": clean_text,
                        "file": os.path.basename(wav),
                    }
                    item["pseudo_t"] = file_t0 + item["start"]
                    segment_items.append(item)
                    narrator.log_event(
                        "Segment dopisany do pergaminu",
                        {
                            "plik": os.path.basename(wav),
                            "zakres_s": f"{item['start']:.2f}-{item['end']:.2f}",
                            "tekst": clean_text,
                        },
                    )

                word_segments: List[List[Dict[str, object]]] = []
                if aligner and segment_items:
                    align_payload = [
                        {"start": it["start"], "end": it["end"], "text": it["text"]}
                        for it in segment_items
                    ]
                    try:
                        word_segments = aligner.align_words(Path(wav), align_payload)
                    except Exception as exc:
                        narrator.log_event(
                            "WhisperX nie zgrał słów",
                            {"plik": os.path.basename(wav), "powód": exc},
                        )
                        word_segments = []

                for idx, item in enumerate(segment_items):
                    words_audio = word_segments[idx] if idx < len(word_segments) else []
                    words_audio = [dict(word) for word in words_audio] if words_audio else []
                    timeline.append(
                        {
                            "pseudo_t": item["pseudo_t"],
                            "start": item["start"],
                            "end": item["end"],
                            "text": item["text"],
                            "file": item["file"],
                            "words_audio": words_audio,
                        }
                    )
            except Exception as e:
                narrator.log_event(
                    "Pomijam uszkodzony plik",
                    {"plik": os.path.basename(wav), "powód": e},
                )
                continue

        timeline.sort(key=lambda x: x["pseudo_t"])

        if not timeline:
            narrator.log_event("Brak segmentów po transkrypcji", {"użytkownik": user_prefix})
            continue

        deduped = []
        last_norm = ""
        last_t = -1e9
        for item in timeline:
            nt = norm_text(item["text"])
            if deduped and nt == last_norm and (item["pseudo_t"] - last_t) < 1.5:
                narrator.log_event(
                    "Pomijam duplikat wypowiedzi",
                    {"tekst": item["text"], "pseudo_t": item["pseudo_t"]},
                )
                continue
            deduped.append(item)
            last_norm = nt
            last_t = item["pseudo_t"]
        timeline = deduped

        session_segments = []
        segments_all = []
        user_id = user_prefix.rsplit("_", 1)[-1]
        for item in timeline:
            conversation_segments.append(
                {
                    "user": user_prefix,
                    "text": item["text"],
                    "start": item["pseudo_t"],
                    "end": item["pseudo_t"] + (item["end"] - item["start"]),
                    "files": [item["file"]],
                    "user_id": user_id,
                }
            )
            segments_all.append(
                {
                    "start": item["start"],
                    "end": item["end"],
                    "text": item["text"],
                    "file": item["file"],
                    "session_epoch": item["pseudo_t"],
                    "words": item.get("words_audio", []),
                }
            )
            session_segments.append(
                {
                    "start": item["pseudo_t"],
                    "end": item["pseudo_t"] + (item["end"] - item["start"]),
                    "text": item["text"],
                }
            )

        write_srt(session_segments, out_dir / f"{file_stub}_session.srt")

        user_payloads.append(
            {
                "user": user_prefix,
                "user_id": user_id,
                "file_stub": file_stub,
                "segments": segments_all,
                "raw_files": raw_wavs,
            }
        )
        summary_index.append({"user": user_prefix, "segments": len(segments_all)})
        total_segments += len(segments_all)
        users_processed += 1

        narrator.log_event(
            "Zamykam rozdział użytkownika",
            {"użytkownik": user_prefix, "segmenty": len(segments_all)},
        )

    if not conversation_segments:
        narrator.log_event("Brak danych do osi czasu rozmowy", None)

    narrator.log_result(
        "Inferencja zakończona",
        {
            "użytkownicy": users_processed,
            "segmenty": total_segments,
            "globalne_segmenty": len(conversation_segments),
        },
        reflection="Po burzy fonemów zostają schludne akapity wspomnień.",
    )

    narrator.log_start("Postprocessing i budowa osi czasu", None)

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
            narrator.log_event(
                "Wiele krótkich globalnych segmentów",
                {"odsetek": f"{short_ratio:.0%}"},
            )

    manifest_path = session_dir / "manifest.json"
    manifest: Dict[str, object] = {}
    manifest_start_iso = None
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest_start_iso = manifest.get("startISO")
            narrator.log_event("Odczytałem manifest sesji", {"plik": manifest_path})
        except (json.JSONDecodeError, OSError) as exc:
            narrator.log_event("Manifest okazał się kapryśny", {"powód": exc})

    session_t0 = parse_iso_to_epoch(manifest_start_iso)
    if session_t0 is None and conversation_segments:
        session_t0 = conversation_segments[0]["start"]
        first_segment_start = session_t0
        manifest_start_iso = (
            datetime.fromtimestamp(first_segment_start, tz=timezone.utc)
            .isoformat()
            .replace("+00:00", "Z")
        )
        narrator.log_event(
            "Szacuję początek sesji na bazie pierwszego segmentu",
            {"epoch": session_t0},
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
            timeline_payload["segments"].append(
                {
                    "user": seg["user"],
                    "user_id": seg.get("user_id"),
                    "text": seg["text"],
                    "start": relative_start,
                    "end": relative_end,
                    "absolute_start": absolute_start,
                    "files": seg.get("files", []),
                }
            )

    conversation_srt_segments: List[Dict[str, object]] = []
    for seg in conversation_segments:
        conversation_srt_segments.append(
            {
                "start": seg["start"],
                "end": seg["end"],
                "text": f"{seg['user']}: {seg['text']}",
            }
        )

    narrator.log_result(
        "Postprocessing zakończony",
        {
            "segmenty_po_scaleniu": len(conversation_segments),
            "segmenty_timeline": len(timeline_payload["segments"]),
        },
        reflection="Fakty układają się w rozdziały niczym kartki spięte złotą nicią.",
    )

    narrator.log_start("Zapis wyników", {"folder": out_dir})

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
        narrator.log_event(
            "Zapisuję indywidualny pergamin",
            {"plik": user_json_path, "segmenty": len(user_data["segments"])},
        )
        user_srt_path = out_dir / f"{user_data['file_stub']}.srt"
        user_vtt_path = out_dir / f"{user_data['file_stub']}.vtt"
        write_srt(user_data["segments"], user_srt_path)
        write_vtt(user_data["segments"], user_vtt_path)
        narrator.log_event(
            "Tworzę formaty SRT i VTT",
            {"srt": user_srt_path, "vtt": user_vtt_path},
        )
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
        narrator.log_event("Generuję wspólne SRT", {"plik": conversation_srt_path})

    conversation_path = out_dir / "conversation.json"
    with open(conversation_path, "w", encoding="utf-8") as f:
        json.dump(timeline_payload, f, ensure_ascii=False, indent=2)
    narrator.log_event(
        "Aktualizuję globalną oś czasu",
        {"plik": conversation_path, "wpisy": len(timeline_payload["segments"])},
    )

    index_path = out_dir / "index.json"
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "session_dir": str(session_dir),
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "items": summary_index,
                "conversation_segments": len(timeline_payload["segments"]),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    narrator.log_event(
        "Spisuję indeks",
        {"plik": index_path, "użytkownicy": len(summary_index)},
    )

    if manifest_transcripts:
        manifest.setdefault("transcripts", {})
        manifest["transcripts"] = manifest_transcripts
        try:
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, ensure_ascii=False, indent=2)
            narrator.log_event("Odświeżam manifest", {"plik": manifest_path})
        except OSError as exc:
            narrator.log_event("Manifest odmówił zapisu", {"powód": exc})

    narrator.log_result(
        "Zapisy zakończone",
        {
            "pliki_użytkowników": len(user_payloads),
            "globalne_wpisy": len(timeline_payload["segments"]),
        },
        reflection="Archiwa napełniły się nowymi rozdziałami – można odkładać pióro.",
    )

if __name__ == "__main__":
    main()
