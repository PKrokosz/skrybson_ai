"""Utilities for word-level alignment using WhisperX."""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from whisperx import DiarizationPipeline, load_align_model, load_audio
from whisperx import align as whisperx_align


class AlignmentError(RuntimeError):
    """Raised when alignment could not be completed."""


@dataclass
class AlignerConfig:
    """Configuration for :class:`WhisperWordAligner`."""

    device: str = "cpu"
    language_code: Optional[str] = None
    diarize: bool = False
    diarization_auth_token: Optional[str] = None


class WhisperWordAligner:
    """Run WhisperX alignment for a given audio file and segments."""

    def __init__(self, config: Optional[AlignerConfig] = None):
        self.config = config or AlignerConfig()
        self._align_model = None
        self._align_metadata = None
        self._diarizer: Optional[DiarizationPipeline] = None

    def _ensure_align_model(self) -> None:
        if self._align_model is None or self._align_metadata is None:
            language = self.config.language_code
            self._align_model, self._align_metadata = load_align_model(
                language_code=language,
                device=self.config.device,
            )

    def _ensure_diarizer(self) -> None:
        if self._diarizer is None:
            auth_token = self.config.diarization_auth_token or os.environ.get("PYANNOTE_AUTH_TOKEN")
            self._diarizer = DiarizationPipeline(use_auth_token=auth_token, device=self.config.device)

    def align_words(
        self,
        audio_path: Path,
        segments: Sequence[Dict[str, Any]],
    ) -> List[List[Dict[str, Any]]]:
        """Return word-level timestamps for ``segments`` aligned to ``audio_path``."""

        if not segments:
            return []

        self._ensure_align_model()
        try:
            audio = load_audio(str(audio_path))
            prepared_segments = [
                {
                    "start": float(seg["start"]),
                    "end": float(seg["end"]),
                    "text": str(seg["text"]),
                }
                for seg in segments
            ]
            result = whisperx_align(
                prepared_segments,
                self._align_model,
                self._align_metadata,
                audio,
                device=self.config.device,
            )
        except Exception as exc:  # pragma: no cover - runtime safeguard
            raise AlignmentError(str(exc)) from exc

        aligned_segments = result.get("segments", [])
        words: List[List[Dict[str, Any]]] = []
        for seg in aligned_segments:
            segment_words: List[Dict[str, Any]] = []
            for word in seg.get("words", []) or []:
                start = word.get("start")
                end = word.get("end")
                token = word.get("word") or word.get("text") or word.get("token")
                if token is None or start is None or end is None:
                    continue
                token = str(token).strip()
                if not token:
                    continue
                segment_words.append({
                    "text": token,
                    "start": float(start),
                    "end": float(end),
                })
            words.append(segment_words)
        return words

    def diarize(self, audio_path: Path) -> Optional[List[Dict[str, Any]]]:
        """Run diarization for ``audio_path`` when enabled in the configuration."""

        if not self.config.diarize:
            return None

        self._ensure_diarizer()
        annotation = self._diarizer(str(audio_path))
        diarization: List[Dict[str, Any]] = []
        for segment, _, speaker in annotation.itertracks(yield_label=True):
            diarization.append(
                {
                    "speaker": str(speaker),
                    "start": float(segment.start),
                    "end": float(segment.end),
                }
            )
        return diarization


def _flatten_words(words: Iterable[Iterable[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    flattened: List[Dict[str, Any]] = []
    for segment_words in words:
        for word in segment_words:
            flattened.append(word)
    return flattened


def main() -> None:
    parser = argparse.ArgumentParser(description="Align Whisper segments to word-level timestamps.")
    parser.add_argument("audio", type=Path, help="Ścieżka do pliku audio (wav/mp3/etc.)")
    parser.add_argument("segments", type=Path, help="Plik JSON zawierający listę segmentów")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Opcjonalna ścieżka wyjściowa dla JSON z wynikami",
    )
    parser.add_argument("--device", default="cpu", help="Urządzenie: cpu lub cuda")
    parser.add_argument("--language", default=None, help="Kod języka (np. pl, en)")
    parser.add_argument(
        "--diarize",
        action="store_true",
        help="Włącz diarization (wymaga PYANNOTE_AUTH_TOKEN)",
    )
    args = parser.parse_args()

    with args.segments.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    segments = payload.get("segments")
    if segments is None:
        raise SystemExit("JSON musi zawierać klucz 'segments'.")

    aligner = WhisperWordAligner(
        AlignerConfig(
            device=args.device,
            language_code=args.language,
            diarize=args.diarize,
        )
    )
    words = aligner.align_words(args.audio, segments)
    result: Dict[str, Any] = {"words": _flatten_words(words)}
    if args.diarize:
        diarization = aligner.diarize(args.audio)
        if diarization is not None:
            result["diarization"] = diarization

    if args.output:
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    else:
        json.dump(result, sys.stdout, ensure_ascii=False, indent=2)


if __name__ == "__main__":  # pragma: no cover - CLI entrypoint
    main()
