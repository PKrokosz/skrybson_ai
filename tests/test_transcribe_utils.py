"""Unit tests for helper utilities in :mod:`transcribe`."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path
from typing import Optional, cast
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from transcribe import (
    MockWhisperModel,
    ModelAttempt,
    TranscribeConfig,
    _cli_overrides,
    _format_timestamp,
    _is_recoverable_model_error,
    _parse_int,
    _resolve_session_path,
    _strtobool_env,
    build_model_attempts,
    load_config,
    norm_text,
    parse_iso_to_epoch,
    pick_latest_session,
    sanitize_text,
    soft_merge_segments,
    write_srt,
    write_vtt,
)


class StrToBoolEnvTest(unittest.TestCase):
    def test_truthy_values(self) -> None:
        self.assertTrue(_strtobool_env("1", False))
        self.assertTrue(_strtobool_env(" yes ", False))

    def test_falsy_values(self) -> None:
        self.assertFalse(_strtobool_env("0", True))
        self.assertFalse(_strtobool_env("No", True))

    def test_defaults(self) -> None:
        self.assertTrue(_strtobool_env(None, True))
        self.assertFalse(_strtobool_env("   ", False))


class ParseIntTest(unittest.TestCase):
    def test_parse_valid_int(self) -> None:
        self.assertEqual(_parse_int("10", 5), 10)

    def test_parse_invalid_int(self) -> None:
        self.assertEqual(_parse_int("foo", 5), 5)
        self.assertEqual(_parse_int(None, 7), 7)


class SanitizeTextTest(unittest.TestCase):
    def test_basic_cleanup(self) -> None:
        self.assertEqual(sanitize_text(" Hello   world!!! "), "Hello world!")

    def test_lower_noise(self) -> None:
        noisy = "Uhm... to jest, eee, test?!"
        self.assertEqual(sanitize_text(noisy, lower_noise=True), "to jest, test?!")


class SoftMergeSegmentsTest(unittest.TestCase):
    def test_merge_short_segments_same_user(self) -> None:
        segments = [
            {
                "start": 0.0,
                "end": 0.4,
                "text": "Cześć",
                "user": "alice",
                "files": ["a.wav"],
                "words": [{"text": "Cześć", "start": 0.0, "end": 0.4}],
            },
            {
                "start": 0.45,
                "end": 0.9,
                "text": "hej",
                "user": "alice",
                "files": ["b.wav"],
                "words": [{"text": "hej", "start": 0.45, "end": 0.9}],
            },
            {"start": 2.0, "end": 3.0, "text": "co tam", "user": "bob"},
        ]

        merged = soft_merge_segments(
            segments,
            user_key="user",
            max_gap=0.6,
            short_threshold=1.0,
            lower_noise=True,
        )

        self.assertEqual(len(merged), 2)
        first = merged[0]
        self.assertEqual(first["text"], "Cześć hej")
        self.assertEqual(first["files"], ["a.wav", "b.wav"])
        words = cast(list, first.get("words", []))
        self.assertEqual(len(words), 2)

    def test_empty_input(self) -> None:
        self.assertEqual(soft_merge_segments([]), [])


class NormalisationHelpersTest(unittest.TestCase):
    def test_norm_text(self) -> None:
        self.assertEqual(norm_text("  Héllo, Wórld!  "), "héllo wórld")

    def test_parse_iso_to_epoch(self) -> None:
        iso_value = "2024-01-01T12:00:00Z"
        epoch = parse_iso_to_epoch(iso_value)
        self.assertIsInstance(epoch, float)
        self.assertEqual(parse_iso_to_epoch(""), None)
        self.assertEqual(parse_iso_to_epoch("not-a-date"), None)


class TimestampFormattingTest(unittest.TestCase):
    def test_format_timestamp_vtt(self) -> None:
        self.assertEqual(_format_timestamp(1.234, separator="."), "00:00:01.234")

    def test_format_timestamp_srt(self) -> None:
        self.assertEqual(_format_timestamp(3661.2, separator=","), "01:01:01,200")


class PickLatestSessionTest(unittest.TestCase):
    def test_pick_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            first = base / "session1"
            second = base / "session2"
            first.mkdir()
            second.mkdir()

            # Ensure the second directory has a newer mtime
            time.sleep(0.01)
            os.utime(second, None)

            self.assertEqual(pick_latest_session(base), second)


class BuildModelAttemptsTest(unittest.TestCase):
    def test_cuda_attempts_include_fallbacks(self) -> None:
        cfg = TranscribeConfig(
            recordings_dir=Path("/tmp/rec"),
            output_dir=Path("/tmp/out"),
            session_dir=None,
            requested_device="cuda",
            model_size="large-v3",
            device="cuda",
            compute_type="int8_float16",
            beam_size=5,
            language="pl",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            sanitize_lower_noise=False,
            align_words=False,
            profile="quality@cuda",
            mock_transcriber=False,
        )

        attempts = build_model_attempts(cfg)

        self.assertEqual(
            attempts[0],
            ModelAttempt("cuda", "large-v3", "int8_float16", "konfiguracja bazowa"),
        )
        self.assertIn(
            ModelAttempt("cuda", "large-v3", "int8", "cuda: wymuszam int8 po OOM"),
            attempts,
        )
        self.assertIn(
            ModelAttempt("cpu", "medium", "int8", "CPU fallback polityki"),
            attempts,
        )


class LoadConfigProfileTest(unittest.TestCase):
    def test_ci_mock_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recordings_dir = Path(tmpdir) / "recordings"
            output_dir = Path(tmpdir) / "out"
            recordings_dir.mkdir()
            output_dir.mkdir()
            with patch.dict(
                os.environ,
                {
                    "RECORDINGS_DIR": str(recordings_dir),
                    "OUTPUT_DIR": str(output_dir),
                    "WHISPER_PROFILE": "ci-mock",
                },
                clear=True,
            ):
                config = load_config(None)

        self.assertEqual(config.profile, "ci-mock")
        self.assertTrue(config.mock_transcriber)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.model_size, "tiny")
        self.assertEqual(config.compute_type, "int8")
        self.assertEqual(config.beam_size, 1)
        self.assertEqual(config.language, "pl")


class LoadConfigFallbackTest(unittest.TestCase):
    def test_cuda_fallback_to_cpu_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recordings_dir = Path(tmpdir) / "recordings"
            output_dir = Path(tmpdir) / "out"
            recordings_dir.mkdir()
            output_dir.mkdir()

            with patch.dict(
                os.environ,
                {
                    "RECORDINGS_DIR": str(recordings_dir),
                    "OUTPUT_DIR": str(output_dir),
                    "WHISPER_DEVICE": "cuda",
                },
                clear=True,
            ):
                with patch("transcribe._cuda_available", return_value=False):
                    config = load_config(None)

        self.assertEqual(config.requested_device, "cuda")
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.model_size, "medium")
        self.assertEqual(config.compute_type, "int8")


class MockWhisperModelTest(unittest.TestCase):
    def test_mock_transcribe_returns_placeholder(self) -> None:
        model = MockWhisperModel(language="pl")
        segments, info = model.transcribe("foo/bar.wav")

        self.assertEqual(len(segments), 1)
        self.assertTrue(segments[0].text.startswith("[mock:pl]"))
        self.assertEqual(info["language"], "pl")


class CliHelpersTest(unittest.TestCase):
    def test_cli_overrides_detects_any_override(self) -> None:
        args = argparse.Namespace(
            recordings=Path("/tmp/rec"),
            output=None,
            session=None,
            profile=None,
            device=None,
            model=None,
            compute_type=None,
            beam_size=None,
            language=None,
            vad_filter=None,
            sanitize_lower_noise=None,
            align_words=None,
        )
        self.assertTrue(_cli_overrides(args))

        none_args = argparse.Namespace(
            recordings=None,
            output=None,
            session=None,
            profile=None,
            device=None,
            model=None,
            compute_type=None,
            beam_size=None,
            language=None,
            vad_filter=None,
            sanitize_lower_noise=None,
            align_words=None,
        )
        self.assertFalse(_cli_overrides(none_args))


class SessionResolutionTest(unittest.TestCase):
    def _make_config(self, recordings_dir: Path, session_dir: Optional[Path]) -> TranscribeConfig:
        return TranscribeConfig(
            recordings_dir=recordings_dir,
            output_dir=recordings_dir / "out",
            session_dir=session_dir,
            requested_device="cpu",
            model_size="small",
            device="cpu",
            compute_type="int8",
            beam_size=1,
            language="pl",
            vad_filter=False,
            vad_parameters={},
            sanitize_lower_noise=False,
            align_words=False,
            profile=None,
            mock_transcriber=False,
        )

    def test_resolve_prefers_explicit_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recordings_dir = Path(tmpdir) / "rec"
            recordings_dir.mkdir()
            explicit = recordings_dir / "sessionA"
            explicit.mkdir()

            config = self._make_config(recordings_dir, explicit)
            resolved = _resolve_session_path(config)

        self.assertEqual(resolved, explicit.resolve())

    def test_resolve_falls_back_to_latest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            recordings_dir = Path(tmpdir) / "rec"
            recordings_dir.mkdir()
            first = recordings_dir / "session1"
            second = recordings_dir / "session2"
            first.mkdir()
            second.mkdir()
            time.sleep(0.01)
            os.utime(second, None)

            config = self._make_config(recordings_dir, None)
            resolved = _resolve_session_path(config)

        self.assertEqual(resolved, second.resolve())


class TimestampWritersTest(unittest.TestCase):
    def test_write_srt_and_vtt_outputs(self) -> None:
        segments = [
            {"start": 0.0, "end": 1.0, "text": "Hello"},
            {"start": 1.5, "end": 3.0, "text": "World"},
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            srt_path = base / "out.srt"
            vtt_path = base / "out.vtt"

            write_srt(segments, srt_path)
            write_vtt(segments, vtt_path, base=0.5)

            srt_content = srt_path.read_text(encoding="utf-8").splitlines()
            vtt_content = vtt_path.read_text(encoding="utf-8").splitlines()

        self.assertEqual(srt_content[0], "1")
        self.assertEqual(srt_content[1], "00:00:00,000 --> 00:00:01,000")
        self.assertEqual(vtt_content[0], "WEBVTT")
        self.assertIn("00:00:01.000 --> 00:00:02.500", vtt_content)


class RecoverableErrorsTest(unittest.TestCase):
    def test_memory_errors_are_recoverable(self) -> None:
        self.assertTrue(_is_recoverable_model_error(MemoryError()))
        oom = RuntimeError("CUDA out of memory while loading model")
        self.assertTrue(_is_recoverable_model_error(oom))
        other = RuntimeError("network unavailable")
        self.assertFalse(_is_recoverable_model_error(other))


if __name__ == "__main__":  # pragma: no cover - manual execution
    unittest.main()
