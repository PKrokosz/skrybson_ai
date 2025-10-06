"""Unit tests for helper utilities in :mod:`transcribe`."""

from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path

from transcribe import (
    _format_timestamp,
    _parse_int,
    _strtobool_env,
    norm_text,
    parse_iso_to_epoch,
    pick_latest_session,
    sanitize_text,
    soft_merge_segments,
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
        self.assertEqual(len(first.get("words", [])), 2)

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


if __name__ == "__main__":  # pragma: no cover - manual execution
    unittest.main()
