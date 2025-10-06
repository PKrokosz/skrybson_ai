"""Application state management for Skrybson GUI."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, RLock
from typing import Any, Dict, List, MutableMapping, Optional

CONFIG_DIR = Path.home() / ".skrybson"
CONFIG_PATH = CONFIG_DIR / "config.json"


@dataclass(slots=True)
class Profile:
    """Representation of a compute profile."""

    name: str
    description: str
    defaults: Dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class AppConfig:
    """Persistent configuration options."""

    recordings_dir: Path = Path.cwd() / "recordings"
    output_dir: Path = Path.cwd() / "outputs"
    active_profile: str = "quality@cuda"
    profiles: List[Profile] = field(default_factory=list)

    @classmethod
    def from_dict(cls, payload: MutableMapping[str, Any]) -> "AppConfig":
        profiles = [
            Profile(
                name=item.get("name", "custom"),
                description=item.get("description", ""),
                defaults=item.get("defaults", {}),
            )
            for item in payload.get("profiles", [])
        ]
        recordings_dir = Path(payload.get("recordings_dir", cls.recordings_dir))
        output_dir = Path(payload.get("output_dir", cls.output_dir))
        active_profile = payload.get("active_profile", cls.active_profile)
        return cls(
            recordings_dir=recordings_dir,
            output_dir=output_dir,
            active_profile=active_profile,
            profiles=profiles,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "recordings_dir": str(self.recordings_dir),
            "output_dir": str(self.output_dir),
            "active_profile": self.active_profile,
            "profiles": [
                {
                    "name": profile.name,
                    "description": profile.description,
                    "defaults": profile.defaults,
                }
                for profile in self.profiles
            ],
        }


class AppState:
    """Thread-safe container for shared runtime state."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._config = AppConfig()
        self._log_buffer: List[str] = []
        self._log_event = Event()
        self._active_view: str = "sessions"
        self._load_config()

    # ------------------------------------------------------------------
    # Configuration
    # ------------------------------------------------------------------
    def _load_config(self) -> None:
        if not CONFIG_PATH.exists():
            CONFIG_DIR.mkdir(parents=True, exist_ok=True)
            self._config.profiles = self._default_profiles()
            self.save_config()
            return
        with CONFIG_PATH.open("r", encoding="utf8") as fp:
            payload = json.load(fp)
        self._config = AppConfig.from_dict(payload)
        if not self._config.profiles:
            self._config.profiles = self._default_profiles()

    def _default_profiles(self) -> List[Profile]:
        return [
            Profile(
                name="quality@cuda",
                description="High quality CUDA profile",
                defaults={
                    "WHISPER_MODEL": "large-v3",
                    "WHISPER_DEVICE": "cuda",
                    "WHISPER_COMPUTE": "float16",
                },
            ),
            Profile(
                name="cpu-fallback",
                description="CPU friendly configuration",
                defaults={
                    "WHISPER_MODEL": "medium",
                    "WHISPER_DEVICE": "cpu",
                    "WHISPER_COMPUTE": "int8",
                },
            ),
            Profile(
                name="custom",
                description="Custom parameters",
                defaults={},
            ),
        ]

    @property
    def config(self) -> AppConfig:
        with self._lock:
            return self._config

    def update_paths(self, *, recordings_dir: Optional[Path] = None, output_dir: Optional[Path] = None) -> None:
        with self._lock:
            if recordings_dir is not None:
                self._config.recordings_dir = recordings_dir
            if output_dir is not None:
                self._config.output_dir = output_dir
        self.save_config()

    def set_active_profile(self, profile_name: str) -> None:
        with self._lock:
            self._config.active_profile = profile_name
        self.save_config()

    def get_active_profile(self) -> Profile:
        with self._lock:
            for profile in self._config.profiles:
                if profile.name == self._config.active_profile:
                    return profile
        return self._config.profiles[0]

    def save_config(self) -> None:
        CONFIG_DIR.mkdir(parents=True, exist_ok=True)
        with CONFIG_PATH.open("w", encoding="utf8") as fp:
            json.dump(self._config.to_dict(), fp, indent=2)

    # ------------------------------------------------------------------
    # Logs
    # ------------------------------------------------------------------
    def push_log(self, message: str) -> None:
        with self._lock:
            self._log_buffer.append(message)
        self._log_event.set()

    def pop_logs(self) -> List[str]:
        with self._lock:
            logs = self._log_buffer[:]
            self._log_buffer.clear()
        self._log_event.clear()
        return logs

    def wait_for_logs(self, timeout: float | None = None) -> bool:
        return self._log_event.wait(timeout)

    # ------------------------------------------------------------------
    # View navigation
    # ------------------------------------------------------------------
    def get_active_view(self) -> str:
        with self._lock:
            return self._active_view

    def set_active_view(self, view: str) -> None:
        with self._lock:
            self._active_view = view


__all__ = ["AppState", "AppConfig", "Profile"]
