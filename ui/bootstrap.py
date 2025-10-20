"""Utility helpers for loading ttkbootstrap with a graceful fallback."""
from __future__ import annotations

from importlib import import_module
from types import ModuleType
from typing import Tuple

__all__ = ["ttk", "ToolTip", "TTK_VERSION", "TTK_IMPORT_ERROR"]


def _load_ttkbootstrap() -> Tuple[ModuleType, type, str, Exception | None]:
    try:
        ttk_module = import_module("ttkbootstrap")
        tooltip_mod = import_module("ttkbootstrap.tooltip")
        tool_tip = tooltip_mod.ToolTip
        version = getattr(ttk_module, "__version__", "unknown")
        return ttk_module, tool_tip, version, None
    except Exception as exc:  # pragma: no cover - executed in headless CI
        ttk_module = import_module("ui._compat.ttkbootstrap_stub")
        tool_tip = ttk_module.ToolTip
        return ttk_module, tool_tip, "stub", exc


ttk, ToolTip, TTK_VERSION, TTK_IMPORT_ERROR = _load_ttkbootstrap()
