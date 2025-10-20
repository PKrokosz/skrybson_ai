"""Utilities for creating text widgets that also work in headless mode."""

from __future__ import annotations

from typing import Any

from ui.bootstrap import TTK_VERSION, ttk

END = "end"


def create_text_widget(master: ttk.Widget, *, height: int = 10) -> Any:
    """Return a text widget compatible with the current ttk backend."""
    global END
    if TTK_VERSION == "stub":
        END = "end"
        return ttk.Text(master, height=height)
    from tkinter import END as TK_END
    from tkinter import Text as TkText

    END = TK_END
    return TkText(master, height=height)


__all__ = ["create_text_widget", "END"]
