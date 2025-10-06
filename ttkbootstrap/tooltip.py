"""Simplified tooltip stub."""
from __future__ import annotations

from tkinter import Toplevel


class ToolTip:
    def __init__(self, widget, text: str | None = None) -> None:  # noqa: ANN001
        self.widget = widget
        self.text = text or ""
        self.tipwindow: Toplevel | None = None

    def show(self) -> None:
        pass

    def hide(self) -> None:
        if self.tipwindow is not None:
            self.tipwindow.destroy()
            self.tipwindow = None


__all__ = ["ToolTip"]
