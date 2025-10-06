"""Navigation sidebar."""
from __future__ import annotations

from typing import Callable, Dict

import ttkbootstrap as ttk
from ttkbootstrap.tooltip import ToolTip
from ui.icons import glyph_icon


class Sidebar(ttk.Frame):
    """Icon based navigation sidebar."""

    def __init__(self, master: ttk.Widget, *, on_navigate: Callable[[str], None]) -> None:
        super().__init__(master, width=80)
        self.on_navigate = on_navigate
        self.configure(padding=10)
        self.buttons: Dict[str, ttk.Button] = {}
        self._build()

    def _build(self) -> None:
        sections = {
            "sessions": "📁",
            "transcribe": "🎙️",
            "align": "📐",
            "results": "📝",
            "bench": "⚙️",
            "logs": "🧾",
            "settings": "⚙",
        }
        for index, (view_id, glyph) in enumerate(sections.items()):
            icon = glyph_icon(view_id, glyph)
            button = ttk.Button(
                self,
                image=icon,
                text=view_id.title(),
                compound="top" if icon else "none",
                width=80,
                command=lambda v=view_id: self.on_navigate(v),
                bootstyle="dark-outline",
            )
            if icon:
                button.image = icon  # type: ignore[attr-defined]
            button.grid(row=index, column=0, sticky="ew", pady=6)
            ToolTip(button, text=view_id.title())
            self.buttons[view_id] = button
        self.columnconfigure(0, weight=1)

    def set_active(self, view_id: str) -> None:
        for vid, button in self.buttons.items():
            if vid == view_id:
                button.configure(bootstyle="success")
            else:
                button.configure(bootstyle="dark-outline")


__all__ = ["Sidebar"]
