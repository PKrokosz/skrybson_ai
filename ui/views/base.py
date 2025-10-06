"""Base view infrastructure for the Skrybson GUI."""
from __future__ import annotations

from typing import Optional

import ttkbootstrap as ttk
from ui.state import AppState


class View(ttk.Frame):
    """Base class for all routed views."""

    view_id: str = "base"
    title: str = ""

    def __init__(self, master: ttk.Widget, state: AppState) -> None:
        super().__init__(master)
        self.state = state
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.main_area = ttk.Frame(self)
        self.main_area.grid(row=0, column=0, sticky="nsew")
        self.right_panel: Optional[ttk.Frame] = None
        if self.has_right_panel:
            self.right_panel = ttk.Frame(self)
            self.right_panel.grid(row=0, column=1, sticky="nsew")
            self.columnconfigure(1, weight=0)
        self.build()

    @property
    def has_right_panel(self) -> bool:
        return True

    # ------------------------------------------------------------------
    # Lifecycle hooks
    # ------------------------------------------------------------------
    def build(self) -> None:  # pragma: no cover - override hook
        """Construct widgets for the view."""

    def on_show(self) -> None:  # pragma: no cover - optional hook
        """Execute logic when the view becomes active."""


__all__ = ["View"]
