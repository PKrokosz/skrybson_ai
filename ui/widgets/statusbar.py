"""Status bar widget."""
from __future__ import annotations

import ttkbootstrap as ttk


class StatusBar(ttk.Frame):
    """Simple three-zone status bar."""

    def __init__(self, master: ttk.Widget) -> None:
        super().__init__(master, padding=(10, 4))
        self.left = ttk.Label(self, text="Ready")
        self.center = ttk.Label(self, text="00:00:00")
        self.right = ttk.Label(self, text="Profile: quality@cuda")
        self.left.grid(row=0, column=0, sticky="w")
        self.center.grid(row=0, column=1)
        self.right.grid(row=0, column=2, sticky="e")
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)

    def set_status(self, text: str) -> None:
        self.left.configure(text=text)

    def set_duration(self, text: str) -> None:
        self.center.configure(text=text)

    def set_profile(self, text: str) -> None:
        self.right.configure(text=text)


__all__ = ["StatusBar"]
