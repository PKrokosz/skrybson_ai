"""Form helpers."""
from __future__ import annotations

from ui.bootstrap import ttk


def labeled_entry(master: ttk.Widget, label: str, textvariable: ttk.StringVar) -> ttk.Frame:
    frame = ttk.Frame(master)
    ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w", padx=(0, 8))
    ttk.Entry(frame, textvariable=textvariable, width=32).grid(row=0, column=1, sticky="ew")
    frame.columnconfigure(1, weight=1)
    return frame


__all__ = ["labeled_entry"]
