"""Minimal shim of ttkbootstrap for headless testing."""
from __future__ import annotations

from tkinter import BooleanVar, IntVar, StringVar, Tk
from tkinter.ttk import *  # noqa: F403,F401


class Window(Tk):
    def __init__(self, *args: object, themename: str | None = None, **kwargs: object) -> None:
        super().__init__(*args, **kwargs)
        self.withdraw()


__all__ = [
    "Window",
    "BooleanVar",
    "StringVar",
    "IntVar",
]
