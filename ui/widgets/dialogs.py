"""Dialog helpers."""
from __future__ import annotations

import tkinter.filedialog as filedialog
from pathlib import Path
from typing import Optional


def ask_directory(initialdir: Path | None = None) -> Optional[Path]:
    path = filedialog.askdirectory(initialdir=str(initialdir) if initialdir else None)
    if not path:
        return None
    return Path(path)


__all__ = ["ask_directory"]
