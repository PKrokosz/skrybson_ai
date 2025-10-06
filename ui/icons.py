"""Icon utilities for Skrybson GUI."""
from __future__ import annotations

from typing import Dict, Optional

from ui.bootstrap import TTK_VERSION

try:
    from PIL import Image, ImageDraw, ImageFont, ImageTk
except ImportError:  # pragma: no cover - fallback for environments without pillow
    Image = ImageDraw = ImageFont = ImageTk = None  # type: ignore[assignment]

_ICON_CACHE: Dict[str, "ImageTk.PhotoImage"] = {}


def _font() -> Optional["ImageFont.FreeTypeFont"]:
    if ImageFont is None:
        return None
    try:
        return ImageFont.truetype("DejaVuSans.ttf", 40)
    except OSError:
        return ImageFont.load_default()


def glyph_icon(key: str, glyph: str, *, size: int = 64) -> Optional["ImageTk.PhotoImage"]:
    """Create and cache an icon rendered from a glyph."""

    cache_key = f"{key}:{size}"
    if cache_key in _ICON_CACHE:
        return _ICON_CACHE[cache_key]
    if Image is None or ImageDraw is None or ImageTk is None or TTK_VERSION == "stub":
        return None
    image = Image.new("RGBA", (size, size), (30, 30, 30, 255))
    draw = ImageDraw.Draw(image)
    font = _font()
    if font is None:
        return None
    try:
        left, top, right, bottom = draw.textbbox((0, 0), glyph, font=font)
        width = right - left
        height = bottom - top
    except AttributeError:  # pragma: no cover - Pillow < 8
        width, height = draw.textsize(glyph, font=font)
        left = top = 0
    draw.text(
        ((size - width) / 2 - left, (size - height) / 2 - top - 4),
        glyph,
        font=font,
        fill="white",
    )
    icon = ImageTk.PhotoImage(image)
    _ICON_CACHE[cache_key] = icon
    return icon


__all__ = ["glyph_icon"]
