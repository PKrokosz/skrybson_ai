from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DISPLAY"),
    reason="Tkinter requires DISPLAY in CI",
)


def test_app_constructs() -> None:
    from ui.app import SkrybsonApp

    app = SkrybsonApp()
    app.update()
    app.destroy()
