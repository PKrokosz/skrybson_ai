"""Fallback, headless-compatible subset of ttkbootstrap used for tests.

This module implements a tiny portion of the ttkbootstrap API that is
sufficient for running the Skrybson UI logic in environments without a
Tk display server.  It intentionally mirrors the public names that are
used throughout the project (Window, Frame, Button, Treeview, etc.) and
implements enough behaviour for unit and smoke tests.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Optional

__all__ = [
    "Window",
    "Frame",
    "Widget",
    "Button",
    "Label",
    "Entry",
    "Treeview",
    "Notebook",
    "Combobox",
    "Checkbutton",
    "Radiobutton",
    "Separator",
    "Text",
    "BooleanVar",
    "IntVar",
    "StringVar",
    "ToolTip",
]


class Variable:
    """Minimal tkinter-like variable."""

    def __init__(self, value: Any = None) -> None:
        self._value = value

    def get(self) -> Any:
        return self._value

    def set(self, value: Any) -> None:
        self._value = value


class BooleanVar(Variable):
    def __init__(self, value: bool = False) -> None:
        super().__init__(bool(value))

    def set(self, value: Any) -> None:  # type: ignore[override]
        super().set(bool(value))


class IntVar(Variable):
    def __init__(self, value: int = 0) -> None:
        super().__init__(int(value))

    def set(self, value: Any) -> None:  # type: ignore[override]
        super().set(int(value))


class StringVar(Variable):
    def __init__(self, value: str = "") -> None:
        super().__init__(str(value))

    def set(self, value: Any) -> None:  # type: ignore[override]
        super().set(str(value))


class Widget:
    """Base widget that mimics Tk's geometry management API."""

    def __init__(self, master: Optional["Widget"] = None, **kwargs: Any) -> None:
        self.master = master
        self.children: list[Widget] = []
        self._options: dict[str, Any] = dict(kwargs)
        self._grid: dict[str, Any] | None = None
        if master is not None:
            master.children.append(self)

    def grid(self, **kwargs: Any) -> None:
        self._grid = dict(kwargs)

    def grid_remove(self) -> None:
        self._grid = None

    def pack(self, **kwargs: Any) -> None:
        self._options.setdefault("pack", {}).update(kwargs)

    def grid_info(self) -> dict[str, Any] | None:
        return self._grid

    def configure(self, **kwargs: Any) -> None:
        self._options.update(kwargs)

    config = configure

    def columnconfigure(self, index: int, weight: int) -> None:  # noqa: D401 - API parity
        self._options.setdefault("columnconfigure", {})[index] = weight

    def rowconfigure(self, index: int, weight: int) -> None:
        self._options.setdefault("rowconfigure", {})[index] = weight

    def bind(self, sequence: str, func: Callable[..., Any]) -> None:
        self._options.setdefault("bindings", {})[sequence] = func

    def after(self, ms: int, func: Callable[..., Any], *args: Any) -> None:
        func(*args)

    def destroy(self) -> None:
        for child in list(self.children):
            child.destroy()
        self.children.clear()

    # Tk clipboard API -------------------------------------------------
    def clipboard_clear(self) -> None:
        self._options["clipboard"] = ""

    def clipboard_append(self, text: str) -> None:
        self._options["clipboard"] = text


class Frame(Widget):
    pass


class Window(Frame):
    """Simplified stand-in for ``ttkbootstrap.Window``."""

    def __init__(self, *args: Any, themename: str | None = None, **kwargs: Any) -> None:
        super().__init__(None, themename=themename, **kwargs)
        self._title = ""
        self._geometry = ""

    def title(self, value: str) -> None:
        self._title = value

    def geometry(self, value: str) -> None:
        self._geometry = value

    def mainloop(self) -> None:
        # No-op in headless mode.
        return None


class Text(Widget):
    """Very small text buffer used by the log window."""

    def __init__(self, master: Optional[Widget] = None, height: int = 0, **kwargs: Any) -> None:
        super().__init__(master, height=height, **kwargs)
        self._buffer: list[str] = []

    def insert(self, index: str, text: str) -> None:
        if index in {"1.0", "0.0"}:
            self._buffer.insert(0, text)
        else:
            self._buffer.append(text)

    def see(self, index: str) -> None:
        # No viewport in headless mode.
        return None

    def getvalue(self) -> str:
        return "".join(self._buffer)

    def delete(self, start: str, end: str) -> None:
        self._buffer.clear()

    def get(self, start: str, end: str) -> str:
        return "".join(self._buffer)


class Button(Widget):
    def __init__(self, master: Optional[Widget] = None, command: Callable[[], Any] | None = None, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self.command = command

    def invoke(self) -> None:
        if self.command:
            self.command()


class Label(Widget):
    pass


class Entry(Widget):
    def __init__(self, master: Optional[Widget] = None, textvariable: Variable | None = None, **kwargs: Any) -> None:
        super().__init__(master, textvariable=textvariable, **kwargs)
        self.textvariable = textvariable or StringVar()

    def get(self) -> str:
        return str(self.textvariable.get())

    def delete(self, start: int, end: int) -> None:
        self.textvariable.set("")

    def insert(self, index: int, value: str) -> None:
        self.textvariable.set(value)


class Checkbutton(Widget):
    def __init__(self, master: Optional[Widget] = None, variable: Variable | None = None, **kwargs: Any) -> None:
        super().__init__(master, variable=variable, **kwargs)
        self.variable = variable or BooleanVar()


class Radiobutton(Checkbutton):
    pass


class Separator(Widget):
    pass


class Combobox(Widget):
    def __init__(self, master: Optional[Widget] = None, values: Iterable[str] | None = None, **kwargs: Any) -> None:
        super().__init__(master, values=list(values or []), **kwargs)
        self.values = list(values or [])
        self.current_index: int | None = None
        self.variable = StringVar()

    def current(self, index: int) -> None:
        self.current_index = index
        if 0 <= index < len(self.values):
            self.variable.set(self.values[index])

    def get(self) -> str:
        return self.variable.get()


@dataclass
class _TreeItem:
    iid: str
    values: tuple[Any, ...]


class Treeview(Widget):
    def __init__(self, master: Optional[Widget] = None, columns: Iterable[str] = (), show: str | None = None, **kwargs: Any) -> None:
        super().__init__(master, columns=list(columns), show=show, **kwargs)
        self.columns = list(columns)
        self._headings: dict[str, str] = {}
        self._items: dict[str, _TreeItem] = {}
        self._order: list[str] = []
        self._tags: dict[str, dict[str, Any]] = {}
        self._selection: list[str] = []

    def heading(self, column: str, text: str, **kwargs: Any) -> None:
        self._headings[column] = text
        if kwargs:
            self._options.setdefault("column_heading", {})[column] = kwargs

    def column(self, column: str, width: int | None = None, **kwargs: Any) -> None:
        options = {"width": width}
        options.update(kwargs)
        self._options.setdefault("column_width", {})[column] = options

    def insert(
        self,
        parent: str,
        index: str,
        iid: str | None = None,
        values: Iterable[Any] = (),
        tags: Iterable[str] | None = None,
    ) -> str:
        new_iid = iid or f"I{len(self._items) + 1}"
        item = _TreeItem(new_iid, tuple(values))
        self._items[new_iid] = item
        self._order.append(new_iid)
        if tags:
            self._options.setdefault("item_tags", {})[new_iid] = tuple(tags)
        return new_iid

    def delete(self, item: str) -> None:
        if item in self._items:
            del self._items[item]
            self._order = [iid for iid in self._order if iid != item]

    def get_children(self) -> list[str]:
        return list(self._order)

    def tag_configure(self, tag: str, **kwargs: Any) -> None:
        self._tags[tag] = kwargs

    def selection(self) -> tuple[str, ...]:
        return tuple(self._selection)

    def selection_set(self, items: Iterable[str]) -> None:
        self._selection = list(items)


class Notebook(Widget):
    def __init__(self, master: Optional[Widget] = None, **kwargs: Any) -> None:
        super().__init__(master, **kwargs)
        self._tabs: list[tuple[Widget, str]] = []

    def add(self, child: Widget, text: str) -> None:
        self._tabs.append((child, text))


class ToolTip:
    """Simplified tooltip implementation."""

    def __init__(self, widget: Widget, text: str) -> None:
        self.widget = widget
        self.text = text


WidgetType = Widget
