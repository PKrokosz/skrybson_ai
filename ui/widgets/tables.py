"""Table helpers."""
from __future__ import annotations

import ttkbootstrap as ttk


def striped_treeview(master: ttk.Widget, columns: tuple[str, ...], headings: dict[str, str]) -> ttk.Treeview:
    tree = ttk.Treeview(master, columns=columns, show="headings")
    for name in columns:
        tree.heading(name, text=headings.get(name, name.title()))
        tree.column(name, width=140, anchor="w")
    tree.tag_configure("odd", background="#2a2a2a")
    return tree


__all__ = ["striped_treeview"]
