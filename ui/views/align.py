"""Alignment view."""
from __future__ import annotations

from tkinter import messagebox

import ttkbootstrap as ttk
from ui.services.align import discover_alignment_candidates
from ui.views.base import View


class AlignView(View):
    view_id = "align"
    title = "Alignment"

    def build(self) -> None:
        self.main_area.columnconfigure(0, weight=1)
        self.main_area.rowconfigure(0, weight=1)
        ttk.Label(self.main_area, text="Pliki do alignu").grid(row=0, column=0, sticky="w")
        self.tree = ttk.Treeview(
            self.main_area,
            columns=("file", "selected"),
            show="headings",
        )
        self.tree.heading("file", text="Plik")
        self.tree.heading("selected", text="Wybrany")
        self.tree.column("file", width=300)
        self.tree.column("selected", width=80)
        self.tree.grid(row=1, column=0, sticky="nsew")

        if self.right_panel is None:
            return
        ttk.Label(self.right_panel, text="Opcje align", font=("TkDefaultFont", 12, "bold")).grid(row=0, column=0, sticky="w")
        self.align_var = ttk.BooleanVar(value=True)
        ttk.Checkbutton(self.right_panel, text="WHISPER_ALIGN", variable=self.align_var).grid(row=1, column=0, sticky="w")
        ttk.Label(self.right_panel, text="PYANNOTE_AUTH_TOKEN").grid(row=2, column=0, sticky="w", pady=(12, 0))
        self.token_var = ttk.StringVar(value="")
        ttk.Entry(self.right_panel, textvariable=self.token_var, show="*").grid(row=3, column=0, sticky="ew")
        ttk.Button(self.right_panel, text="Start", command=self._start).grid(row=4, column=0, sticky="ew", pady=6)
        ttk.Button(self.right_panel, text="Stop", command=self._stop).grid(row=5, column=0, sticky="ew")
        self.right_panel.columnconfigure(0, weight=1)

    def on_show(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for candidate in discover_alignment_candidates(self.state.config.output_dir):
            self.tree.insert("", "end", values=(candidate.path.name, "✔" if candidate.selected else ""))

    def _start(self) -> None:
        messagebox.showinfo("Align", "Stub: align start")

    def _stop(self) -> None:
        messagebox.showinfo("Align", "Stub: align stop")


__all__ = ["AlignView"]
