"""Benchmark view."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

from ui.bootstrap import ttk
from ui.views.base import View


class BenchView(View):
    view_id = "bench"
    title = "Bench"

    def build(self) -> None:
        self.main_area.columnconfigure(0, weight=1)
        ttk.Label(self.main_area, text="Ostatnie benchmarki").grid(row=0, column=0, sticky="w")
        self.tree = ttk.Treeview(
            self.main_area,
            columns=("model", "device", "compute", "duration"),
            show="headings",
        )
        for col in ("model", "device", "compute", "duration"):
            self.tree.heading(col, text=col.title())
            self.tree.column(col, width=140)
        self.tree.grid(row=1, column=0, sticky="nsew")
        self.main_area.rowconfigure(1, weight=1)
        self._populate()

        if self.right_panel is None:
            return
        self.right_panel.columnconfigure(0, weight=1)
        ttk.Button(self.right_panel, text="Uruchom benchmark", command=self._run).grid(row=0, column=0, sticky="ew")
        self.status = ttk.Label(self.right_panel, text="Nie uruchomiono")
        self.status.grid(row=1, column=0, sticky="w", pady=(8, 0))

    def _populate(self) -> None:
        self.tree.insert("", "end", values=("large-v3", "cuda", "float16", "15.2s"))
        self.tree.insert("", "end", values=("large-v3", "cpu", "int8", "64.0s"))

    def _run(self) -> None:
        report_path = Path("docs/bench.md")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(f"# Bench report\n\nUruchomiono: {datetime.now().isoformat()}\n")
        self.status.configure(text=f"Zapisano raport do {report_path}")


__all__ = ["BenchView"]
