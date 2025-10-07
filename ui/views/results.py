"""Results view."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from tkinter import messagebox

from ui.bootstrap import ttk
from ui.views.base import View
from ui.widgets.textutil import END, create_text_widget
from ui.widgets.dialogs import ask_directory


class ResultsView(View):
    view_id = "results"
    title = "Wyniki"

    def build(self) -> None:
        self.notebook = ttk.Notebook(self.main_area)
        self.notebook.grid(row=0, column=0, sticky="nsew")
        self.main_area.rowconfigure(0, weight=1)
        self.main_area.columnconfigure(0, weight=1)

        self.srt_text = create_text_widget(self.notebook)
        self.vtt_text = create_text_widget(self.notebook)
        self.json_tree = ttk.Treeview(self.notebook, columns=("speaker", "start", "end", "text"), show="headings")
        for col in ("speaker", "start", "end", "text"):
            self.json_tree.heading(col, text=col.title())
            self.json_tree.column(col, width=160)
        self.notebook.add(self.srt_text, text="SRT")
        self.notebook.add(self.vtt_text, text="VTT")
        self.notebook.add(self.json_tree, text="JSON")

        if self.right_panel is None:
            return
        self.right_panel.columnconfigure(0, weight=1)
        ttk.Label(self.right_panel, text="Eksport", font=("TkDefaultFont", 12, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Button(self.right_panel, text="Eksportuj", command=self._export).grid(row=1, column=0, sticky="ew")
        ttk.Label(self.right_panel, text="Statystyki", font=("TkDefaultFont", 12, "bold")).grid(row=2, column=0, sticky="w", pady=(12, 0))
        self.stats_label = ttk.Label(self.right_panel, text="Brak danych")
        self.stats_label.grid(row=3, column=0, sticky="w")

    def on_show(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        output_dir = self.state.config.output_dir
        self._load_text_files(output_dir, "srt", self.srt_text)
        self._load_text_files(output_dir, "vtt", self.vtt_text)
        self._load_json(output_dir)

    def _load_text_files(self, directory: Path, extension: str, widget: object) -> None:
        widget.delete("1.0", END)
        files = sorted(directory.glob(f"*.{extension}"))
        for path in files:
            widget.insert(END, f"## {path.name}\n")
            widget.insert(END, path.read_text(encoding="utf8"))
            widget.insert(END, "\n\n")
        if not files:
            widget.insert(END, "Brak plików")

    def _load_json(self, directory: Path) -> None:
        for item in self.json_tree.get_children():
            self.json_tree.delete(item)
        stats_segments = 0
        for json_path in sorted(directory.glob("*.json")):
            try:
                payload = json.loads(json_path.read_text(encoding="utf8"))
            except json.JSONDecodeError:
                continue
            segments = payload.get("segments", [])
            for segment in segments:
                stats_segments += 1
                self.json_tree.insert(
                    "",
                    "end",
                    values=(
                        segment.get("speaker", "-"),
                        f"{segment.get('start', 0):.2f}",
                        f"{segment.get('end', 0):.2f}",
                        segment.get("text", ""),
                    ),
                )
        self.stats_label.configure(text=f"Segmentów: {stats_segments}")

    def _export(self) -> None:
        destination = ask_directory(self.state.config.output_dir)
        if not destination:
            return
        output_dir = self.state.config.output_dir
        exported = 0
        errors: list[str] = []
        for path in output_dir.rglob("*"):
            if not path.is_file():
                continue
            if path.suffix.lower() not in {".json", ".srt", ".vtt"}:
                continue
            target = destination / path.relative_to(output_dir)
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(path, target)
                exported += 1
            except OSError as exc:
                errors.append(f"{path.name}: {exc}")
        if errors:
            messagebox.showerror(
                "Eksport",
                "\n".join(["Niektóre pliki nie zostały skopiowane:"] + errors[:10]),
            )
        elif exported == 0:
            messagebox.showwarning("Eksport", "Brak plików do eksportu w katalogu wynikowym.")
        else:
            messagebox.showinfo("Eksport", f"Wyeksportowano {exported} plików do {destination}")


__all__ = ["ResultsView"]
