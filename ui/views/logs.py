"""Logs view."""
from __future__ import annotations

from pathlib import Path

from ui.bootstrap import ttk
from ui.views.base import View
from ui.widgets.textutil import END, create_text_widget


class LogsView(View):
    view_id = "logs"
    title = "Log"

    def build(self) -> None:
        self.main_area.columnconfigure(0, weight=1)
        self.main_area.rowconfigure(1, weight=1)
        ttk.Label(self.main_area, text="Filtry").grid(row=0, column=0, sticky="w")
        self.filter_var = ttk.StringVar(value="info")
        filter_frame = ttk.Frame(self.main_area)
        filter_frame.grid(row=0, column=1, sticky="e")
        for value in ("narration", "info", "warning", "error"):
            ttk.Radiobutton(filter_frame, text=value.title(), value=value, variable=self.filter_var).pack(side="left")
        self.text = create_text_widget(self.main_area)
        self.text.grid(row=1, column=0, columnspan=2, sticky="nsew")
        ttk.Button(self.main_area, text="Kopiuj", command=self._copy).grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Button(self.main_area, text="Zapisz log", command=self._save).grid(row=2, column=1, sticky="e", pady=(6, 0))

    def on_show(self) -> None:
        self._refresh_logs()

    def _refresh_logs(self) -> None:
        self.text.delete("1.0", END)
        for line in self.state.pop_logs():
            self.text.insert("end", line + "\n")
        self.text.see(END)

    def _copy(self) -> None:
        content = self.text.get("1.0", "end-1c")
        self.clipboard_clear()
        self.clipboard_append(content)

    def _save(self) -> None:
        path = Path("logs.txt")
        path.write_text(self.text.get("1.0", "end-1c"), encoding="utf8")
        messagebox = __import__("tkinter.messagebox", fromlist=["messagebox"]).messagebox
        messagebox.showinfo("Log", f"Zapisano do {path}")


__all__ = ["LogsView"]
