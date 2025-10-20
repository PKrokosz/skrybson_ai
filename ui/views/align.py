"""Alignment view."""
from __future__ import annotations

import queue
from pathlib import Path
from tkinter import messagebox

from ui.bootstrap import ttk
from ui.services.align import (
    AlignItem,
    AlignmentMessage,
    AlignmentWorker,
    discover_alignment_candidates,
)
from ui.views.base import View
from ui.widgets.textutil import END, create_text_widget


class AlignView(View):
    view_id = "align"
    title = "Alignment"

    def build(self) -> None:
        self._candidates: list[Path] = []
        self._item_map: dict[str, Path] = {}
        self._queue: "queue.Queue[AlignmentMessage]" = queue.Queue()
        self._worker: AlignmentWorker | None = None
        self.log_widget = None

        self.main_area.columnconfigure(0, weight=1)
        self.main_area.rowconfigure(0, weight=1)
        self.main_area.rowconfigure(3, weight=1)

        ttk.Label(self.main_area, text="Pliki do alignu").grid(row=0, column=0, sticky="w")
        self.tree = ttk.Treeview(self.main_area, columns=("file", "status"), show="headings")
        self.tree.heading("file", text="Plik")
        self.tree.heading("status", text="Status")
        self.tree.column("file", width=320)
        self.tree.column("status", width=140)
        self.tree.grid(row=1, column=0, sticky="nsew")

        ttk.Label(self.main_area, text="Log").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.log_widget = create_text_widget(self.main_area, height=12)
        self.log_widget.grid(row=3, column=0, sticky="nsew")

        if self.right_panel is None:
            return

        ttk.Label(self.right_panel, text="Opcje align", font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, sticky="w"
        )
        self.align_var = ttk.BooleanVar(value=True)
        ttk.Checkbutton(self.right_panel, text="WHISPER_ALIGN", variable=self.align_var).grid(
            row=1, column=0, sticky="w"
        )
        ttk.Label(self.right_panel, text="PYANNOTE_AUTH_TOKEN").grid(
            row=2, column=0, sticky="w", pady=(12, 0)
        )
        self.token_var = ttk.StringVar(value="")
        ttk.Entry(self.right_panel, textvariable=self.token_var, show="*").grid(row=3, column=0, sticky="ew")
        self.start_button = ttk.Button(self.right_panel, text="Start", command=self._start)
        self.start_button.grid(row=4, column=0, sticky="ew", pady=6)
        self.stop_button = ttk.Button(self.right_panel, text="Stop", command=self._stop, state="disabled")
        self.stop_button.grid(row=5, column=0, sticky="ew")
        self.right_panel.columnconfigure(0, weight=1)

    def on_show(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        self._candidates.clear()
        self._item_map.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)
        output_dir = self.state.config.output_dir
        for candidate in discover_alignment_candidates(output_dir):
            iid = str(candidate.path)
            aligned = candidate.path.with_suffix(".aligned.json").exists()
            self.tree.insert("", "end", iid=iid, values=(candidate.path.name, "✔" if aligned else ""))
            self._item_map[iid] = candidate.path
            self._candidates.append(candidate.path)
        if self.log_widget:
            self.log_widget.delete("1.0", END)

    def _start(self) -> None:
        if not self.align_var.get():
            messagebox.showinfo("Align", "Opcja WHISPER_ALIGN jest wyłączona.")
            return
        if self._worker and self._worker.is_running:
            messagebox.showinfo("Align", "Proces align jest już uruchomiony.")
            return
        selected = self.tree.selection()
        if selected:
            paths = [self._item_map[item] for item in selected if item in self._item_map]
        else:
            paths = list(self._candidates)
        if not paths:
            messagebox.showwarning("Align", "Brak kandydatów do alignu.")
            return
        align_items = [AlignItem(path=path) for path in paths]
        self._queue = queue.Queue()
        self._worker = AlignmentWorker(
            align_items,
            recordings_dir=self.state.config.recordings_dir,
            output_dir=self.state.config.output_dir,
            queue_=self._queue,
            diarization_token=self.token_var.get().strip() or None,
        )
        self._worker.start()
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._append_log("info", "Uruchomiono proces align.")
        self.after(150, self._poll_queue)

    def _stop(self) -> None:
        if not self._worker:
            return
        self._worker.stop()
        self._append_log("info", "Wysłano sygnał zatrzymania align.")

    def _poll_queue(self) -> None:
        while True:
            try:
                message = self._queue.get_nowait()
            except queue.Empty:
                break
            self._handle_message(message)
        if self._worker and self._worker.is_running:
            self.after(200, self._poll_queue)
        else:
            self._worker = None
            self.start_button.configure(state="normal")
            self.stop_button.configure(state="disabled")

    def _handle_message(self, message: AlignmentMessage) -> None:
        if message.level == "state":
            if message.message == "done":
                self._append_log("info", "Zakończono align.")
            elif message.message == "stopped":
                self._append_log("warn", "Align zatrzymany przez użytkownika.")
            return

        candidate_path = message.candidate
        if candidate_path is not None:
            iid = str(candidate_path)
            if message.level == "success":
                self.tree.set(iid, "status", "✔")
            elif message.level == "error":
                self.tree.set(iid, "status", "✖")

        normalized_level = {
            "success": "info",
            "warning": "warn",
            "error": "error",
        }.get(message.level, message.level)
        suffix = f" ({candidate_path.name})" if candidate_path else ""
        self._append_log(normalized_level, f"{message.message}{suffix}")

    def _append_log(self, level: str, text: str) -> None:
        if not self.log_widget:
            return
        prefix = {
            "info": "[info]",
            "warn": "[warn]",
            "error": "[error]",
        }.get(level, "[info]")
        self.log_widget.insert(END, f"{prefix} {text}\n")
        self.log_widget.see(END)


__all__ = ["AlignView"]
