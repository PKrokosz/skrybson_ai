"""Transcription view."""
from __future__ import annotations

from pathlib import Path
from tkinter import messagebox
from typing import Dict

from ui.bootstrap import ttk
from ui.widgets.textutil import END, create_text_widget
from ui.services.tasks import TaskManager
from ui.state import AppState
from ui.views.base import View


class TranscribeView(View):
    view_id = "transcribe"
    title = "Transkrypcja"

    def __init__(self, master: ttk.Widget, state: AppState, task_manager: TaskManager | None = None) -> None:
        self.task_manager = task_manager or TaskManager()
        self._env_vars: Dict[str, ttk.StringVar] = {}
        self.log_widget: object | None = None
        super().__init__(master, state)

    def build(self) -> None:
        self.main_area.columnconfigure(0, weight=1)
        self.main_area.rowconfigure(0, weight=1)
        ttk.Label(self.main_area, text="Kolejka zadań").grid(row=0, column=0, sticky="w")
        self.queue_tree = ttk.Treeview(
            self.main_area,
            columns=("file", "user", "status"),
            show="headings",
            height=8,
        )
        for col in ("file", "user", "status"):
            self.queue_tree.heading(col, text=col.title())
            self.queue_tree.column(col, width=200)
        self.queue_tree.grid(row=1, column=0, sticky="nsew")
        self.main_area.rowconfigure(1, weight=1)
        ttk.Label(self.main_area, text="Log").grid(row=2, column=0, sticky="w", pady=(8, 0))
        self.log_widget = create_text_widget(self.main_area, height=14)
        if self.log_widget:
            self.log_widget.grid(row=3, column=0, sticky="nsew")
        self.main_area.rowconfigure(3, weight=1)

        if self.right_panel is None:
            return
        self.right_panel.columnconfigure(0, weight=1)
        ttk.Label(self.right_panel, text="Parametry", font=("TkDefaultFont", 12, "bold")).grid(row=0, column=0, sticky="w")
        params = [
            "WHISPER_MODEL",
            "WHISPER_DEVICE",
            "WHISPER_COMPUTE",
            "WHISPER_SEGMENT_BEAM",
            "WHISPER_LANG",
            "WHISPER_VAD",
            "WHISPER_VAD_MIN_SILENCE_MS",
            "WHISPER_VAD_SPEECH_PAD_MS",
            "SANITIZE_LOWER_NOISE",
            "WHISPER_ALIGN",
        ]
        for index, key in enumerate(params, start=1):
            ttk.Label(self.right_panel, text=key).grid(row=index, column=0, sticky="w")
            var = ttk.StringVar(value=self.state.get_active_profile().defaults.get(key, ""))
            entry = ttk.Entry(self.right_panel, textvariable=var)
            entry.grid(row=index, column=1, sticky="ew", padx=(8, 0))
            self._env_vars[key] = var
        self.right_panel.columnconfigure(1, weight=1)
        ttk.Button(
            self.right_panel,
            text="Dry-Run (Mock)",
            command=lambda: self._start_task(dry_run=True),
            bootstyle="secondary",
        ).grid(row=len(params) + 2, column=0, columnspan=2, sticky="ew", pady=(12, 0))
        ttk.Button(
            self.right_panel,
            text="Start",
            command=self._start_task,
            bootstyle="success",
        ).grid(row=len(params) + 3, column=0, columnspan=2, sticky="ew", pady=4)
        ttk.Button(
            self.right_panel,
            text="Stop",
            command=self._stop_task,
            bootstyle="danger",
        ).grid(row=len(params) + 4, column=0, columnspan=2, sticky="ew")

    def _start_task(self, dry_run: bool = False) -> None:
        env = {key: var.get() for key, var in self._env_vars.items() if var.get()}
        config = self.state.config
        env.update(
            {
                "RECORDINGS_DIR": str(config.recordings_dir),
                "OUTPUT_DIR": str(config.output_dir),
            },
        )
        session_dir = self._detect_session_dir(config.recordings_dir)
        if session_dir:
            env["SESSION_DIR"] = str(session_dir)
        try:
            self.task_manager.start(env, dry_run=dry_run)
        except RuntimeError as exc:
            if self.log_widget:
                self.log_widget.insert(END, f"\n[error] {exc}")
                self.log_widget.see(END)
            messagebox.showerror("Transkrypcja", str(exc))
            return
        self.after(100, self._poll_logs)
        if self.log_widget:
            self.log_widget.insert(END, f"\n[info] Uruchomiono transkrypcję (dry_run={dry_run})")
            self.log_widget.see(END)

    def _stop_task(self) -> None:
        self.task_manager.stop()
        if self.log_widget:
            self.log_widget.insert(END, "\n[info] Zatrzymano transkrypcję")
            self.log_widget.see(END)

    def _poll_logs(self) -> None:
        task = self.task_manager.get_task()
        if not task:
            return
        for message in task.logs():
            if self.log_widget:
                self.log_widget.insert(END, f"\n[{message.stream}] {message.content}")
                self.log_widget.see(END)
        if task.is_running:
            self.after(200, self._poll_logs)

    def _detect_session_dir(self, recordings_dir: Path) -> Path | None:
        sessions = list(recordings_dir.glob("*/"))
        if not sessions:
            return None
        return sessions[-1]


__all__ = ["TranscribeView"]
