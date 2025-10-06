"""Sessions view."""
from __future__ import annotations

import json

from ui.bootstrap import ttk
from ui.services.sessions import SessionSummary, discover_sessions
from ui.state import AppState
from ui.views.base import View
from ui.widgets.dialogs import ask_directory
from ui.widgets.tables import striped_treeview
from ui.widgets.textutil import END, create_text_widget


class SessionsView(View):
    view_id = "sessions"
    title = "Sesje"

    def __init__(self, master: ttk.Widget, state: AppState) -> None:
        self._sessions: list[SessionSummary] = []
        super().__init__(master, state)

    def build(self) -> None:
        self.main_area.columnconfigure(0, weight=1)
        self.main_area.rowconfigure(0, weight=1)
        self.tree = striped_treeview(
            self.main_area,
            columns=("id", "date", "channel", "users", "duration", "size", "status"),
            headings={
                "id": "Session ID",
                "date": "Data",
                "channel": "Kanał",
                "users": "Użytkownicy",
                "duration": "Długość",
                "size": "Rozmiar",
                "status": "Status",
            },
        )
        self.tree.grid(row=0, column=0, sticky="nsew")
        self.tree.bind("<<TreeviewSelect>>", self._on_select)
        self.preview = create_text_widget(self.main_area, height=12)
        self.preview.grid(row=1, column=0, sticky="nsew", pady=(8, 0))
        self.main_area.rowconfigure(1, weight=1)

        if self.right_panel is None:
            return
        self.right_panel.columnconfigure(0, weight=1)
        ttk.Label(self.right_panel, text="Ścieżki", font=("TkDefaultFont", 12, "bold")).grid(row=0, column=0, sticky="w")
        ttk.Button(
            self.right_panel,
            text="Wybierz RECORDINGS_DIR",
            command=self._pick_recordings,
            bootstyle="outline",
        ).grid(row=1, column=0, sticky="ew", pady=4)
        ttk.Button(
            self.right_panel,
            text="Wybierz OUTPUT_DIR",
            command=self._pick_output,
            bootstyle="outline",
        ).grid(row=2, column=0, sticky="ew", pady=4)
        ttk.Label(self.right_panel, text="Profil").grid(row=3, column=0, sticky="w", pady=(12, 0))
        self.profile_var = ttk.StringVar(value=self.state.config.active_profile)
        self.profile_combo = ttk.Combobox(
            self.right_panel,
            textvariable=self.profile_var,
            values=[profile.name for profile in self.state.config.profiles],
            state="readonly",
        )
        self.profile_combo.grid(row=4, column=0, sticky="ew")
        self.profile_combo.bind("<<ComboboxSelected>>", self._on_profile_change)
        ttk.Button(
            self.right_panel,
            text="Sprawdź spójność manifestu",
            command=self._check_manifest,
            bootstyle="secondary",
        ).grid(row=5, column=0, sticky="ew", pady=(12, 0))

    def on_show(self) -> None:
        self.refresh()

    def refresh(self) -> None:
        config = self.state.config
        self._sessions = discover_sessions(config.recordings_dir)
        for item in self.tree.get_children():
            self.tree.delete(item)
        for index, session in enumerate(self._sessions):
            self.tree.insert(
                "",
                "end",
                iid=session.session_id,
                values=(
                    session.session_id,
                    session.created_at.isoformat() if session.created_at else "-",
                    session.channel or "-",
                    ", ".join(session.users) or "-",
                    f"{session.duration:.1f}" if session.duration else "-",
                    f"{session.size_bytes / 1024:.1f} KB",
                    session.status_glyph,
                ),
                tags=("odd" if index % 2 else "even",),
            )
        self.preview.delete("1.0", END)

    def _on_select(self, event: object) -> None:
        selection = self.tree.selection()
        if not selection:
            return
        session_id = selection[0]
        session = next((s for s in self._sessions if s.session_id == session_id), None)
        if not session:
            return
        self.preview.delete("1.0", END)
        if session.manifest:
            formatted = json.dumps(session.manifest, indent=2, ensure_ascii=False)
            self.preview.insert("1.0", formatted)
        else:
            self.preview.insert("1.0", "Brak manifestu")

    def _pick_recordings(self) -> None:
        new_dir = ask_directory(self.state.config.recordings_dir)
        if new_dir:
            self.state.update_paths(recordings_dir=new_dir)
            self.refresh()

    def _pick_output(self) -> None:
        new_dir = ask_directory(self.state.config.output_dir)
        if new_dir:
            self.state.update_paths(output_dir=new_dir)

    def _on_profile_change(self, event: object) -> None:
        self.state.set_active_profile(self.profile_var.get())

    def _check_manifest(self) -> None:
        self.preview.insert("end", "\n[info] Sprawdzanie manifestu niezaimplementowane")


__all__ = ["SessionsView"]
