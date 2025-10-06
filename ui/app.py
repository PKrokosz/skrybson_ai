"""Entry point for the Skrybson Tkinter UI."""
from __future__ import annotations

from typing import Dict

import ttkbootstrap as ttk
from ui.services.tasks import TaskManager
from ui.state import AppState
from ui.views.align import AlignView
from ui.views.base import View
from ui.views.bench import BenchView
from ui.views.logs import LogsView
from ui.views.results import ResultsView
from ui.views.sessions import SessionsView
from ui.views.settings import SettingsView
from ui.views.sidebar import Sidebar
from ui.views.transcribe import TranscribeView
from ui.widgets.statusbar import StatusBar


class SkrybsonApp(ttk.Window):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__(themename="darkly")
        self.title("Skrybson")
        self.geometry("1400x860")
        self.state = AppState()
        self.task_manager = TaskManager()
        self.state.set_active_view("sessions")

        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self.sidebar = Sidebar(self, on_navigate=self.show_view)
        self.sidebar.grid(row=0, column=0, sticky="ns")

        self.main_container = ttk.Frame(self)
        self.main_container.grid(row=0, column=1, sticky="nsew")
        self.main_container.columnconfigure(0, weight=1)
        self.main_container.rowconfigure(0, weight=1)

        self.statusbar = StatusBar(self)
        self.statusbar.grid(row=1, column=0, columnspan=3, sticky="ew")

        self.views: Dict[str, View] = {}
        self._init_views()
        self._register_shortcuts()
        self.show_view(self.state.get_active_view())

    def _init_views(self) -> None:
        view_classes: list[type] = [
            SessionsView,
            TranscribeView,
            AlignView,
            ResultsView,
            BenchView,
            LogsView,
            SettingsView,
        ]
        for view_cls in view_classes:
            if view_cls is TranscribeView:
                view = view_cls(self.main_container, self.state, task_manager=self.task_manager)
            else:
                view = view_cls(self.main_container, self.state)
            view.grid(row=0, column=0, sticky="nsew")
            view.grid_remove()
            self.views[view_cls.view_id] = view

    def _register_shortcuts(self) -> None:
        shortcuts = {
            "<Control-o>": lambda event: self.show_view("sessions"),
            "<Control-r>": lambda event: self._refresh_sessions(),
            "<Control-t>": lambda event: self.show_view("transcribe"),
            "<Control-e>": lambda event: self.show_view("results"),
            "<Control-l>": lambda event: self.show_view("logs"),
        }
        for sequence, handler in shortcuts.items():
            self.bind(sequence, handler)

    def _refresh_sessions(self) -> None:
        view = self.views.get("sessions")
        if view:
            view.refresh()

    def show_view(self, view_id: str) -> None:
        if view_id not in self.views:
            return
        for view in self.views.values():
            view.grid_remove()
        view = self.views[view_id]
        view.grid()
        view.on_show()
        self.sidebar.set_active(view_id)
        self.state.set_active_view(view_id)
        self.statusbar.set_status(f"Aktywny widok: {view.title}")
        self.statusbar.set_profile(f"Profil: {self.state.get_active_profile().name}")


def main() -> None:
    app = SkrybsonApp()
    app.mainloop()


if __name__ == "__main__":
    main()
