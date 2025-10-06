"""Settings view."""
from __future__ import annotations

import ttkbootstrap as ttk
from ui.views.base import View


class SettingsView(View):
    view_id = "settings"
    title = "Ustawienia"

    def build(self) -> None:
        notebook = ttk.Notebook(self.main_area)
        notebook.grid(row=0, column=0, sticky="nsew")
        self.main_area.rowconfigure(0, weight=1)
        self.main_area.columnconfigure(0, weight=1)

        self.paths_frame = ttk.Frame(notebook, padding=10)
        self.profile_frame = ttk.Frame(notebook, padding=10)
        self.deps_frame = ttk.Frame(notebook, padding=10)
        self.security_frame = ttk.Frame(notebook, padding=10)
        self.experiments_frame = ttk.Frame(notebook, padding=10)

        notebook.add(self.paths_frame, text="Ścieżki")
        notebook.add(self.profile_frame, text="Profile")
        notebook.add(self.deps_frame, text="Zależności")
        notebook.add(self.security_frame, text="Bezpieczeństwo")
        notebook.add(self.experiments_frame, text="Eksperymenty")

        ttk.Label(self.paths_frame, text=f"Recordings: {self.state.config.recordings_dir}").grid(row=0, column=0, sticky="w")
        ttk.Label(self.paths_frame, text=f"Outputs: {self.state.config.output_dir}").grid(row=1, column=0, sticky="w")

        ttk.Label(self.profile_frame, text="Profile dostępne:").grid(row=0, column=0, sticky="w")
        for index, profile in enumerate(self.state.config.profiles, start=1):
            ttk.Label(self.profile_frame, text=f"{profile.name}: {profile.description}").grid(row=index, column=0, sticky="w")

        ttk.Label(self.deps_frame, text="Wersje").grid(row=0, column=0, sticky="w")
        ttk.Label(self.deps_frame, text="Python 3.11").grid(row=1, column=0, sticky="w")
        ttk.Button(self.deps_frame, text="Test GPU", command=self._test_gpu).grid(row=2, column=0, sticky="w")

        ttk.Label(self.security_frame, text=".env informacje: brak zmian").grid(row=0, column=0, sticky="w")
        ttk.Label(self.experiments_frame, text="Brak aktywnych eksperymentów").grid(row=0, column=0, sticky="w")

    def _test_gpu(self) -> None:
        messagebox = __import__("tkinter.messagebox", fromlist=["messagebox"]).messagebox
        messagebox.showinfo("GPU", "Test GPU niezaimplementowany")


__all__ = ["SettingsView"]
